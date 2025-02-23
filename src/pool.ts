import process from 'node:process'
import PQueue from 'p-queue'
import EventEmitter3 from 'eventemitter3'
import { ModelInstance } from '#package/instance.js'
import { ModelConfig, InferenceParams, ModelInstanceRequest, ModelEngine } from '#package/types/index.js'
import { Logger, LogLevels, createSublogger, LogLevel } from '#package/lib/logger.js'
import { mergeAbortSignals } from '#package/lib/util.js'

export interface ModelInstanceHandle {
	instance: ModelInstance
	release: () => Promise<void>
}

interface ModelTask {
	instance: ModelInstance
	request: ModelInstanceRequest
}

type PrepareModelInstanceCallback = (instance: ModelInstance, signal?: AbortSignal) => Promise<void>

interface ModelPoolConfig {
	concurrency: number
	models: Record<string, ModelConfig>
}

export interface ModelPoolOptions {
	concurrency?: number
	models: Record<string, ModelConfig>
	log?: Logger | LogLevel
}

type ModelPoolEvent = 'ready' | 'spawn' | 'release'

export class ModelPool extends EventEmitter3<ModelPoolEvent> {
	queue: PQueue
	config: ModelPoolConfig
	instances: Record<string, ModelInstance>
	private engines?: Record<string, ModelEngine>
	private cleanupInterval?: NodeJS.Timeout
	private log: Logger
	private requestSequence: number = 0
	private pendingRequests: Set<ModelInstanceRequest> = new Set()
	private shutdownController: AbortController = new AbortController()
	private gpuLock: boolean = false // TODO could derive this from "is there any instance that has gpu=true"
	private prepareInstance?: PrepareModelInstanceCallback

	constructor(options: ModelPoolOptions, prepareInstance?: PrepareModelInstanceCallback) {
		super()
		this.log = createSublogger(options.log)
		const models: Record<string, ModelConfig> = {}
		for (const id in options.models) {
			const modelConfig = options.models[id]
			models[id] = {
				...modelConfig,
				id: modelConfig.id ?? id,
			}
		}
		const config: ModelPoolConfig = {
			concurrency: 1,
			...options,
			models,
		}
		this.queue = new PQueue({
			concurrency: config.concurrency,
		})
		this.config = config
		this.instances = {}
		this.prepareInstance = prepareInstance
	}

	// start up pool, creating instances and loading models
	async init(engines: Record<string, ModelEngine>) {
		const initPromises = []
		const modelConfigs = this.config.models
		this.engines = engines

		// making sure id is set.
		for (const modelId in modelConfigs) {
			const modelConfig = modelConfigs[modelId]
			if (!modelConfig.id) {
				modelConfig.id = modelId
			}
		}

		// prioritize initializing the first model defined that has gpu explicitly set
		// so lock cant be acquired first by another model that has gpu=auto/undefined
		const firstGpuModel = Object.entries(modelConfigs).find(
			([id, config]) => !!config.device?.gpu && config.device?.gpu !== 'auto',
		)
		if (firstGpuModel) {
			const modelConfig = modelConfigs[firstGpuModel[0]]
			const spawnPromises = this.ensureModelInstances(modelConfig)
			initPromises.push(...spawnPromises)
		}

		// then handle other models in the order they were defined
		for (const modelId in modelConfigs) {
			if (firstGpuModel && modelId === firstGpuModel[0]) {
				continue
			}
			const modelConfig = modelConfigs[modelId]
			const spawnPromises = this.ensureModelInstances(modelConfig)
			initPromises.push(...spawnPromises)
		}

		// resolve when all initial instances are loaded
		await Promise.allSettled(initPromises)
		this.emit('ready')
		this.cleanupInterval = setInterval(() => {
			this.disposeOutdatedInstances()
		}, 1000 * 60) // every minute
	}

	// see if the minInstances for a models are spawned. if not, spawn them.
	ensureModelInstances(model: ModelConfig) {
		const spawnPromises = []
		const instanceCount = model.minInstances ?? 0
		for (let i = 0; i < instanceCount; i++) {
			if (this.canSpawnInstance(model.id)) {
				const spawnPromise = this.spawnInstance(model.id)
				spawnPromises.push(spawnPromise)
			} else {
				this.log(LogLevels.warn, 'Failed to spawn min instances for', {
					model: model.id,
				})
				break
			}
		}
		return spawnPromises
	}

	async dispose() {
		this.log(LogLevels.debug, 'Disposing pool')
		clearInterval(this.cleanupInterval)
		super.removeAllListeners()
		this.queue.pause()
		this.queue.clear()
		this.shutdownController.abort()
		for (const request of this.pendingRequests) {
			request.abortController.abort()
		}
		const disposePromises: Array<Promise<void>> = []
		for (const key in this.instances) {
			const instance = this.instances[key]
			disposePromises.push(this.disposeInstance(instance))
		}
		await Promise.allSettled(disposePromises)
	}

	// disposes instances that have been idle for longer than their ttl
	private disposeOutdatedInstances() {
		const now = new Date().getTime()
		for (const key in this.instances) {
			const instance = this.instances[key]
			const instanceAge = (now - instance.lastUsed) / 1000
			const modelInstanceCount = Object.values(this.instances).filter((i) => i.modelId === instance.modelId).length
			const minInstanceCount = this.config.models[instance.modelId].minInstances ?? 0
			if (modelInstanceCount > minInstanceCount && instanceAge > instance.ttl && instance.status === 'idle') {
				this.log(LogLevels.info, 'Auto disposing instance', {
					instance: instance.id,
				})
				this.disposeInstance(instance)
			}
		}
	}

	getStatus() {
		const processingInstances = Object.values(this.instances).filter((instance) => instance.status === 'busy')
		const poolStatusInfo = {
			processing: processingInstances.length,
			pending: this.pendingRequests.size,
			instances: Object.fromEntries(
				Object.entries(this.instances).map(([key, instance]) => {
					return [
						key,
						{
							model: instance.modelId,
							status: instance.status,
							engine: instance.config.engine,
							device: instance.gpu ? 'gpu' : 'cpu',
							contextState: instance.getContextStateIdentity(),
							lastUsed: new Date(instance.lastUsed).toISOString(),
						},
					]
				}),
			),
		}
		return poolStatusInfo
	}

	// checks if another instance can be spawned for given model
	canSpawnInstance(modelId: string) {
		const modelConfig = this.config.models[modelId]
		// if the model is configured with gpu=true, interpret that as "it MUST run on gpu"
		// and prevent spawning more instances if the gpu is already locked.
		const requiresGpu = !!modelConfig.device?.gpu
		if (requiresGpu && this.gpuLock && modelConfig.device?.gpu !== 'auto') {
			this.log(LogLevels.verbose, 'Cannot spawn new instance: model requires gpu, but its already in use', {
				model: modelId,
			})
			return false
		}
		// see if we're within maxInstances limit
		const maxInstances = modelConfig.maxInstances ?? 1
		const currentInstances = Object.values(this.instances).filter((instance) => instance.modelId === modelId)
		if (currentInstances.length >= maxInstances) {
			this.log(LogLevels.verbose, 'Cannot spawn new instance: maxInstances reached', {
				model: modelId,
				curent: currentInstances.length,
				max: maxInstances,
			})
			return false
		}
		return true
	}

	private async disposeInstance(instance: ModelInstance) {
		this.log(LogLevels.debug, 'Disposing instance', {
			instance: instance.id,
		})
		await instance.dispose()
		if (instance.gpu) {
			this.gpuLock = false
		}
		delete this.instances[instance.id]
	}

	// spawns a new instance for the given model, without checking whether it's allowed
	private async spawnInstance(modelId: string, options: { signal?: AbortSignal; emit?: boolean } = {}) {
		if (!this.engines) {
			throw new Error(
				'No engines available. Make sure the pool is initialized and inferenceServer.start() or ModelPool.init() were called.',
			)
		}
		const model = this.config.models[modelId]
		const engine = this.engines[model.engine]
		if (!engine) {
			throw new Error(`Engine not found: ${model.engine}`)
		}
		const autoGpuEnabled = !!engine.autoGpu

		// if the model is configured with gpu=auto (or unset), we can use the gpu if its not locked
		const autoGpu = model.device?.gpu === undefined || model.device?.gpu === 'auto'
		let useGpu = autoGpu ? autoGpuEnabled && !this.gpuLock : false

		if (!!model.device?.gpu) {
			useGpu = true
		}

		const instance = new ModelInstance(engine, {
			...model,
			gpu: useGpu,
			log: this.log,
		})
		this.instances[instance.id] = instance

		if (useGpu) {
			this.gpuLock = true
		}
		const signals = [this.shutdownController.signal]
		if (options.signal) {
			signals.push(options.signal)
		}
		const abortSignal = mergeAbortSignals(signals)
		if (this.prepareInstance) {
			this.log(LogLevels.debug, 'Preparing instance', {
				instance: instance.id,
			})
			try {
				await this.prepareInstance(instance, abortSignal)
				instance.status = 'idle'
			} catch (error) {
				this.log(LogLevels.error, 'Error preparing instance', {
					model: modelId,
					instance: instance.id,
					error,
				})
				instance.status = 'error'
				return instance
			}
		}
		await instance.load(abortSignal)
		if (options.emit !== false) {
			this.emit('spawn', instance)
		}
		return instance
	}

	// wait to acquire a gpu instance for the given request
	private acquireGpuInstance(request: ModelInstanceRequest, signal?: AbortSignal): Promise<ModelInstance> {
		return new Promise(async (resolve, reject) => {
			// if we have an idle gpu instance and the model matches we can lock and return immediately
			const gpuInstance = Object.values(this.instances).find((instance) => instance.gpu === true)!
			if (gpuInstance.status === 'idle') {
				if (gpuInstance.modelId === request.model) {
					gpuInstance.lock(request)
					resolve(gpuInstance)
					return
				} else {
					await this.disposeInstance(gpuInstance)

					const newInstance = await this.spawnInstance(request.model, {
						emit: false,
					})
					newInstance.lock(request)
					resolve(newInstance)
					return
				}
			}

			// otherwise attach the listener and wait until gpu slot becomes available
			const listener = async (instance: ModelInstance) => {
				if (instance.gpu === true && instance.status === 'idle') {
					if (instance.matchesRequirements(request)) {
						// model matches whats needed, lock and resolve
						this.off('release', listener)
						instance.lock(request)
						resolve(instance)
					} else {
						// model doesnt match, dispose and spawn new instance
						this.off('release', listener)
						await this.disposeInstance(instance)
						const newInstance = await this.spawnInstance(request.model, {
							emit: false,
						})
						newInstance.lock(request)
						resolve(newInstance)
					}
				}
			}
			this.on('release', listener)
			if (signal) {
				signal.addEventListener('abort', () => {
					this.off('release', listener)
					reject(signal.reason)
				})
			}
		})
	}

	// wait to acquire an idle instance for the given request
	private acquireIdleInstance(request: ModelInstanceRequest, signal?: AbortSignal): Promise<ModelInstance> {
		return new Promise((resolve, reject) => {
			const listener = (instance: ModelInstance) => {
				if (instance.matchesRequirements(request) && instance.status === 'idle') {
					this.off('release', listener)
					this.off('spawn', listener)
					try {
						instance.lock(request)
						resolve(instance)
					} catch (error: any) {
						this.log(LogLevels.error, 'Error acquiring idle instance', {
							error,
						})
						reject(error)
					}
				}
			}
			this.on('spawn', listener)
			this.on('release', listener)
			if (signal) {
				signal.addEventListener('abort', () => {
					this.off('release', listener)
					this.off('spawn', listener)
					reject(signal.reason)
				})
			}
		})
	}

	// acquire an instance for the given request
	private async acquireInstance(request: ModelInstanceRequest, signal?: AbortSignal) {
		if ('messages' in request || 'prompt' in request) {
			// for text and chat completions first search for an instance that has the context ready
			for (const key in this.instances) {
				const instance = this.instances[key]
				if (
					instance.matchesRequirements(request) &&
					instance.status === 'idle' &&
					instance.matchesContextState(request)
				) {
					this.log(LogLevels.debug, 'Cache hit - reusing cached instance', {
						instance: instance.id,
						sequence: request.sequence,
					})
					instance.lock(request)
					return instance
				}
			}
			this.log(LogLevels.debug, 'Cache miss - continue acquiring model instance', { sequence: request.sequence })
		}

		// prefer an instance of the model that has no context state.
		for (const key in this.instances) {
			const instance = this.instances[key]
			if (instance.matchesRequirements(request) && instance.status === 'idle' && !instance.hasContextState()) {
				this.log(LogLevels.debug, 'Reusing idle instance without context state', {
					instance: instance.id,
					sequence: request.sequence,
				})
				instance.lock(request)
				return instance
			}
		}

		// still havent found any, see if we're allowed to spawn a new instance
		if (this.canSpawnInstance(request.model)) {
			const instance = await this.spawnInstance(request.model, {
				emit: false,
			})
			// reset the context if the request doesnt match the instances preloaded context state
			const hasInitialContextState = instance.config.initialMessages?.length || instance.config.prefix
			if (hasInitialContextState && !instance.matchesContextState(request)) {
				instance.resetContext()
			}
			this.log(LogLevels.debug, 'Spawned instance acquired', {
				instance: instance.id,
				sequence: request.sequence,
			})
			instance.lock(request)
			return instance
		}

		// if all instances have cached state, prefer the one that was used the longest time ago
		const availableInstances = Object.values(this.instances).filter(
			(instance) => instance.matchesRequirements(request) && instance.status === 'idle',
		)
		if (availableInstances.length > 0) {
			const lruInstance = availableInstances.reduce((prev, current) =>
				prev.lastUsed < current.lastUsed ? prev : current,
			)
			this.log(LogLevels.debug, 'Reusing least recently used instance', {
				instance: lruInstance.id,
				sequence: request.sequence,
			})
			lruInstance.lock(request)
			lruInstance.resetContext() // make sure we reset its cache.
			return lruInstance
		}

		const requiresGpu = this.config.models[request.model].device?.gpu === true
		if (requiresGpu && this.gpuLock) {
			const gpuInstance = Object.values(this.instances).find((instance) => instance.gpu === true)!

			if (gpuInstance.modelId !== request.model) {
				this.log(LogLevels.debug, 'GPU already in use, waiting ...', {
					sequence: request.sequence,
				})
				const instance = await this.acquireGpuInstance(request, signal)
				this.log(LogLevels.debug, 'GPU instance acquired', {
					instance: instance.id,
					sequence: request.sequence,
				})
				if (signal?.aborted) {
					instance.unlock()
					throw signal.reason
				} else {
					return instance
				}
			}
		}

		// before starting to wait, make sure sure we're not stuck with an error'd instance (and wait forever)
		// currently instances only enter error state if prepareInstance throws an error
		const errorInstance = Object.values(this.instances).find(
			(instance) => instance.modelId === request.model && instance.status === 'error',
		)
		if (errorInstance) {
			throw new Error('Instance is in error state')
		}

		// wait until an instance of our model is released or spawned
		this.log(LogLevels.debug, 'Awaiting idle instance', {
			model: request.model,
			sequence: request.sequence,
		})
		const instance = await this.acquireIdleInstance(request, signal)
		this.log(LogLevels.debug, 'Idle instance acquired', {
			instance: instance.id,
			sequence: request.sequence,
		})

		if (signal?.aborted) {
			instance.unlock()
			throw signal.reason
		} else {
			return instance
		}
	}

	private createRequestSequence() {
		if (this.requestSequence > 999999) {
			this.requestSequence = 0
		}
		return ++this.requestSequence
	}

	// requests a model instance from the pool
	async requestInstance(params: Partial<InferenceParams>, signal?: AbortSignal): Promise<ModelInstanceHandle> {
		if (this.shutdownController.signal.aborted) {
			throw new Error('Pool is disposed')
		}
		const requestSequence = this.createRequestSequence()
		const request = {
			...params as InferenceParams,
			sequence: requestSequence,
			abortController: new AbortController(),
		}
		if (!this.config.models[request.model]) {
			this.log(LogLevels.error, `Model not found: ${request.model}`)
			throw new Error(`Model not found: ${request.model}`)
		}

		this.log(LogLevels.info, 'Incoming request', {
			model: request.model,
			sequence: request.sequence,
		})

		this.pendingRequests.add(request)
		const abortSignal = mergeAbortSignals([request.abortController.signal, signal])
		abortSignal.addEventListener('abort', () => {
			this.log(LogLevels.info, 'Request aborted', {
				model: request.model,
				sequence: request.sequence,
			})
			this.pendingRequests.delete(request)
		})
		const instance = await this.acquireInstance(request, abortSignal)

		// once instance is acquired & locked, we can pass it on to the caller
		// the queue task promise will be forwarded as releaseInstance
		let resolveQueueTask: (value: ModelTask) => void = () => {}

		this.queue
			.add((): Promise<ModelTask> => {
				this.pendingRequests.delete(request)
				return new Promise((resolve, reject) => {
					resolveQueueTask = resolve
				})
			})
			.then((task) => {
				// if there are more requests waiting, prioritize handling them before spawning new instances
				// deferred to avoid AbortError when the pool is disposed right after the operation
				process.nextTick(() => {
					if (
						!this.pendingRequests.size &&
						this.canSpawnInstance(request.model) &&
						!this.shutdownController.signal.aborted
					) {
						this.spawnInstance(request.model)
					}
				})
				if (task?.instance) {
					this.emit('release', instance)
				}
			})

		// TODO what if user never calls release? automatically resolve or reject after a timeout?
		const releaseInstance = () => {
			return new Promise<void>((resolve, reject) => {
				process.nextTick(() => {
					resolveQueueTask({ instance, request })
					this.log(LogLevels.info, 'Task completed, releasing', {
						instance: instance.id,
						sequence: request.sequence,
					})
					if (instance.config.ttl === 0) {
						this.disposeInstance(instance)
					} else {
						instance.unlock()
					}
					resolve()
				})
			})
		}

		return {
			instance,
			release: releaseInstance,
		}
	}
}
