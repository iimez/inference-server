import path from 'node:path'
import { builtInEngineNames } from '#package/engines/index.js'
import { ModelPool } from '#package/pool.js'
import { ModelInstance } from '#package/instance.js'
import { ModelStore, StoredModel } from '#package/store.js'
import {
	ModelOptions,
	IncomingRequest,
	CompletionProcessingOptions,
	ChatCompletionRequest,
	EmbeddingRequest,
	ProcessingOptions,
	TextCompletionRequest,
	ModelEngine,
	ImageToTextRequest,
	SpeechToTextRequest,
	SpeechToTextProcessingOptions,
	BuiltInModelOptions,
	CustomEngineModelOptions,
	ModelConfigBase,
	TextToImageRequest,
	ImageToImageRequest,
} from '#package/types/index.js'
import { Logger, LogLevel, createSublogger, LogLevels } from '#package/lib/logger.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { validateModelOptions } from '#package/lib/validateModelOptions.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'

export interface ModelServerOptions {
	engines?: Record<string, ModelEngine>
	models: Record<string, ModelOptions>
	concurrency?: number
	cachePath?: string
	log?: Logger | LogLevel
}

export function startModelServer(options: ModelServerOptions) {
	const server = new ModelServer(options)
	server.start()
	return server
}

export class ModelServer {
	pool: ModelPool
	store: ModelStore
	engines: Record<string, ModelEngine> = {}
	log: Logger

	constructor(options: ModelServerOptions) {
		this.log = createSublogger(options.log)
		let modelsCachePath = getCacheDirPath('models')
		if (options.cachePath) {
			modelsCachePath = path.join(options.cachePath, 'models')
		}

		const modelsWithDefaults: Record<string, ModelConfigBase> = {}
		const usedEngines: Array<{ model: string; engine: string }> = []
		for (const modelId in options.models) {
			const modelOptions = options.models[modelId]
			const isBuiltIn = builtInEngineNames.includes(modelOptions.engine)
			if (isBuiltIn) {
				const builtInModelOptions = modelOptions as BuiltInModelOptions
				// can validate and resolve location of model files if a built-in engine is used
				validateModelOptions(modelId, builtInModelOptions)
				modelsWithDefaults[modelId] = {
					id: modelId,
					minInstances: 0,
					maxInstances: 1,
					modelsCachePath,
					location: resolveModelFileLocation({
						url: builtInModelOptions.url,
						filePath: builtInModelOptions.location,
						modelsCachePath,
					}),
					...builtInModelOptions,
				}
			} else {
				const customEngineOptions = modelOptions as CustomEngineModelOptions
				modelsWithDefaults[modelId] = {
					id: modelId,
					minInstances: 0,
					maxInstances: 1,
					modelsCachePath,
					...customEngineOptions,
				}
			}
			usedEngines.push({
				model: modelId,
				engine: modelOptions.engine,
			})
		}

		const customEngines = Object.keys(options.engines ?? {})
		for (const ref of usedEngines) {
			const isBuiltIn = builtInEngineNames.includes(ref.engine)
			const isCustom = customEngines.includes(ref.engine)
			if (!isBuiltIn && !isCustom) {
				throw new Error(`Engine "${ref.engine}" used by model "${ref.model}" does not exist`)
			}
			if (isCustom) {
				this.engines[ref.engine] = options.engines![ref.engine]
			}
		}

		this.store = new ModelStore({
			log: this.log,
			// TODO expose this? or remove it?
			// prepareConcurrency: 2,
			models: modelsWithDefaults,
			modelsCachePath,
		})
		this.pool = new ModelPool(
			{
				log: this.log,
				concurrency: options.concurrency ?? 1,
				models: modelsWithDefaults,
			},
			this.prepareInstance.bind(this),
		)
	}

	modelExists(modelId: string) {
		return !!this.pool.config.models[modelId]
	}

	async start() {
		const engineStartPromises = []
		// call startEngine on custom engines
		for (const [key, methods] of Object.entries(this.engines)) {
			if (methods.start) {
				engineStartPromises.push(methods.start(this))
			}
		}
		// import built-in engines
		for (const key of builtInEngineNames) {
			// skip unused engines
			const modelUsingEngine = Object.keys(this.store.models).find(
				(modelId) => this.store.models[modelId].engine === key,
			)
			if (!modelUsingEngine) {
				continue
			}
			engineStartPromises.push(
				new Promise(async (resolve, reject) => {
					try {
						const engine = await import(`./engines/${key}/engine.js`)
						this.engines[key] = engine
						resolve({
							key,
							engine,
						})
					} catch (err) {
						reject(err)
					}
				}),
			)
		}
		await Promise.all(engineStartPromises)
		await Promise.all([this.store.init(this.engines), this.pool.init(this.engines)])
	}

	async stop() {
		this.log(LogLevels.info, 'Stopping model server')
		this.pool.queue.clear()
		this.store.dispose()
		// need to make sure all tasks are canceled, waiting for idle can make stop hang
		// await this.pool.queue.onIdle() // would wait until all completions are done
		try {
			await this.pool.dispose() // might cause abort errors when there are still running tasks
		} catch (err) {
			this.log(LogLevels.error, 'Error while stopping model server', err)
		}

		this.log(LogLevels.debug, 'Model server stopped')
	}

	async requestInstance(request: IncomingRequest, signal?: AbortSignal) {
		return this.pool.requestInstance(request, signal)
	}

	// gets called by the pool right before a new instance is created
	private async prepareInstance(instance: ModelInstance, signal?: AbortSignal) {
		const model = instance.config
		const modelStoreStatus = this.store.models[model.id].status
		if (modelStoreStatus === 'unloaded') {
			await this.store.prepareModel(model.id, signal)
		}
		if (modelStoreStatus === 'preparing') {
			const modelReady = new Promise<void>((resolve, reject) => {
				const onCompleted = async (storeModel: StoredModel) => {
					if (storeModel.id === model.id) {
						this.store.prepareQueue.off('completed', onCompleted)
						if (storeModel.status === 'ready') {
							resolve()
						} else {
							reject()
						}
					}
				}
				this.store.prepareQueue.on('completed', onCompleted)
			})
			await modelReady
		}
	}

	async processChatCompletionTask(args: ChatCompletionRequest, options?: CompletionProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processChatCompletionTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processTextCompletionTask(args: TextCompletionRequest, options?: CompletionProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processTextCompletionTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processEmbeddingTask(args: EmbeddingRequest, options?: ProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processEmbeddingTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processImageToTextTask(args: ImageToTextRequest, options?: ProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processImageToTextTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processSpeechToTextTask(args: SpeechToTextRequest, options?: SpeechToTextProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processSpeechToTextTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processTextToImageTask(args: TextToImageRequest, options?: ProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processTextToImageTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processImageToImageTask(args: ImageToImageRequest, options?: ProcessingOptions) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processImageToImageTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	getStatus() {
		const poolStatus = this.pool.getStatus()
		const storeStatus = this.store.getStatus()
		return {
			pool: poolStatus,
			store: storeStatus,
		}
	}
}
