import crypto from 'node:crypto'
import { customAlphabet } from 'nanoid'
import {
	ModelEngine,
	ModelConfig,
	ModelInstanceRequest,
	ChatCompletionTaskResult,
	TextCompletionTaskResult,
	ChatCompletionTaskArgs,
	TextCompletionTaskArgs,
	EmbeddingTaskArgs,
	ImageToTextTaskArgs,
	ImageToImageTaskArgs,
	SpeechToTextTaskArgs,
	TextToSpeechTaskArgs,
	TextToImageTaskArgs,
	ObjectDetectionTaskArgs,
} from '#package/types/index.js'
import { calculateContextIdentity } from '#package/lib/calculateContextIdentity.js'
import { LogLevels, Logger, createLogger, withLogMeta } from '#package/lib/logger.js'
import { elapsedMillis, mergeAbortSignals } from '#package/lib/util.js'

const idAlphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
const generateId = customAlphabet(idAlphabet, 8)

type ModelInstanceStatus = 'idle' | 'busy' | 'error' | 'loading' | 'preparing'

interface ModelInstanceOptions extends ModelConfig {
	log?: Logger
	gpu: boolean
}

export class ModelInstance<TEngineRef = unknown> {
	id: string
	status: ModelInstanceStatus
	modelId: string
	config: ModelConfig
	fingerprint: string
	createdAt: Date
	lastUsed: number = 0
	gpu: boolean
	ttl: number
	log: Logger

	private engine: ModelEngine
	private engineRef?: TEngineRef | unknown
	private contextIdentity?: string
	private needsContextReset: boolean = false
	private currentRequest?: ModelInstanceRequest | null
	private shutdownController: AbortController

	constructor(engine: ModelEngine, { log, gpu, ...options }: ModelInstanceOptions) {
		this.modelId = options.id
		this.id = this.generateInstanceId()
		this.engine = engine
		this.config = options
		this.gpu = gpu
		this.ttl = options.ttl ?? 300
		this.status = 'preparing'
		this.createdAt = new Date()
		this.log = withLogMeta(log ?? createLogger(LogLevels.warn), {
			instance: this.id,
		})
		this.shutdownController = new AbortController()

		// TODO to implement this properly we should only include what changes the "behavior" of the model
		this.fingerprint = crypto.createHash('sha1').update(JSON.stringify(options)).digest('hex')
		this.log(LogLevels.info, 'Initializing new instance', {
			model: this.modelId,
			engine: this.config.engine,
			device: this.config.device,
			hasGpuLock: this.gpu,
		})
	}

	private generateInstanceId() {
		return this.modelId + ':' + generateId(8)
	}

	private generateTaskId() {
		return this.id + '-' + generateId(8)
	}

	getEngineRef() {
		return this.engineRef
	}

	async load(signal?: AbortSignal) {
		if (this.engineRef) {
			throw new Error('Instance is already loaded')
		}
		this.status = 'loading'
		const loadBegin = process.hrtime.bigint()
		const abortSignal = mergeAbortSignals([this.shutdownController.signal, signal])
		try {
			this.engineRef = await this.engine.createInstance(
				{
					log: withLogMeta(this.log, {
						instance: this.id,
					}),
					config: {
						...this.config,
						device: {
							...this.config.device,
							gpu: this.gpu ? this.config.device?.gpu : false,
						},
					},
				},
				abortSignal,
			)
			this.status = 'idle'
			if (this.config.initialMessages?.length) {
				this.contextIdentity = calculateContextIdentity({
					messages: this.config.initialMessages,
				})
			}
			if (this.config.prefix) {
				this.contextIdentity = calculateContextIdentity({
					text: this.config.prefix,
				})
			}
			this.log(LogLevels.debug, 'Instance loaded', {
				elapsed: elapsedMillis(loadBegin),
			})
		} catch (error: any) {
			this.status = 'error'
			this.log(LogLevels.error, 'Failed to load instance:', {
				error,
			})
			throw error
		}
	}

	dispose() {
		this.status = 'busy'
		if (!this.engineRef) {
			return Promise.resolve()
		}
		this.shutdownController.abort()
		return this.engine.disposeInstance(this.engineRef)
	}

	lock(request: ModelInstanceRequest) {
		if (this.status !== 'idle') {
			throw new Error(`Cannot lock: Instance ${this.id} is not idle`)
		}
		this.currentRequest = request
		this.status = 'busy'
	}

	unlock() {
		this.status = 'idle'
		this.currentRequest = null
	}

	resetContext() {
		this.needsContextReset = true
	}

	getContextStateIdentity() {
		return this.contextIdentity
	}

	hasContextState() {
		return this.contextIdentity !== undefined
	}

	matchesContextState(request: ModelInstanceRequest) {
		if (!this.contextIdentity) {
			return false
		}
		let incomingContextIdentity = ''
		if ('messages' in request && request.messages?.length) {
			incomingContextIdentity = calculateContextIdentity({
				messages: request.messages,
				dropLastUserMessage: true,
			})
		} else if ('prompt' in request && request.prompt) {
			incomingContextIdentity = calculateContextIdentity({
				text: request.prompt,
			})
		}

		if (!incomingContextIdentity) {
			return false
		}

		return this.contextIdentity === incomingContextIdentity || incomingContextIdentity.startsWith(this.contextIdentity)
	}

	matchesRequirements(request: ModelInstanceRequest) {
		const requiresGpu = !!this.config.device?.gpu && this.config.device?.gpu !== 'auto'
		const modelMatches = this.modelId === request.model
		const gpuMatches = requiresGpu ? this.gpu : true
		return modelMatches && gpuMatches
	}

	private createTaskController(args: { timeout?: number; signal?: AbortSignal }) {
		const cancelController = new AbortController()
		const timeoutController = new AbortController()
		const abortSignals = [cancelController.signal, this.shutdownController.signal]
		if (args.signal) {
			abortSignals.push(args.signal)
		}
		let timeout: NodeJS.Timeout | undefined
		if (args.timeout) {
			timeout = setTimeout(() => {
				timeoutController.abort('timeout')
			}, args.timeout)
			abortSignals.push(timeoutController.signal)
		}
		return {
			cancel: () => {
				cancelController.abort('cancel')
				if (timeout) {
					clearTimeout(timeout)
				}
			},
			complete: () => {
				if (timeout) {
					clearTimeout(timeout)
				}
			},
			signal: mergeAbortSignals(abortSignals),
			timeoutSignal: timeoutController.signal,
			cancelSignal: cancelController.signal,
		}
	}

	processChatCompletionTask(args: ChatCompletionTaskArgs) {
		if (!('processChatCompletionTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement chat completions`)
		}
		if (!args.messages?.length) {
			throw new Error('Messages are required for chat completions')
		}
		const id = this.generateTaskId()
		this.lastUsed = Date.now()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		// checking if this instance has been flagged for reset
		let resetContext = false
		if (this.needsContextReset) {
			this.contextIdentity = undefined
			this.needsContextReset = false
			resetContext = true
		}
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		// start completion processing
		taskLogger(LogLevels.verbose, 'Creating chat completion')
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			resetContext,
			log: taskLogger,
		}
		const completionPromise = this.engine.processChatCompletionTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				if (controller.timeoutSignal.aborted) {
					result.finishReason = 'timeout'
				} else if (controller.cancelSignal.aborted) {
					result.finishReason = 'cancel'
				}
				this.contextIdentity = calculateContextIdentity({
					messages: [...args.messages, result.message],
				})
				return result
			})
			.catch((error) => {
				if (error.name === 'AbortError') {
					const emptyResponse: ChatCompletionTaskResult = {
						finishReason: 'abort',
						message: {
							role: 'assistant',
							content: '',
						},
						promptTokens: 0,
						completionTokens: 0,
						contextTokens: 0,
					}
					if (controller.timeoutSignal.aborted) {
						emptyResponse.finishReason = 'timeout'
						return emptyResponse
					}
					if (controller.cancelSignal.aborted) {
						emptyResponse.finishReason = 'cancel'
						return emptyResponse
					}
					return emptyResponse
				}
				taskLogger(LogLevels.error, 'Error while processing task - ', {
					error,
				})
				throw error
			})
			.finally(() => {
				const elapsedTime = elapsedMillis(taskBegin)
				controller.complete()
				taskLogger(LogLevels.info, 'Chat completion task done', {
					elapsed: elapsedTime,
				})
			})
		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			result: completionPromise,
			cancel: controller.cancel,
		}
	}

	processTextCompletionTask(args: TextCompletionTaskArgs) {
		if (!('processTextCompletionTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement text completion`)
		}
		if (!args.prompt) {
			throw new Error('Prompt is required for text completion')
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		taskLogger(LogLevels.verbose, 'Creating text completion task')
		// pass on resetContext if this instance has been flagged for reset
		let resetContext = false
		if (this.needsContextReset) {
			this.contextIdentity = undefined
			this.needsContextReset = false
			resetContext = true
		}
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			resetContext,
			log: taskLogger,
		}
		const completionPromise = this.engine.processTextCompletionTask!(args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				if (controller.timeoutSignal.aborted) {
					result.finishReason = 'timeout'
				} else if (controller.cancelSignal.aborted) {
					result.finishReason = 'cancel'
				}
				this.contextIdentity = calculateContextIdentity({
					text: args.prompt + result.text,
				})
				return result
			})
			.catch((error) => {
				if (error.name === 'AbortError') {
					const emptyResponse: TextCompletionTaskResult = {
						finishReason: 'abort',
						text: '',
						promptTokens: 0,
						completionTokens: 0,
						contextTokens: 0,
					}
					if (controller.timeoutSignal.aborted) {
						emptyResponse.finishReason = 'timeout'
						return emptyResponse
					}
					if (controller.cancelSignal.aborted) {
						emptyResponse.finishReason = 'cancel'
						return emptyResponse
					}
					return emptyResponse
				}
				taskLogger(LogLevels.error, 'Error while processing task - ', {
					error,
				})
				throw error
			})
			.finally(() => {
				const elapsedTime = elapsedMillis(taskBegin)
				controller.complete()
				taskLogger(LogLevels.info, 'Text completion task done', {
					elapsed: elapsedTime,
				})
			})
		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result: completionPromise,
		}
	}

	processEmbeddingTask(args: EmbeddingTaskArgs) {
		if (!('processEmbeddingTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement embedding`)
		}
		if (!args.input) {
			throw new Error('Input is required for embedding')
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		taskLogger(LogLevels.verbose, 'Creating embedding task')
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processEmbeddingTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'Embedding task timed out')
				}
				taskLogger(LogLevels.verbose, 'Embedding task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}

	processImageToTextTask(args: ImageToTextTaskArgs) {
		if (!('processImageToTextTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement image to text`)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processImageToTextTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'ImageToText task timed out')
				}
				taskLogger(LogLevels.verbose, 'ImageToText task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}

	processImageToImageTask(args: ImageToImageTaskArgs) {
		if (!('processImageToImageTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement image to image`)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processImageToImageTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'ImageToImage task timed out')
				}
				taskLogger(LogLevels.verbose, 'ImageToImage task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}

	processSpeechToTextTask(args: SpeechToTextTaskArgs) {
		if (!('processSpeechToTextTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement speech to text`)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processSpeechToTextTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'SpeechToText task timed out')
				}
				taskLogger(LogLevels.verbose, 'SpeechToText task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}

	processTextToSpeechTask(args: TextToSpeechTaskArgs) {
		if (!('processTextToSpeechTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement text to speech`)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processTextToSpeechTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'TextToSpeech task timed out')
				}
				taskLogger(LogLevels.verbose, 'TextToSpeech task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})
		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}

	processTextToImageTask(args: TextToImageTaskArgs) {
		if (!('processTextToImageTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement text to image`)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processTextToImageTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'TextToImage task timed out')
				}
				taskLogger(LogLevels.verbose, 'TextToImage task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}
	
	processObjectDetectionTask(args: ObjectDetectionTaskArgs) {
		if (!('processObjectDetectionTask' in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement object detection`)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: args?.timeout,
			signal: args?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const taskContext = {
			instance: this.engineRef,
			config: this.config,
			log: taskLogger,
		}
		const result = this.engine.processObjectDetectionTask!(
			args,
			taskContext,
			controller.signal,
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, 'object-detection task timed out')
				}
				taskLogger(LogLevels.verbose, 'object-detection task done', {
					elapsed: timeElapsed,
				})
				return result
			})
			.catch((error) => {
				taskLogger(LogLevels.error, 'Task failed - ', {
					error,
				})
				throw error
			})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}
}
