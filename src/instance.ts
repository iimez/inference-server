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
	TextClassificationTaskArgs,
	TaskKind,
	TaskProcessor,
	TaskProcessorName,
	TaskArgs,
	EmbeddingTaskResult,
	TaskResult,
	InferenceTask,
	ImageToTextTaskResult,
	ImageToImageTaskResult,
	SpeechToTextTaskResult,
	TextToSpeechTaskResult,
	TextToImageTaskResult,
	TextClassificationTaskResult,
	ObjectDetectionTaskResult,
} from '#package/types/index.js'
import { calculateChatContextIdentity } from '#package/lib/calculateChatContextIdentity.js'
import { LogLevels, Logger, createLogger, withLogMeta } from '#package/lib/logger.js'
import { elapsedMillis, mergeAbortSignals } from '#package/lib/util.js'
import { getLargestCommonPrefix } from '#package/lib/getLargestCommonPrefix.js'

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
				this.contextIdentity = calculateChatContextIdentity({
					messages: this.config.initialMessages,
				})
			}
			if (this.config.prefix) {
				this.contextIdentity = this.config.prefix
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
		if ('messages' in request && request.messages?.length) {
			const incomingContextIdentity = calculateChatContextIdentity({
				messages: request.messages,
				dropLastUserMessage: true,
			})
			return this.contextIdentity === incomingContextIdentity
		} else if ('prompt' in request && request.prompt) {
			const commonPrefix = getLargestCommonPrefix(this.contextIdentity, request.prompt)
			return commonPrefix.length > 0
		}
		
		return false
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
		taskLogger(LogLevels.info, 'Processing chat completion task')
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
				this.contextIdentity = calculateChatContextIdentity({
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
		taskLogger(LogLevels.info, 'Processing text completion task')
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
				this.contextIdentity = args.prompt + result.text
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

	private processTask<T extends TaskResult>(
		taskType: TaskKind,
		processorName: TaskProcessorName,
		args: TaskArgs,
	): InferenceTask<T> {
		if (!(processorName in this.engine)) {
			throw new Error(`Engine "${this.config.engine}" does not implement ${taskType}`)
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
		
		taskLogger(LogLevels.info, `Processing ${taskType} task`)
		
		const processor = this.engine[processorName] as TaskProcessor<any, any, any>
		const result: Promise<TaskResult> = processor(
			args,
			taskContext,
			controller.signal
		)
			.then((result) => {
				const timeElapsed = elapsedMillis(taskBegin)
				controller.complete()
				if (controller.timeoutSignal.aborted) {
					taskLogger(LogLevels.warn, `${taskType} task timed out`)
				}
				taskLogger(LogLevels.info, `${taskType} task done`, {
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
			result: result as Promise<T>,
		}
	}
	
	processEmbeddingTask(args: EmbeddingTaskArgs) {
		return this.processTask<EmbeddingTaskResult>('embedding', 'processEmbeddingTask', args)
	}
	
	processImageToTextTask(args: ImageToTextTaskArgs) {
		return this.processTask<ImageToTextTaskResult>('image-to-text', 'processImageToTextTask', args)
	}
	
	processImageToImageTask(args: ImageToImageTaskArgs) {
		return this.processTask<ImageToImageTaskResult>('image-to-image', 'processImageToImageTask', args)
	}
	
	processSpeechToTextTask(args: SpeechToTextTaskArgs) {
		return this.processTask<SpeechToTextTaskResult>('speech-to-text', 'processSpeechToTextTask', args)
	}
	
	processTextToSpeechTask(args: TextToSpeechTaskArgs) {
		return this.processTask<TextToSpeechTaskResult>('text-to-speech', 'processTextToSpeechTask', args)
	}
	
	processTextToImageTask(args: TextToImageTaskArgs) {
		return this.processTask<TextToImageTaskResult>('text-to-image', 'processTextToImageTask', args)
	}
	
	processTextClassificationTask(args: TextClassificationTaskArgs) {
		return this.processTask<TextClassificationTaskResult>('text-classification', 'processTextClassificationTask', args)
	}
	
	processObjectDetectionTask(args: ObjectDetectionTaskArgs) {
		return this.processTask<ObjectDetectionTaskResult>('object-detection', 'processObjectDetectionTask', args)
	}
}
