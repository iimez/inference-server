import path from 'node:path'
import { builtInEngineNames } from '#package/engines/index.js'
import { ModelPool } from '#package/pool.js'
import { ModelInstance } from '#package/instance.js'
import { ModelStore, StoredModel } from '#package/store.js'
import {
	ModelOptions,
	InferenceParams,
	ModelEngine,
	BuiltInModelOptions,
	CustomEngineModelOptions,
	ModelConfigBase,
	InferenceTaskArgs,
	ChatCompletionTaskArgs,
	TextCompletionTaskArgs,
	InferenceTask,
	EmbeddingTaskArgs,
	ImageToTextTaskArgs,
	TextToSpeechTaskArgs,
	SpeechToTextTaskArgs,
	TextToImageTaskArgs,
	ImageToImageTaskArgs,
	ObjectDetectionTaskArgs,
	ChatCompletionTaskResult,
	TextCompletionTaskResult,
	EmbeddingTaskResult,
	TextToImageTaskResult,
	TextToSpeechTaskResult,
	SpeechToTextTaskResult,
	ImageToTextTaskResult,
	ImageToImageTaskResult,
	ObjectDetectionTaskResult,
	ChatCompletionInferenceTaskArgs,
	TextCompletionInferenceTaskArgs,
	EmbeddingInferenceTaskArgs,
	ImageToTextInferenceTaskArgs,
	SpeechToTextInferenceTaskArgs,
	TextToSpeechInferenceTaskArgs,
	TextToImageInferenceTaskArgs,
	ImageToImageInferenceTaskArgs,
	ObjectDetectionInferenceTaskArgs,
	TextClassificationTaskArgs,
	TextClassificationTaskResult,
	TextClassificationInferenceTaskArgs,
} from '#package/types/index.js'
import { Logger, LogLevel, createSublogger, LogLevels } from '#package/lib/logger.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { validateModelOptions } from '#package/lib/validateModelOptions.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'

/**
 * Configuration options for initializing a `InferenceServer`.
 * @interface InferenceServerOptions
 */
export interface InferenceServerOptions {
	/**
	 * A record of custom engines to be used for processing tasks. Each engine is identified by a unique name.
	 * @type {Record<string, ModelEngine>}
	 * @optional
	 */
	engines?: Record<string, ModelEngine>

	/**
	 * A record of model configurations, where each model is identified by a unique ID, defined by the user.
	 * @type {Record<string, ModelOptions>}
	 */
	models: Record<string, ModelOptions>
	/**
	 * The maximum number of concurrent tasks allowed in the model pool.
	 * @type {number}
	 * @optional
	 */
	concurrency?: number
	/**
	 * The path to the cache directory where model files and related data will be stored.
	 * @type {string}
	 * @optional
	 */
	cachePath?: string
	/**
	 * A logger instance or log level to control logging for the server. If a log level is provided,
	 * a default logger will be created with that level.
	 * @type {Logger | LogLevel}
	 * @optional
	 */
	log?: Logger | LogLevel
}

/**
 * Represents a server for managing and serving machine learning models, including model initialization,
 * file downloads, request handling, and task processing. The example provided starts an inference server
 * using llama.cpp as the engine, with the task of text-completion and two instances of smollm.
 *
 * @class InferenceServer
 * @example
 * const inferenceServer = new InferenceServer({
 *   log: 'info',
 *   concurrency: 2,
 *   models: {
 *     'smollm-135m': {
 *       task: 'text-completion',
 *       url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
 *       engine: 'node-llama-cpp',
 *       maxInstances: 2,
 *   },
 * })
 * inferenceServer.start()
 */
export class InferenceServer {
	/** @property {ModelPool} pool - A pool for managing model instances and concurrency. */
	pool: ModelPool

	/** @property {ModelStore} store - A store for managing model metadata, preparation, and storage. */
	store: ModelStore

	/** @property {Record<string, ModelEngine>} engines - A record of engines (custom and built-in) used for processing tasks. */
	engines: Record<string, ModelEngine> = {}

	/** @property {Logger} log - Logger for tracking the server's activities and errors. */
	log: Logger

	/**
	 * Constructs a `InferenceServer` instance with the specified options.
	 * @param {InferenceServerOptions} options - Configuration options for the server.
	 */
	constructor(options: InferenceServerOptions) {
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
	/**
	 * Starts the inference server, initializing engines and preparing the model store and pool.
	 * @returns {Promise<void>} Resolves when the server is fully started.
	 */
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
	/**
	 * Stops the server. disposes all resources. Clears the queue of working tasks.
	 **/
	async stop() {
		this.log(LogLevels.info, 'Stopping model server')
		this.pool.queue.clear()
		this.store.dispose()

		try {
			await this.pool.dispose() // might cause abort errors when there are still running tasks
		} catch (err) {
			this.log(LogLevels.error, 'Error while stopping model server', err)
		}
		this.log(LogLevels.debug, 'Waiting for running tasks to finish')
		// wait until all running tasks are cancelled
		await this.pool.queue.onIdle()

		this.log(LogLevels.debug, 'Model server stopped')
	}
	/**
	 * Requests an available model instance from the pool for a specific task.
	 * Use this for manual control over when to release the instance back to the pool.
	 * @param {InferenceParams} args - The inference task arguments.
	 * @param {AbortSignal} [signal] - An optional signal to abort the request.
	 * @returns {Promise<ModelInstance>} A model instance that can fulfill the task.
	 */
	requestInstance(args: InferenceParams, signal?: AbortSignal) {
		return this.pool.requestInstance(args, signal)
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

	async processTask(args: TextCompletionInferenceTaskArgs): Promise<TextCompletionTaskResult>
	async processTask(args: ChatCompletionInferenceTaskArgs): Promise<ChatCompletionTaskResult>
	async processTask(args: EmbeddingInferenceTaskArgs): Promise<EmbeddingTaskResult>
	async processTask(args: ImageToTextInferenceTaskArgs): Promise<ImageToTextTaskResult>
	async processTask(args: SpeechToTextInferenceTaskArgs): Promise<SpeechToTextTaskResult>
	async processTask(args: TextToSpeechInferenceTaskArgs): Promise<TextToSpeechTaskResult>
	async processTask(args: TextToImageInferenceTaskArgs): Promise<TextToImageTaskResult>
	async processTask(args: ImageToImageInferenceTaskArgs): Promise<ImageToImageTaskResult>
	async processTask(args: ObjectDetectionInferenceTaskArgs): Promise<ObjectDetectionTaskResult>
	async processTask(args: TextClassificationInferenceTaskArgs): Promise<TextClassificationTaskResult>

	async processTask(args: InferenceTaskArgs) {
		const lock = await this.requestInstance(args)
		const waitForResult = async <T>(task: InferenceTask): Promise<T> => {
			const result = await task.result
			await lock.release()
			return result as T
		}

		if (args.task === 'text-completion') {
			const task = lock.instance.processTextCompletionTask(args)
			return await waitForResult<TextCompletionTaskResult>(task)
		}
		if (args.task === 'chat-completion') {
			const task = lock.instance.processChatCompletionTask(args)
			return await waitForResult<ChatCompletionTaskResult>(task)
		}
		if (args.task === 'embedding') {
			const task = lock.instance.processEmbeddingTask(args)
			return await waitForResult<EmbeddingTaskResult>(task)
		}
		if (args.task === 'image-to-text') {
			const task = lock.instance.processImageToTextTask(args)
			return await waitForResult<ImageToTextTaskResult>(task)
		}
		if (args.task === 'speech-to-text') {
			const task = lock.instance.processSpeechToTextTask(args)
			return await waitForResult<SpeechToTextTaskResult>(task)
		}
		if (args.task === 'text-to-speech') {
			const task = lock.instance.processTextToSpeechTask(args)
			return await waitForResult<TextToSpeechTaskResult>(task)
		}
		if (args.task === 'text-to-image') {
			const task = lock.instance.processTextToImageTask(args)
			return await waitForResult<TextToImageTaskResult>(task)
		}
		if (args.task === 'image-to-image') {
			const task = lock.instance.processImageToImageTask(args)
			return await waitForResult<ImageToImageTaskResult>(task)
		}
		if (args.task === 'object-detection') {
			const task = lock.instance.processObjectDetectionTask(args)
			return await waitForResult<ObjectDetectionTaskResult>(task)
		}
		if (args.task === 'text-classification') {
			const task = lock.instance.processTextClassificationTask(args)
			return await waitForResult<TextClassificationTaskResult>(task)
		}

		// @ts-expect-error
		throw new Error(`Unknown task type: ${args.task}`)
	}

	processChatCompletionTask(args: ChatCompletionTaskArgs) {
		return this.processTask({
			task: 'chat-completion',
			...args,
		})
	}

	processTextCompletionTask(args: TextCompletionTaskArgs) {
		return this.processTask({
			task: 'text-completion',
			...args,
		})
	}

	processEmbeddingTask(args: EmbeddingTaskArgs) {
		return this.processTask({
			task: 'embedding',
			...args,
		})
	}

	processImageToTextTask(args: ImageToTextTaskArgs) {
		return this.processTask({
			task: 'image-to-text',
			...args,
		})
	}

	processSpeechToTextTask(args: SpeechToTextTaskArgs) {
		return this.processTask({
			task: 'speech-to-text',
			...args,
		})
	}

	processTextToSpeechTask(args: TextToSpeechTaskArgs) {
		return this.processTask({
			task: 'text-to-speech',
			...args,
		})
	}

	processTextToImageTask(args: TextToImageTaskArgs) {
		return this.processTask({
			task: 'text-to-image',
			...args,
		})
	}
	processImageToImageTask(args: ImageToImageTaskArgs) {
		return this.processTask({
			task: 'image-to-image',
			...args,
		})
	}

	processObjectDetectionTask(args: ObjectDetectionTaskArgs) {
		return this.processTask({
			task: 'object-detection',
			...args,
		})
	}
	
	processTextClassificationTask(args: TextClassificationTaskArgs) {
		return this.processTask({
			task: 'text-classification',
			...args,
		})
	}

	/**
	 * Retrieves the current status of the model server, including pool and store status.
	 *
	 * @returns {Object} The status object containing pool and store information.
	 */
	getStatus() {
		const poolStatus = this.pool.getStatus()
		const storeStatus = this.store.getStatus()
		return {
			pool: poolStatus,
			store: storeStatus,
		}
	}
}
