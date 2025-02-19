import path from 'node:path'
import fs from 'node:fs'
import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	TextCompletionTaskResult,
	EmbeddingTaskResult,
	ImageEmbeddingInput,
	TransformersJsModel,
	TextEmbeddingInput,
	TransformersJsSpeechModel,
	SpeakerEmbeddings,
	TextCompletionTaskArgs,
	EngineTextCompletionTaskContext,
	EmbeddingTaskArgs,
	EngineTaskContext,
	ImageToTextTaskArgs,
	SpeechToTextTaskArgs,
	TextToSpeechTaskArgs,
	ObjectDetectionResult,
	ObjectDetectionTaskArgs,
	ChatCompletionTaskArgs,
	ChatCompletionTaskResult,
	ChatMessage,
	MessageTextContentPart,
	CompletionFinishReason,
	TextClassificationTaskArgs,
	ObjectDetectionTaskResult,
	TextClassificationTaskResult,
	TextToSpeechTaskResult,
} from '#package/types/index.js'
import {
	env,
	AutoModel,
	AutoProcessor,
	AutoTokenizer,
	RawImage,
	TextStreamer,
	mean_pooling,
	Processor,
	PreTrainedModel,
	PreTrainedTokenizer,
	Tensor,
	SpeechT5ForTextToSpeech,
	WhisperForConditionalGeneration,
	ModelOutput,
	StoppingCriteria,
	ProgressInfo,
	softmax,
	TextClassificationPipeline,
	TextClassificationSingle,
} from '@huggingface/transformers'
import { LogLevels } from '#package/lib/logger.js'
import { resampleAudioBuffer } from '#package/lib/loadAudio.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { moveDirectoryContents } from '#package/lib/moveDirectoryContents.js'
import {
	fetchBuffer,
	normalizeTransformersJsClass,
	parseHuggingfaceModelIdAndBranch,
	remoteFileExists,
} from './util.js'
import { validateModelFiles, ModelValidationResult } from './validateModelFiles.js'
import { acquireModelFileLocks } from './acquireModelFileLocks.js'
import { loadModelComponents, loadSpeechModelComponents } from './loadModelComponents.js'
import {
	TransformersJsModelClass,
	TransformersJsProcessorClass,
	TransformersJsTokenizerClass,
} from '#package/engines/transformers-js/types.js'
import { flattenMessageTextContent } from '#package/lib/flattenMessageTextContent.js'
import { Text } from 'openai/resources/beta/threads/messages.js'

export interface TransformersJsModelComponents<TModel = PreTrainedModel> {
	model?: TModel
	processor?: Processor
	tokenizer?: PreTrainedTokenizer
}

export interface TextToSpeechModel {
	generate_speech: SpeechT5ForTextToSpeech['generate_speech']
}

export interface SpeechToTextModel {
	generate: WhisperForConditionalGeneration['generate']
}

export interface SpeechModelComponents extends TransformersJsModelComponents<TextToSpeechModel | SpeechToTextModel> {
	vocoder?: PreTrainedModel
	speakerEmbeddings?: Record<string, Float32Array>
}

interface TransformersJsInstance {
	primary?: TransformersJsModelComponents
	text?: TransformersJsModelComponents
	vision?: TransformersJsModelComponents
	speech?: SpeechModelComponents
}

interface ModelFile {
	file: string
	size: number
}

// TODO model metadata
// interface TransformersJsModelMeta {
// 	modelType: string
// 	files: ModelFile[]
// }

export interface TransformersJsModelConfig extends ModelConfig, TransformersJsModel, TransformersJsSpeechModel {
	location: string
	url: string
	textModel?: TransformersJsModel
	visionModel?: TransformersJsModel
	speechModel?: TransformersJsSpeechModel
	device?: {
		gpu?: boolean | 'auto' | (string & {})
	}
}

export const autoGpu = true

let didConfigureEnvironment = false
function configureEnvironment(modelsPath: string) {
	// console.debug({
	// 	cacheDir: env.cacheDir,
	// 	localModelPaths: env.localModelPath,
	// })
	// env.useFSCache = false
	// env.useCustomCache = true
	// env.customCache = new TransformersFileCache(modelsPath)
	env.localModelPath = ''
	didConfigureEnvironment = true
}

async function disposeModelComponents(modelComponents: TransformersJsModelComponents) {
	if (modelComponents.model && 'dispose' in modelComponents.model) {
		await modelComponents.model.dispose()
	}
}

export async function prepareModel(
	{ config, log }: EngineContext<TransformersJsModelConfig>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	if (!didConfigureEnvironment) {
		configureEnvironment(config.modelsCachePath)
	}
	fs.mkdirSync(config.location, { recursive: true })
	const releaseFileLocks = await acquireModelFileLocks(config, signal)
	if (signal?.aborted) {
		releaseFileLocks()
		return
	}
	log(LogLevels.info, `Preparing transformers.js model at ${config.location}`, {
		model: config.id,
	})

	const downloadModelFiles = async (
		modelOpts: TransformersJsModel | (TransformersJsModel & TransformersJsSpeechModel),
		{ modelId, branch }: { modelId: string; branch: string },
		requiredComponents: string[] = ['model', 'tokenizer', 'processor', 'vocoder'],
	) => {
		const modelClass = normalizeTransformersJsClass<TransformersJsModelClass>(modelOpts.modelClass, AutoModel)
		const downloadPromises: Record<string, Promise<any> | undefined> = {}
		const progressCallback = (progress: ProgressInfo) => {
			if (onProgress && progress.status === 'progress') {
				onProgress({
					file: env.cacheDir + progress.name + '/' + progress.file,
					loadedBytes: progress.loaded || 0,
					totalBytes: progress.total || 0,
				})
			}
		}
		if (requiredComponents.includes('model')) {
			const modelDownloadPromise = modelClass.from_pretrained(modelId, {
				revision: branch,
				dtype: modelOpts.dtype || 'fp32',
				progress_callback: progressCallback,
				// use_external_data_format: true, // https://github.com/xenova/transformers.js/blob/38a3bf6dab2265d9f0c2f613064535863194e6b9/src/models.js#L205-L207
			})
			downloadPromises.model = modelDownloadPromise
		}
		if (requiredComponents.includes('tokenizer')) {
			const hasTokenizer = await remoteFileExists(`${config.url}/blob/${branch}/tokenizer.json`)
			if (hasTokenizer) {
				const tokenizerClass = normalizeTransformersJsClass<TransformersJsTokenizerClass>(
					modelOpts.tokenizerClass,
					AutoTokenizer,
				)
				const tokenizerDownload = tokenizerClass.from_pretrained(modelId, {
					revision: branch,
					progress_callback: progressCallback,
				})
				downloadPromises.tokenizer = tokenizerDownload
			}
		}

		if (requiredComponents.includes('processor')) {
			const processorClass = normalizeTransformersJsClass<TransformersJsProcessorClass>(
				modelOpts.processorClass,
				AutoProcessor,
			)
			if (modelOpts.processor?.url) {
				const { modelId, branch } = parseHuggingfaceModelIdAndBranch(modelOpts.processor.url)
				const processorDownload = processorClass.from_pretrained(modelId, {
					revision: branch,
					progress_callback: progressCallback,
				})
				downloadPromises.processor = processorDownload
			} else {
				const [hasProcessor, hasPreprocessor] = await Promise.all([
					remoteFileExists(`${config.url}/blob/${branch}/processor_config.json`),
					remoteFileExists(`${config.url}/blob/${branch}/preprocessor_config.json`),
				])
				if (hasProcessor || hasPreprocessor) {
					const processorDownload = processorClass.from_pretrained(modelId, {
						revision: branch,
						progress_callback: progressCallback,
					})
					downloadPromises.processor = processorDownload
				}
			}
		}
		if (requiredComponents.includes('vocoder') && 'vocoder' in modelOpts) {
			if (modelOpts.vocoder?.url) {
				const { modelId, branch } = parseHuggingfaceModelIdAndBranch(modelOpts.vocoder.url)
				const vocoderDownload = AutoModel.from_pretrained(modelId, {
					revision: branch,
					progress_callback: progressCallback,
				})
				downloadPromises.vocoder = vocoderDownload
			}
		}
		await Promise.all(Object.values(downloadPromises))
		const modelComponents: TransformersJsModelComponents = {}
		if (downloadPromises.model) {
			modelComponents.model = (await downloadPromises.model) as PreTrainedModel
		}
		if (downloadPromises.tokenizer) {
			modelComponents.tokenizer = (await downloadPromises.tokenizer) as PreTrainedTokenizer
		}
		if (downloadPromises.processor) {
			modelComponents.processor = (await downloadPromises.processor) as Processor
		}
		disposeModelComponents(modelComponents)
		// return modelComponents
	}

	const downloadSpeakerEmbeddings = async (speakerEmbeddings: SpeakerEmbeddings) => {
		const speakerEmbeddingsPromises = []
		for (const speakerEmbedding of Object.values(speakerEmbeddings)) {
			if (speakerEmbedding instanceof Float32Array) {
				// nothing to download if we have the embeddings already
				continue
			}
			if (!speakerEmbedding.url) {
				continue
			}
			const speakerEmbeddingsPath = resolveModelFileLocation({
				url: speakerEmbedding.url,
				filePath: speakerEmbedding.file,
				modelsCachePath: config.modelsCachePath,
			})
			const url = speakerEmbedding.url
			speakerEmbeddingsPromises.push(
				(async () => {
					const buffer = await fetchBuffer(url)
					await fs.promises.writeFile(speakerEmbeddingsPath, buffer)
				})(),
			)
		}
		return Promise.all(speakerEmbeddingsPromises)
	}

	const downloadModel = async (validationResult: ModelValidationResult) => {
		log(LogLevels.info, `${validationResult.message} - Downloading files`, {
			model: config.id,
			url: config.url,
			location: config.location,
			errors: validationResult.errors,
		})
		const modelDownloadPromises = []
		if (!config.url) {
			throw new Error(`Missing URL for model ${config.id}`)
		}
		const { modelId, branch } = parseHuggingfaceModelIdAndBranch(config.url)
		const directoriesToCopy: Record<string, string> = {}
		const modelCacheDir = path.join(env.cacheDir, modelId)
		directoriesToCopy[modelCacheDir] = config.location
		const prepareDownloadedVocoder = (vocoderOpts: { url?: string; file?: string }) => {
			if (!vocoderOpts.url) {
				return
			}
			const vocoderPath = resolveModelFileLocation({
				url: vocoderOpts.url,
				filePath: vocoderOpts.file,
				modelsCachePath: config.modelsCachePath,
			})
			const { modelId } = parseHuggingfaceModelIdAndBranch(vocoderOpts.url)
			const vocoderCacheDir = path.join(env.cacheDir, modelId)
			directoriesToCopy[vocoderCacheDir] = vocoderPath
		}
		const prepareDownloadedProcessor = (processorOpts: { url?: string; file?: string }) => {
			if (!processorOpts.url) {
				return
			}
			const processorPath = resolveModelFileLocation({
				url: processorOpts.url,
				filePath: processorOpts.file,
				modelsCachePath: config.modelsCachePath,
			})
			const { modelId } = parseHuggingfaceModelIdAndBranch(processorOpts.url)
			const processorCacheDir = path.join(env.cacheDir, modelId)
			directoriesToCopy[processorCacheDir] = processorPath
		}
		const requiredComponents = validationResult.errors?.primaryModel
			? Object.keys(validationResult.errors.primaryModel)
			: undefined
		modelDownloadPromises.push(downloadModelFiles(config, { modelId, branch }, requiredComponents))
		if (config.processor?.url) {
			prepareDownloadedProcessor(config.processor)
		}
		if (config.vocoder?.url) {
			prepareDownloadedVocoder(config.vocoder)
		}
		if (config?.speakerEmbeddings) {
			modelDownloadPromises.push(downloadSpeakerEmbeddings(config.speakerEmbeddings))
		}
		if (config.textModel) {
			const requiredComponents = validationResult.errors?.textModel
				? Object.keys(validationResult.errors.textModel)
				: undefined
			modelDownloadPromises.push(downloadModelFiles(config.textModel, { modelId, branch }, requiredComponents))
		}
		if (config.visionModel) {
			const requiredComponents = validationResult.errors?.visionModel
				? Object.keys(validationResult.errors.visionModel)
				: undefined
			modelDownloadPromises.push(downloadModelFiles(config.visionModel, { modelId, branch }, requiredComponents))
			if (config.processor?.url) {
				prepareDownloadedProcessor(config.processor)
			}
		}
		if (config.speechModel) {
			const requiredComponents = validationResult.errors?.speechModel
				? Object.keys(validationResult.errors.speechModel)
				: undefined
			modelDownloadPromises.push(downloadModelFiles(config.speechModel, { modelId, branch }, requiredComponents))
			if (config.speechModel.vocoder?.url) {
				prepareDownloadedVocoder(config.speechModel.vocoder)
			}
			if (config.speechModel?.speakerEmbeddings) {
				modelDownloadPromises.push(downloadSpeakerEmbeddings(config.speechModel.speakerEmbeddings))
			}
		}

		await Promise.all(modelDownloadPromises)

		if (signal?.aborted) {
			return
		}
		// move all downloads to their final location
		await Promise.all(
			Object.entries(directoriesToCopy).map(([from, to]) => {
				if (fs.existsSync(from)) {
					return moveDirectoryContents(from, to)
				}
				return Promise.resolve()
			}),
		)
	}

	try {
		const validationResults = await validateModelFiles(config)
		if (signal?.aborted) {
			return
		}
		if (validationResults) {
			if (config.url) {
				await downloadModel(validationResults)
			} else {
				throw new Error(`Model files are invalid: ${validationResults.message}`)
			}
		}
	} catch (error) {
		throw error
	} finally {
		releaseFileLocks()
	}
	const configMeta: Record<string, any> = {}
	const fileList: ModelFile[] = []
	const modelFiles = fs.readdirSync(config.location, { recursive: true })

	const pushFile = (file: string) => {
		const targetFile = path.join(config.location, file)
		const targetStat = fs.statSync(targetFile)
		fileList.push({
			file: targetFile,
			size: targetStat.size,
		})
		if (targetFile.endsWith('.json')) {
			const key = path.basename(targetFile).replace('.json', '')
			configMeta[key] = JSON.parse(fs.readFileSync(targetFile, 'utf8'))
		}
	}
	// add model files to the list
	for (const file of modelFiles) {
		pushFile(file.toString())
	}

	// add extra stuff from external repos
	if (config.visionModel?.processor) {
		const processorPath = resolveModelFileLocation({
			url: config.visionModel.processor.url,
			filePath: config.visionModel.processor.file,
			modelsCachePath: config.modelsCachePath,
		})
		const processorFiles = fs.readdirSync(processorPath, { recursive: true })
		for (const file of processorFiles) {
			pushFile(file.toString())
		}
	}
	return {
		files: modelFiles,
		...configMeta,
	}
}

export async function createInstance({ config, log }: EngineContext<TransformersJsModelConfig>, signal?: AbortSignal) {
	const modelLoadPromises: Promise<unknown>[] = []

	modelLoadPromises.push(loadModelComponents(config, config))
	if (config.textModel) {
		modelLoadPromises.push(loadModelComponents(config.textModel, config))
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}
	if (config.visionModel) {
		modelLoadPromises.push(loadModelComponents(config.visionModel, config))
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}
	if (config.speechModel) {
		modelLoadPromises.push(loadSpeechModelComponents(config.speechModel, config))
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}

	const models = await Promise.all(modelLoadPromises)
	const instance: TransformersJsInstance = {
		primary: models[0] as TransformersJsModelComponents,
		text: models[1] as TransformersJsModelComponents,
		vision: models[2] as TransformersJsModelComponents,
		speech: models[3] as SpeechModelComponents & TransformersJsModelComponents,
	}

	// warm up model by doing a tiny generation
	if (config.task === 'chat-completion') {
		const chatModel = instance.text || (instance.primary as TransformersJsModelComponents)
		if (chatModel.tokenizer && !chatModel.processor) {
			// TODO figure out a way to warm up using processor?
			const inputs = chatModel.tokenizer('a')
			await chatModel.model!.generate({ ...inputs, max_new_tokens: 1 })
		}
	}
	if (config.task === 'text-completion') {
		const textModel = instance.text || (instance.primary as TransformersJsModelComponents)
		const inputs = textModel.tokenizer!('a')
		await textModel.model!.generate({ ...inputs, max_new_tokens: 1 })
	}

	// TODO warm up other model types
	// ie for whisper, this seems to speed up the initial response time
	// await model.generate({
	// 	input_features: full([1, 80, 3000], 0.0),
	// 	max_new_tokens: 1,
	// });

	return instance
}

export async function disposeInstance(instance: TransformersJsInstance) {
	const disposePromises = []
	if (instance.primary) {
		disposePromises.push(disposeModelComponents(instance.primary))
	}
	if (instance.vision) {
		disposePromises.push(disposeModelComponents(instance.vision))
	}
	if (instance.speech) {
		disposePromises.push(disposeModelComponents(instance.speech as TransformersJsModelComponents))
	}
	await Promise.all(disposePromises)
}

class CustomStoppingCriteria extends StoppingCriteria {
	stopped: boolean
	constructor() {
		super()
		this.stopped = false
	}
	stop() {
		this.stopped = true
	}
	reset() {
		this.stopped = false
	}
	override _call(inputIds: any, scores: any) {
		return new Array(inputIds.length).fill(this.stopped)
	}
}

function prepareInputMessages(messages: ChatMessage[]) {
	const images: RawImage[] = []
	const inputMessages = messages.map((message) => {
		if (typeof message.content === 'string') {
			return {
				role: message.role,
				content: message.content,
			}
		} else if (Array.isArray(message.content)) {
			return {
				role: message.role,
				content: message.content.map((part) => {
					if (part.type === 'text') {
						return part
					} else if (part.type === 'image') {
						const rawImage = new RawImage(
							new Uint8ClampedArray(part.image.data),
							part.image.width,
							part.image.height,
							part.image.channels as 1 | 2 | 3 | 4,
						)
						images.push(rawImage)
						return {
							type: 'text',
							text: '<image_placeholder>',
						} as MessageTextContentPart
					} else {
						throw new Error('Invalid message content: unknown part type')
					}
				}),
			}
		} else {
			throw new Error('Invalid message content: must be string or array')
		}
	})
	return {
		inputMessages: inputMessages.map((message) => {
			return {
				role: message.role,
				content: flattenMessageTextContent(message.content),
			}
		}),
		images,
	}
}

export async function processChatCompletionTask(
	task: ChatCompletionTaskArgs,
	ctx: EngineTextCompletionTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
): Promise<ChatCompletionTaskResult> {
	const { instance } = ctx
	if (!task.messages) {
		throw new Error('Messages are required for chat completion.')
	}
	const chatModel = instance.text || (instance.primary as TransformersJsModelComponents)
	if (!(chatModel.tokenizer && chatModel.model)) {
		throw new Error('Chat model is not loaded.')
	}

	const { images, inputMessages } = prepareInputMessages(task.messages)

	let inputs
	let inputTokenCount = 0
	const inputText: any = chatModel.tokenizer.apply_chat_template(inputMessages, {
		tokenize: false,
		add_generation_prompt: true,
		return_dict: true,
	})
	if (chatModel.processor) {
		inputs = await chatModel.processor(inputMessages, {
			images,
		})
	} else {
		inputs = chatModel.tokenizer(inputText, {
			return_tensor: true,
			add_special_tokens: false,
		})
	}
	inputTokenCount = inputs.input_ids.size

	const stoppingCriteria = new CustomStoppingCriteria()
	signal?.addEventListener('abort', () => {
		stoppingCriteria.stop()
	})

	let responseText = ''
	let finishReason: CompletionFinishReason = 'cancel'
	const streamer = new TextStreamer(chatModel.tokenizer, {
		skip_prompt: true,
		decode_kwargs: {
			skip_special_tokens: true,
		},
		callback_function: (output: string) => {
			responseText += output
			if (task.stop && task.stop.some((stopToken) => output.includes(stopToken))) {
				stoppingCriteria.stop()
				finishReason = 'stopTrigger'
			}
			if (task.onChunk) {
				const tokens = chatModel.tokenizer!.encode(output)
				task.onChunk({ text: output, tokens: tokens })
			}
		},
	})

	const maxTokens = task.maxTokens ?? 128
	const outputs = (await chatModel.model.generate({
		...inputs,
		// common params
		max_new_tokens: maxTokens,
		repetition_penalty: task.repeatPenalty ?? 1.0, // 1 = no penalty
		temperature: task.temperature ?? 1.0,
		top_k: task.topK ?? 50,
		top_p: task.topP ?? 1.0,
		// do_sample: true,
		// num_beams: 1,
		// num_return_sequences: 2, // TODO https://github.com/huggingface/transformers.js/issues/1007
		// eos_token_id: stopTokens[0], // TODO implement stop
		streamer,
		// transformers-exclusive params
		// Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
		// length_penalty: -64,
		// The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay.
		// exponential_decay_length_penalty: [1, 64],
		// typical_p: 1,
		// epsilon_cutoff: 0,
		// eta_cutoff: 0,
		// diversity_penalty: 0,
		// encoder_repetition_penalty: 1.0, // 1 = no penalty
		// no_repeat_ngram_size: 0,
		// forced_eos_token_id: [],
		// bad_words_ids: [],
		// force_words_ids: [],
		// suppress_tokens: [],
		stopping_criteria: stoppingCriteria,
	})) as Tensor
	const outputTokenCount = outputs.size
	// const hasEogToken = outputs.
	// @ts-ignore
	const outputTexts = chatModel.tokenizer.batch_decode(outputs, { skip_special_tokens: false })
	const eosToken = chatModel.tokenizer._tokenizer_config.eos_token
	const hasEogToken = outputTexts[0].endsWith(eosToken)
	const completionTokenCount = outputTokenCount - inputTokenCount

	if (hasEogToken) {
		finishReason = 'eogToken'
	} else if (completionTokenCount >= maxTokens) {
		finishReason = 'maxTokens'
	}
	return {
		finishReason,
		message: {
			role: 'assistant',
			content: responseText,
		},
		promptTokens: inputTokenCount,
		completionTokens: outputTokenCount - inputTokenCount,
		contextTokens: outputTokenCount,
	}
}

// TextGenerationPipeline https://github.com/huggingface/transformers.js/blob/705cfc456f8b8f114891e1503b0cdbaa97cf4b11/src/pipelines.js#L977
// Generation Args https://github.com/huggingface/transformers.js/blob/705cfc456f8b8f114891e1503b0cdbaa97cf4b11/src/generation/configuration_utils.js#L11
// default Model.generate https://github.com/huggingface/transformers.js/blob/705cfc456f8b8f114891e1503b0cdbaa97cf4b11/src/models.js#L1378
export async function processTextCompletionTask(
	task: TextCompletionTaskArgs,
	ctx: EngineTextCompletionTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
): Promise<TextCompletionTaskResult> {
	const { instance } = ctx
	if (!task.prompt) {
		throw new Error('Prompt is required for text completion.')
	}
	const textModel = instance.text || (instance.primary as TransformersJsModelComponents)
	if (!(textModel?.tokenizer && textModel?.model)) {
		throw new Error('Text model is not loaded.')
	}
	if (!('generate' in textModel.model)) {
		throw new Error('Text model does not support generation.')
	}
	textModel.tokenizer.padding_side = 'left'
	const inputs = textModel.tokenizer(task.prompt, {
		add_special_tokens: false,
		padding: true,
		truncation: true,
	})

	const stoppingCriteria = new CustomStoppingCriteria()
	signal?.addEventListener('abort', () => {
		stoppingCriteria.stop()
	})

	let finishReason: CompletionFinishReason = 'cancel'

	const streamer = new TextStreamer(textModel.tokenizer, {
		skip_prompt: true,
		callback_function: (output: string) => {
			if (task.stop && task.stop.some((stopToken) => output.includes(stopToken))) {
				stoppingCriteria.stop()
				finishReason = 'stopTrigger'
			}
			if (task.onChunk) {
				const tokens = textModel.tokenizer!.encode(output)
				task.onChunk({ text: output, tokens: tokens })
			}
		},
	})

	const maxTokens = task.maxTokens ?? 128
	const outputs: ModelOutput = await textModel.model.generate({
		...inputs,
		renormalize_logits: true,
		output_scores: true, // TODO currently no effect
		return_dict_in_generate: true,
		// common params
		max_new_tokens: maxTokens,
		repetition_penalty: task.repeatPenalty ?? 1.0, // 1 = no penalty
		temperature: task.temperature,
		top_k: task.topK,
		top_p: task.topP,
		// num_beams: 1,
		// num_return_sequences: 2, // TODO https://github.com/huggingface/transformers.js/issues/1007
		// eos_token_id: stopTokens[0], // TODO implement stop
		length_penalty: 0,
		streamer,
		// transformers-exclusive params
		// length_penalty: -64, // Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
		// The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay.
		// exponential_decay_length_penalty: [1, 64],
		// typical_p: 1,
		// epsilon_cutoff: 0,
		// eta_cutoff: 0,
		// diversity_penalty: 0,
		// encoder_repetition_penalty: 1.0, // 1 = no penalty
		// no_repeat_ngram_size: 0,
		// forced_eos_token_id: [],
		// bad_words_ids: [],
		// force_words_ids: [],
		// suppress_tokens: [],
	})
	// @ts-ignore
	const outputTexts = instance.primary.tokenizer.batch_decode(outputs.sequences, {
		skip_special_tokens: true,
		clean_up_tokenization_spaces: true,
	})
	const generatedText = outputTexts[0].slice(task.prompt.length)
	// @ts-ignore
	const outputTokenCount = outputs.sequences.tolist().reduce((acc, sequence) => acc + sequence.length, 0)
	const inputTokenCount = inputs.input_ids.size

	// const outputTexts = chatModel.tokenizer.batch_decode(outputs, { skip_special_tokens: false })
	const eosToken = textModel.tokenizer._tokenizer_config.eos_token
	const hasEogToken = outputTexts[0].endsWith(eosToken)
	const completionTokenCount = outputTokenCount - inputTokenCount

	if (hasEogToken) {
		finishReason = 'eogToken'
	} else if (completionTokenCount >= maxTokens) {
		finishReason = 'maxTokens'
	}

	return {
		finishReason,
		text: generatedText,
		promptTokens: inputTokenCount,
		completionTokens: outputTokenCount,
		contextTokens: inputTokenCount + outputTokenCount,
	}
}

// see https://github.com/xenova/transformers.js/blob/v3/src/utils/tensor.js
// https://github.com/xenova/transformers.js/blob/v3/src/pipelines.js#L1284
export async function processEmbeddingTask(
	task: EmbeddingTaskArgs,
	ctx: EngineTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
): Promise<EmbeddingTaskResult> {
	const { instance, config } = ctx
	if (!task.input) {
		throw new Error('Input is required for embedding.')
	}
	const inputs = Array.isArray(task.input) ? task.input : [task.input]
	const normalizedInputs: Array<TextEmbeddingInput | ImageEmbeddingInput> = inputs.map((input) => {
		if (typeof input === 'string') {
			return {
				type: 'text',
				content: input,
			}
		} else if (input.type) {
			return input
		} else {
			throw new Error('Invalid input type')
		}
	})

	const embeddings: Float32Array[] = []
	let inputTokens = 0

	const applyPooling = (result: any, pooling: string, modelInputs: any) => {
		if (pooling === 'mean') {
			return mean_pooling(result, modelInputs.attention_mask)
		} else if (pooling === 'cls') {
			return result.slice(null, 0)
		} else {
			throw Error(`Pooling method '${pooling}' not supported.`)
		}
	}

	const truncateDimensions = (result: any, dimensions: number) => {
		const truncatedData = new Float32Array(dimensions)
		truncatedData.set(result.data.slice(0, dimensions))
		return truncatedData
	}

	for (const embeddingInput of normalizedInputs) {
		if (signal?.aborted) {
			break
		}
		let result
		let modelInputs
		if (embeddingInput.type === 'text') {
			const modelComponents = instance.text || instance.primary
			if (!modelComponents?.tokenizer || !modelComponents?.model) {
				throw new Error('Text model is not loaded.')
			}
			modelInputs = modelComponents.tokenizer(embeddingInput.content, {
				padding: true, // pads input if it is shorter than context window
				truncation: true, // truncates input if it exceeds context window
			})
			inputTokens += modelInputs.input_ids.size
			// @ts-ignore TODO check _call
			const modelOutputs = await instance.primary.model(modelInputs)
			result =
				modelOutputs.last_hidden_state ??
				modelOutputs.logits ??
				modelOutputs.token_embeddings ??
				modelOutputs.text_embeds
		} else if (embeddingInput.type === 'image') {
			const modelComponents = instance.vision || instance.primary
			if (!modelComponents?.processor || !modelComponents?.model) {
				throw new Error('Vision model is not loaded.')
			}
			// const
			// const { data, info } = await sharp(embeddingInput.content.data).raw().toBuffer({ resolveWithObject: true })
			const image = embeddingInput.content
			const rawImage = new RawImage(
				new Uint8ClampedArray(image.data),
				image.width,
				image.height,
				image.channels as 1 | 2 | 3 | 4,
			)
			modelInputs = await modelComponents.processor!(rawImage)
			// @ts-ignore TODO check _call
			const modelOutputs = await instance.vision.model(modelInputs)
			result = modelOutputs.last_hidden_state ?? modelOutputs.logits ?? modelOutputs.image_embeds
		}

		if (task.pooling) {
			result = applyPooling(result, task.pooling, modelInputs)
		}
		if (task.dimensions && result.data.length > task.dimensions) {
			embeddings.push(truncateDimensions(result, task.dimensions))
		} else {
			embeddings.push(result.data)
		}
	}

	return {
		embeddings,
		inputTokens,
	}
}

export async function processImageToTextTask(
	task: ImageToTextTaskArgs,
	ctx: EngineTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
) {
	const { instance } = ctx
	if (!task.image) {
		throw new Error('No image provided')
	}
	const image = task.image
	const rawImage = new RawImage(
		new Uint8ClampedArray(image.data),
		image.width,
		image.height,
		image.channels as 1 | 2 | 3 | 4,
	)

	if (signal?.aborted) {
		return
	}

	const modelComponents = instance.vision || instance.primary
	if (!(modelComponents && modelComponents.tokenizer && modelComponents.processor && modelComponents.model)) {
		throw new Error('No model loaded')
	}
	if (!('generate' in modelComponents.model)) {
		throw new Error('Model does not support generation')
	}
	let textInputs = {}
	if (task.prompt) {
		textInputs = modelComponents!.tokenizer(task.prompt)
	}
	const imageInputs = await modelComponents.processor(rawImage)
	const outputTokens = await modelComponents.model.generate({
		...textInputs,
		...imageInputs,
		max_new_tokens: task.maxTokens ?? 128,
	})
	// @ts-ignore
	const outputText = modelComponents.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})

	return {
		text: outputText[0],
	}
}

// see examples
// https://huggingface.co/docs/transformers.js/guides/node-audio-processing
// https://github.com/xenova/transformers.js/tree/v3/examples/node-audio-processing
export async function processSpeechToTextTask(
	task: SpeechToTextTaskArgs,
	ctx: EngineTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
) {
	const { instance } = ctx
	if (!task.audio) {
		throw new Error('No audio provided')
	}
	const modelComponents = instance.speech || instance.primary
	if (!(modelComponents?.tokenizer && modelComponents?.model)) {
		throw new Error('No speech model loaded')
	}
	const streamer = new TextStreamer(modelComponents.tokenizer, {
		skip_prompt: true,
		// skip_special_tokens: true,
		callback_function: (output: any) => {
			if (task.onChunk) {
				task.onChunk({ text: output })
			}
		},
	})

	let inputSamples = task.audio.samples

	if (task.audio.sampleRate !== 16000) {
		inputSamples = await resampleAudioBuffer(task.audio.samples, {
			inputSampleRate: task.audio.sampleRate,
			outputSampleRate: 16000,
			nChannels: 1,
		})
	}
	const inputs = await modelComponents.processor!(inputSamples)

	if (!('generate' in modelComponents.model)) {
		throw new Error('Speech model class does not support text generation')
	}

	const outputs = await modelComponents.model.generate({
		...inputs,
		max_new_tokens: task.maxTokens ?? 128,
		language: task.language ?? 'en',
		streamer,
	})

	// @ts-ignore
	const outputText = modelComponents.tokenizer.batch_decode(outputs, {
		skip_special_tokens: true,
	})

	return {
		text: outputText[0],
	}
}

// TextGenerationPipeline https://github.com/huggingface/transformers.js/blob/e129c47c65a049173f35e6263fd8d9f660dfc1a7/src/pipelines.js#L2663
export async function processTextToSpeechTask(
	task: TextToSpeechTaskArgs,
	ctx: EngineTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
): Promise<TextToSpeechTaskResult> {
	const { instance } = ctx
	const modelComponents = instance.speech || instance.primary
	if (!modelComponents?.model || !modelComponents?.tokenizer) {
		throw new Error('No speech model loaded')
	}

	if (!('generate_speech' in modelComponents.model)) {
		throw new Error('The model does not support speech generation')
	}

	const encodedInputs = modelComponents.tokenizer(task.text, {
		padding: true,
		truncation: true,
	})

	if (!('speakerEmbeddings' in modelComponents)) {
		throw new Error('No speaker embeddings supplied')
	}

	let speakerEmbeddings = modelComponents.speakerEmbeddings?.[Object.keys(modelComponents.speakerEmbeddings)[0]]

	if (!speakerEmbeddings) {
		throw new Error('No speaker embeddings supplied')
	}

	if (task.voice) {
		speakerEmbeddings = modelComponents.speakerEmbeddings?.[task.voice]
		if (!speakerEmbeddings) {
			throw new Error(`No speaker embeddings found for voice ${task.voice}`)
		}
	}
	
	if (signal?.aborted) {
		throw new Error('Task aborted')
	}
	const speakerEmbeddingsTensor = new Tensor('float32', speakerEmbeddings, [1, speakerEmbeddings.length])
	const outputs = await modelComponents.model.generate_speech(encodedInputs.input_ids, speakerEmbeddingsTensor, {
		vocoder: modelComponents.vocoder,
	})

	if (!outputs.waveform) {
		throw new Error('No waveform generated')
	}

	const sampleRate = modelComponents.processor!.feature_extractor!.config.sampling_rate

	return {
		audio: {
			samples: outputs.waveform.data as Float32Array,
			sampleRate,
			channels: 1,
		},
	}
}

// ObjectDetectionPipeline https://github.com/huggingface/transformers.js/blob/6bd45ac66a861f37f3f95b81ac4b6d796a4ee231/src/pipelines.js#L2336
// ZeroShotObjectDetection https://github.com/huggingface/transformers.js/blob/6bd45ac66a861f37f3f95b81ac4b6d796a4ee231/src/pipelines.js#L2471
export async function processObjectDetectionTask(
	task: ObjectDetectionTaskArgs,
	ctx: EngineTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
): Promise<ObjectDetectionTaskResult> {
	const { instance } = ctx
	if (!task.image) {
		throw new Error('No image provided')
	}
	const image = task.image
	const rawImage = new RawImage(new Uint8ClampedArray(image.data), image.width, image.height, image.channels)
	const modelComponents = instance.vision || instance.primary
	if (!(modelComponents && modelComponents.model)) {
		throw new Error('No model loaded')
	}

	if (signal?.aborted) {
		throw new Error('Task aborted')
	}

	const results: ObjectDetectionResult[] = []

	if (task?.labels?.length) {
		if (!modelComponents.tokenizer || !modelComponents.processor) {
			throw new Error('Model components not loaded.')
		}
		const labelInputs = modelComponents.tokenizer(task.labels, {
			padding: true,
			truncation: true,
		})
		const imageInputs = await modelComponents.processor([rawImage])
		const output = await modelComponents.model({
			...labelInputs,
			pixel_values: imageInputs.pixel_values[0].unsqueeze_(0),
		})

		// @ts-ignore
		const processed = modelComponents.processor.image_processor.post_process_object_detection(
			output,
			task.threshold ?? 0.5,
			[[image.height, image.width]],
			true,
		)[0]
		for (let i = 0; i < processed.boxes.length; i++) {
			results.push({
				score: processed.scores[i],
				label: task.labels[processed.classes[i]],
				box: {
					x: processed.boxes[i][0],
					y: processed.boxes[i][1],
					width: processed.boxes[i][2] - processed.boxes[i][0],
					height: processed.boxes[i][3] - processed.boxes[i][1],
				},
			})
		}
	} else {
		// @ts-ignore
		const { pixel_values, pixel_mask } = await modelComponents.processor([rawImage])
		const output = await modelComponents.model({ pixel_values, pixel_mask })
		// @ts-ignore
		const processed = modelComponents.processor.image_processor.post_process_object_detection(
			output,
			task.threshold ?? 0.5,
			[[image.height, image.width]],
			null,
			false,
		)
		// Add labels
		// @ts-ignore
		const id2label = modelComponents.model.config.id2label
		for (const batch of processed) {
			for (let i = 0; i < batch.boxes.length; i++) {
				results.push({
					score: batch.scores[i],
					label: id2label[batch.classes[i]],
					box: {
						x: batch.boxes[i][0],
						y: batch.boxes[i][1],
						width: batch.boxes[i][2] - batch.boxes[i][0],
						height: batch.boxes[i][3] - batch.boxes[i][1],
					},
				})
			}
		}
	}

	return {
		detections: results,
	}
}

// https://github.com/huggingface/transformers.js/blob/6f43f244e04522545d3d939589c761fdaff057d4/src/pipelines.js#L1135
export async function processTextClassificationTask(
	task: TextClassificationTaskArgs,
	ctx: EngineTaskContext<TransformersJsInstance, TransformersJsModelConfig>,
	signal?: AbortSignal,
): Promise<TextClassificationTaskResult> {
	const { instance } = ctx
	const modelComponents = instance.text || instance.primary
	if (!modelComponents?.tokenizer || !modelComponents?.model) {
		throw new Error('No text model loaded')
	}
	
	if (signal?.aborted) {
		throw new Error('Task aborted')
	}

	if (!task.labels?.length) {
		// Reuse the pipeline for normal text classification
		const pipeline = new TextClassificationPipeline({
			task: 'text-classification',
			model: modelComponents.model,
			tokenizer: modelComponents.tokenizer,
		})
		const pipelineRes = await pipeline(task.input, { top_k: task.topK })
		if (Array.isArray(pipelineRes)) {
			const resultItems = pipelineRes as TextClassificationSingle[]
			const classifications = resultItems.map((item) => {
				const labels = [
					{
						name: item.label,
						score: item.score,
					},
				]
				return { labels }
			})
			return { classifications }
		}

		const singleResultItem = pipelineRes as TextClassificationSingle
		const labels = [
			{
				name: singleResultItem.label,
				score: singleResultItem.score,
			},
		]
		return {
			classifications: [{ labels }],
		}
	}

	// Zero shot classification
	// @ts-ignore
	const label2id = modelComponents.model.config.label2id
	let entailmentId = label2id['entailment']
	if (entailmentId === undefined) {
		console.warn("Could not find 'entailment' in label2id mapping. Using 2 as entailment_id.")
		entailmentId = 2
	}
	let contradictionId = label2id['contradiction'] ?? label2id['not_entailment']
	if (contradictionId === undefined) {
		console.warn("Could not find 'contradiction' in label2id mapping. Using 0 as contradiction_id.")
		contradictionId = 0
	}

	const texts = []
	if (typeof task.input === 'string') {
		texts.push(task.input)
	} else if (Array.isArray(task.input)) {
		texts.push(...task.input)
	} else {
		throw new Error('Invalid input')
	}
	const hypotheses = task.labels!.map((label) => task.hypothesisTemplate!.replace('{}', label))

	// How to perform the softmax over the logits:
	//  - true:  softmax over the entailment vs. contradiction dim for each label independently
	//  - false: softmax the "entailment" logits over all candidate labels
	const softmaxEach = task.labels!.length === 1

	const toReturn = []
	for (const premise of texts) {
		const entailsLogits = []

		for (const hypothesis of hypotheses) {
			const inputs = modelComponents.tokenizer(premise, {
				text_pair: hypothesis,
				padding: true,
				truncation: true,
			})
			const outputs = await modelComponents.model(inputs)

			if (softmaxEach) {
				entailsLogits.push([outputs.logits.data[contradictionId], outputs.logits.data[entailmentId]])
			} else {
				entailsLogits.push(outputs.logits.data[entailmentId])
			}
		}

		const scores = softmaxEach ? entailsLogits.map((x) => softmax(x)[1]) : softmax(entailsLogits)
		const scoresSorted = scores.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0])

		toReturn.push({
			sequence: premise,
			labels: scoresSorted.map((x) => task.labels![x[1]]),
			scores: scoresSorted.map((x) => x[0]),
		})
	}

	const classifications = toReturn.map((x) => {
		let labels = x.labels.map((label, i) => {
			return {
				name: label,
				score: x.scores[i],
			}
		})
		if (task.topK) {
			labels = labels.slice(0, task.topK)
		}
		return { labels }
	})
	return { classifications }
}
