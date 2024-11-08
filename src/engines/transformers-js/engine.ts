import path, { normalize } from 'node:path'
import fs from 'node:fs'
import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	EngineImageToTextArgs,
	EngineSpeechToTextArgs,
	EngineTextToSpeechArgs,
	EngineTextCompletionResult,
	EngineTextCompletionArgs,
	EngineEmbeddingArgs,
	EngineEmbeddingResult,
	ImageEmbeddingInput,
	TransformersJsModel,
	TextEmbeddingInput,
	TransformersJsSpeechModel,
	EngineObjectRecognitionArgs,
	ObjectRecognitionResult,
	SpeakerEmbeddings,
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

interface TransformersJsDownloadProgress {
	status: 'progress' | 'done' | 'initiate'
	name: string
	file: string
	progress: number
	loaded: number
	total: number
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
		console.debug({ modelOpts, modelId, branch, requiredComponents })
		const modelClass = normalizeTransformersJsClass<TransformersJsModelClass>(modelOpts.modelClass, AutoModel)
		const downloadPromises: Record<string, Promise<any> | undefined> = {}
		const progressCallback = (progress: TransformersJsDownloadProgress) => {
			if (onProgress && progress.status === 'progress') {
				onProgress({
					file: env.cacheDir + progress.name + '/' + progress.file,
					loadedBytes: progress.loaded,
					totalBytes: progress.total,
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
		const prepareDownloadedVocoder = (vocoderOpts: { url?: string, file?: string }) => {
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
		const prepareDownloadedProcessor = (processorOpts: { url?: string, file?: string }) => {
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
	// const noModelConfigured = !config.textModel && !config.visionModel && !config.speechModel

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

	// TODO preload whisper / any speech to text?
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

export async function processTextCompletionTask(
	{ request, config, log, onChunk }: EngineTextCompletionArgs<TransformersJsModelConfig>,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
): Promise<EngineTextCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for text completion.')
	}
	if (!(instance.primary?.tokenizer && instance.primary?.model)) {
		throw new Error('Text model is not loaded.')
	}
	if (!('generate' in instance.primary.model)) {
		throw new Error('Text model does not support generation.')
	}
	const inputTokens = instance.primary.tokenizer(request.prompt)
	const outputTokens = await instance.primary.model.generate({
		...inputTokens,
		max_new_tokens: request.maxTokens ?? 128,
	})
	// @ts-ignore
	const outputText = instance.primary.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})

	return {
		finishReason: 'eogToken',
		text: outputText[0],
		promptTokens: inputTokens.length,
		// @ts-ignore
		completionTokens: outputTokens.length,
		// @ts-ignore
		contextTokens: inputTokens.length + outputTokens.length,
	}
}

// see https://github.com/xenova/transformers.js/blob/v3/src/utils/tensor.js
// https://github.com/xenova/transformers.js/blob/v3/src/pipelines.js#L1284
export async function processEmbeddingTask(
	{ request, config }: EngineEmbeddingArgs<TransformersJsModelConfig>,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
): Promise<EngineEmbeddingResult> {
	if (!request.input) {
		throw new Error('Input is required for embedding.')
	}
	const inputs = Array.isArray(request.input) ? request.input : [request.input]
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

		if (request.pooling) {
			result = applyPooling(result, request.pooling, modelInputs)
		}
		if (request.dimensions && result.data.length > request.dimensions) {
			embeddings.push(truncateDimensions(result, request.dimensions))
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
	{ request, config, log }: EngineImageToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	if (!request.image) {
		throw new Error('No image provided')
	}
	const image = request.image
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
	if (request.prompt) {
		textInputs = modelComponents!.tokenizer(request.prompt)
	}
	const imageInputs = await modelComponents.processor(rawImage)
	const outputTokens = await modelComponents.model.generate({
		...textInputs,
		...imageInputs,
		max_new_tokens: request.maxTokens ?? 128,
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
	{ request, onChunk }: EngineSpeechToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	if (!request.audio) {
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
			if (onChunk) {
				onChunk({ text: output })
			}
		},
	})

	let inputSamples = request.audio.samples

	if (request.audio.sampleRate !== 16000) {
		inputSamples = await resampleAudioBuffer(request.audio.samples, {
			inputSampleRate: request.audio.sampleRate,
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
		max_new_tokens: request.maxTokens ?? 128,
		language: request.language ?? 'en',
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

// see https://github.com/huggingface/transformers.js/blob/e129c47c65a049173f35e6263fd8d9f660dfc1a7/src/pipelines.js#L2663
export async function processTextToSpeechTask(
	{ request, config, log }: EngineTextToSpeechArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	const modelComponents = instance.speech || instance.primary
	if (!modelComponents?.model || !modelComponents?.tokenizer) {
		throw new Error('No speech model loaded')
	}

	if (!('generate_speech' in modelComponents.model)) {
		throw new Error('The model does not support speech generation')
	}

	const encodedInputs = modelComponents.tokenizer(request.text, {
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

	if (request.voice) {
		speakerEmbeddings = modelComponents.speakerEmbeddings?.[request.voice]
		if (!speakerEmbeddings) {
			throw new Error(`No speaker embeddings found for voice ${request.voice}`)
		}
	}
	const speakerEmbeddingsTensor = new Tensor('float32', speakerEmbeddings, [1, speakerEmbeddings.length])
	const outputs = await modelComponents.model.generate_speech(encodedInputs.input_ids, speakerEmbeddingsTensor, {
		vocoder: modelComponents.vocoder,
	})

	if (!outputs.waveform) {
		throw new Error('No waveform generated')
	}

	const sampleRate = modelComponents.processor!.feature_extractor.config.sampling_rate

	return {
		audio: {
			samples: outputs.waveform.data,
			sampleRate,
			channels: 1,
		},
	}
}

// ObjectDetectionPipeline https://github.com/huggingface/transformers.js/blob/6bd45ac66a861f37f3f95b81ac4b6d796a4ee231/src/pipelines.js#L2336
// ZeroShotObjectDetection https://github.com/huggingface/transformers.js/blob/6bd45ac66a861f37f3f95b81ac4b6d796a4ee231/src/pipelines.js#L2471
export async function processObjectRecognitionTask(
	{ request, config, log }: EngineObjectRecognitionArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	if (!request.image) {
		throw new Error('No image provided')
	}
	// const { data, info } = await request.image.handle.raw().toBuffer({ resolveWithObject: true })
	// const image = new RawImage(new Uint8ClampedArray(data), info.width, info.height, info.channels)
	const image = request.image
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
	if (!(modelComponents && modelComponents.model)) {
		throw new Error('No model loaded')
	}

	const results: ObjectRecognitionResult[] = []

	if (request?.labels?.length) {
		if (!modelComponents.tokenizer || !modelComponents.processor) {
			throw new Error('Model components not loaded.')
		}
		const labelInputs = modelComponents.tokenizer(request.labels, {
			padding: true,
			truncation: true,
		})
		const imageInputs = await modelComponents.processor([rawImage])
		const output = await modelComponents.model({
			...labelInputs,
			pixel_values: imageInputs.pixel_values[0].unsqueeze_(0),
		})

		// @ts-ignore
		const processed = modelComponents.processor.feature_extractor.post_process_object_detection(
			output,
			request.threshold ?? 0.5,
			[[image.height, image.width]],
			true,
		)[0]
		for (let i = 0; i < processed.boxes.length; i++) {
			results.push({
				score: processed.scores[i],
				label: request.labels[processed.classes[i]],
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
		const { pixel_values, pixel_mask } = await modelComponents.processor.feature_extractor([rawImage])
		const output = await modelComponents.model({ pixel_values, pixel_mask })
		// @ts-ignore
		const processed = modelComponents.processor.feature_extractor.post_process_object_detection(
			output,
			request.threshold ?? 0.5,
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
		objects: results,
	}
}
