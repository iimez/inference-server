import path from 'node:path'
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
} from '@huggingface/transformers'
import { LogLevels } from '#package/lib/logger.js'
import { decodeAudio } from '#package/lib/decodeAudio.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { moveDirectoryContents } from '#package/lib/moveDirectoryContents.js'
import { TransformersJsModelComponents, SpeechModelInstance } from './types.js'
import { fetchBuffer, parseHuggingfaceModelIdAndBranch, remoteFileExists } from './util.js'
import { validateModelFiles, ModelValidationResult } from './validateModelFiles.js'
import { acquireModelFileLocks } from './acquireModelFileLocks.js'
import { loadModelComponents, loadSpeechModelComponents } from './loadModelComponents.js'

interface TransformersJsInstance {
	textModel?: TransformersJsModelComponents
	visionModel?: TransformersJsModelComponents
	speechModel?: SpeechModelInstance
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

export interface TransformersJsModelConfig extends ModelConfig {
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
		modelOpts: TransformersJsModel | TransformersJsSpeechModel,
		{ modelId, branch }: { modelId: string; branch: string },
		requiredComponents: string[] = ['model', 'tokenizer', 'processor', 'vocoder'],
	) => {
		const modelClass = modelOpts.modelClass ?? AutoModel
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
				const tokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
				const tokenizerDownload = tokenizerClass.from_pretrained(modelId, {
					revision: branch,
					progress_callback: progressCallback,
				})
				downloadPromises.tokenizer = tokenizerDownload
			}
		}

		if (requiredComponents.includes('processor')) {
			if (modelOpts.processor?.url) {
				const { modelId, branch } = parseHuggingfaceModelIdAndBranch(modelOpts.processor.url)
				const processorDownload = AutoProcessor.from_pretrained(modelId, {
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
					const processorDownload = AutoProcessor.from_pretrained(modelId, {
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
		return modelComponents
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
		const noModelConfigured = !config.textModel && !config.visionModel && !config.speechModel
		if (config.textModel || noModelConfigured) {
			const requiredComponents = validationResult.errors?.textModel
				? Object.keys(validationResult.errors.textModel)
				: undefined
			modelDownloadPromises.push(downloadModelFiles(config.textModel || {}, { modelId, branch }, requiredComponents))
		}
		if (config.visionModel) {
			const requiredComponents = validationResult.errors?.visionModel
				? Object.keys(validationResult.errors.visionModel)
				: undefined
			modelDownloadPromises.push(downloadModelFiles(config.visionModel, { modelId, branch }, requiredComponents))
			if (config.visionModel.processor?.url) {
				const processorPath = resolveModelFileLocation({
					url: config.visionModel.processor.url,
					filePath: config.visionModel.processor.file,
					modelsCachePath: config.modelsCachePath,
				})
				const { modelId } = parseHuggingfaceModelIdAndBranch(config.visionModel.processor.url)
				const processorCacheDir = path.join(env.cacheDir, modelId)
				directoriesToCopy[processorCacheDir] = processorPath
			}
		}
		if (config.speechModel) {
			const requiredComponents = validationResult.errors?.speechModel
				? Object.keys(validationResult.errors.speechModel)
				: undefined
			modelDownloadPromises.push(downloadModelFiles(config.speechModel, { modelId, branch }, requiredComponents))
			if (config.speechModel.vocoder?.url) {
				const vocoderPath = resolveModelFileLocation({
					url: config.speechModel.vocoder.url,
					filePath: config.speechModel.vocoder.file,
					modelsCachePath: config.modelsCachePath,
				})
				const { modelId } = parseHuggingfaceModelIdAndBranch(config.speechModel.vocoder.url)
				const vocoderCacheDir = path.join(env.cacheDir, modelId)
				directoriesToCopy[vocoderCacheDir] = vocoderPath
			}
			if (config.speechModel?.speakerEmbeddings) {
				const speakerEmbeddingsPromises = []
				for (const speakerEmbedding of Object.values(config.speechModel.speakerEmbeddings)) {
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
				modelDownloadPromises.push(Promise.all(speakerEmbeddingsPromises))
			}
		}

		await Promise.all(modelDownloadPromises)

		if (signal?.aborted) {
			return
		}
		// move all downloads to their final location
		// console.debug('Copying directories', directoriesToCopy)
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
	const modelLoadPromises = []
	const noModelConfigured = !config.textModel && !config.visionModel && !config.speechModel

	if (config.textModel || noModelConfigured) {
		modelLoadPromises.push(loadModelComponents(config.textModel || {}, config))
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
		textModel: models[0],
		visionModel: models[1],
		speechModel: models[2],
		// vocoderModel: models[3],
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
	if (instance.textModel) {
		disposePromises.push(disposeModelComponents(instance.textModel))
	}
	if (instance.visionModel) {
		disposePromises.push(disposeModelComponents(instance.visionModel))
	}
	if (instance.speechModel) {
		disposePromises.push(disposeModelComponents(instance.speechModel))
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
	if (!(instance.textModel?.tokenizer && instance.textModel?.model)) {
		throw new Error('Text model is not loaded.')
	}
	if (!('generate' in instance.textModel.model)) {
		throw new Error('Text model does not support generation.')
	}
	const inputTokens = instance.textModel.tokenizer(request.prompt)
	const outputTokens = await instance.textModel.model.generate({
		...inputTokens,
		max_new_tokens: request.maxTokens ?? 128,
	})
	// @ts-ignore
	const outputText = instance.textModel.tokenizer.batch_decode(outputTokens, {
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
			if (!instance.textModel?.tokenizer || !instance.textModel?.model) {
				throw new Error('Text model is not loaded.')
			}
			modelInputs = instance.textModel.tokenizer(embeddingInput.content, {
				padding: true, // pads input if it is shorter than context window
				truncation: true, // truncates input if it exceeds context window
			})
			inputTokens += modelInputs.input_ids.size
			// @ts-ignore TODO check _call
			const modelOutputs = await instance.textModel.model(modelInputs)
			result =
				modelOutputs.last_hidden_state ??
				modelOutputs.logits ??
				modelOutputs.token_embeddings ??
				modelOutputs.text_embeds
		} else if (embeddingInput.type === 'image') {
			if (!instance.visionModel?.processor || !instance.visionModel?.model) {
				throw new Error('Vision model is not loaded.')
			}
			const { data, info } = await embeddingInput.content.handle.raw().toBuffer({ resolveWithObject: true })
			const image = new RawImage(new Uint8ClampedArray(data), info.width, info.height, info.channels)
			modelInputs = await instance.visionModel.processor!(image)
			// @ts-ignore TODO check _call
			const modelOutputs = await instance.visionModel.model(modelInputs)
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
	const { data, info } = await request.image.handle.raw().toBuffer({ resolveWithObject: true })
	const image = new RawImage(new Uint8ClampedArray(data), info.width, info.height, info.channels)

	if (signal?.aborted) {
		return
	}

	const model = instance.visionModel || instance.textModel
	if (!(model && model.tokenizer && model.processor && model.model)) {
		throw new Error('No model loaded')
	}
	if (!('generate' in model.model)) {
		throw new Error('Model does not support generation')
	}
	let textInputs = {}
	if (request.prompt) {
		textInputs = model!.tokenizer(request.prompt)
	}
	const imageInputs = await model.processor(image)
	const outputTokens = await model.model.generate({
		...textInputs,
		...imageInputs,
		max_new_tokens: request.maxTokens ?? 128,
	})
	// @ts-ignore
	const outputText = model.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})

	return {
		text: outputText[0],
	}
}

async function readAudioFile(filePath: string) {
	const WHISPER_SAMPLING_RATE = 16_000
	const MAX_AUDIO_LENGTH = 30 // seconds
	const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH
	// Read the file into a buffer
	const fileBuffer = fs.readFileSync(filePath)

	// Decode the audio data
	let decodedAudio = await decodeAudio(fileBuffer, WHISPER_SAMPLING_RATE)

	// Trim the audio data if it exceeds MAX_SAMPLES
	if (decodedAudio.length > MAX_SAMPLES) {
		decodedAudio = decodedAudio.slice(-MAX_SAMPLES)
	}

	return decodedAudio
}

// see examples
// https://huggingface.co/docs/transformers.js/guides/node-audio-processing
// https://github.com/xenova/transformers.js/tree/v3/examples/node-audio-processing
export async function processSpeechToTextTask(
	{ request, onChunk }: EngineSpeechToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	if (!(instance.speechModel?.tokenizer && instance.speechModel?.model)) {
		throw new Error('No speech model loaded')
	}
	const streamer = new TextStreamer(instance.speechModel.tokenizer, {
		skip_prompt: true,
		// skip_special_tokens: true,
		callback_function: (output: any) => {
			if (onChunk) {
				onChunk({ text: output })
			}
		},
	})
	let inputs
	if (request.file) {
		const audio = await readAudioFile(request.file)
		inputs = await instance.speechModel.processor!(audio)
	}

	if (!('generate' in instance.speechModel.model)) {
		throw new Error('Speech model class does not support text generation')
	}

	const outputs = await instance.speechModel.model.generate({
		...inputs,
		max_new_tokens: request.maxTokens ?? 128,
		language: request.language ?? 'en',
		streamer,
	})

	// @ts-ignore
	const outputText = instance.speechModel.tokenizer.batch_decode(outputs, {
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
	if (!instance.speechModel || !instance.speechModel?.model || !instance.speechModel?.tokenizer) {
		throw new Error('No speech model loaded')
	}

	if (!('generate_speech' in instance.speechModel.model)) {
		console.debug(instance.speechModel.model)
		throw new Error('The model does not support speech generation')
	}

	const encodedInputs = instance.speechModel.tokenizer(request.text, {
		padding: true,
		truncation: true,
	})

	let speakerEmbeddings =
		instance.speechModel.speakerEmbeddings?.[Object.keys(instance.speechModel.speakerEmbeddings)[0]]

	if (!speakerEmbeddings) {
		throw new Error('No speaker embeddings supplied')
	}

	if (request.voice) {
		speakerEmbeddings = instance.speechModel.speakerEmbeddings?.[request.voice]
		if (!speakerEmbeddings) {
			throw new Error(`No speaker embeddings found for voice ${request.voice}`)
		}
	}
	const speakerEmbeddingsTensor = new Tensor('float32', speakerEmbeddings, [1, speakerEmbeddings.length])
	const outputs = await instance.speechModel.model.generate_speech(encodedInputs.input_ids, speakerEmbeddingsTensor, {
		vocoder: instance.speechModel.vocoder,
	})

	if (!outputs.waveform) {
		throw new Error('No waveform generated')
	}

	const sampleRate = instance.speechModel.processor!.feature_extractor.config.sampling_rate

	return {
		audio: {
			samples: outputs.waveform.data,
			sampleRate,
			channels: 1,
		},
	}
}
