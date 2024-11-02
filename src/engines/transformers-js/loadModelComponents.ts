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
	SpeechT5ForTextToSpeech,
	PreTrainedTokenizer,
	WhisperForConditionalGeneration,
	Tensor,
} from '@huggingface/transformers'
import { LogLevels } from '#package/lib/logger.js'
import { TransformersJsModelConfig } from './engine.js'
import { TransformersJsModelComponents, SpeechModelInstance } from './types.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'

export async function loadModelComponents<TModel extends TransformersJsModelComponents = TransformersJsModelComponents>(
	modelOpts: TransformersJsModel | TransformersJsSpeechModel,
	config: TransformersJsModelConfig,
): Promise<TModel> {
	const device = config.device?.gpu ? 'gpu' : 'cpu'
	const modelClass = modelOpts.modelClass ?? AutoModel
	let modelPath = config.location
	if (!modelPath.endsWith('/')) {
		modelPath += '/'
	}
	const loadPromises = []
	const modelPromise = modelClass.from_pretrained(modelPath, {
		local_files_only: true,
		device: device,
		dtype: modelOpts.dtype || 'fp32',
	})
	loadPromises.push(modelPromise)
	const tokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
	const tokenizerPromise = tokenizerClass.from_pretrained(modelPath, {
		local_files_only: true,
	})
	loadPromises.push(tokenizerPromise)

	const hasPreprocessor = fs.existsSync(modelPath + 'preprocessor_config.json')
	const hasProcessor = fs.existsSync(modelPath + 'processor_config.json')

	if (hasProcessor || hasPreprocessor || modelOpts.processor) {
		const processorClass = modelOpts.processorClass ?? AutoProcessor
		if (modelOpts.processor) {
			const processorPath = resolveModelFileLocation({
				url: modelOpts.processor.url,
				filePath: modelOpts.processor.file,
				modelsCachePath: config.modelsCachePath,
			})
			const processorPromise = processorClass.from_pretrained(processorPath, {
				local_files_only: true,
			})
			loadPromises.push(processorPromise)
		} else {
			const processorPromise = processorClass.from_pretrained(modelPath, {
				local_files_only: true,
			})
			loadPromises.push(processorPromise)
		}
	}

	const loadedComponents = await Promise.all(loadPromises)
	const modelComponents: TransformersJsModelComponents = {}
	if (loadedComponents[0]) {
		modelComponents.model = loadedComponents[0] as PreTrainedModel
	}
	if (loadedComponents[1]) {
		modelComponents.tokenizer = loadedComponents[1] as PreTrainedTokenizer
	}
	if (loadedComponents[2]) {
		modelComponents.processor = loadedComponents[2] as Processor
	}
	return modelComponents as TModel
}

export async function loadSpeechModelComponents(
	modelOpts: TransformersJsSpeechModel,
	config: TransformersJsModelConfig,
): Promise<SpeechModelInstance> {
	const loadPromises: Promise<unknown>[] = [loadModelComponents(modelOpts, config)]

	if (modelOpts.vocoder) {
		const vocoderClass = modelOpts.vocoderClass ?? AutoModel
		const vocoderPath = resolveModelFileLocation({
			url: modelOpts.vocoder.url,
			filePath: modelOpts.vocoder.file,
			modelsCachePath: config.modelsCachePath,
		})
		const vocoderPromise = vocoderClass.from_pretrained(vocoderPath, {
			local_files_only: true,
		})
		loadPromises.push(vocoderPromise)
	} else {
		loadPromises.push(Promise.resolve(undefined))
	}

	if ('speakerEmbeddings' in modelOpts) {
		const speakerEmbeddings = modelOpts.speakerEmbeddings
		const speakerEmbeddingsPromises = []

		for (const speakerEmbedding of Object.values(speakerEmbeddings)) {
			if (speakerEmbedding instanceof Float32Array) {
				speakerEmbeddingsPromises.push(Promise.resolve(speakerEmbedding))
				continue
			}
			const speakerEmbeddingPath = resolveModelFileLocation({
				url: speakerEmbedding.url,
				filePath: speakerEmbedding.file,
				modelsCachePath: config.modelsCachePath,
			})
			const speakerEmbeddingPromise = fs.promises
				.readFile(speakerEmbeddingPath)
				.then((data) => new Float32Array(data.buffer))
			speakerEmbeddingsPromises.push(speakerEmbeddingPromise)
		}
		loadPromises.push(Promise.all(speakerEmbeddingsPromises))
	}
	const loadedComponents = await Promise.all(loadPromises)
	const speechModelInstance: SpeechModelInstance = loadedComponents[0] as TransformersJsModelComponents
	if (loadedComponents[1]) {
		speechModelInstance.vocoder = loadedComponents[1] as TransformersJsModelComponents
	}
	if (loadedComponents[2]) {
		const loadedSpeakerEmbeddings = loadedComponents[2] as Float32Array[]
		speechModelInstance.speakerEmbeddings = Object.fromEntries(
			Object.keys(modelOpts.speakerEmbeddings).map((key, index) => [key, loadedSpeakerEmbeddings[index]]),
		)
	}
	return speechModelInstance
}