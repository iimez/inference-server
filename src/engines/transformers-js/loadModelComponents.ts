import fs from 'node:fs'
import {
	AutoModel,
	AutoProcessor,
	AutoTokenizer,
	Processor,
	PreTrainedModel,
	PreTrainedTokenizer,
} from '@huggingface/transformers'
import { TransformersJsModel, TransformersJsSpeechModel } from '#package/types/index.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { TransformersJsModelConfig, TransformersJsModelComponents, SpeechModelComponents } from './engine.js'
import { TransformersJsModelClass, TransformersJsProcessorClass, TransformersJsTokenizerClass } from './types.js'
import { normalizeTransformersJsClass } from './util.js'

export async function loadModelComponents<TModel extends TransformersJsModelComponents = TransformersJsModelComponents>(
	modelOpts: TransformersJsModel | TransformersJsModel & TransformersJsSpeechModel,
	config: TransformersJsModelConfig,
): Promise<TModel> {
	const device = config.device?.gpu ? 'gpu' : 'cpu'
	const modelClass = normalizeTransformersJsClass<TransformersJsModelClass>(modelOpts.modelClass, AutoModel)
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

	const hasTokenizer = fs.existsSync(modelPath + 'tokenizer.json')
	if (hasTokenizer) {
		const tokenizerClass = normalizeTransformersJsClass<TransformersJsTokenizerClass>(modelOpts.tokenizerClass, AutoTokenizer)
		const tokenizerPromise = tokenizerClass.from_pretrained(modelPath, {
			local_files_only: true,
		})
		loadPromises.push(tokenizerPromise)
	} else {
		loadPromises.push(Promise.resolve(undefined))
	}

	const hasPreprocessor = fs.existsSync(modelPath + 'preprocessor_config.json')
	const hasProcessor = fs.existsSync(modelPath + 'processor_config.json')
	if (hasProcessor || hasPreprocessor || modelOpts.processor) {
		const processorClass = normalizeTransformersJsClass<TransformersJsProcessorClass>(modelOpts.processorClass, AutoProcessor)
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
	} else {
		loadPromises.push(Promise.resolve(undefined))
	}
	

	if ('vocoder' in modelOpts && modelOpts.vocoder) {
		const vocoderClass = normalizeTransformersJsClass<TransformersJsModelClass>(modelOpts.vocoderClass, AutoModel)
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

	if ('speakerEmbeddings' in modelOpts && modelOpts.speakerEmbeddings) {
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
	} else {
		loadPromises.push(Promise.resolve(undefined))
	}

	const loadedComponents = await Promise.all(loadPromises)
	const modelComponents: SpeechModelComponents = {}
	if (loadedComponents[0]) {
		modelComponents.model = loadedComponents[0] as PreTrainedModel
	}
	if (loadedComponents[1]) {
		modelComponents.tokenizer = loadedComponents[1] as PreTrainedTokenizer
	}
	if (loadedComponents[2]) {
		modelComponents.processor = loadedComponents[2] as Processor
	}
	if (loadedComponents[3]) {
		modelComponents.vocoder = loadedComponents[3] as PreTrainedModel
	}
	if (loadedComponents[4] && 'speakerEmbeddings' in modelOpts && modelOpts.speakerEmbeddings) {
		const loadedSpeakerEmbeddings = loadedComponents[4] as Float32Array[]
		modelComponents.speakerEmbeddings = Object.fromEntries(
			Object.keys(modelOpts.speakerEmbeddings).map((key, index) => [key, loadedSpeakerEmbeddings[index]]),
		)
	}
	return modelComponents as TModel
}

export async function loadSpeechModelComponents(
	modelOpts: TransformersJsSpeechModel,
	config: TransformersJsModelConfig,
): Promise<SpeechModelComponents> {
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

	if ('speakerEmbeddings' in modelOpts && modelOpts.speakerEmbeddings) {
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
	const speechModelInstance: SpeechModelComponents = loadedComponents[0] as TransformersJsModelComponents
	if (loadedComponents[1]) {
		speechModelInstance.vocoder = loadedComponents[1] as PreTrainedModel
	}
	if (loadedComponents[2] && 'speakerEmbeddings' in modelOpts && modelOpts.speakerEmbeddings) {
		const loadedSpeakerEmbeddings = loadedComponents[2] as Float32Array[]
		speechModelInstance.speakerEmbeddings = Object.fromEntries(
			Object.keys(modelOpts.speakerEmbeddings).map((key, index) => [key, loadedSpeakerEmbeddings[index]]),
		)
	}
	return speechModelInstance
}
