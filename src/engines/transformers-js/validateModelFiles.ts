import fs from 'node:fs'
import {
	AutoModel,
	AutoProcessor,
	AutoTokenizer,
	TextToAudioPipeline,
	TextToAudioPipelineConstructorArgs,
	TextToAudioPipelineType,
} from '@huggingface/transformers'
import { TransformersJsModel, TransformersJsSpeechModel } from '#package/types/index.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { TransformersJsModelConfig } from './engine.js'
import { parseHuggingfaceModelIdAndBranch, remoteFileExists } from './util.js'

async function validateModel(
	modelOpts: TransformersJsModel,
	config: TransformersJsModelConfig,
	modelPath: string,
): Promise<string | undefined> {
	const modelClass = modelOpts.modelClass ?? AutoModel
	const device = config.device?.gpu ? 'gpu' : 'cpu'
	try {
		const model = await modelClass.from_pretrained(modelPath, {
			local_files_only: true,
			device: device,
			dtype: modelOpts.dtype || 'fp32',
		})
		await model.dispose()
	} catch (error) {
		return `Failed to load model (${error})`
	}
	return undefined
}

async function validateTokenizer(
	modelOpts: TransformersJsModel,
	config: TransformersJsModelConfig,
	modelPath: string,
): Promise<string | undefined> {
	const tokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
	try {
		await tokenizerClass.from_pretrained(modelPath, {
			local_files_only: true,
		})
	} catch (error) {
		return `Failed to load tokenizer (${error})`
	}
	return undefined
}

async function validateProcessor(
	modelOpts: TransformersJsModel,
	config: TransformersJsModelConfig,
	modelPath: string,
): Promise<string | undefined> {
	const processorClass = modelOpts.processorClass ?? AutoProcessor
	try {
		if (modelOpts.processor) {
			const processorPath = resolveModelFileLocation({
				url: modelOpts.processor.url,
				filePath: modelOpts.processor.file,
				modelsCachePath: config.modelsCachePath,
			})
			await processorClass.from_pretrained(processorPath, {
				local_files_only: true,
			})
		} else {
			if (modelOpts.processorClass) {
				await processorClass.from_pretrained(modelPath, {
					local_files_only: true,
				})
			} else if (config.url) {
				const { branch } = parseHuggingfaceModelIdAndBranch(config.url)
				const [hasProcessor, hasPreprocessor] = await Promise.all([
					remoteFileExists(`${config.url}/blob/${branch}/processor_config.json`),
					remoteFileExists(`${config.url}/blob/${branch}/preprocessor_config.json`),
				])
				if (hasProcessor || hasPreprocessor) {
					await processorClass.from_pretrained(modelPath, {
						local_files_only: true,
					})
				}
			}
		}
	} catch (error) {
		return `Failed to load processor (${error})`
	}
	return undefined
}

async function validateVocoder(
	modelOpts: TransformersJsSpeechModel,
	config: TransformersJsModelConfig,
	modelPath: string,
): Promise<string | undefined> {
	const vocoderClass = modelOpts.vocoderClass ?? AutoModel
	if (modelOpts.vocoder) {
		const vocoderPath = resolveModelFileLocation({
			url: modelOpts.vocoder.url,
			filePath: modelOpts.vocoder.file,
			modelsCachePath: config.modelsCachePath,
		})
		try {
			await vocoderClass.from_pretrained(vocoderPath, {
				local_files_only: true,
			})
		} catch (error) {
			return `Failed to load vocoder (${error})`
		}
	}
	return undefined
}

async function validateModelComponents(
	modelOpts: TransformersJsModel,
	config: TransformersJsModelConfig,
	modelPath: string,
) {
	const componentValidationPromises = [
		validateModel(modelOpts, config, modelPath),
		validateTokenizer(modelOpts, config, modelPath),
		validateProcessor(modelOpts, config, modelPath),
	]

	if ('vocoder' in modelOpts) {
		componentValidationPromises.push(validateVocoder(modelOpts as TransformersJsSpeechModel, config, modelPath))
	}

	const [model, tokenizer, processor, vocoder] = await Promise.all(componentValidationPromises)
	const result: ComponentValidationErrors = {}
	if (model) result.model = model
	if (tokenizer) result.tokenizer = tokenizer
	if (processor) result.processor = processor
	if (vocoder) result.vocoder = vocoder
	return result
}

async function validateSpeechModel(
	modelOpts: TransformersJsSpeechModel,
	config: TransformersJsModelConfig,
	modelPath: string,
) {
	if (modelOpts.speakerEmbeddings) {
		
		for (const voice of Object.values(modelOpts.speakerEmbeddings)) {
			const speakerEmbeddingsPath = resolveModelFileLocation({
				url: voice.url,
				filePath: voice.file,
				modelsCachePath: config.modelsCachePath,
			})
			if (!fs.existsSync(speakerEmbeddingsPath)) {
				return `Speaker embeddings file does not exist: ${speakerEmbeddingsPath}`
			}
		}
	}
	return validateModelComponents(modelOpts, config, modelPath)
}

interface ComponentValidationErrors {
	model?: string
	tokenizer?: string
	processor?: string
	vocoder?: string
}

interface ModelValidationErrors {
	textModel?: ComponentValidationErrors
	visionModel?: ComponentValidationErrors
	speechModel?: ComponentValidationErrors
	vocoderModel?: ComponentValidationErrors
}

export interface ModelValidationResult {
	message: string
	errors?: ModelValidationErrors
}

export async function validateModelFiles(
	config: TransformersJsModelConfig,
): Promise<ModelValidationResult | undefined> {
	if (!fs.existsSync(config.location)) {
		return {
			message: `model directory does not exist: ${config.location}`,
		}
	}

	let modelPath = config.location
	if (!modelPath.endsWith('/')) {
		modelPath += '/'
	}

	const modelValidationPromises: any = {}
	const noModelConfigured = !config.textModel && !config.visionModel && !config.speechModel
	if (config.textModel || noModelConfigured) {
		modelValidationPromises.textModel = validateModelComponents(config.textModel || {}, config, modelPath)
	}
	if (config.visionModel) {
		modelValidationPromises.visionModel = validateModelComponents(config.visionModel, config, modelPath)
	}
	if (config.speechModel) {
		modelValidationPromises.speechModel = validateSpeechModel(config.speechModel, config, modelPath)
	}

	await Promise.all(Object.values(modelValidationPromises))
	const validationErrors: ModelValidationErrors = {}
	const textModelErrors = await modelValidationPromises.textModel
	if (textModelErrors && Object.keys(textModelErrors).length) {
		validationErrors.textModel = textModelErrors
	}
	const visionModelErrors = await modelValidationPromises.visionModel
	if (visionModelErrors && Object.keys(visionModelErrors).length) {
		validationErrors.visionModel = visionModelErrors
	}
	const speechModelErrors = await modelValidationPromises.speechModel
	if (speechModelErrors && Object.keys(speechModelErrors).length) {
		validationErrors.speechModel = speechModelErrors
	}

	const vocoderModelErrors = await modelValidationPromises.vocoderModel
	if (vocoderModelErrors && Object.keys(vocoderModelErrors).length) {
		validationErrors.vocoderModel = vocoderModelErrors
	}

	if (Object.keys(validationErrors).length > 0) {
		return {
			message: 'Failed to validate model components',
			errors: validationErrors,
		}
	}
	return undefined
}
