import fs from 'node:fs'
import { AutoModel, AutoProcessor, AutoTokenizer } from '@huggingface/transformers'
import { TransformersJsModel } from '#package/types/index.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { TransformersJsModelConfig } from './engine.js'

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

async function validateTokenizer(modelOpts: TransformersJsModel, modelPath: string): Promise<string | undefined> {
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

async function validateProcessor(modelOpts: TransformersJsModel, modelPath: string): Promise<string | undefined> {
	const processorClass = modelOpts.processorClass ?? AutoProcessor
	try {
		if (modelOpts.processor) {
			const processorPath = resolveModelFileLocation({
				url: modelOpts.processor.url,
				filePath: modelOpts.processor.file,
				modelsPath: modelPath,
			})
			await processorClass.from_pretrained(processorPath, {
				local_files_only: true,
			})
		} else {
			await processorClass.from_pretrained(modelPath, {
				local_files_only: true,
			})
		}
	} catch (error) {
		return `Failed to load processor (${error})`
	}
	return undefined
}

interface ComponentValidationErrors {
	model?: string
	tokenizer?: string
	processor?: string
}

interface ModelValidationErrors {
	textModel?: ComponentValidationErrors
	visionModel?: ComponentValidationErrors
	speechModel?: ComponentValidationErrors
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

	const validateModelComponents = async (modelOpts: TransformersJsModel) => {
		const componentValidationPromises = [
			validateModel(modelOpts, config, modelPath),
			validateTokenizer(modelOpts, modelPath),
		]
		if (modelOpts.processor) {
			componentValidationPromises.push(validateProcessor(modelOpts, modelPath))
		}
		const [model, tokenizer, processor] = await Promise.all(componentValidationPromises)
		const result: ComponentValidationErrors = {}
		if (model) result.model = model
		if (tokenizer) result.tokenizer = tokenizer
		if (processor) result.processor = processor
		return result
	}

	const modelValidationPromises: any = {}
	const noModelConfigured = !config.textModel && !config.visionModel && !config.speechModel
	if (config.textModel || noModelConfigured) {
		modelValidationPromises.textModel = validateModelComponents(config.textModel || {})
	}
	if (config.visionModel) {
		modelValidationPromises.visionModel = validateModelComponents(config.visionModel)
	}
	if (config.speechModel) {
		modelValidationPromises.speechModel = validateModelComponents(config.speechModel)
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

	if (Object.keys(validationErrors).length > 0) {
		return {
			message: 'failed to load model components',
			errors: validationErrors,
		}
	}
	return undefined
}
