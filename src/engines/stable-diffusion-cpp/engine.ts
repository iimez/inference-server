import StableDiffusion from '@lmagder/node-stable-diffusion-cpp'
import { gguf } from '@huggingface/gguf'
import fs from 'node:fs'
import path from 'node:path'
import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	TextToImageTaskResult,
	ModelFileSource,
	Image,
	TextToImageTaskArgs,
	EngineTaskContext,
	ImageToImageTaskArgs,
} from '#package/types/index.js'
import { LogLevel, LogLevels } from '#package/lib/logger.js'
import { downloadModelFile } from '#package/lib/downloadModelFile.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { acquireFileLock } from '#package/lib/acquireFileLock.js'
import { getRandomNumber } from '#package/lib/util.js'
import { StableDiffusionSamplingMethod, StableDiffusionSchedule, StableDiffusionWeightType } from './types.js'
import { validateModelFiles, ModelValidationResult } from './validateModelFiles.js'
import { parseQuantization, getWeightType, getSamplingMethod } from './util.js'

export interface StableDiffusionInstance {
	context: StableDiffusion.Context
}

export interface StableDiffusionModelConfig extends ModelConfig {
	location: string
	sha256?: string
	clipL?: ModelFileSource
	clipG?: ModelFileSource
	vae?: ModelFileSource
	t5xxl?: ModelFileSource
	controlNet?: ModelFileSource
	taesd?: ModelFileSource
	diffusionModel?: boolean
	model?: ModelFileSource
	loras?: ModelFileSource[]
	samplingMethod?: StableDiffusionSamplingMethod
	weightType?: StableDiffusionWeightType
	schedule?: StableDiffusionSchedule
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		cpuThreads?: number
	}
}

interface StableDiffusionModelMeta {
	gguf: any
}

export const autoGpu = true

export async function prepareModel(
	{ config, log }: EngineContext<StableDiffusionModelConfig, StableDiffusionModelMeta>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(path.dirname(config.location), { recursive: true })
	const releaseFileLock = await acquireFileLock(config.location)
	if (signal?.aborted) {
		releaseFileLock()
		return
	}
	log(LogLevels.info, `Preparing stable-diffusion model at ${config.location}`, {
		model: config.id,
	})

	const downloadModel = (url: string, validationResult: ModelValidationResult) => {
		log(LogLevels.info, `${validationResult.message} - Downloading model files`, {
			model: config.id,
			url: config.url,
			location: config.location,
			errors: validationResult.errors,
		})
		const downloadPromises = []
		if (validationResult.errors.model && config.location) {
			downloadPromises.push(
				downloadModelFile({
					url: url,
					filePath: config.location,
					modelsCachePath: config.modelsCachePath,
					onProgress,
					signal,
				}),
			)
		}
		const pushDownload = (src: ModelFileSource) => {
			if (!src.url) {
				return
			}
			downloadPromises.push(
				downloadModelFile({
					url: src.url,
					filePath: src.file,
					modelsCachePath: config.modelsCachePath,
					onProgress,
					signal,
				}),
			)
		}
		if (validationResult.errors.clipG && config.clipG) {
			pushDownload(config.clipG)
		}
		if (validationResult.errors.clipL && config.clipL) {
			pushDownload(config.clipL)
		}
		if (validationResult.errors.vae && config.vae) {
			pushDownload(config.vae)
		}
		if (validationResult.errors.t5xxl && config.t5xxl) {
			pushDownload(config.t5xxl)
		}
		if (validationResult.errors.controlNet && config.controlNet) {
			pushDownload(config.controlNet)
		}
		if (validationResult.errors.taesd && config.taesd) {
			pushDownload(config.taesd)
		}
		if (config.loras) {
			for (const lora of config.loras) {
				if (!lora.url) {
					continue
				}
				pushDownload(lora)
			}
		}
		return Promise.all(downloadPromises)
	}
	try {
		if (signal?.aborted) {
			return
		}

		const validationResults = await validateModelFiles(config)
		if (signal?.aborted) {
			return
		}
		if (validationResults) {
			if (config.url) {
				await downloadModel(config.url, validationResults)
			} else {
				throw new Error(`${validationResults.message} - No URL provided`)
			}
		}

		const finalValidationError = await validateModelFiles(config)
		if (finalValidationError) {
			throw new Error(`Downloaded files are invalid: ${finalValidationError}`)
		}

		const result: any = {}
		if (config.location.endsWith('.gguf')) {
			const { metadata, tensorInfos } = await gguf(config.location, {
				allowLocalFile: true,
			})
			result.gguf = metadata
		}
		return result
	} catch (error) {
		throw error
	} finally {
		releaseFileLock()
	}
}

export async function createInstance({ config, log }: EngineContext<StableDiffusionModelConfig>, signal?: AbortSignal) {
	log(LogLevels.debug, 'Load Stable Diffusion model', config)
	const handleLog = (level: string, message: string) => {
		log(level as LogLevel, message)
	}
	const handleProgress = (step: number, steps: number, time: number) => {
		log(LogLevels.debug, `Progress: ${step}/${steps} (${time}ms)`)
	}

	const resolveComponentLocation = (src?: ModelFileSource) => {
		if (src) {
			return resolveModelFileLocation({
				url: src.url,
				filePath: src.file,
				modelsCachePath: config.modelsCachePath,
			})
		}
		return undefined
	}

	const vaeFilePath = resolveComponentLocation(config.vae)
	const clipLFilePath = resolveComponentLocation(config.clipL)
	const clipGFilePath = resolveComponentLocation(config.clipG)
	const t5xxlFilePath = resolveComponentLocation(config.t5xxl)
	const controlNetFilePath = resolveComponentLocation(config.controlNet)
	const taesdFilePath = resolveComponentLocation(config.taesd)

	let weightType = config.weightType ? getWeightType(config.weightType) : undefined
	if (typeof weightType === 'undefined') {
		const quantization = parseQuantization(config.location)
		if (quantization) {
			weightType = getWeightType(quantization)
		}
	}

	if (typeof weightType === 'undefined') {
		log(LogLevels.warn, 'Failed to parse model weight type (quantization) from file name, falling back to f32', {
			file: config.location,
		})
	}

	const loraDir = path.join(path.dirname(config.location), 'loras')
	const contextParams = {
		model: !config.diffusionModel ? config.location : undefined,
		diffusionModel: config.diffusionModel ? config.location : undefined,
		numThreads: config.device?.cpuThreads,
		vae: vaeFilePath,
		clipL: clipLFilePath,
		clipG: clipGFilePath,
		t5xxl: t5xxlFilePath,
		controlNet: controlNetFilePath,
		taesd: taesdFilePath,
		weightType: weightType,
		loraDir: loraDir,
		// TODO how to expose?
		// keepClipOnCpu: true,
		// keepControlNetOnCpu: true,
		// keepVaeOnCpu: true,
	}
	log(LogLevels.debug, 'Creating context with', contextParams)
	const context = await StableDiffusion.createContext(
		// @ts-ignore
		contextParams,
		handleLog,
		handleProgress,
	)

	return {
		context,
	}
}

export async function processTextToImageTask(
	task: TextToImageTaskArgs,
	ctx: EngineTaskContext<StableDiffusionInstance, StableDiffusionModelConfig, StableDiffusionModelMeta>,
	signal?: AbortSignal,
): Promise<TextToImageTaskResult> {
	const { instance, config, log } = ctx
	const seed = task.seed ?? getRandomNumber(0, 1000000)
	const results = await instance.context.txt2img({
		prompt: task.prompt,
		negativePrompt: task.negativePrompt,
		width: task.width || 512,
		height: task.height || 512,
		batchCount: task.batchCount,
		sampleMethod: getSamplingMethod(task.samplingMethod || config.samplingMethod),
		sampleSteps: task.sampleSteps,
		cfgScale: task.cfgScale,
		guidance: task.guidance,
		styleRatio: task.styleRatio,
		controlStrength: task.controlStrength,
		normalizeInput: false,
		seed,
	})

	const images: Image[] = []
	for (const [idx, img] of results.entries()) {
		images.push({
			data: img.data,
			width: img.width,
			height: img.height,
			channels: img.channel,
		})
	}
	if (!images.length) {
		throw new Error('No images generated')
	}
	return {
		images: images,
		seed,
	}
}

export async function processImageToImageTask(
	task: ImageToImageTaskArgs,
	ctx: EngineTaskContext<StableDiffusionInstance, StableDiffusionModelConfig, StableDiffusionModelMeta>,
	signal?: AbortSignal,
): Promise<TextToImageTaskResult> {
	const { instance, config, log } = ctx
	const seed = task.seed ?? getRandomNumber(0, 1000000)
	console.debug('processImageToImageTask', {
		width: task.image.width,
		height: task.image.height,
		channel: task.image.channels as 3 | 4,
	})
	const initImage = {
		// data: await request.image.handle.raw().toBuffer(),
		data: task.image.data,
		width: task.image.width,
		height: task.image.height,
		channel: task.image.channels as 3 | 4,
	}
	const results = await instance.context.img2img({
		initImage,
		prompt: task.prompt,
		width: task.width || 512,
		height: task.height || 512,
		batchCount: task.batchCount,
		sampleMethod: getSamplingMethod(task.samplingMethod || config.samplingMethod),
		cfgScale: task.cfgScale,
		sampleSteps: task.sampleSteps,
		guidance: task.guidance,
		strength: task.strength,
		styleRatio: task.styleRatio,
		controlStrength: task.controlStrength,
		seed,
	})

	const images: Image[] = []

	for (const [idx, img] of results.entries()) {
		// console.debug('img', {
		// 	id: idx,
		// 	width: img.width,
		// 	height: img.height,
		// 	channels: img.channel,
		// })

		images.push({
			data: img.data,
			width: img.width,
			height: img.height,
			channels: img.channel,
		})
	}
	if (!images.length) {
		throw new Error('No images generated')
	}
	return {
		images: images,
		seed,
	}
}
