import StableDiffusion from '@lmagder/node-stable-diffusion-cpp'
import { gguf } from '@huggingface/gguf'
import sharp from 'sharp'
import fs from 'node:fs'
import path from 'node:path'
import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	EngineTextToImageResult,
	ModelFileSource,
	EngineTextToImageArgs,
	Image,
	EngineImageToImageArgs,
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
		await releaseFileLock()
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
					modelsPath: config.modelsPath,
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
					modelsPath: config.modelsPath,
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
		const validationResults = await validateModelFiles(config)
		if (signal?.aborted) {
			return
		}
		if (validationResults) {
			if (config.url) {
				await downloadModel(config.url, validationResults)
			} else {
				throw new Error(`Model files are invalid: ${validationResults.message}`)
			}
		}

		const finalValidationError = await validateModelFiles(config)
		if (finalValidationError) {
			throw new Error(`Model files are invalid: ${finalValidationError}`)
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
		await releaseFileLock()
		throw error
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

	const vaeFilePath = config.vae
		? resolveModelFileLocation({ url: config.vae.url, filePath: config.vae.file, modelsPath: config.modelsPath })
		: undefined
	const clipLFilePath = config.clipL
		? resolveModelFileLocation({ url: config.clipL.url, filePath: config.clipL.file, modelsPath: config.modelsPath })
		: undefined
	const clipGFilePath = config.clipG
		? resolveModelFileLocation({ url: config.clipG.url, filePath: config.clipG.file, modelsPath: config.modelsPath })
		: undefined
	const t5xxlFilePath = config.t5xxl
		? resolveModelFileLocation({ url: config.t5xxl.url, filePath: config.t5xxl.file, modelsPath: config.modelsPath })
		: undefined
	const controlNetFilePath = config.controlNet
		? resolveModelFileLocation({
				url: config.controlNet.url,
				filePath: config.controlNet.file,
				modelsPath: config.modelsPath,
		  })
		: undefined
	const taesdFilePath = config.taesd
		? resolveModelFileLocation({ url: config.taesd.url, filePath: config.taesd.file, modelsPath: config.modelsPath })
		: undefined

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
	{ request, config, log }: EngineTextToImageArgs<StableDiffusionModelConfig>,
	instance: StableDiffusionInstance,
	signal?: AbortSignal,
): Promise<EngineTextToImageResult> {
	const seed = request.seed ?? getRandomNumber(0, 1000000)
	const results = await instance.context.txt2img({
		prompt: request.prompt,
		negativePrompt: request.negativePrompt,
		width: request.width || 512,
		height: request.height || 512,
		batchCount: request.batchCount,
		sampleMethod: getSamplingMethod(request.samplingMethod || config.samplingMethod),
		sampleSteps: request.sampleSteps,
		cfgScale: request.cfgScale,
		// @ts-ignore
		guidance: request.guidance,
		styleRatio: request.styleRatio,
		controlStrength: request.controlStrength,
		normalizeInput: false,
		seed,
	})

	const images: Image[] = []
	for (const [idx, img] of results.entries()) {
		images.push({
			handle: sharp(img.data, {
				raw: {
					width: img.width,
					height: img.height,
					channels: img.channel,
				},
			}),
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
	{ request, config, log }: EngineImageToImageArgs<StableDiffusionModelConfig>,
	instance: StableDiffusionInstance,
	signal?: AbortSignal,
): Promise<EngineTextToImageResult> {
	const seed = request.seed ?? getRandomNumber(0, 1000000)
	console.debug('processImageToImageTask', {
		width: request.image.width,
		height: request.image.height,
		channel: request.image.channels as 3 | 4,
	})
	const initImage = {
		data: await request.image.handle.raw().toBuffer(),
		width: request.image.width,
		height: request.image.height,
		channel: request.image.channels as 3 | 4,
	}
	const results = await instance.context.img2img({
		initImage,
		prompt: request.prompt,
		width: request.width || 512,
		height: request.height || 512,
		batchCount: request.batchCount,
		sampleMethod: getSamplingMethod(request.samplingMethod || config.samplingMethod),
		cfgScale: request.cfgScale,
		sampleSteps: request.sampleSteps,
		// @ts-ignore
		guidance: request.guidance,
		strength: request.strength,
		styleRatio: request.styleRatio,
		controlStrength: request.controlStrength,
		seed,
	})

	const images: Image[] = []
	// to sharp
	// const imagePromises = results.map(async (img, idx) => {
	// 	return await sharp(img.data, {
	// 			raw: {
	// 				width: img.width,
	// 				height: img.height,
	// 				channels: img.channel,
	// 			},
	// 		})
	// 	})

	for (const [idx, img] of results.entries()) {
		console.debug('img', {
			id: idx,
			width: img.width,
			height: img.height,
			channels: img.channel,
		})

		images.push({
			handle: sharp(img.data, {
				raw: {
					width: img.width,
					height: img.height,
					channels: img.channel,
				},
			}),
			width: img.width,
			height: img.height,
			channels: img.channel,
		})

		// images.push({
		// 	data: img.data,
		// 	width: img.width,
		// 	height: img.height,
		// 	channels: img.channel,
		// })
	}
	if (!images.length) {
		throw new Error('No images generated')
	}
	return {
		images: images,
		seed,
	}
}
