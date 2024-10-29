import fs from 'node:fs'
import { calculateFileChecksum } from '#package/lib/calculateFileChecksum.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { ModelFileSource } from '#package/types/index.js'
import { StableDiffusionModelConfig } from './engine.js'

interface ModelValidationErrors {
	model?: string
	clipL?: string
	clipG?: string
	vae?: string
	t5xxl?: string
	controlNet?: string
	taesd?: string
	lora?: {
		[index: number]: string
	}
}

export interface ModelValidationResult {
	message: string
	errors: ModelValidationErrors
}

export async function validateModelFiles(
	config: StableDiffusionModelConfig,
): Promise<ModelValidationResult | undefined> {
	const validateFile = async (component: string, src: ModelFileSource) => {
		const fileLocation = resolveModelFileLocation({
			url: src.url,
			filePath: src.file,
			modelsCachePath: config.modelsCachePath,
		})
		if (!fs.existsSync(fileLocation)) {
			return {
				component,
				message: `${component} file missing at ${fileLocation}`,
			}
		}
		const ipullFile = fileLocation + '.ipull'
		let validatedChecksum = false
		if (fs.existsSync(ipullFile)) {
			// if we have a valid file at the download destination, we can remove the ipull file
			if (src.sha256) {
				const fileHash = await calculateFileChecksum(fileLocation, 'sha256')
				if (fileHash === src.sha256) {
					fs.unlinkSync(ipullFile)
					validatedChecksum = true
				}
			}
			if (!validatedChecksum) {
				return {
					component,
					message: `${component} with incomplete download`,
				}
			}
		}

		if (!validatedChecksum && src.sha256) {
			const fileHash = await calculateFileChecksum(fileLocation, 'sha256')
			if (fileHash !== src.sha256) {
				return {
					component,
					message: `${component} file sha256 checksum mismatch: expected ${src.sha256} got ${fileHash} at ${fileLocation}`,
				}
			}
		}
		return undefined
	}

	const validationPromises = []
	if (config.clipL) {
		validationPromises.push(validateFile('clipL', config.clipL))
	}
	if (config.clipG) {
		validationPromises.push(validateFile('clipG', config.clipG))
	}
	if (config.vae) {
		validationPromises.push(validateFile('vae', config.vae))
	}
	if (config.t5xxl) {
		validationPromises.push(validateFile('t5xxl', config.t5xxl))
	}
	if (config.controlNet) {
		validationPromises.push(validateFile('controlNet', config.controlNet))
	}
	if (config.taesd) {
		validationPromises.push(validateFile('taesd', config.taesd))
	}
	if (config.sha256) {
		validationPromises.push(validateFile('model', { file: config.location, sha256: config.sha256 }))
	}

	// const loraDir = path.join(path.dirname(config.location), 'lora')
	if (config.loras) {
		for (const lora of config.loras) {
			validationPromises.push(validateFile('lora', lora))

			// const loraFile = path.join(loraDir, getModelFileName(lora))
			// if (!fs.existsSync(loraFile)) {
			// 	return `lora file missing: ${loraFile}`
			// }
		}
	}

	const res = await Promise.all(validationPromises)
	const validationErrors = res.filter((e) => !!e)
	if (validationErrors.length) {
		return {
			message: 'Invalid model files',
			errors: validationErrors.reduce((acc, e) => {
				acc[e.component as keyof ModelValidationErrors] = e.message
				return acc
			}, {} as ModelValidationErrors),
		}
	}

	return undefined
}
