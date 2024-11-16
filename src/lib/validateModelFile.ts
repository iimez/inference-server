import fs from 'node:fs'
import { calculateFileChecksum } from '#package/lib/calculateFileChecksum.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { readGgufFileInfo } from 'node-llama-cpp'

interface ValidatableModelConfig {
	url?: string
	location?: string
	modelsCachePath: string
	sha256?: string
	md5?: string
}

interface ValidationResult {
	error?: string
	meta?: {
		gguf?: any
	}
}

export async function validateModelFile(
	config: ValidatableModelConfig,
	signal?: AbortSignal,
): Promise<ValidationResult> {
	const fileLocation = resolveModelFileLocation({
		url: config.url,
		filePath: config.location,
		modelsCachePath: config.modelsCachePath,
	})
	if (!fs.existsSync(fileLocation)) {
		return {
			error: `Model file missing at ${fileLocation}`,
		}
	}
	const stats = fs.statSync(fileLocation)
	if (stats.size === 0) {
		return {
			error: `Model file at ${fileLocation} is empty`,
		}
	}
	const hasChecksum = config.sha256 || config.md5
	let validatedChecksum = false
	const validateChecksum = async () => {
		if (config.md5) {
			const fileHash = await calculateFileChecksum(fileLocation, 'md5')
			if (fileHash === config.md5) {
				validatedChecksum = true
				return true
			}
		} else if (config.sha256) {
			const fileHash = await calculateFileChecksum(fileLocation, 'sha256')
			if (fileHash === config.sha256) {
				validatedChecksum = true
				return true
			}
		}
		return false
	}
	const ipullFile = fileLocation + '.ipull'
	if (fs.existsSync(ipullFile)) {
		// if we have a valid file at the download destination, we can remove the ipull file
		if (hasChecksum) {
			const isValid = await validateChecksum()
			if (isValid) {
				fs.unlinkSync(ipullFile)
				validatedChecksum = true
			}
		}
		if (!validatedChecksum) {
			return {
				error: `Model file with incomplete download`,
			}
		}
	}

	if (!validatedChecksum && hasChecksum) {
		if (config.sha256) {
			const fileHash = await calculateFileChecksum(fileLocation, 'sha256')
			if (fileHash !== config.sha256) {
				return {
					error: `File sha256 checksum mismatch: expected ${config.sha256} got ${fileHash} at ${fileLocation}`,
				}
			}
		} else if (config.md5) {
			const fileHash = await calculateFileChecksum(fileLocation, 'md5')
			if (fileHash !== config.md5) {
				return {
					error: `File md5 checksum mismatch: expected ${config.md5} got ${fileHash} at ${fileLocation}`,
				}
			}
		}
	}

	if (config.location?.endsWith('.gguf')) {
		try {
			const ggufMeta = await readGgufFileInfo(config.location!, {
				signal,
				ignoreKeys: [
					'gguf.tokenizer.ggml.merges',
					'gguf.tokenizer.ggml.tokens',
					'gguf.tokenizer.ggml.scores',
					'gguf.tokenizer.ggml.token_type',
				],
			})
			return { meta: { gguf: ggufMeta } }
		} catch (err) {
			return {
				error: `Invalid GGUF: ${err}`,
			}
		}
	}

	return { meta: {} }
}
