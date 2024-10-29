

import fs from 'node:fs'
import { calculateFileChecksum } from '#package/lib/calculateFileChecksum.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'

interface ValidatableModelConfig {
	url?: string
	location?: string
	modelsCachePath: string
	sha256?: string
}

export async function validateModelFile(config: ValidatableModelConfig): Promise<string | undefined> {
	const fileLocation = resolveModelFileLocation({
		url: config.url,
		filePath: config.location,
		modelsCachePath: config.modelsCachePath,
	})
	if (!fs.existsSync(fileLocation)) {
		return `Model file missing at ${fileLocation}`
	}
	const ipullFile = fileLocation + '.ipull'
	let validatedChecksum = false
	if (fs.existsSync(ipullFile)) {
		// if we have a valid file at the download destination, we can remove the ipull file
		if (config.sha256) {
			const fileHash = await calculateFileChecksum(fileLocation, 'sha256')
			if (fileHash === config.sha256) {
				fs.unlinkSync(ipullFile)
				validatedChecksum = true
			}
		}
		if (!validatedChecksum) {
			return `Model file with incomplete download`
		}
	}

	if (!validatedChecksum && config.sha256) {
		const fileHash = await calculateFileChecksum(fileLocation, 'sha256')
		if (fileHash !== config.sha256) {
			return `File sha256 checksum mismatch: expected ${config.sha256} got ${fileHash} at ${fileLocation}`
		}
	}
	return undefined
}
