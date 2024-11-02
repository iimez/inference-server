import fs from 'node:fs'
import path from 'node:path'
import { env } from '@huggingface/transformers'
import { acquireFileLock } from '#package/lib/acquireFileLock.js'
import { TransformersJsModelConfig } from './engine.js'
import { parseHuggingfaceModelIdAndBranch } from './util.js'

export async function acquireModelFileLocks(config: TransformersJsModelConfig, signal?: AbortSignal) {
	const requestedLocks: Array<Promise<() => void>> = []
	const modelId = config.id
	const modelCacheDir = path.join(env.cacheDir, modelId)
	fs.mkdirSync(modelCacheDir, { recursive: true })
	requestedLocks.push(acquireFileLock(modelCacheDir, signal))
	if (config.visionModel?.processor?.url) {
		const { modelId } = parseHuggingfaceModelIdAndBranch(config.visionModel.processor.url)
		const processorCacheDir = path.join(env.cacheDir, modelId)
		fs.mkdirSync(processorCacheDir, { recursive: true })
		requestedLocks.push(acquireFileLock(processorCacheDir, signal))
	}
	const acquiredLocks = await Promise.all(requestedLocks)
	return () => {
		for (const releaseLock of acquiredLocks) {
			releaseLock()
		}
	}
}
