import type { ModelServer } from '#package/server.js'
import { ChatCompletionRequest, TextCompletionRequest } from '#package/types/index.js'

const testDefaults = {
	model: 'test',
	temperature: 0,
	maxTokens: 64,
}
const defaultTimeout = 30000

export async function createChatCompletion(
	server: ModelServer,
	args: Omit<ChatCompletionRequest, 'model'> & { model?: string },
	timeout = defaultTimeout
) {
	const mergedArgs = {
		...testDefaults,
		...args,
	}
	const lock = await server.pool.requestInstance(mergedArgs)
	const task = lock.instance.processChatCompletionTask(mergedArgs, { timeout })
	const device = lock.instance.gpu ? 'gpu' : 'cpu'
	try {
		await task.result
	} catch (error) {
		console.debug('error happened', error.message)
		console.error('Error in createChatCompletion', error)
		await lock.release()
		throw error
	}
	const result = await task.result
	await lock.release()
	return { task, result, device }
}

export async function createTextCompletion(
	server: ModelServer,
	args: Omit<TextCompletionRequest, 'model'> & { model?: string },
	timeout = defaultTimeout
) {
	const mergedArgs = {
		...testDefaults,
		...args,
	}
	const lock = await server.pool.requestInstance(mergedArgs)
	const task = lock.instance.processTextCompletionTask(mergedArgs, { timeout })
	const device = lock.instance.gpu ? 'gpu' : 'cpu'
	const result = await task.result
	await lock.release()
	return { task, result, device }
}

export function parseInstanceId(completionId: string) {
	// part after the last ":" because model names may include colons
	const afterModelName = completionId.split(':').pop()
	const instanceId = afterModelName?.split('-')[0] // rest is instanceId-completionId
	return instanceId
}