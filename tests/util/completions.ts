import type { InferenceServer } from '#package/server.js'
import { ChatCompletionTaskArgs, TextCompletionTaskArgs } from '#package/types/index.js'

const testDefaults = {
	model: 'test',
	temperature: 0,
	maxTokens: 64,
}
const defaultTimeout = 30000

export async function createChatCompletion(
	server: InferenceServer,
	args: Omit<ChatCompletionTaskArgs, 'model'> & { model?: string },
	timeout = defaultTimeout
) {
	const mergedArgs = {
		...testDefaults,
		...args,
		timeout,
	}
	const lock = await server.pool.requestInstance(mergedArgs)
	const task = lock.instance.processChatCompletionTask(mergedArgs)
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
	server: InferenceServer,
	args: Omit<TextCompletionTaskArgs, 'model'> & { model?: string },
	timeout = defaultTimeout
) {
	const mergedArgs = {
		...testDefaults,
		...args,
		timeout,
	}
	const lock = await server.pool.requestInstance(mergedArgs)
	const task = lock.instance.processTextCompletionTask(mergedArgs)
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