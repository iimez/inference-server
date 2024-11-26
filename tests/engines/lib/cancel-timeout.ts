import { expect } from 'vitest'
import { InferenceServer } from '#package/server.js'
import { ChatCompletionTaskArgs } from '#package/types/index.js'

export async function runTimeoutTest(inferenceServer: InferenceServer) {
	const args: ChatCompletionTaskArgs = {
		model: 'test',
		messages: [
			{
				role: 'user',
				content: 'Tell me a long story.',
			},
		],
	}
	const lock = await inferenceServer.pool.requestInstance(args)
	const task = lock.instance.processChatCompletionTask({ ...args, timeout: 500 })
	const result = await task.result
	expect(result.message.content).toBeDefined()
	expect(result.finishReason).toBe('timeout')
	await lock.release()
	// console.debug({
	// 	response: result.message,
	// })
}

export async function runCancellationTest(inferenceServer: InferenceServer) {
	const args: ChatCompletionTaskArgs = {
		model: 'test',
		messages: [
			{
				role: 'user',
				content: 'Tell me a long story.',
			},
		],
	}
	const lock = await inferenceServer.pool.requestInstance(args)
	const task = lock.instance.processChatCompletionTask(args)
	setTimeout(() => {
		task.cancel()
	}, 500)
	const result = await task.result
	expect(result.message.content).toBeDefined()
	expect(result.finishReason).toBe('cancel')
	await lock.release()
	// console.debug({
	// 	response: result.message,
	// })
}
