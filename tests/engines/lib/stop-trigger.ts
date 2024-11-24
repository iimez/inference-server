import { expect } from 'vitest'
import { InferenceServer } from '#package/server.js'
import { createChatCompletion, createTextCompletion } from '../../util/completions.js'

export async function runStopTriggerTest(inferenceServer: InferenceServer) {
	const chat = await createChatCompletion(inferenceServer, {
		messages: [
			{
				role: 'user',
				content: "This is a test. Please only answer with 'OK'.",
			},
		],
		stop: ['OK'],
		maxTokens: 10,
	})
	// console.debug({
	// 	response: chat.result.message.content,
	// })
	expect(chat.result.finishReason).toBe('stopTrigger')
	expect(chat.result.message.content).toBe('')
	const completion = await createTextCompletion(inferenceServer, {
		prompt: "Let's count to four. One, two,",
		stop: [' three'],
		maxTokens: 10,
	})
	expect(completion.result.finishReason).toBe('stopTrigger')
	expect(completion.result.text).toBe('')
}
