import { expect } from 'vitest'
import { ModelServer } from '#package/server.js'
import { createChatCompletion, createTextCompletion } from '../../util.js'

export async function runStopTriggerTest(modelServer: ModelServer) {
	const chat = await createChatCompletion(modelServer, {
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
	const completion = await createTextCompletion(modelServer, {
		prompt: "Let's count to four. One, two,",
		stop: [' three'],
		maxTokens: 10,
	})
	expect(completion.result.finishReason).toBe('stopTrigger')
	expect(completion.result.text).toBe('')
}
