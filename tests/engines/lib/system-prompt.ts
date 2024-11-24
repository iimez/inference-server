import { expect } from 'vitest'
import { InferenceServer } from '#package/server.js'
import { createChatCompletion } from '../../util/completions.js'

export async function runSystemMessageTest(inferenceServer: InferenceServer) {
	const chat = await createChatCompletion(inferenceServer, {
		messages: [
			{
				role: 'system',
				content:
					'Roleplay as Batman. Answer as if you are Batman, never say you are an Assistant or AI model.',
			},
			{
				role: 'user',
				content: 'Hey im Robin! Whats the plan tonight?',
			},
		],
	})
	// console.debug({
	// 	response: chat.result.message.content,
	// })
	expect(chat.result.message.content).toMatch(/Gotham|Joker|Riddler|criminal/)
	expect(chat.result.message.content).not.toMatch(/Assistant|AI|model/)
}
