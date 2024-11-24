import { expect } from 'vitest'
import { InferenceServer } from '#package/server.js'
import { createChatCompletion } from '../../util/completions.js'

export async function runTokenBiasTest(inferenceServer: InferenceServer) {
	const unbiasedChat = await createChatCompletion(inferenceServer, {
		messages: [
			{
				role: 'user',
				content: 'Please finish my sentence: "Once upon a..."',
			},
		],
	})
	// console.debug({
	// 	unbiasedResponse: unbiasedChat.result.message.content,
	// })
	expect(unbiasedChat.result.message.content).toMatch(/time/)
	const biasedChat = await createChatCompletion(inferenceServer, {
		tokenBias: {
			'time': -100,
			'a time...': -100,
			'...times': -100,
		},
		messages: [
			{
				role: 'user',
				content: 'Please finish my sentence: "Once upon a..."',
			},
		],
	})
	// console.debug({
	// 	biasedResponse: biasedChat.result.message.content,
	// })
	expect(biasedChat.result.message.content).not.toMatch(/time/)
}
