import { expect } from 'vitest'
import { ModelServer } from '#package/server.js'
import { ChatMessage } from '#package/types/index.js'
import { createChatCompletion, parseInstanceId } from '../../util.js'

// conversation that tests whether the instance cache will be kept around for a follow up
// while also handling intermediate incoming completion requests.
export async function runContextReuseTest(
	modelServer: ModelServer,
	model: string = 'test',
) {
	// middle part of the completion id is the instance uid.
	// we'll use this to verify which instance handled a completion.
	const messagesA: ChatMessage[] = [
		{ role: 'user', content: 'Tell me a fun fact about bears.' },
	]
	const responseA1 = await createChatCompletion(modelServer, {
		maxTokens: 100,
		messages: messagesA,
	})
	const instanceIdA1 = parseInstanceId(responseA1.task.id)
	// do a unrelated chat completion that should be picked up by the other instance
	const responseB1 = await createChatCompletion(modelServer, {
		maxTokens: 100,
		messages: [
			{
				role: 'user',
				content: 'Write a haiku about pancakes.',
			},
		],
	})
	const instanceIdB1 = parseInstanceId(responseB1.task.id)
	expect(instanceIdA1).not.toBe(instanceIdB1)
	// send a follow up turn to see if context is still there
	messagesA.push(responseA1.result.message, {
		role: 'user',
		content: 'Continue.',
	})
	const responseA2 = await createChatCompletion(modelServer, {
		maxTokens: 100,
		messages: messagesA,
	})
	const instanceIdA2 = parseInstanceId(responseA2.task.id)
	expect(instanceIdA1).toBe(instanceIdA2)
	// assert its still about bears
	expect(responseA2.result.message.content).toMatch(/bear/i)
}
