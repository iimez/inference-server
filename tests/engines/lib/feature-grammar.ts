import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { createChatCompletion } from '../../util.js'
import { ChatMessage } from '#lllms/types/index.js'

export async function runGrammarTest(llms: LLMServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Answer with a JSON object containing the key "test" with the value "test". And an array of cats, just strings with names.',
		},
	]
	const turn1 = await createChatCompletion(llms, {
		grammar: 'json',
		messages,
	})
	// console.debug({
	// 	turn1: turn1.result.message.content,
	// })
	messages.push(turn1.result.message)
	expect(turn1.result.message.content).toBeTruthy()
	const turn1Data = JSON.parse(turn1.result.message.content!)
	expect(turn1Data.test).toMatch(/test/)
	expect(turn1Data.cats).toBeInstanceOf(Array)
	
	const secondCat = turn1Data.cats[1]
	messages.push({
		role: 'user',
		content: 'Write a haiku on the second cat in the array.',
	})
	const turn2 = await createChatCompletion(llms, {
		messages,
	})
	// console.debug({
	// 	turn2: turn2.result.message.content,
	// })
	expect(turn2.result.message.content).toContain(secondCat)
	
}