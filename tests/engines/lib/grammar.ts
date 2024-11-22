import { expect } from 'vitest'
import { ModelServer } from '#package/server.js'
import { createChatCompletion } from '../../util/completions.js'
import { ChatMessage } from '#package/types/index.js'

export async function runBuiltInGrammarTest(modelServer: ModelServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Answer with a JSON object containing the key "test" with the value "test". And an array of cats, just strings with names.',
		},
	]
	const turn1 = await createChatCompletion(modelServer, {
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

	const firstCat = turn1Data.cats[0]
	messages.push({
		role: 'user',
		content: 'Write a haiku using the name of the first cat in the array.',
	})
	const turn2 = await createChatCompletion(modelServer, {
		messages,
	})
	// console.debug({
	// 	turn2: turn2.result.message.content,
	// })
	expect(turn2.result.message.content).toContain(firstCat)

}

export async function runRawGBNFGrammarTest(modelServer: ModelServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Generate a {name, age}[] JSON array with at most 10 famous actors of all ages.',
		},
	]
	const turn1 = await createChatCompletion(modelServer, {
		grammar: 'custom-gbnf-string',
		messages,
		maxTokens: 512,
	})
	// console.debug({
	// 	turn1: turn1.result.message.content,
	// })
	expect(turn1.result.message.content).toBeTruthy()
	const data = JSON.parse(turn1.result.message.content!)
	expect(data).toBeInstanceOf(Array)
	expect(data[0].name).toBeTruthy()
	expect(data[0].age).toBeDefined()
}

export async function runJsonSchemaGrammarTest(modelServer: ModelServer) {

	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Generate a {name, age} JSON object of a famous actor.',
		},
	]
	const turn1 = await createChatCompletion(modelServer, {
		grammar: 'custom-json-schema',
		messages,
		maxTokens: 512,
	})
	expect(turn1.result.message.content).toBeTruthy()
	const data = JSON.parse(turn1.result.message.content!)

	expect(data.name).toBeTruthy()
	expect(data.age).toBeGreaterThan(0)
}
