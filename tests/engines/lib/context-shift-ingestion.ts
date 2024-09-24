import fs from 'node:fs'
import { expect } from 'vitest'
import { ModelServer } from '#lllms/server.js'
import { ChatMessage } from '#lllms/types/index.js'
import { createChatCompletion } from '../../util.js'

// conversation that tests behavior when context window is exceeded while model is ingesting text
export async function runIngestionContextShiftTest(
	llms: ModelServer,
	model: string = 'test',
) {
	
	const text = fs.readFileSync('tests/fixtures/lovecraft.txt', 'utf-8')
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: text + '\n\nWhats that? Tell me more.',
		},
	]
	const ingestionResponse = await createChatCompletion(llms, {
		model,
		messages,
		maxTokens: 512,
	})
	
	messages.push({
		role: 'assistant',
		content: ingestionResponse.result.message.content,
	})
	
	messages.push({
		role: 'user',
		content: 'Did your last response look sane? If so, answer only with "OK".',
	})
	
	const validationResponse = await createChatCompletion(llms, {
		model,
		messages,
		maxTokens: 8,
	})
	expect(validationResponse.result.message.content).toMatch(/OK/i)
}
