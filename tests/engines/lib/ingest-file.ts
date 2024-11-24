import fs from 'node:fs'
import { InferenceServer } from '#package/server.js'
import { ChatMessage } from '#package/types/index.js'
import { createChatCompletion } from '../../util/completions.js'

export async function runFileIngestionTest(
	inferenceServer: InferenceServer,
	file: string,
	prompt: string = 'Whats that?',
	model: string = 'test',
) {
	const text = fs.readFileSync(`tests/fixtures/${file}.txt`, 'utf-8')
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: text + '\n---\n\n' + prompt,
		},
	]
	const response = await createChatCompletion(inferenceServer, {
		model,
		messages,
		maxTokens: 256,
	})
	return response.result
}
