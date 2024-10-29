import { ChatMessage as GPT4AllChatMessage } from 'gpt4all'
import { ChatMessage } from '#package/types/index.js'
import { flattenMessageTextContent } from '#package/lib/flattenMessageTextContent.js'

export function createChatMessageArray(
	messages: ChatMessage[],
): GPT4AllChatMessage[] {
	const chatMessages: GPT4AllChatMessage[] = []
	let systemPrompt: string | undefined
	for (const message of messages) {
		if (message.role === 'user' || message.role === 'assistant') {
			chatMessages.push({
				role: message.role,
				content: flattenMessageTextContent(message.content),
			})
		} else if (message.role === 'system') {
			if (systemPrompt) {
				systemPrompt += '\n\n' + message.content
			} else {
				systemPrompt = flattenMessageTextContent(message.content)
			}
		}
	}
	if (systemPrompt) {
		chatMessages.unshift({
			role: 'system',
			content: systemPrompt,
		})
	}
	return chatMessages
}