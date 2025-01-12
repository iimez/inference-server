import { CompletionFinishReason, ChatMessage } from '#package/types/index.js'
import OpenAI from 'openai'

export const finishReasonMap: Record<CompletionFinishReason, OpenAI.ChatCompletion.Choice['finish_reason']> = {
	maxTokens: 'length',
	toolCalls: 'tool_calls',
	eogToken: 'stop',
	stopTrigger: 'stop',
	timeout: 'stop',
	cancel: 'stop',
	abort: 'stop',
} as const

export const messageRoleMap: Record<string, string> = {
	function: 'tool',
}