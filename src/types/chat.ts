import type { SomeJSONSchema } from 'ajv/dist/types/json-schema'
import { Image } from './index.js'

export type CompletionFinishReason =
	| 'maxTokens'
	| 'toolCalls'
	| 'eogToken'
	| 'stopTrigger'
	| 'abort'
	| 'cancel'
	| 'timeout'

export interface AssistantToolCall {
	id: string
	name: string
	parameters?: Record<string, any>
}

export type ChatMessage = UserMessage | SystemMessage | AssistantMessage | ToolCallResultMessage

export interface MessageTextContentPart {
	type: 'text'
	text: string
}

export interface MessageImageContentPart {
	type: 'image'
	image: Image
}

export type MessageContentPart = MessageTextContentPart | MessageImageContentPart

export interface UserMessage {
	role: 'user'
	content: string | MessageContentPart[]
}

export interface SystemMessage {
	role: 'system'
	content: string | MessageContentPart[]
}

export interface AssistantMessage {
	role: 'assistant'
	content: string
	toolCalls?: AssistantToolCall[]
}

export interface ToolCallResultMessage {
	role: 'tool'
	content: string | MessageContentPart[]
	callId: string
	// name: string
}

// TODO figure out how to type this better.
// export type ToolDefinitionParams<TParams = any> = JSONSchemaType<TParams>
export type ToolDefinitionParams<TParams> = SomeJSONSchema

export interface ToolCallResultObject {
	text: string
	preventFurtherCalls?: boolean
}

export type ToolCallResult = string | ToolCallResultObject

export interface ToolDefinition<TParams extends Record<string, any> = any> {
	description?: string
	parameters?: ToolDefinitionParams<TParams>
	handler?: (params: TParams) => Promise<ToolCallResult>
}

