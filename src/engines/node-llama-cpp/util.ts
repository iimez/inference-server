import fs from 'node:fs'
import path from 'node:path'
import {
	ChatHistoryItem,
	ChatModelResponse,
	LlamaTextJSON,
	ChatModelFunctionCall,
} from 'node-llama-cpp'
import { CompletionFinishReason, ChatMessage, ToolCallResult } from '#package/types/index.js'
import { flattenMessageTextContent } from '#package/lib/flattenMessageTextContent.js'
import { LlamaChatResult } from './types.js'

export function mapFinishReason(
	nodeLlamaCppFinishReason: LlamaChatResult['stopReason'],
): CompletionFinishReason {
	switch (nodeLlamaCppFinishReason) {
		case 'functionCalls':
			return 'toolCalls'
		case 'stopGenerationTrigger':
			return 'stopTrigger'
		case 'customStopTrigger':
			return 'stopTrigger'
		default:
			return nodeLlamaCppFinishReason
	}
}

export function addFunctionCallToChatHistory({
	chatHistory,
	functionName,
	functionDescription,
	callParams,
	callResult,
	rawCall,
	startsNewChunk
}: {
	chatHistory: ChatHistoryItem[],
	functionName: string,
	functionDescription?: string,
	callParams: any,
	callResult: ToolCallResult,
	rawCall?: LlamaTextJSON,
	startsNewChunk?: boolean
}) {
	const newChatHistory = chatHistory.slice();
	if (newChatHistory.length === 0 || newChatHistory[newChatHistory.length - 1]!.type !== "model")
			newChatHistory.push({
					type: "model",
					response: []
			});

	const lastModelResponseItem = newChatHistory[newChatHistory.length - 1] as ChatModelResponse;
	const newLastModelResponseItem = {...lastModelResponseItem};
	newChatHistory[newChatHistory.length - 1] = newLastModelResponseItem;

	const modelResponse = newLastModelResponseItem.response.slice();
	newLastModelResponseItem.response = modelResponse;

	const functionCall: ChatModelFunctionCall = {
			type: "functionCall",
			name: functionName,
			description: functionDescription,
			params: callParams,
			result: typeof callResult === "string" ? callResult : callResult.text,
			rawCall
	};

	if (startsNewChunk)
			functionCall.startsNewChunk = true;

	modelResponse.push(functionCall);

	return newChatHistory;
}

export function createChatMessageArray(
	messages: ChatMessage[],
): ChatHistoryItem[] {
	const items: ChatHistoryItem[] = []
	let systemPrompt: string | undefined
	for (const message of messages) {
		if (message.role === 'user') {
			items.push({
				type: 'user',
				text: flattenMessageTextContent(message.content),
			})
		} else if (message.role === 'assistant') {
			items.push({
				type: 'model',
				response: [message.content],
			})
		} else if (message.role === 'system') {
			if (systemPrompt) {
				systemPrompt += '\n\n' + flattenMessageTextContent(message.content)
			} else {
				systemPrompt = flattenMessageTextContent(message.content)
			}
		}
	}

	if (systemPrompt) {
		items.unshift({
			type: 'system',
			text: systemPrompt,
		})
	}

	return items
}

export async function readGBNFFiles(directoryPath: string) {
	const gbnfFiles = fs
		.readdirSync(directoryPath)
		.filter((f) => f.endsWith('.gbnf'))
	const fileContents = await Promise.all(
		gbnfFiles.map((file) =>
			fs.promises.readFile(path.join(directoryPath, file), 'utf-8'),
		),
	)
	return gbnfFiles.reduce((acc, file, i) => {
		acc[file.replace('.gbnf', '')] = fileContents[i]
		return acc
	}, {} as Record<string, string>)
}


