import type { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import type { ModelServer } from '#lllms/server.js'
import {
	ChatCompletionRequest,
	ToolDefinition,
	ChatMessage,
	ToolCallResultMessage,
	UserMessage,
	AssistantMessage,
	SystemMessage,
	MessageContentPart,
} from '#lllms/types/index.js'
import { parseJSONRequestBody } from '#lllms/api/parseJSONRequestBody.js'
import { omitEmptyValues } from '#lllms/lib/util.js'
import { finishReasonMap, messageRoleMap } from '../enums.js'

interface OpenAIChatCompletionParams
	extends Omit<OpenAI.ChatCompletionCreateParamsStreaming, 'stream'> {
	stream?: boolean
	top_k?: number
	min_p?: number
	repeat_penalty_num?: number
}

interface OpenAIChatCompletionChunk extends OpenAI.ChatCompletionChunk {
	usage?: OpenAI.CompletionUsage
}

// v1/chat/completions
// https://platform.openai.com/docs/api-reference/chat/create
export function createChatCompletionHandler(llms: ModelServer) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args: OpenAIChatCompletionParams

		try {
			const body = await parseJSONRequestBody(req)
			args = body
		} catch (e) {
			console.error(e)
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}

		// TODO ajv schema validation?
		if (!args.model || !args.messages) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request (need at least model and messages)' }))
			return
		}

		if (!llms.modelExists(args.model)) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Model does not exist' }))
			return
		}

		const controller = new AbortController()
		req.on('close', () => {
			console.debug('Client closed connection')
			controller.abort()
		})
		req.on('end', () => {
			console.debug('Client ended connection')
			controller.abort()
		})
		req.on('aborted', () => {
			console.debug('Client aborted connection')
			controller.abort()
		})
		req.on('error', () => {
			console.debug('Client error')
			controller.abort()
		})

		try {
			let ssePing: NodeJS.Timeout | undefined
			if (args.stream) {
				res.writeHead(200, {
					'Content-Type': 'text/event-stream',
					'Cache-Control': 'no-cache',
					Connection: 'keep-alive',
				})
				res.flushHeaders()
				ssePing = setInterval(() => {
					res.write(':ping\n\n')
				}, 30000)
			}

			let stop = args.stop ? args.stop : undefined
			if (typeof stop === 'string') {
				stop = [stop]
			}

			let completionGrammar: 'json' | undefined
			if (args.response_format) {
				if (args.response_format.type === 'json_object') {
					completionGrammar = 'json'
				}
			}

			let completionTools:
				| Record<string, ToolDefinition>
				| undefined

			if (args.tools) {
				const functionTools = args.tools
					.filter((tool) => tool.type === 'function')
					.map((tool) => {
						return {
							name: tool.function.name,
							description: tool.function.description,
							parameters: tool.function.parameters,
						}
					})
				if (functionTools.length) {
					if (!completionTools) {
						completionTools = {}
					}
					for (const tool of functionTools) {
						completionTools[tool.name] = {
							description: tool.description,
							parameters: tool.parameters,
						} as ToolDefinition
					}
				}
			}

			const completionReq = omitEmptyValues<ChatCompletionRequest>({
				model: args.model,
				messages: args.messages.map((msg) => {
					const role = messageRoleMap[msg.role]
					let content: ChatMessage['content']
					if (Array.isArray(msg.content)) {
						content = msg.content.map((part) => {
							if (typeof part === 'string') {
								return {
									type: 'text',
									text: part,
								} as MessageContentPart
							}
							if (part.type === 'image_url') {
								return {
									type: 'image',
									url: part.image_url.url,
								} as MessageContentPart
							}
							return part as MessageContentPart
						})
					} else {
						content = msg.content || ''
					}
					if (role === 'tool' && 'tool_call_id' in msg) {
						return {
							role,
							content,
							callId: msg.tool_call_id,
						} as ToolCallResultMessage
					}
					return {
						role,
						content,
					} as UserMessage | AssistantMessage | SystemMessage
				}),
				temperature: args.temperature ? args.temperature : undefined,
				stream: args.stream ? Boolean(args.stream) : false,
				maxTokens: args.max_tokens ? args.max_tokens : undefined,
				seed: args.seed ? args.seed : undefined,
				stop,
				frequencyPenalty: args.frequency_penalty
					? args.frequency_penalty
					: undefined,
				presencePenalty: args.presence_penalty
					? args.presence_penalty
					: undefined,
				topP: args.top_p ? args.top_p : undefined,
				tokenBias: args.logit_bias ? args.logit_bias : undefined,
				grammar: completionGrammar,
				tools: completionTools,
				// additional non-spec params
				repeatPenaltyNum: args.repeat_penalty_num
					? args.repeat_penalty_num
					: undefined,
				minP: args.min_p ? args.min_p : undefined,
				topK: args.top_k ? args.top_k : undefined,
			})
			const { instance, release } = await llms.requestInstance(
				completionReq,
				controller.signal,
			)
			
			if (ssePing) {
				clearInterval(ssePing)
			}
			const task = instance.processChatCompletionTask(completionReq, {
				signal: controller.signal,
				onChunk: (chunk) => {
					if (args.stream) {
						const chunkData: OpenAIChatCompletionChunk = {
							id: task.id,
							object: 'chat.completion.chunk',
							model: task.model,
							created: Math.floor(task.createdAt.getTime() / 1000),
							choices: [
								{
									index: 0,
									delta: {
										role: 'assistant',
										content: chunk.text,
									},
									logprobs: null,
									finish_reason: null,
								},
							],
						}
						res.write(`data: ${JSON.stringify(chunkData)}\n\n`)
					}
				},
			})

			const result = await task.result

			release()

			if (args.stream) {
				if (result.finishReason === 'toolCalls') {
					// currently not possible to stream function calls
					// imitating a stream here by sending two chunks. makes it work with the openai client
					const streamedToolCallChunk: OpenAIChatCompletionChunk = {
						id: task.id,
						object: 'chat.completion.chunk',
						model: task.model,
						created: Math.floor(task.createdAt.getTime() / 1000),
						choices: [
							{
								index: 0,
								delta: {
									role: 'assistant',
									content: null,
								},
								logprobs: null,
								finish_reason: result.finishReason
									? finishReasonMap[result.finishReason]
									: 'stop',
							},
						],
					}

					const toolCalls: OpenAI.ChatCompletionChunk.Choice.Delta.ToolCall[] =
						result.message.toolCalls!.map((call, index) => {
							return {
								index,
								id: call.id,
								type: 'function',
								function: {
									name: call.name,
									arguments: JSON.stringify(call.parameters),
								},
							}
						})
					streamedToolCallChunk.choices[0].delta.tool_calls = toolCalls
					res.write(`data: ${JSON.stringify(streamedToolCallChunk)}\n\n`)
				}
				if (args.stream_options?.include_usage) {
					const finalChunk: OpenAIChatCompletionChunk = {
						id: task.id,
						object: 'chat.completion.chunk',
						model: task.model,
						created: Math.floor(task.createdAt.getTime() / 1000),
						system_fingerprint: instance.fingerprint,
						choices: [
							{
								index: 0,
								delta: {},
								logprobs: null,
								finish_reason: result.finishReason
									? finishReasonMap[result.finishReason]
									: 'stop',
							},
						],
						usage: {
							prompt_tokens: result.promptTokens,
							completion_tokens: result.completionTokens,
							total_tokens: result.contextTokens,
						},
					}
					res.write(`data: ${JSON.stringify(finalChunk)}\n\n`)
				}
				res.write('data: [DONE]')
				res.end()
			} else {
				const response: OpenAI.ChatCompletion = {
					id: task.id,
					model: task.model,
					object: 'chat.completion',
					created: Math.floor(task.createdAt.getTime() / 1000),
					system_fingerprint: instance.fingerprint,
					choices: [
						{
							index: 0,
							message: {
								role: 'assistant',
								content: result.message.content || null,
								refusal: null,
							},
							logprobs: null,
							finish_reason: result.finishReason
								? finishReasonMap[result.finishReason]
								: 'stop',
						},
					],
					usage: {
						prompt_tokens: result.promptTokens,
						completion_tokens: result.completionTokens,
						total_tokens: result.contextTokens,
					},
				}
				if (
					'toolCalls' in result.message &&
					result.message.toolCalls?.length
				) {
					response.choices[0].message.tool_calls =
						result.message.toolCalls.map((call) => {
							return {
								id: call.id,
								type: 'function',
								function: {
									name: call.name,
									arguments: JSON.stringify(call.parameters),
								},
							}
						})
				}
				res.writeHead(200, { 'Content-Type': 'application/json' })
				res.end(JSON.stringify(response, null, 2))
			}
		} catch (e) {
			console.error(e)
			if (args.stream) {
				res.write('data: [ERROR]')
			} else {
				res.writeHead(500, { 'Content-Type': 'application/json' })
				res.end(JSON.stringify({ error: 'Internal server error' }))
			}
		}
	}
}
