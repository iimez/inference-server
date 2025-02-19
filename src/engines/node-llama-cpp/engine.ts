import path from 'node:path'
import fs from 'node:fs'
import { nanoid } from 'nanoid'
import {
	getLlama,
	LlamaOptions,
	LlamaChat,
	LlamaModel,
	LlamaContext,
	LlamaCompletion,
	LlamaLogLevel,
	LlamaChatResponseFunctionCall,
	TokenBias,
	Token,
	LlamaContextSequence,
	LlamaGrammar,
	ChatHistoryItem,
	LlamaChatResponse,
	ChatModelResponse,
	LlamaEmbeddingContext,
	defineChatSessionFunction,
	GbnfJsonSchema,
	ChatSessionModelFunction,
	createModelDownloader,
	GgufFileInfo,
	LlamaJsonSchemaGrammar,
	LLamaChatContextShiftOptions,
	LlamaContextOptions,
	ChatWrapper,
	ChatModelFunctionCall,
} from 'node-llama-cpp'
import { StopGenerationTrigger } from 'node-llama-cpp/dist/utils/StopGenerationDetector'
import {
	ChatCompletionTaskResult,
	TextCompletionTaskResult,
	EngineContext,
	ToolDefinition,
	ToolCallResultMessage,
	AssistantMessage,
	EmbeddingTaskResult,
	FileDownloadProgress,
	ModelConfig,
	TextCompletionGrammar,
	ChatMessage,
	EngineTaskContext,
	EngineTextCompletionTaskContext,
	TextCompletionParamsBase,
	ChatCompletionTaskArgs,
	TextCompletionTaskArgs,
	EmbeddingTaskArgs,
} from '#package/types/index.js'
import { LogLevels } from '#package/lib/logger.js'
import { flattenMessageTextContent } from '#package/lib/flattenMessageTextContent.js'
import { acquireFileLock } from '#package/lib/acquireFileLock.js'
import { getRandomNumber } from '#package/lib/util.js'
import { validateModelFile } from '#package/lib/validateModelFile.js'
import { createChatMessageArray, addFunctionCallToChatHistory, mapFinishReason } from './util.js'
import { LlamaChatResult } from './types.js'

export interface NodeLlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	chat?: LlamaChat
	chatHistory: ChatHistoryItem[]
	grammars: Record<string, LlamaGrammar>
	pendingFunctionCalls: Record<string, any>
	lastEvaluation?: LlamaChatResponse['lastEvaluation']
	embeddingContext?: LlamaEmbeddingContext
	completion?: LlamaCompletion
	contextSequence: LlamaContextSequence
	chatWrapper?: ChatWrapper
}

export interface NodeLlamaCppModelMeta {
	gguf: GgufFileInfo
}

export interface NodeLlamaCppModelConfig extends ModelConfig {
	location: string
	grammars?: Record<string, TextCompletionGrammar>
	sha256?: string
	completionDefaults?: TextCompletionParamsBase
	initialMessages?: ChatMessage[]
	prefix?: string
	tools?: {
		definitions?: Record<string, ToolDefinition>
		documentParams?: boolean
		maxParallelCalls?: number
	}
	contextSize?: number
	batchSize?: number
	lora?: LlamaContextOptions['lora']
	contextShiftStrategy?: LLamaChatContextShiftOptions['strategy']
	chatWrapper?: ChatWrapper
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		gpuLayers?: number
		cpuThreads?: number
		memLock?: boolean
	}
}

export const autoGpu = true

export async function prepareModel(
	{ config, log }: EngineContext<NodeLlamaCppModelConfig>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(path.dirname(config.location), { recursive: true })
	const releaseFileLock = await acquireFileLock(config.location, signal)

	if (signal?.aborted) {
		releaseFileLock()
		return
	}
	log(LogLevels.info, `Preparing node-llama-cpp model at ${config.location}`, {
		model: config.id,
	})
	const downloadModel = async (url: string, validationResult: string) => {
		log(LogLevels.info, `Downloading model files`, {
			model: config.id,
			url: url,
			location: config.location,
			error: validationResult,
		})

		const downloader = await createModelDownloader({
			modelUrl: url,
			dirPath: path.dirname(config.location),
			fileName: path.basename(config.location),
			deleteTempFileOnCancel: false,
			onProgress: (status) => {
				if (onProgress) {
					onProgress({
						file: config.location,
						loadedBytes: status.downloadedSize,
						totalBytes: status.totalSize,
					})
				}
			},
		})
		await downloader.download()
	}
	try {
		if (signal?.aborted) {
			return
		}

		const validationRes = await validateModelFile(config)
		let modelMeta = validationRes.meta
		if (signal?.aborted) {
			return
		}
		if (validationRes.error) {
			if (!config.url) {
				throw new Error(`${validationRes.error} - No URL provided`)
			}
			await downloadModel(config.url, validationRes.error)
			const revalidationRes = await validateModelFile(config)
			if (revalidationRes.error) {
				throw new Error(`Downloaded files are invalid: ${revalidationRes.error}`)
			}
			modelMeta = revalidationRes.meta
		}

		return modelMeta
	} catch (err) {
		throw err
	} finally {
		releaseFileLock()
	}
}

export async function createInstance({ config, log }: EngineContext<NodeLlamaCppModelConfig>, signal?: AbortSignal) {
	log(LogLevels.debug, 'Load Llama model', config.device)
	// takes "auto" | "metal" | "cuda" | "vulkan"
	const gpuSetting = (config.device?.gpu ?? 'auto') as LlamaOptions['gpu']
	const llama = await getLlama({
		gpu: gpuSetting,
		// forwarding llama logger
		logLevel: LlamaLogLevel.debug,
		logger: (level, message) => {
			if (level === LlamaLogLevel.warn) {
				log(LogLevels.warn, message)
			} else if (level === LlamaLogLevel.error || level === LlamaLogLevel.fatal) {
				log(LogLevels.error, message)
			} else if (level === LlamaLogLevel.info || level === LlamaLogLevel.debug) {
				log(LogLevels.verbose, message)
			}
		},
	})

	const llamaGrammars: Record<string, LlamaGrammar> = {
		json: await LlamaGrammar.getFor(llama, 'json'),
	}

	if (config.grammars) {
		for (const key in config.grammars) {
			const input = config.grammars[key]
			if (typeof input === 'string') {
				llamaGrammars[key] = new LlamaGrammar(llama, {
					grammar: input,
				})
			} else {
				// assume input is a JSON schema object
				llamaGrammars[key] = new LlamaJsonSchemaGrammar<GbnfJsonSchema>(llama, input as GbnfJsonSchema)
			}
		}
	}

	const llamaModel = await llama.loadModel({
		modelPath: config.location, // full model absolute path
		loadSignal: signal,
		useMlock: config.device?.memLock ?? false,
		gpuLayers: config.device?.gpuLayers,
		// onLoadProgress: (percent) => {}
	})

	const context = await llamaModel.createContext({
		sequences: 1,
		lora: config.lora,
		threads: config.device?.cpuThreads,
		batchSize: config.batchSize,
		contextSize: config.contextSize,
		flashAttention: true,
		createSignal: signal,
	})

	const instance: NodeLlamaCppInstance = {
		model: llamaModel,
		context,
		grammars: llamaGrammars,
		chat: undefined,
		chatHistory: [],
		pendingFunctionCalls: {},
		lastEvaluation: undefined,
		completion: undefined,
		contextSequence: context.getSequence(),
		chatWrapper: config.chatWrapper,
	}

	if (config.initialMessages) {
		const initialChatHistory = createChatMessageArray(config.initialMessages)
		const chat = new LlamaChat({
			contextSequence: instance.contextSequence!,
			chatWrapper: instance.chatWrapper,
			// autoDisposeSequence: true,
		})

		let inputFunctions: Record<string, ChatSessionModelFunction> | undefined

		if (config.tools?.definitions && Object.keys(config.tools.definitions).length > 0) {
			const functionDefs = config.tools.definitions
			inputFunctions = {}
			for (const functionName in functionDefs) {
				const functionDef = functionDefs[functionName]
				inputFunctions[functionName] = defineChatSessionFunction<any>({
					description: functionDef.description,
					params: functionDef.parameters,
					handler: functionDef.handler || (() => {}),
				}) as ChatSessionModelFunction
			}
		}

		const loadMessagesRes = await chat.loadChatAndCompleteUserMessage(initialChatHistory, {
			initialUserPrompt: '',
			functions: inputFunctions,
			documentFunctionParams: config.tools?.documentParams,
		})

		instance.chat = chat
		instance.chatHistory = initialChatHistory
		instance.lastEvaluation = {
			cleanHistory: initialChatHistory,
			contextWindow: loadMessagesRes.lastEvaluation.contextWindow,
			contextShiftMetadata: loadMessagesRes.lastEvaluation.contextShiftMetadata,
		}
	}

	if (config.prefix) {
		const contextSequence = instance.contextSequence!
		const completion = new LlamaCompletion({
			contextSequence: contextSequence,
		})
		await completion.generateCompletion(config.prefix, {
			maxTokens: 0,
		})
		instance.completion = completion
		instance.contextSequence = contextSequence
	}

	return instance
}

export async function disposeInstance(instance: NodeLlamaCppInstance) {
	await instance.model.dispose()
}

export async function processChatCompletionTask(
	task: ChatCompletionTaskArgs,
	ctx: EngineTextCompletionTaskContext<NodeLlamaCppInstance, NodeLlamaCppModelConfig, NodeLlamaCppModelMeta>,
	signal?: AbortSignal,
): Promise<ChatCompletionTaskResult> {
	const { instance, resetContext, config, log } = ctx
	if (!instance.chat || resetContext) {
		log(LogLevels.debug, 'Recreating chat context', {
			resetContext: resetContext,
			willDisposeChat: !!instance.chat,
		})
		// if context reset is requested, dispose the chat instance
		if (instance.chat) {
			instance.chat.dispose()
		}
		let contextSequence = instance.contextSequence
		if (!contextSequence || contextSequence.disposed) {
			if (instance.context.sequencesLeft) {
				contextSequence = instance.context.getSequence()
				instance.contextSequence = contextSequence
			} else {
				throw new Error('No context sequence available')
			}
		} else {
			contextSequence.clearHistory()
		}
		instance.chat = new LlamaChat({
			contextSequence: contextSequence,
			chatWrapper: instance.chatWrapper,
			// autoDisposeSequence: true,
		})
		// reset state and reingest the conversation history
		instance.lastEvaluation = undefined
		instance.pendingFunctionCalls = {}
		instance.chatHistory = createChatMessageArray(task.messages)
		// drop last user message. its gonna be added later, after resolved function calls
		if (instance.chatHistory[instance.chatHistory.length - 1].type === 'user') {
			instance.chatHistory.pop()
		}
	}

	// set additional stop generation triggers for this completion
	const customStopTriggers: StopGenerationTrigger[] = []
	const stopTrigger = task.stop ?? config.completionDefaults?.stop
	if (stopTrigger) {
		customStopTriggers.push(...stopTrigger.map((t) => [t]))
	}
	// setting up logit/token bias dictionary
	let tokenBias: TokenBias | undefined
	const completionTokenBias = task.tokenBias ?? config.completionDefaults?.tokenBias
	if (completionTokenBias) {
		tokenBias = new TokenBias(instance.model.tokenizer)
		for (const key in completionTokenBias) {
			const bias = completionTokenBias[key] / 10
			const tokenId = parseInt(key) as Token
			if (!isNaN(tokenId)) {
				tokenBias.set(tokenId, bias)
			} else {
				tokenBias.set(key, bias)
			}
		}
	}

	// setting up available function definitions
	const functionDefinitions: Record<string, ToolDefinition> = {
		...config.tools?.definitions,
		...task.tools?.definitions,
	}

	// see if the user submitted any function call results
	const maxParallelCalls = task.tools?.maxParallelCalls ?? config.tools?.maxParallelCalls
	const chatWrapperSupportsParallelism = !!instance.chat.chatWrapper.settings.functions.parallelism
	const supportsParallelFunctionCalling = chatWrapperSupportsParallelism && !!maxParallelCalls
	const resolvedFunctionCalls: ChatModelFunctionCall[] = []
	const functionCallResultMessages = task.messages.filter((m) => m.role === 'tool') as ToolCallResultMessage[]
	let startsNewChunk = supportsParallelFunctionCalling
	for (const message of functionCallResultMessages) {
		if (!instance.pendingFunctionCalls[message.callId]) {
			log(LogLevels.warn, `Received function result for non-existing call id "${message.callId}`)
			continue
		}
		log(LogLevels.debug, 'Resolving pending function call', {
			id: message.callId,
			result: message.content,
		})
		const functionCall = instance.pendingFunctionCalls[message.callId]
		const functionDef = functionDefinitions[functionCall.functionName]
		const resolvedFunctionCall: ChatModelFunctionCall = {
			type: 'functionCall',
			name: functionCall.functionName,
			description: functionDef?.description,
			params: functionCall.params,
			result: message.content,
			rawCall: functionCall.raw,
		}
		if (startsNewChunk) {
			resolvedFunctionCall.startsNewChunk = true
			startsNewChunk = false
		}
		resolvedFunctionCalls.push(resolvedFunctionCall)
		delete instance.pendingFunctionCalls[message.callId]
	}

	// only grammar or functions can be used, not both.
	// currently ignoring function definitions if grammar is provided

	let inputGrammar: LlamaGrammar | undefined
	let inputFunctions: Record<string, ChatSessionModelFunction> | undefined

	if (task.grammar) {
		if (!instance.grammars[task.grammar]) {
			throw new Error(`Grammar "${task.grammar}" not found.`)
		}
		inputGrammar = instance.grammars[task.grammar]
	} else if (Object.keys(functionDefinitions).length > 0) {
		inputFunctions = {}
		for (const functionName in functionDefinitions) {
			const functionDef = functionDefinitions[functionName]
			inputFunctions[functionName] = defineChatSessionFunction<any>({
				description: functionDef.description,
				params: functionDef.parameters,
				handler: functionDef.handler || (() => {}),
			})
		}
	}

	let lastEvaluation: LlamaChatResponse['lastEvaluation'] | undefined = instance.lastEvaluation

	const appendResolvedFunctionCalls = (history: ChatHistoryItem[], response: ChatModelFunctionCall[]) => {
		const lastMessage = history[history.length - 1]
		// append to existing response item if last message in history is a model response
		if (lastMessage.type === 'model') {
			const lastMessageResponse = lastMessage as ChatModelResponse
			if (Array.isArray(response)) {
				lastMessageResponse.response.push(...response)
				// if we dont add a fresh empty message llama 3.2 3b will keep trying to call functions, not sure why this is
				history.push({
					type: 'model',
					response: [],
				})
			}
			return
		}
		// otherwise append a new one with the calls
		history.push({
			type: 'model',
			response: response,
		})
	}

	// if the incoming messages resolved any pending function calls, add them to history
	if (resolvedFunctionCalls.length) {
		appendResolvedFunctionCalls(instance.chatHistory, resolvedFunctionCalls)
		if (lastEvaluation?.contextWindow) {
			appendResolvedFunctionCalls(lastEvaluation.contextWindow, resolvedFunctionCalls)
		}
	}

	// add the new user message to the chat history
	let assistantPrefill: string = ''
	const lastMessage = task.messages[task.messages.length - 1]
	if (lastMessage.role === 'user' && lastMessage.content) {
		const newUserText = flattenMessageTextContent(lastMessage.content)
		if (newUserText) {
			instance.chatHistory.push({
				type: 'user',
				text: newUserText,
			})
		}
	} else if (lastMessage.role === 'assistant') {
		// use last message as prefill for response, if its an assistant message
		assistantPrefill = flattenMessageTextContent(lastMessage.content)
	} else if (!resolvedFunctionCalls.length) {
		log(LogLevels.warn, 'Last message is not valid for chat completion. This is likely a mistake.', lastMessage)
		throw new Error('Invalid last chat message')
	}

	const defaults = config.completionDefaults ?? {}
	let newChatHistory = instance.chatHistory.slice()
	let newContextWindowChatHistory = !lastEvaluation?.contextWindow ? undefined : instance.chatHistory.slice()

	if (instance.chatHistory[instance.chatHistory.length - 1].type !== 'model' || assistantPrefill) {
		const newModelResponse = assistantPrefill ? [assistantPrefill] : []
		newChatHistory.push({
			type: 'model',
			response: newModelResponse,
		})
		if (newContextWindowChatHistory) {
			newContextWindowChatHistory.push({
				type: 'model',
				response: newModelResponse,
			})
		}
	}

	const functionsOrGrammar = inputFunctions
		? {
				// clone the input funcs because the dict gets mutated in the loop below to enable preventFurtherCalls
				functions: { ...inputFunctions },
				documentFunctionParams: task.tools?.documentParams ?? config.tools?.documentParams,
				maxParallelFunctionCalls: maxParallelCalls,
				onFunctionCall: (functionCall: LlamaChatResponseFunctionCall<any>) => {
					// log(LogLevels.debug, 'Called function', functionCall)
				},
		  }
		: {
				grammar: inputGrammar,
		  }

	const initialTokenMeterState = instance.chat.sequence.tokenMeter.getState()
	let completionResult: LlamaChatResult
	while (true) {
		// console.debug('before eval newChatHistory', JSON.stringify(newChatHistory, null, 2))
		// console.debug('before eval newContextWindowChatHistory', JSON.stringify(newContextWindowChatHistory, null, 2))

		const {
			functionCalls,
			lastEvaluation: currentLastEvaluation,
			metadata,
		} = await instance.chat.generateResponse(newChatHistory, {
			signal,
			stopOnAbortSignal: true, // this will make aborted completions resolve (with a partial response)
			maxTokens: task.maxTokens ?? defaults.maxTokens,
			temperature: task.temperature ?? defaults.temperature,
			topP: task.topP ?? defaults.topP,
			topK: task.topK ?? defaults.topK,
			minP: task.minP ?? defaults.minP,
			seed: task.seed ?? config.completionDefaults?.seed ?? getRandomNumber(0, 1000000),
			tokenBias,
			customStopTriggers,
			trimWhitespaceSuffix: false,
			...functionsOrGrammar,
			repeatPenalty: {
				lastTokens: task.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
				frequencyPenalty: task.frequencyPenalty ?? defaults.frequencyPenalty,
				presencePenalty: task.presencePenalty ?? defaults.presencePenalty,
			},
			contextShift: {
				strategy: config.contextShiftStrategy,
				lastEvaluationMetadata: lastEvaluation?.contextShiftMetadata,
			},
			lastEvaluationContextWindow: {
				history: newContextWindowChatHistory,
				minimumOverlapPercentageToPreventContextShift: 0.5,
			},
			onToken: (tokens) => {
				const text = instance.model.detokenize(tokens)
				if (task.onChunk) {
					task.onChunk({
						tokens,
						text,
					})
				}
			},
		})

		lastEvaluation = currentLastEvaluation
		newChatHistory = lastEvaluation.cleanHistory

		// console.debug('after eval newChatHistory', JSON.stringify(newChatHistory, null, 2))
		// console.debug('after eval newContextWindowChatHistory', JSON.stringify(newContextWindowChatHistory, null, 2))

		if (functionCalls) {
			// find leading immediately invokable function calls (=have a handler function)
			const invokableFunctionCalls = []
			for (const functionCall of functionCalls) {
				const functionDef = functionDefinitions[functionCall.functionName]
				if (functionDef.handler) {
					invokableFunctionCalls.push(functionCall)
				} else {
					break
				}
			}

			// if the model output text before the call, pass it on into the function handlers
			// the response tokens will also be available via onChunk but this is more convenient
			const lastMessage = newChatHistory[newChatHistory.length - 1] as ChatModelResponse
			const lastResponsePart = lastMessage.response[lastMessage.response.length - 1]
			let leadingResponseText: string | undefined
			if (typeof lastResponsePart === 'string' && lastResponsePart) {
				leadingResponseText = lastResponsePart
			}

			// resolve function call results
			const results = await Promise.all(
				invokableFunctionCalls.map(async (functionCall) => {
					const functionDef = functionDefinitions[functionCall.functionName]
					if (!functionDef) {
						throw new Error(`The model tried to call undefined function "${functionCall.functionName}"`)
					}
					let functionCallResult = await functionDef.handler!(functionCall.params, leadingResponseText)
					log(LogLevels.debug, 'Function handler resolved', {
						function: functionCall.functionName,
						args: functionCall.params,
						result: functionCallResult,
					})
					if (typeof functionCallResult !== 'string') {
						if (functionsOrGrammar.functions && functionCallResult.preventFurtherCalls) {
							// remove the function we just called from the list of available functions
							functionsOrGrammar.functions = Object.fromEntries(
								Object.entries(functionsOrGrammar.functions).filter(([key]) => key !== functionCall.functionName),
							)
							if (Object.keys(functionsOrGrammar.functions).length === 0) {
								// @ts-ignore
								functionsOrGrammar.functions = undefined
							}
							functionCallResult = functionCallResult.text
						}
					}
					return {
						functionDef,
						functionCall,
						functionCallResult,
					}
				}),
			)
			newContextWindowChatHistory = lastEvaluation.contextWindow
			let startsNewChunk = supportsParallelFunctionCalling
			// add results to chat history in the order they were called
			for (const callResult of results) {
				newChatHistory = addFunctionCallToChatHistory({
					chatHistory: newChatHistory,
					functionName: callResult.functionCall.functionName,
					functionDescription: callResult.functionDef.description,
					callParams: callResult.functionCall.params,
					callResult: callResult.functionCallResult,
					rawCall: callResult.functionCall.raw,
					startsNewChunk: startsNewChunk,
				})
				newContextWindowChatHistory = addFunctionCallToChatHistory({
					chatHistory: newContextWindowChatHistory,
					functionName: callResult.functionCall.functionName,
					functionDescription: callResult.functionDef.description,
					callParams: callResult.functionCall.params,
					callResult: callResult.functionCallResult,
					rawCall: callResult.functionCall.raw,
					startsNewChunk: startsNewChunk,
				})
				startsNewChunk = false
			}

			// if functions without handler have been called, return the calls as messages
			const remainingFunctionCalls = functionCalls.slice(invokableFunctionCalls.length)

			if (remainingFunctionCalls.length === 0) {
				// if yes, continue with generation
				lastEvaluation.cleanHistory = newChatHistory
				lastEvaluation.contextWindow = newContextWindowChatHistory!
				continue
			} else {
				// if no, return the function calls and skip generation
				instance.lastEvaluation = lastEvaluation
				instance.chatHistory = newChatHistory
				completionResult = {
					responseText: null,
					stopReason: 'functionCalls',
					functionCalls: remainingFunctionCalls,
				}
				break
			}
		}

		// no function calls happened, we got a model response.
		instance.lastEvaluation = lastEvaluation
		instance.chatHistory = newChatHistory
		const lastMessage = instance.chatHistory[instance.chatHistory.length - 1] as ChatModelResponse
		const responseText = lastMessage.response.filter((item: any) => typeof item === 'string').join('')
		completionResult = {
			responseText,
			stopReason: metadata.stopReason,
		}
		break
	}

	const assistantMessage: AssistantMessage = {
		role: 'assistant',
		content: completionResult.responseText || '',
	}

	if (completionResult.functionCalls) {
		// TODO its possible that there are trailing immediately-evaluatable function calls.
		// function call results need to be added in the order the functions were called, so
		// we need to wait for the pending calls to complete before we can add the trailing calls.
		// as is, these may never resolve
		const pendingFunctionCalls = completionResult.functionCalls.filter((call) => {
			const functionDef = functionDefinitions[call.functionName]
			return !functionDef.handler
		})

		// TODO write a test that triggers a parallel call to a handlerless function and to a function with one
		const trailingFunctionCalls = completionResult.functionCalls.filter((call) => {
			const functionDef = functionDefinitions[call.functionName]
			return functionDef.handler
		})
		if (trailingFunctionCalls.length) {
			console.debug(trailingFunctionCalls)
			log(LogLevels.warn, 'Trailing function calls not resolved')
		}

		assistantMessage.toolCalls = pendingFunctionCalls.map((call) => {
			const callId = nanoid()
			instance.pendingFunctionCalls[callId] = call
			log(LogLevels.debug, 'Saving pending tool call', {
				id: callId,
				function: call.functionName,
				args: call.params,
			})
			return {
				id: callId,
				name: call.functionName,
				parameters: call.params,
			}
		})
	}

	const tokenDifference = instance.chat.sequence.tokenMeter.diff(initialTokenMeterState)
	// console.debug('final chatHistory', JSON.stringify(instance.chatHistory, null, 2))
	// console.debug('final lastEvaluation', JSON.stringify(instance.lastEvaluation, null, 2))
	return {
		finishReason: mapFinishReason(completionResult.stopReason),
		message: assistantMessage,
		promptTokens: tokenDifference.usedInputTokens,
		completionTokens: tokenDifference.usedOutputTokens,
		contextTokens: instance.chat.sequence.contextTokens.length,
	}
}

export async function processTextCompletionTask(
	task: TextCompletionTaskArgs,
	ctx: EngineTextCompletionTaskContext<NodeLlamaCppInstance, NodeLlamaCppModelConfig, NodeLlamaCppModelMeta>,
	signal?: AbortSignal,
): Promise<TextCompletionTaskResult> {
	const { instance, resetContext, config, log } = ctx
	if (!task.prompt) {
		throw new Error('Prompt is required for text completion.')
	}

	let completion: LlamaCompletion
	let contextSequence: LlamaContextSequence

	if (resetContext && instance.contextSequence) {
		instance.contextSequence.clearHistory()
	}

	if (!instance.completion || instance.completion.disposed) {
		if (instance.contextSequence) {
			contextSequence = instance.contextSequence
		} else if (instance.context.sequencesLeft) {
			contextSequence = instance.context.getSequence()
		} else {
			throw new Error('No context sequence available')
		}
		instance.contextSequence = contextSequence
		completion = new LlamaCompletion({
			contextSequence,
		})
		instance.completion = completion
	} else {
		completion = instance.completion
		contextSequence = instance.contextSequence!
	}

	if (!contextSequence || contextSequence.disposed) {
		contextSequence = instance.context.getSequence()
		instance.contextSequence = contextSequence
		completion = new LlamaCompletion({
			contextSequence,
		})
		instance.completion = completion
	}

	const stopGenerationTriggers: StopGenerationTrigger[] = []
	const stopTrigger = task.stop ?? config.completionDefaults?.stop
	if (stopTrigger) {
		stopGenerationTriggers.push(...stopTrigger.map((t) => [t]))
	}

	const initialTokenMeterState = contextSequence.tokenMeter.getState()
	const defaults = config.completionDefaults ?? {}
	const result = await completion.generateCompletionWithMeta(task.prompt, {
		maxTokens: task.maxTokens ?? defaults.maxTokens,
		temperature: task.temperature ?? defaults.temperature,
		topP: task.topP ?? defaults.topP,
		topK: task.topK ?? defaults.topK,
		minP: task.minP ?? defaults.minP,
		repeatPenalty: {
			lastTokens: task.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
			frequencyPenalty: task.frequencyPenalty ?? defaults.frequencyPenalty,
			presencePenalty: task.presencePenalty ?? defaults.presencePenalty,
		},
		signal: signal,
		customStopTriggers: stopGenerationTriggers.length ? stopGenerationTriggers : undefined,
		seed: task.seed ?? config.completionDefaults?.seed ?? getRandomNumber(0, 1000000),
		onToken: (tokens) => {
			const text = instance.model.detokenize(tokens)
			if (task.onChunk) {
				task.onChunk({
					tokens,
					text,
				})
			}
		},
	})

	const tokenDifference = contextSequence.tokenMeter.diff(initialTokenMeterState)

	return {
		finishReason: mapFinishReason(result.metadata.stopReason),
		text: result.response,
		promptTokens: tokenDifference.usedInputTokens,
		completionTokens: tokenDifference.usedOutputTokens,
		contextTokens: contextSequence.contextTokens.length,
	}
}

export async function processEmbeddingTask(
	task: EmbeddingTaskArgs,
	ctx: EngineTaskContext<NodeLlamaCppInstance, NodeLlamaCppModelConfig, NodeLlamaCppModelMeta>,
	signal?: AbortSignal,
): Promise<EmbeddingTaskResult> {
	const { instance, config, log } = ctx
	if (!task.input) {
		throw new Error('Input is required for embedding.')
	}
	const texts: string[] = []
	if (typeof task.input === 'string') {
		texts.push(task.input)
	} else if (Array.isArray(task.input)) {
		for (const input of task.input) {
			if (typeof input === 'string') {
				texts.push(input)
			} else if (input.type === 'text') {
				texts.push(input.content)
			} else if (input.type === 'image') {
				throw new Error('Image inputs not implemented.')
			}
		}
	}

	if (!instance.embeddingContext) {
		instance.embeddingContext = await instance.model.createEmbeddingContext({
			batchSize: config.batchSize,
			createSignal: signal,
			threads: config.device?.cpuThreads,
			contextSize: config.contextSize,
		})
	}

	// @ts-ignore - private property
	const contextSize = instance.embeddingContext._llamaContext.contextSize

	const embeddings: Float32Array[] = []
	let inputTokens = 0

	for (const text of texts) {
		let tokenizedInput = instance.model.tokenize(text)
		if (tokenizedInput.length > contextSize) {
			log(LogLevels.warn, 'Truncated input that exceeds context size')
			tokenizedInput = tokenizedInput.slice(0, contextSize)
		}
		inputTokens += tokenizedInput.length
		const embedding = await instance.embeddingContext.getEmbeddingFor(tokenizedInput)
		embeddings.push(new Float32Array(embedding.vector))
		if (signal?.aborted) {
			break
		}
	}

	return {
		embeddings,
		inputTokens,
	}
}
