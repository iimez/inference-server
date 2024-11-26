import path from 'node:path'
import fs from 'node:fs'
import {
	loadModel,
	createCompletion,
	createEmbedding,
	InferenceModel,
	LoadModelOptions,
	CompletionInput,
	EmbeddingModel,
	DEFAULT_MODEL_LIST_URL,
} from 'gpt4all'
import {
	ChatCompletionTaskResult,
	TextCompletionTaskResult,
	CompletionFinishReason,
	EngineContext,
	EmbeddingTaskResult,
	FileDownloadProgress,
	ModelConfig,
	ChatMessage,
	TextCompletionTaskArgs,
	EngineTextCompletionTaskContext,
	TextCompletionParamsBase,
	ChatCompletionTaskArgs,
	EmbeddingTaskArgs,
	EngineTaskContext,
} from '#package/types/index.js'
import { LogLevels } from '#package/lib/logger.js'
import { downloadModelFile } from '#package/lib/downloadModelFile.js'
import { acquireFileLock } from '#package/lib/acquireFileLock.js'
import { validateModelFile } from '#package/lib/validateModelFile.js'
import { createChatMessageArray } from './util.js'

export type GPT4AllInstance = InferenceModel | EmbeddingModel

export interface GPT4AllModelMeta {
	url: string
	md5sum: string
	filename: string
	promptTemplate: string
	systemPrompt: string
	filesize: number
	ramrequired: number
}

export interface GPT4AllModelConfig extends ModelConfig {
	location: string
	md5?: string
	url?: string
	contextSize?: number
	batchSize?: number
	task: 'text-completion' | 'embedding'
	initialMessages?: ChatMessage[]
	completionDefaults?: TextCompletionParamsBase
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		gpuLayers?: number
		cpuThreads?: number
	}
}

export const autoGpu = true

export async function prepareModel(
	{ config, log }: EngineContext<GPT4AllModelConfig>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(path.dirname(config.location), { recursive: true })
	const releaseFileLock = await acquireFileLock(config.location)
	if (signal?.aborted) {
		releaseFileLock()
		return
	}
	log(LogLevels.info, `Preparing gpt4all model at ${config.location}`, {
		model: config.id,
	})
	let gpt4allMeta: GPT4AllModelMeta | undefined
	let modelList: GPT4AllModelMeta[]
	const modelMetaPath = path.join(path.dirname(config.location), 'models.json')
	try {

		if (!fs.existsSync(modelMetaPath)) {
			const res = await fetch(DEFAULT_MODEL_LIST_URL)
			modelList = (await res.json()) as GPT4AllModelMeta[]
			fs.writeFileSync(modelMetaPath, JSON.stringify(modelList, null, 2))
		} else {
			modelList = JSON.parse(fs.readFileSync(modelMetaPath, 'utf-8'))
		}
		const foundModelMeta = modelList.find((item) => {
			if (config.md5 && item.md5sum) {
				return item.md5sum === config.md5
			}
			if (config.url && item.url) {
				return item.url === config.url
			}
			return item.filename === path.basename(config.location)
		})
		if (foundModelMeta) {
			gpt4allMeta = foundModelMeta
		}

		const validationRes = await validateModelFile({
			...config,
			md5: config.md5 || gpt4allMeta?.md5sum,
		})
		let modelMeta = validationRes.meta
		if (signal?.aborted) {
			return
		}
		if (validationRes.error) {
			if (!config.url) {
				throw new Error(`${validationRes.error} - No URL provided`)
			}
				log(LogLevels.info, 'Downloading', {
					model: config.id,
					url: config.url,
					location: config.location,
					error: validationRes.error,
				})
				await downloadModelFile({
					url: config.url,
					filePath: config.location,
					modelsCachePath: config.modelsCachePath,
					onProgress,
					signal,
				})
				const revalidationRes = await validateModelFile({
					...config,
					md5: config.md5 || gpt4allMeta?.md5sum,
				})
				if (revalidationRes.error) {
					throw new Error(`Downloaded files are invalid: ${revalidationRes.error}`)
				}
				modelMeta = revalidationRes.meta
		}

		if (signal?.aborted) {
			return
		}

		return {
			gpt4allMeta,
			...modelMeta,
		}
	} catch (error) {
		throw error
	} finally {
		releaseFileLock()
	}
}

export async function createInstance({ config, log }: EngineContext<GPT4AllModelConfig>, signal?: AbortSignal) {
	log(LogLevels.info, `Load GPT4All model ${config.location}`)
	let device = config.device?.gpu ?? 'cpu'
	if (typeof device === 'boolean') {
		device = device ? 'gpu' : 'cpu'
	} else if (device === 'auto') {
		device = 'cpu'
	}
	const loadOpts: LoadModelOptions = {
		modelPath: path.dirname(config.location),
		// file: config.file,
		modelConfigFile: path.dirname(config.location) + '/models.json',
		allowDownload: false,
		device: device,
		ngl: config.device?.gpuLayers ?? 100,
		nCtx: config.contextSize ?? 2048,
		// verbose: true,
		// signal?: // TODO no way to cancel load
	}

	let modelType: 'inference' | 'embedding'
	if (config.task === 'text-completion') {
		modelType = 'inference'
	} else if (config.task === 'embedding') {
		modelType = 'embedding'
	} else {
		throw new Error(`Unsupported task type: ${config.task}`)
	}

	const instance = await loadModel(path.basename(config.location), {
		...loadOpts,
		type: modelType,
	})
	if (config.device?.cpuThreads) {
		instance.llm.setThreadCount(config.device.cpuThreads)
	}

	if ('generate' in instance) {
		if (config.initialMessages?.length) {
			let messages = createChatMessageArray(config.initialMessages)
			let systemPrompt
			if (messages[0].role === 'system') {
				systemPrompt = messages[0].content
				messages = messages.slice(1)
			}
			await instance.createChatSession({
				systemPrompt,
				messages,
			})
		} else if (config.prefix) {
			await instance.generate(config.prefix, {
				nPredict: 0,
			})
		} else {
			await instance.generate('', {
				nPredict: 0,
			})
		}
	}

	return instance
}

export async function disposeInstance(instance: GPT4AllInstance) {
	instance.dispose()
}

export async function processTextCompletionTask(
	task: TextCompletionTaskArgs,
	ctx: EngineTextCompletionTaskContext<GPT4AllInstance, GPT4AllModelConfig, GPT4AllModelMeta>,
	signal?: AbortSignal,
): Promise<TextCompletionTaskResult> {
	const { instance, config } = ctx
	if (!('generate' in instance)) {
		throw new Error('Instance does not support text completion.')
	}
	if (!task.prompt) {
		throw new Error('Prompt is required for text completion.')
	}

	let finishReason: CompletionFinishReason = 'eogToken'
	let suffixToRemove: string | undefined

	const defaults = config.completionDefaults ?? {}
	const stopTriggers = task.stop ?? defaults.stop ?? []
	const includesStopTriggers = (text: string) => stopTriggers.find((t) => text.includes(t))
	const result = await instance.generate(task.prompt, {
		// @ts-ignore
		special: true, // allows passing in raw prompt (including <|start|> etc.)
		promptTemplate: '%1',
		temperature: task.temperature ?? defaults.temperature,
		nPredict: task.maxTokens ?? defaults.maxTokens,
		topP: task.topP ?? defaults.topP,
		topK: task.topK ?? defaults.topK,
		minP: task.minP ?? defaults.minP,
		nBatch: config?.batchSize,
		repeatLastN: task.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
		// repeat penalty is doing something different than both frequency and presence penalty
		// so not falling back to them here.
		repeatPenalty: task.repeatPenalty ?? defaults.repeatPenalty,
		// seed: args.seed, // https://github.com/nomic-ai/gpt4all/issues/1952
		// @ts-ignore
		onResponseToken: (tokenId, text) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (task.onChunk) {
				task.onChunk({
					text,
					tokens: [tokenId],
				})
			}
			return !signal?.aborted
		},
		// @ts-ignore
		onResponseTokens: ({ tokenIds, text }) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (task.onChunk) {
				task.onChunk({
					text,
					tokens: tokenIds,
				})
			}
			return !signal?.aborted
		},
	})

	if (result.tokensGenerated === task.maxTokens) {
		finishReason = 'maxTokens'
	}

	let responseText = result.text
	if (suffixToRemove) {
		responseText = responseText.slice(0, -suffixToRemove.length)
	}

	return {
		finishReason,
		text: responseText,
		promptTokens: result.tokensIngested,
		completionTokens: result.tokensGenerated,
		contextTokens: instance.activeChatSession?.promptContext.nPast ?? 0,
	}
}

export async function processChatCompletionTask(
	task: ChatCompletionTaskArgs,
	ctx: EngineTextCompletionTaskContext<GPT4AllInstance, GPT4AllModelConfig, GPT4AllModelMeta>,
	signal?: AbortSignal,
): Promise<ChatCompletionTaskResult> {
	const { config, instance, resetContext, log } = ctx
	if (!('createChatSession' in instance)) {
		throw new Error('Instance does not support chat completion.')
	}
	let session = instance.activeChatSession
	if (!session || resetContext) {
		log(LogLevels.debug, 'Resetting chat context')
		let messages = createChatMessageArray(task.messages)
		let systemPrompt
		if (messages[0].role === 'system') {
			systemPrompt = messages[0].content
			messages = messages.slice(1)
		}
		// drop last user message
		if (messages[messages.length - 1].role === 'user') {
			messages = messages.slice(0, -1)
		}

		session = await instance.createChatSession({
			systemPrompt,
			messages,
		})
	}

	const conversationMessages = createChatMessageArray(task.messages).filter((m) => m.role !== 'system')

	const lastMessage = conversationMessages[conversationMessages.length - 1]
	if (!(lastMessage.role === 'user' && lastMessage.content)) {
		throw new Error('Chat completions require a final user message.')
	}
	const input: CompletionInput = lastMessage.content

	let finishReason: CompletionFinishReason = 'eogToken'
	let suffixToRemove: string | undefined

	const defaults = config.completionDefaults ?? {}
	const stopTriggers = task.stop ?? defaults.stop ?? []
	const includesStopTriggers = (text: string) => stopTriggers.find((t) => text.includes(t))
	const result = await createCompletion(session, input, {
		temperature: task.temperature ?? defaults.temperature,
		nPredict: task.maxTokens ?? defaults.maxTokens,
		topP: task.topP ?? defaults.topP,
		topK: task.topK ?? defaults.topK,
		minP: task.minP ?? defaults.minP,
		nBatch: config.batchSize,
		repeatLastN: task.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
		repeatPenalty: task.repeatPenalty ?? defaults.repeatPenalty,
		// seed: args.seed, // see https://github.com/nomic-ai/gpt4all/issues/1952
		// @ts-ignore
		onResponseToken: (tokenId, text) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (task.onChunk) {
				task.onChunk({
					text,
					tokens: [tokenId],
				})
			}
			return !signal?.aborted
		},
		// @ts-ignore
		onResponseTokens: ({ tokenIds, text }) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (task.onChunk) {
				task.onChunk({
					tokens: tokenIds,
					text,
				})
			}

			return !signal?.aborted
		},
	})

	if (result.usage.completion_tokens === task.maxTokens) {
		finishReason = 'maxTokens'
	}

	let response = result.choices[0].message.content
	if (suffixToRemove) {
		response = response.slice(0, -suffixToRemove.length)
	}

	return {
		finishReason,
		message: {
			role: 'assistant',
			content: response,
		},
		promptTokens: result.usage.prompt_tokens,
		completionTokens: result.usage.completion_tokens,
		contextTokens: session.promptContext.nPast,
	}
}

export async function processEmbeddingTask(
	task: EmbeddingTaskArgs,
	ctx: EngineTaskContext<GPT4AllInstance, GPT4AllModelConfig, GPT4AllModelMeta>,
	signal?: AbortSignal,
): Promise<EmbeddingTaskResult> {
	const { instance, config } = ctx
	if (!('embed' in instance)) {
		throw new Error('Instance does not support embedding.')
	}
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

	const res = await createEmbedding(instance, texts, {
		dimensionality: task.dimensions,
	})

	return {
		embeddings: res.embeddings,
		inputTokens: res.n_prompt_tokens,
	}
}
