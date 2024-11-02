import type { SomeJSONSchema } from 'ajv/dist/types/json-schema'
import type { Sharp } from 'sharp'
import type { BuiltInEngineName } from '#package/engines/index.js'
import type { Logger } from '#package/lib/logger.js'
import type { ModelPool } from '#package/pool.js'
import type { ModelStore } from '#package/store.js'
import {
	AssistantMessage,
	ChatMessage,
	CompletionFinishReason,
	TextCompletionParams,
	ToolDefinition,
} from '#package/types/completions.js'
import type { ContextShiftStrategy } from '#package/engines/node-llama-cpp/types.js'
import type {
	StableDiffusionWeightType,
	StableDiffusionSamplingMethod,
	StableDiffusionSchedule,
} from '#package/engines/stable-diffusion-cpp/types.js'
import type {
	TransformersJsModelClass,
	TransformersJsTokenizerClass,
	TransformersJsProcessorClass,
	TransformersJsDataType,
} from '#package/engines/transformers-js/types.js'
export * from '#package/types/completions.js'

export type ModelTaskType =
	| 'text-completion'
	| 'embedding'
	| 'image-to-text'
	| 'image-to-image'
	| 'text-to-image'
	| 'speech-to-text'
	| 'text-to-speech'

export interface ModelOptionsBase {
	engine: BuiltInEngineName | (string & {})
	task: ModelTaskType | (string & {})
	prepare?: 'blocking' | 'async' | 'on-demand'
	minInstances?: number
	maxInstances?: number
	location?: string
}

export interface BuiltInModelOptionsBase extends ModelOptionsBase {
	engine: BuiltInEngineName
	task: ModelTaskType
	url?: string
	location?: string
}

export interface ModelConfigBase extends ModelOptionsBase {
	id: string
	minInstances: number
	maxInstances: number
	modelsCachePath: string
}

export interface ModelConfig extends ModelConfigBase {
	url?: string
	location?: string
	task: ModelTaskType | (string & {})
	engine: BuiltInEngineName | (string & {})
	// minInstances: number
	// maxInstances: number
	ttl?: number
	prefix?: string
	initialMessages?: ChatMessage[]
	device?: {
		gpu?: boolean | 'auto' | (string & {}) // TODO rename to backend?
		// gpuLayers?: number
		// cpuThreads?: number
		// memLock?: boolean
	}
}

// export interface ChatModelConfig extends ModelConfig {
// 	initialMessages?: ChatMessage[]
// }

export interface CompletionChunk {
	tokens: number[]
	text: string
}

export interface ProcessingOptions {
	timeout?: number
	signal?: AbortSignal
}

export interface Image {
	handle: Sharp
	width: number
	height: number
	channels: 1 | 2 | 3 | 4
}

export interface Audio {
	sampleRate: number
	channels: 1 | 2
	samples: Float32Array
}

export interface CompletionProcessingOptions extends ProcessingOptions {
	onChunk?: (chunk: CompletionChunk) => void
}

export interface SpeechToTextProcessingOptions extends ProcessingOptions {
	onChunk?: (chunk: { text: string }) => void
}

export interface TextToSpeechProcessingOptions extends ProcessingOptions {
	onChunk?: (chunk: { audio: Buffer }) => void
}

export interface EngineContext<TModelConfig = ModelConfig, TModelMeta = unknown> {
	config: TModelConfig
	meta?: TModelMeta
	log: Logger
}

export interface TextCompletionRequestBase extends TextCompletionParams {
	model: string
	stream?: boolean
}

export interface TextCompletionRequest extends TextCompletionRequestBase {
	prompt?: string
}

export interface ChatCompletionRequest extends TextCompletionRequestBase {
	messages: ChatMessage[]
	grammar?: string
	tools?: Record<string, ToolDefinition>
}

export interface TextEmbeddingInput {
	type: 'text'
	content: string
}

export interface ImageEmbeddingInput {
	type: 'image'
	content: Image
}

export type EmbeddingInput = TextEmbeddingInput | ImageEmbeddingInput | string

export interface EmbeddingRequest {
	model: string
	input: EmbeddingInput | EmbeddingInput[]
	dimensions?: number
	pooling?: 'cls' | 'mean'
}

export interface ImageToTextRequest {
	model: string
	image: Image
	prompt?: string
	maxTokens?: number
}

export interface StableDiffusionRequest {
	negativePrompt?: string
	guidance?: number
	styleRatio?: number
	strength?: number
	sampleSteps?: number
	batchCount?: number
	samplingMethod?: StableDiffusionSamplingMethod
	cfgScale?: number
	controlStrength?: number
}

export interface TextToImageRequest extends StableDiffusionRequest {
	model: string
	prompt: string
	width?: number
	height?: number
	seed?: number
}

export interface ImageToImageRequest extends StableDiffusionRequest {
	model: string
	image: Image
	prompt: string
	width?: number
	height?: number
	seed?: number
}

export interface SpeechToTextRequest {
	model: string
	url?: string
	file?: string
	language?: string
	prompt?: string
	maxTokens?: number
}

export interface TextToSpeechRequest {
	model: string
	text: string
	voice?: string
}

export interface ModelRequestMeta {
	sequence: number
	abortController: AbortController
}
export type IncomingRequest =
	| TextCompletionRequest
	| ChatCompletionRequest
	| EmbeddingRequest
	| ImageToTextRequest
	| SpeechToTextRequest
export type ModelInstanceRequest = ModelRequestMeta & IncomingRequest

export interface EngineTextCompletionArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	onChunk?: (chunk: CompletionChunk) => void
	resetContext?: boolean
	request: TextCompletionRequest
}

export interface EngineChatCompletionArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	onChunk?: (chunk: CompletionChunk) => void
	resetContext?: boolean
	request: ChatCompletionRequest
}

export interface EngineEmbeddingArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: EmbeddingRequest
}

export interface EngineImageToTextArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: ImageToTextRequest
}

export interface EngineTextToImageArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: TextToImageRequest
}

export interface EngineImageToImageArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: ImageToImageRequest
}

export interface EngineSpeechToTextArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: SpeechToTextRequest
	onChunk?: (chunk: { text: string }) => void
}

export interface EngineTextToSpeechArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: TextToSpeechRequest
	onChunk?: (chunk: { text: string }) => void
}

export interface FileDownloadProgress {
	file: string
	loadedBytes: number
	totalBytes: number
}

export interface EngineStartContext {
	pool: ModelPool
	store: ModelStore
}

export interface ModelEngine<
	TInstance = unknown,
	TModelConfig extends ModelConfig = ModelConfig,
	TModelMeta = unknown,
> {
	autoGpu?: boolean
	start?: (ctx: EngineStartContext) => Promise<void>
	prepareModel: (
		ctx: EngineContext<TModelConfig, TModelMeta>,
		onProgress?: (progress: FileDownloadProgress) => void,
		signal?: AbortSignal,
	) => Promise<TModelMeta>
	createInstance: (ctx: EngineContext<TModelConfig, TModelMeta>, signal?: AbortSignal) => Promise<TInstance>
	disposeInstance: (instance: TInstance) => Promise<void>
	processChatCompletionTask?: (
		args: EngineChatCompletionArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineChatCompletionResult>
	processTextCompletionTask?: (
		args: EngineTextCompletionArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextCompletionResult>
	processEmbeddingTask?: (
		args: EngineEmbeddingArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineEmbeddingResult>
	processImageToTextTask?: (
		args: EngineImageToTextArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineImageToTextResult>
	processSpeechToTextTask?: (
		args: EngineSpeechToTextArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineSpeechToTextResult>
	processTextToSpeechTask?: (
		args: EngineTextToSpeechArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextToSpeechResult>
	processTextToImageTask?: (
		args: EngineTextToImageArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextToImageResult>
	processImageToImageTask?: (
		args: EngineImageToImageArgs<TModelConfig, TModelMeta>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineImageToImageResult>
}

interface EmbeddingModelOptions {
	task: 'embedding'
}

export type TextCompletionGrammar = string | SomeJSONSchema

interface TextCompletionModelOptions {
	task: 'text-completion'
	contextSize?: number
	grammars?: Record<string, TextCompletionGrammar>
	completionDefaults?: TextCompletionParams
	initialMessages?: ChatMessage[]
	prefix?: string
	batchSize?: number
}

interface LlamaCppModelOptionsBase extends BuiltInModelOptionsBase {
	engine: 'node-llama-cpp'
	task: 'text-completion' | 'embedding'
	sha256?: string
	batchSize?: number
	contextShiftStrategy?: ContextShiftStrategy
	tools?: {
		definitions: Record<string, ToolDefinition>
		includeParamsDocumentation?: boolean
		parallelism?: number
	}
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		gpuLayers?: number
		cpuThreads?: number
		memLock?: boolean
	}
}

interface LlamaCppEmbeddingModelOptions extends LlamaCppModelOptionsBase, EmbeddingModelOptions {
	task: 'embedding'
}

export interface LlamaCppTextCompletionModelOptions extends LlamaCppModelOptionsBase, TextCompletionModelOptions {
	task: 'text-completion'
}

interface GPT4AllModelOptions extends BuiltInModelOptionsBase {
	engine: 'gpt4all'
	task: 'text-completion' | 'embedding'
	md5?: string
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		gpuLayers?: number
		cpuThreads?: number
	}
}

type GPT4AllTextCompletionModelOptions = TextCompletionModelOptions & GPT4AllModelOptions

type GPT4AllEmbeddingModelOptions = GPT4AllModelOptions & EmbeddingModelOptions

export interface TransformersJsModel {
	processor?: {
		url?: string
		file?: string
	}
	processorClass?: TransformersJsProcessorClass
	tokenizerClass?: TransformersJsTokenizerClass
	modelClass?: TransformersJsModelClass
	dtype?: Record<string, TransformersJsDataType> | TransformersJsDataType
}

export interface TransformersJsSpeechModel extends TransformersJsModel {
	speakerEmbeddings: Record<string, {
		url?: string
		file?: string
	}>
	vocoderClass?: TransformersJsModelClass
	vocoder?: {
		url?: string
		file?: string
	}
}

// TODO improve, split these by task and create union?
interface TransformersJsModelOptions extends BuiltInModelOptionsBase {
	engine: 'transformers-js'
	task: 'image-to-text' | 'speech-to-text' | 'text-to-speech' | 'text-completion' | 'embedding'
	textModel?: TransformersJsModel
	visionModel?: TransformersJsModel
	speechModel?: TransformersJsSpeechModel
	device?: {
		gpu?: boolean | 'auto' | (string & {})
	}
}

export interface ModelFileSource {
	url?: string
	file?: string
	sha256?: string
}

interface StableDiffusionModelOptions extends BuiltInModelOptionsBase {
	engine: 'stable-diffusion-cpp'
	task: 'image-to-text' | 'text-to-image' | 'image-to-image'
	sha256?: string
	url?: string
	diffusionModel?: boolean
	vae?: ModelFileSource
	clipL?: ModelFileSource
	clipG?: ModelFileSource
	t5xxl?: ModelFileSource
	taesd?: ModelFileSource
	controlNet?: ModelFileSource
	samplingMethod?: StableDiffusionSamplingMethod
	weightType?: StableDiffusionWeightType
	schedule?: StableDiffusionSchedule
	loras?: ModelFileSource[]
}

export interface CustomEngineModelOptions extends ModelOptionsBase {}

export type BuiltInModelOptions =
	| LlamaCppTextCompletionModelOptions
	| LlamaCppEmbeddingModelOptions
	| GPT4AllTextCompletionModelOptions
	| GPT4AllEmbeddingModelOptions
	| TransformersJsModelOptions
	| StableDiffusionModelOptions

export type ModelOptions = BuiltInModelOptions | CustomEngineModelOptions

export interface EngineEmbeddingResult {
	embeddings: Float32Array[]
	inputTokens: number
}

export interface ChatCompletionResult extends EngineChatCompletionResult {
	id: string
	model: string
}

export interface EngineChatCompletionResult {
	message: AssistantMessage
	finishReason: CompletionFinishReason
	promptTokens: number
	completionTokens: number
	contextTokens: number
}

export interface EngineTextCompletionResult {
	text: string
	finishReason?: CompletionFinishReason
	promptTokens: number
	completionTokens: number
	contextTokens: number
}

export interface EngineImageToTextResult {
	text: string
}

export interface EngineTextToImageResult {
	images: Image[]
	seed: number
}

export interface EngineImageToImageResult {
	images: Image[]
	seed: number
}

export interface EngineSpeechToTextResult {
	text: string
}

export interface EngineTextToSpeechResult {
	audio: Audio
}
