import type { ModelPool } from '#package/pool.js'
import type { ModelStore } from '#package/store.js'
import type { Logger } from '#package/lib/logger.js'
import type {
	StableDiffusionWeightType,
	StableDiffusionSamplingMethod,
	StableDiffusionSchedule,
} from '#package/engines/stable-diffusion-cpp/types.js'
import {
	AssistantMessage,
	ChatMessage,
	CompletionFinishReason,
	TextCompletionParams,
	ToolDefinition,
} from '#package/types/completions.js'

import { Image, Audio, ModelConfig } from '#package/types/index.js'

export interface TextCompletionRequestBase extends TextCompletionParams {
	model: string
	stream?: boolean
}

export interface TextCompletionRequest extends TextCompletionRequestBase {
	prompt?: string
}

export interface CompletionChunk {
	tokens: number[]
	text: string
}

export interface ChatCompletionRequest extends TextCompletionRequestBase {
	messages: ChatMessage[]
	grammar?: string
	tools?: {
		definitions: Record<string, ToolDefinition>
		documentParams?: boolean
		maxParallelCalls?: number
	}
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

export interface ObjectRecognitionRequest {
	model: string
	image: Image
	threshold?: number
	labels?: string[]
}

export interface SpeechToTextRequest {
	model: string
	audio: Audio
	language?: string
	prompt?: string
	maxTokens?: number
}

export interface TextToSpeechRequest {
	model: string
	text: string
	voice?: string
}

export type InferenceRequest =
	| TextCompletionRequest
	| ChatCompletionRequest
	| EmbeddingRequest
	| ImageToTextRequest
	| SpeechToTextRequest

export interface EngineContext<TModelConfig = ModelConfig, TModelMeta = unknown> {
	config: TModelConfig
	meta?: TModelMeta
	log: Logger
}

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

export interface EngineObjectRecognitionArgs<TModelConfig = unknown, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	request: ObjectRecognitionRequest
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

export interface ProcessingOptions {
	timeout?: number
	signal?: AbortSignal
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
	TModelInstance = unknown,
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
	createInstance: (ctx: EngineContext<TModelConfig, TModelMeta>, signal?: AbortSignal) => Promise<TModelInstance>
	disposeInstance: (instance: TModelInstance) => Promise<void>
	processChatCompletionTask?: (
		args: EngineChatCompletionArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineChatCompletionResult>
	processTextCompletionTask?: (
		args: EngineTextCompletionArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextCompletionResult>
	processEmbeddingTask?: (
		args: EngineEmbeddingArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineEmbeddingResult>
	processImageToTextTask?: (
		args: EngineImageToTextArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineImageToTextResult>
	processSpeechToTextTask?: (
		args: EngineSpeechToTextArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineSpeechToTextResult>
	processTextToSpeechTask?: (
		args: EngineTextToSpeechArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextToSpeechResult>
	processTextToImageTask?: (
		args: EngineTextToImageArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextToImageResult>
	processImageToImageTask?: (
		args: EngineImageToImageArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineImageToImageResult>
	processObjectRecognitionTask?: (
		args: EngineObjectRecognitionArgs<TModelConfig, TModelMeta>,
		instance: TModelInstance,
		signal?: AbortSignal,
	) => Promise<EngineObjectRecognitionResult>
}

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

export interface ObjectRecognitionResult {
	label: string
	score: number
	box: {
		x: number
		y: number
		width: number
		height: number
	}
}

export interface EngineObjectRecognitionResult {
	objects: ObjectRecognitionResult[]
}