import type { ModelPool } from '#package/pool.js'
import type { ModelStore } from '#package/store.js'
import type { Logger } from '#package/lib/logger.js'
import type { StableDiffusionSamplingMethod } from '#package/engines/stable-diffusion-cpp/types.js'
import { AssistantMessage, ChatMessage, CompletionFinishReason, ToolDefinition } from '#package/types/chat.js'

import {
	Image,
	Audio,
	ModelConfig,
	ChatCompletionTaskArgs,
	TextCompletionTaskArgs,
	EmbeddingTaskArgs,
	ImageToTextTaskArgs,
	SpeechToTextTaskArgs,
	TextToSpeechTaskArgs,
	TextToImageTaskArgs,
	ImageToImageTaskArgs,
	ObjectDetectionTaskArgs,
} from '#package/types/index.js'

export interface TextCompletionParamsBase {
	temperature?: number
	maxTokens?: number
	seed?: number
	stop?: string[]
	repeatPenalty?: number
	repeatPenaltyNum?: number
	frequencyPenalty?: number
	presencePenalty?: number
	grammar?: string
	topP?: number
	minP?: number
	topK?: number
	tokenBias?: Record<string, number>
}

export interface TextCompletionParams extends TextCompletionParamsBase {
	model: string
	prompt?: string
}

export interface ChatCompletionParams extends TextCompletionParamsBase {
	model: string
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

export interface EmbeddingParams {
	model: string
	input: EmbeddingInput | EmbeddingInput[]
	dimensions?: number
	pooling?: 'cls' | 'mean'
}

export interface ImageToTextParams {
	model: string
	image: Image
	prompt?: string
	maxTokens?: number
}

export interface StableDiffusionParams {
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

export interface TextToImageParams extends StableDiffusionParams {
	model: string
	prompt: string
	width?: number
	height?: number
	seed?: number
}

export interface ImageToImageParams extends StableDiffusionParams {
	model: string
	image: Image
	prompt: string
	width?: number
	height?: number
	seed?: number
}

export interface ObjectDetectionParams {
	model: string
	image: Image
	threshold?: number
	labels?: string[]
}

export interface SpeechToTextParams {
	model: string
	audio: Audio
	language?: string
	prompt?: string
	maxTokens?: number
}

export interface TextToSpeechParams {
	model: string
	text: string
	voice?: string
}

export type InferenceParams =
	| TextCompletionParams
	| ChatCompletionParams
	| EmbeddingParams
	| ImageToTextParams
	| SpeechToTextParams
	| TextToSpeechParams
	| TextToImageParams
	| ImageToImageParams
	| ObjectDetectionParams

export interface EngineContext<TModelConfig = ModelConfig, TModelMeta = unknown> {
	config: TModelConfig
	meta?: TModelMeta
	log: Logger
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

export interface EngineTaskContext<TModelInstance = unknown, TModelConfig = ModelConfig, TModelMeta = unknown>
	extends EngineContext<TModelConfig, TModelMeta> {
	instance: TModelInstance
}

export interface EngineTextCompletionTaskContext<
	TModelInstance = unknown,
	TModelConfig = ModelConfig,
	TModelMeta = unknown,
> extends EngineTaskContext<TModelInstance, TModelConfig, TModelMeta> {
	resetContext?: boolean
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
		task: ChatCompletionTaskArgs,
		ctx: EngineTextCompletionTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineChatCompletionResult>
	processTextCompletionTask?: (
		task: TextCompletionTaskArgs,
		ctx: EngineTextCompletionTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineTextCompletionResult>
	processEmbeddingTask?: (
		task: EmbeddingTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineEmbeddingResult>
	processImageToTextTask?: (
		task: ImageToTextTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineImageToTextResult>
	processSpeechToTextTask?: (
		task: SpeechToTextTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineSpeechToTextResult>
	processTextToSpeechTask?: (
		task: TextToSpeechTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineTextToSpeechResult>
	processTextToImageTask?: (
		task: TextToImageTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineTextToImageResult>
	processImageToImageTask?: (
		task: ImageToImageTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineImageToImageResult>
	processObjectDetectionTask?: (
		task: ObjectDetectionTaskArgs,
		ctx: EngineTaskContext<TModelInstance, TModelConfig, TModelMeta>,
		signal?: AbortSignal,
	) => Promise<EngineObjectDetectionResult>
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

export interface ObjectDetection {
	label: string
	score: number
	box: {
		x: number
		y: number
		width: number
		height: number
	}
}

export interface EngineObjectDetectionResult {
	objects: ObjectDetection[]
}

export type EngineInferenceResult =
	| EngineChatCompletionResult
	| EngineTextCompletionResult
	| EngineEmbeddingResult
	| EngineImageToTextResult
	| EngineSpeechToTextResult
	| EngineTextToSpeechResult
	| EngineTextToImageResult
	| EngineImageToImageResult
	| EngineObjectDetectionResult
