import type { SomeJSONSchema } from 'ajv/dist/types/json-schema'
import type { Sharp } from 'sharp'
import type { BuiltInEngineName } from '#package/engines/index.js'
import { ChatMessage, TextCompletionParams, ToolDefinition } from '#package/types/completions.js'
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
import type { InferenceRequest } from '#package/types/engine.js'
export * from '#package/types/completions.js'
export * from '#package/types/engine.js'

export type ModelTaskType =
	| 'text-completion'
	| 'embedding'
	| 'image-to-text'
	| 'image-to-image'
	| 'text-to-image'
	| 'speech-to-text'
	| 'text-to-speech'
	| 'object-detection'

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

// TODO could split this up, but its only used internally?
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

export interface Image {
	data: Buffer
	width: number
	height: number
	channels: 1 | 2 | 3 | 4
}

export interface Audio {
	sampleRate: number
	channels: 1 | 2
	samples: Float32Array
}

export interface ModelRequestMeta {
	sequence: number
	abortController: AbortController
}

export type ModelInstanceRequest = ModelRequestMeta & InferenceRequest

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
		documentParams?: boolean
		maxParallelCalls?: number
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
	processorClass?: TransformersJsProcessorClass | string
	tokenizerClass?: TransformersJsTokenizerClass | string
	modelClass?: TransformersJsModelClass | string
	dtype?: Record<string, TransformersJsDataType> | TransformersJsDataType
}

export type SpeakerEmbeddings = Record<
	string,
	| {
			url?: string
			file?: string
	  }
	| Float32Array
>

export interface TransformersJsSpeechModel {
	speakerEmbeddings?: SpeakerEmbeddings
	vocoderClass?: TransformersJsModelClass
	vocoder?: {
		url?: string
		file?: string
	}
}

// TODO improve, split these by task and create union?
interface TransformersJsModelOptions extends BuiltInModelOptionsBase, TransformersJsModel, TransformersJsSpeechModel {
	engine: 'transformers-js'
	task: 'image-to-text' | 'speech-to-text' | 'text-to-speech' | 'text-completion' | 'embedding' | 'object-detection'
	textModel?: TransformersJsModel
	visionModel?: TransformersJsModel
	speechModel?: TransformersJsModel & TransformersJsSpeechModel
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
