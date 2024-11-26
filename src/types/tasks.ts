import {
	TextCompletionParams,
	ChatCompletionParams,
	EmbeddingParams,
	ObjectDetectionParams,
	ImageToTextParams,
	SpeechToTextParams,
	TextToSpeechParams,
	TextToImageParams,
	ImageToImageParams,
	InferenceTaskResult,
} from '#package/types/engine.js'

export type TaskKind =
	| 'text-completion'
	| 'chat-completion'
	| 'embedding'
	| 'image-to-text'
	| 'image-to-image'
	| 'text-to-image'
	| 'speech-to-text'
	| 'text-to-speech'
	| 'object-detection'

export interface TextCompletionChunk {
	tokens: number[]
	text: string
}

export interface ProcessingOptions {
	timeout?: number
	signal?: AbortSignal
}

export interface TextCompletionProcessingOptions extends ProcessingOptions {
	onChunk?: (chunk: TextCompletionChunk) => void
}

export interface SpeechToTextProcessingOptions extends ProcessingOptions {
	onChunk?: (chunk: { text: string }) => void
}

export interface TextToSpeechProcessingOptions extends ProcessingOptions {
	onChunk?: (chunk: { audio: Buffer }) => void
}

export type TextCompletionTaskArgs = TextCompletionParams & TextCompletionProcessingOptions
export type ChatCompletionTaskArgs = ChatCompletionParams & TextCompletionProcessingOptions
export type EmbeddingTaskArgs = EmbeddingParams & ProcessingOptions
export type ObjectDetectionTaskArgs = ObjectDetectionParams & ProcessingOptions
export type ImageToTextTaskArgs = ImageToTextParams & ProcessingOptions
export type SpeechToTextTaskArgs = SpeechToTextParams & SpeechToTextProcessingOptions
export type TextToSpeechTaskArgs = TextToSpeechParams & TextToSpeechProcessingOptions
export type TextToImageTaskArgs = TextToImageParams & ProcessingOptions
export type ImageToImageTaskArgs = ImageToImageParams & ProcessingOptions

export interface TextCompletionInferenceTaskArgs extends TextCompletionTaskArgs {
	task: 'text-completion'
}

export interface ChatCompletionInferenceTaskArgs extends ChatCompletionTaskArgs {
	task: 'chat-completion'
}

export interface EmbeddingInferenceTaskArgs extends EmbeddingTaskArgs {
	task: 'embedding'
}

export interface TextToImageInferenceTaskArgs extends TextToImageTaskArgs {
	task: 'text-to-image'
}

export interface ImageToTextInferenceTaskArgs extends ImageToTextTaskArgs {
	task: 'image-to-text'
}

export interface ImageToImageInferenceTaskArgs extends ImageToImageTaskArgs {
	task: 'image-to-image'
}

export interface ObjectDetectionInferenceTaskArgs extends ObjectDetectionTaskArgs {
	task: 'object-detection'
}

export interface TextToSpeechInferenceTaskArgs extends TextToSpeechTaskArgs {
	task: 'text-to-speech'
}

export interface SpeechToTextInferenceTaskArgs extends SpeechToTextTaskArgs {
	task: 'speech-to-text'
}

export type InferenceTaskArgs =
	| TextCompletionInferenceTaskArgs
	| ChatCompletionInferenceTaskArgs
	| EmbeddingInferenceTaskArgs
	| ObjectDetectionInferenceTaskArgs
	| ImageToTextInferenceTaskArgs
	| ImageToImageInferenceTaskArgs
	| TextToImageInferenceTaskArgs
	| TextToSpeechInferenceTaskArgs
	| SpeechToTextInferenceTaskArgs

export interface InferenceTask<TResult = InferenceTaskResult> {
	id: string
	model: string
	createdAt: Date
	result: Promise<TResult>
	cancel: () => void
}
