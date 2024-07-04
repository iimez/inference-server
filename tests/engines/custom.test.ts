import { suite, it, test, beforeAll, afterAll, expect } from 'vitest'
// @ts-ignore
import {
	Florence2ForConditionalGeneration,
	WhisperForConditionalGeneration,
} from '@xenova/transformers'
import { ModelServer } from '#lllms/server.js'
import {
	ChatCompletionRequest,
	ChatMessage,
	ModelEngine,
	ModelOptions,
} from '#lllms/types/index.js'
import { ChatWithVisionEngine } from '#lllms/lib/custom-engines/ChatWithVision.js'
import { VoiceFunctionCallEngine } from '#lllms/lib/custom-engines/VoiceFunctionCall.js'
import { createChatCompletion } from '../util'

suite('chat with vision', () => {
	// florence2 generates a description of the image and passes it to llama3-8b
	const llms = new ModelServer({
		// log: 'debug',
		concurrency: 2,
		engines: {
			'chat-with-vision': new ChatWithVisionEngine({
				chatModel: 'llama3-8b',
				imageToTextModel: 'florence2',
			}),
		},
		models: {
			'llama3-8b': {
				url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
				md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
				engine: 'gpt4all',
				task: 'text-completion',
			},
			florence2: {
				url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
				engine: 'transformers-js',
				task: 'image-to-text',
				engineOptions: {
					modelClass: Florence2ForConditionalGeneration,
					gpu: false,
					dtype: {
						embed_tokens: 'fp16',
						vision_encoder: 'fp32',
						encoder_model: 'fp16',
						decoder_model_merged: 'q4',
					},
				},
			},
			'vision-at-home': {
				engine: 'chat-with-vision',
				task: 'text-completion',
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	it('can see', async () => {
		const messages: ChatMessage[] = [
			{
				role: 'user',
				content: [
					{
						type: 'image',
						url: 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true',
					},
					{
						type: 'text',
						text: 'WHAT DO YOUR ELF EYES SEE?',
					},
				],
			},
		]
		const response = await createChatCompletion(llms, {
			model: 'vision-at-home',
			temperature: 0,
			messages,
		})
		console.debug({ response: response.result.message.content })
		expect(response.result.message.content).toContain('car')
	})
})

suite('voice functions', () => {
	const llms = new ModelServer({
		// log: 'debug',
		engines: {
			'voice-function-calling': new VoiceFunctionCallEngine({
				speechToTextModel: 'whisper-base',
				chatModel: 'functionary',
				tools: {
					search: {
						description: 'Search',
						parameters: {
							type: 'object',
							properties: {
								query: {
									type: 'string',
								},
								sources: {
									type: 'enum',
									enum: ['web', 'all databases', 'local files'],
								}
							},
						},
						handler: async (params) => {
							// console.debug('called', { params })
							return `Searching for: ${params.query}` +
								'1. A dessert on Darmok\n' +
								'2. A continent on Etobicoke\n' +
								'3. A city on Risa'
						},
					},
				}
			}),
		},
		models: {
			'voice-function-calling': {
				engine: 'voice-function-calling',
				task: 'speech-to-text',
			},
			'whisper-base': {
				url: 'https://huggingface.co/onnx-community/whisper-base',
				engine: 'transformers-js',
				task: 'speech-to-text',
				prepare: 'async',
				minInstances: 1,
				engineOptions: {
					modelClass: WhisperForConditionalGeneration,
					gpu: false,
					dtype: {
						encoder_model: 'fp32', // 'fp16' works too
						decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
					},
				},
			},
			'functionary': {
				url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/raw/main/functionary-small-v2.5.Q4_0.gguf',
				sha256: '3941bf2a5d1381779c60a7ccb39e8c34241e77f918d53c7c61601679b7160c48',
				engine: 'node-llama-cpp',
				task: 'text-completion',
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	it('can hear', async () => {
		const result = await llms.processSpeechToTextTask({
			file: 'tests/fixtures/tenabra.mp3',
			model: 'voice-function-calling',
			// model: 'whisper-base',
		})
		// console.debug({ result })
		// expect(result.text).toContain('Risa')
	})
})