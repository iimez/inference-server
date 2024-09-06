import { suite, it, test, beforeAll, afterAll, expect } from 'vitest'
import {
	Florence2ForConditionalGeneration,
	WhisperForConditionalGeneration,
} from '@huggingface/transformers'
import { ModelServer } from '#lllms/server.js'
import {
	ChatMessage,
} from '#lllms/types/index.js'
import { ChatWithVisionEngine } from '#lllms/experiments/ChatWithVision.js'
import { VoiceFunctionCallEngine } from '#lllms/experiments/VoiceFunctionCall.js'
import { createChatCompletion } from '../util'

suite('chat with vision', () => {
	// florence2 generates a description of the image and passes it to phi3
	const llms = new ModelServer({
		// log: 'debug',
		concurrency: 2,
		engines: {
			'chat-with-vision': new ChatWithVisionEngine({
				chatModel: 'phi3',
				imageToTextModel: 'florence2',
			}),
		},
		models: {
			phi3: {
				url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
				md5: 'f8347badde9bfc2efbe89124d78ddaf5',
				engine: 'gpt4all',
				task: 'text-completion',
			},
			florence2: {
				url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
				engine: 'transformers-js',
				task: 'image-to-text',
				visionModel: {
					modelClass: Florence2ForConditionalGeneration,
					dtype: {
						embed_tokens: 'fp16',
						vision_encoder: 'fp32',
						encoder_model: 'fp16',
						decoder_model_merged: 'q4',
					},
				},
				device: {
					gpu: false,
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

// suite('voice functions', () => {
// 	let searchSources: string
// 	const llms = new ModelServer({
// 		// log: 'debug',
// 		engines: {
// 			'voice-function-calling': new VoiceFunctionCallEngine({
// 				speechToTextModel: 'whisper-base',
// 				chatModel: 'functionary',
// 				tools: {
// 					search: {
// 						description: 'Search',
// 						parameters: {
// 							type: 'object',
// 							properties: {
// 								query: {
// 									type: 'string',
// 								},
// 								sources: {
// 									type: 'string',
// 									enum: ['web', 'all databases', 'local files'],
// 								},
// 							},
// 							required: ['query', 'sources'],
// 						},
// 						handler: async (params) => {
// 							searchSources = params.sources
// 							// console.debug('called', { params })
// 							return (
// 								`Searching for: ${params.query}` +
// 								'1. A dessert on Darmok\n' +
// 								'2. A continent on Etobicoke\n' +
// 								'3. A city on Risa'
// 							)
// 						},
// 					},
// 				},
// 			}),
// 		},
// 		models: {
// 			'voice-function-calling': {
// 				engine: 'voice-function-calling',
// 				task: 'speech-to-text',
// 			},
// 			'whisper-base': {
// 				url: 'https://huggingface.co/onnx-community/whisper-base',
// 				engine: 'transformers-js',
// 				task: 'speech-to-text',
// 				prepare: 'async',
// 				minInstances: 1,
// 				speechModel: {
// 					modelClass: WhisperForConditionalGeneration,
// 					dtype: {
// 						encoder_model: 'fp32', // 'fp16' works too
// 						decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
// 					},
// 				},
// 				device: {
// 					gpu: false,
// 				},
// 			},
// 			functionary: {
// 				url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/raw/main/functionary-small-v2.5.Q4_0.gguf',
// 				sha256:
// 					'3941bf2a5d1381779c60a7ccb39e8c34241e77f918d53c7c61601679b7160c48',
// 				engine: 'node-llama-cpp',
// 				task: 'text-completion',
// 			},
// 		},
// 	})

// 	beforeAll(async () => {
// 		await llms.start()
// 	})
// 	afterAll(async () => {
// 		await llms.stop()
// 	})
// 	it('can hear', async () => {
// 		const result = await llms.processSpeechToTextTask({
// 			file: 'tests/fixtures/tenabra.mp3',
// 			model: 'voice-function-calling',
// 			// model: 'whisper-base',
// 		})
// 		console.debug({ result })
// 		expect(result.text).toContain('Risa')
// 		expect(searchSources).toMatch('all databases')
// 	})
// })
