import { suite, it, beforeAll, afterAll, expect } from 'vitest'
import { Florence2ForConditionalGeneration, WhisperForConditionalGeneration } from '@huggingface/transformers'
import { ModelServer } from '#package/server.js'
import { ChatMessage, ToolDefinition } from '#package/types/index.js'
import { ChatWithVisionEngine } from '#package/experiments/ChatWithVision.js'
import { VoiceFunctionCallEngine } from '#package/experiments/VoiceFunctionCall.js'
import { createChatCompletion } from '../util'
import { loadImageFromUrl } from '#package/lib/loadImage.js'
import { loadAudioFromFile } from '#package/lib/loadAudio.js'

suite('chat with vision', () => {
	// florence2 generates a description of the image and passes it to phi3
	const modelServer = new ModelServer({
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
				modelClass: Florence2ForConditionalGeneration,
				dtype: {
					embed_tokens: 'fp16',
					vision_encoder: 'fp32',
					encoder_model: 'fp16',
					decoder_model_merged: 'q4',
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
		await modelServer.start()
	})
	afterAll(async () => {
		await modelServer.stop()
	})

	it('can see', async () => {
		const image = await loadImageFromUrl(
			'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true',
		)
		const messages: ChatMessage[] = [
			{
				role: 'user',
				content: [
					{
						type: 'image',
						image,
					},
					{
						type: 'text',
						text: 'WHAT DO YOUR ELF EYES SEE?',
					},
				],
			},
		]
		const response = await createChatCompletion(modelServer, {
			model: 'vision-at-home',
			temperature: 0,
			messages,
		})
		// Based on the description provided, my elf eyes would see ...
		console.debug({ response: response.result.message.content })
		expect(response.result.message.content).toContain('car')
	})
})

suite('voice functions', () => {
	type SearchParams = { query: string, sources: string }
	const searchCalls: SearchParams[] = []
	const searchTool: ToolDefinition<SearchParams> = {
		description: 'Search',
		parameters: {
			type: 'object',
			properties: {
				query: {
					type: 'string',
				},
				sources: {
					type: 'string',
					enum: ['web', 'all databases', 'local files'],
				},
				test: {
					type: 'number',
					multipleOf: 2,
				}
			},
			required: ['query', 'sources'],
		},
		handler: async (params) => {
			// console.debug('called', { params })
			searchCalls.push(params)
			return (
				`Searching for: ${params.query}` +
				'1. A dessert on Darmok\n' +
				'2. A continent on Etobicoke\n' +
				'3. A city on Risa'
			)
		},
	}

	const modelServer = new ModelServer({
		// log: 'debug',
		engines: {
			'voice-function-calling': new VoiceFunctionCallEngine({
				speechToTextModel: 'whisper-base',
				chatModel: 'functionary',
				tools: {
					search: searchTool,
				},
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
				modelClass: WhisperForConditionalGeneration,
				dtype: {
					encoder_model: 'fp16', // 'fp16' works too
					decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
				},
				device: {
					gpu: false,
				},
			},
			functionary: {
				task: 'text-completion',
				engine: 'node-llama-cpp',
				url: 'https://huggingface.co/meetkai/functionary-small-v3.2-GGUF/blob/main/functionary-small-v3.2.Q4_0.gguf',
				sha256: 'c0afdbbffa498a8490dea3401e34034ac0f2c6e337646513a7dbc04fcef1c3a4',
				// device: {
				// 	gpu: 'vulkan',
				// }
			},
		},
	})

	beforeAll(async () => {
		await modelServer.start()
	})
	afterAll(async () => {
		await modelServer.stop()
	})
	it('can hear', async () => {
		const audio = await loadAudioFromFile('tests/fixtures/tenagra.mp3', {
			sampleRate: 16000,
		})
		const result = await modelServer.processSpeechToTextTask({
			model: 'voice-function-calling',
			audio,
		})
		expect(result.text).toContain('Risa')
		expect(searchCalls).toHaveLength(1)
		expect(searchCalls[0].sources).toMatch('all databases')
		expect(searchCalls[0].query).toMatch(/tenagra/i)
	}, 120000)
})
