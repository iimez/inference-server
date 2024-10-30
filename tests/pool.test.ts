import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { ModelServer } from '#package/server.js'
import { createChatCompletion } from './util.js'

suite('basic', () => {
	const modelServer = new ModelServer({
		log: 'debug',
		models: {
			test: {
				task: 'text-completion',
				url: 'https://huggingface.co/mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf',
				sha256: '56e1a31ac6e5037174344ac2153c33d873f301f2a312ef2619775190aade51c7',
				engine: 'node-llama-cpp',
			},
		},
	})

	beforeAll(async () => {
		await modelServer.start()
	})
	afterAll(async () => {
		await modelServer.stop()
	})

	test('does a completion', async () => {
		const chat = await createChatCompletion(modelServer, {
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat.result.message.content.length).toBeGreaterThan(0)
	})

	test('does two consecutive completions', async () => {
		const chat1 = await createChatCompletion(modelServer, {
			temperature: 1,
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat1.result.message.content.length).toBeGreaterThan(0)
		const chat2 = await createChatCompletion(modelServer, {
			temperature: 1,
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat2.result.message.content.length).toBeGreaterThan(0)
	})

	test('handles 10 simultaneous completion requests', async () => {
		const results = await Promise.all(
			Array.from({ length: 10 }, () =>
				createChatCompletion(modelServer, {
					temperature: 1,
					messages: [
						{
							role: 'user',
							content: 'Tell me a story, but just its title.',
						},
					],
				}),
			),
		)
		expect(results.length).toBe(10)
		expect(results.every((r) => r.result.message.content.length > 0)).toBe(true)
	})
})

suite('gpu', () => {
	const modelServer = new ModelServer({
		log: 'debug',
		models: {
			gpt4all: {
				url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
				task: 'text-completion',
				md5: 'f8347badde9bfc2efbe89124d78ddaf5',
				engine: 'gpt4all',
				device: { gpu: true },
			},
			'node-llama-cpp': {
				url: 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/blob/main/Phi-3.5-mini-instruct-Q4_K_M.gguf',
				sha256:
					'e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5',
				task: 'text-completion',
				engine: 'node-llama-cpp',
				device: { gpu: true },
			},
		},
	})

	beforeAll(async () => {
		await modelServer.start()
	})
	afterAll(async () => {
		await modelServer.stop()
	})

	test('gpu completion', async () => {
		const chat = await createChatCompletion(modelServer, {
			model: 'gpt4all',
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat.device).toBe('gpu')
	})

	test('switch to different gpu model when necessary', async () => {
		const chat = await createChatCompletion(modelServer, {
			model: 'node-llama-cpp',
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat.device).toBe('gpu')
	})

	test('handle simultaneous requests to two gpu models', async () => {
		const [chat1, chat2] = await Promise.all([
			createChatCompletion(modelServer, {
				model: 'node-llama-cpp',
				messages: [
					{
						role: 'user',
						content: 'Tell me a story, but just its title.',
					},
				],
			}),
			createChatCompletion(modelServer, {
				model: 'gpt4all',
				messages: [
					{
						role: 'user',
						content: 'Tell me a story, but just its title.',
					},
				],
			}),
		])
		expect(chat1.device).toBe('gpu')
		expect(chat2.device).toBe('gpu')
	})
})
