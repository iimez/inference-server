import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { ModelServer } from '#package/server.js'
import { createChatCompletion } from './util/completions.js'

suite('basic', () => {
	const modelServer = new ModelServer({
		log: 'debug',
		models: {
			test: {
				task: 'text-completion',
				url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
				sha256: '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff',
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
				url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
				sha256: '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff',
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
