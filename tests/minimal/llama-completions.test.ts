import path from 'node:path'
import os from 'node:os'
import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import {
	getLlama,
	Llama,
	LlamaChatSession,
	LlamaCompletion,
	LlamaContext,
	LlamaModel,
} from 'node-llama-cpp'

suite('basic', () => {
	let llama: Llama
	let model: LlamaModel

	beforeAll(async () => {
		llama = await getLlama({
			gpu: 'vulkan',
			// gpu: false,
		})
		model = await llama.loadModel({
			modelPath: path.resolve(
				os.homedir(),
				'.cache/inference-server/huggingface.co/mradermacher/Meta-Llama-3-8B-Instruct-GGUF-main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
			),
		})
	})
	afterAll(async () => {
		await llama.dispose()
	})

	test('completion', async () => {
		const context = await model.createContext()
		const completion = new LlamaCompletion({
			contextSequence: context.getSequence(),
		})
		const res = await completion.generateCompletion('"All animals are equal,', {
			maxTokens: 15,
		})
		context.dispose()
		console.debug({
			completion: res
		})
	})

	test('chat', async () => {
		const context = await model.createContext()
		const session = new LlamaChatSession({
			contextSequence: context.getSequence(),
		})
		const res = await session.prompt(
			'Tell me something about yourself.',
		)
		context.dispose()
		session.dispose()
		console.debug({
			chat: res
		})
	}, 30000)
}, 60000)
