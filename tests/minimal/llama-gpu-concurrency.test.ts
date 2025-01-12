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

suite('gpu concurrency', () => {
	let llama: Llama
	let modelOne: LlamaModel
	let modelTwo: LlamaModel
	let context: LlamaContext

	beforeAll(async () => {
		llama = await getLlama({
			gpu: 'vulkan',
			// gpu: false,
		})
		modelOne = await llama.loadModel({
			modelPath: path.resolve(
				os.homedir(),
				// '.cache/inference-server/models/huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF-main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
				// '.cache/inference-server/models/huggingface.co/mradermacher/Ministral-8B-Instruct-2410-GGUF-main/Ministral-8B-Instruct-2410.Q4_K_M.gguf',
				'.cache/inference-server/models/huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF-main/smollm2-1.7b-instruct-q4_k_m.gguf',
			),
		})
		modelTwo = await llama.loadModel({
			modelPath: path.resolve(
				os.homedir(),
				// '.cache/inference-server/models/huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF-main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
				// '.cache/inference-server/models/huggingface.co/mradermacher/Ministral-8B-Instruct-2410-GGUF-main/Ministral-8B-Instruct-2410.Q4_K_M.gguf',
				'.cache/inference-server/models/huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF-main/smollm2-1.7b-instruct-q4_k_m.gguf',
			),
		})
		

	})
	afterAll(async () => {
		await llama.dispose()
	})

	test('parallel completions', async () => {
		context = await modelOne.createContext({
			sequences: 2,
		})
		const createCompletion = async (text: string) => {
			const completion = new LlamaCompletion({
				contextSequence: context.getSequence(),
			})
			const res = await completion.generateCompletion(text, {
				maxTokens: 15,
				// onTextChunk: (text) => {
				// 	console.debug({
				// 		text,
				// 	})
				// }
			})
			return res
		}
		
		const completions = await Promise.all([
			createCompletion('The quick brown fox'),
			createCompletion('All animals are equal,'),
		])

		// const completion = new LlamaCompletion({
		// 	contextSequence: context.getSequence(),
		// })
		// const res = await completion.generateCompletion('"All animals are equal,', {
		// 	maxTokens: 15,
		// })
		// context.dispose()
		console.debug({
			completions,
		})
	})

	test('chat', async () => {
		context = await modelOne.createContext({
			sequences: 1,
			// sequences: 1,
		})
		const createCompletion = async (text: string) => {
			const session = new LlamaChatSession({
				contextSequence: context.getSequence(),
			})
			const res = await session.promptWithMeta(text, {
				maxTokens: 15,
				// onTextChunk: (text) => {
				// 	console.debug({
				// 		text,
				// 	})
				// },
			})
			return res.responseText
		}
		
		const chatCompletions = await Promise.all([
			createCompletion('I need a recipe for a cake.'),
			// createCompletion('Write a story about a cat.'),
		])
		console.debug({
			completions: chatCompletions,
		})
		// const context = await model.createContext()
		// const session = new LlamaChatSession({
		// 	contextSequence: context.getSequence(),
		// })
		// const res = await session.prompt(
		// 	'Tell me something about yourself.',
		// )
		// context.dispose()
		// session.dispose()
		// console.debug({
		// 	chat: res
		// })
	}, 30000)
}, 60000)
