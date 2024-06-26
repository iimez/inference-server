import { suite, test, expect, beforeAll, afterAll } from 'vitest'

import { ModelServer } from '#lllms/server.js'
import { ChatMessage, ModelOptions } from '#lllms/types/index.js'
import {
	runStopTriggerTest,
	runTokenBiasTest,
	runSystemMessageTest,
	runContextLeakTest,
	runContextReuseTest,
	runFileIngestionTest,
	runContextShiftGenerationTest,
	runContextShiftIngestionTest,
	runFunctionCallTest,
	runSequentialFunctionCallTest,
	runParallelFunctionCallTest,
	runGrammarTest,
} from './lib/index.js'
import { createChatCompletion } from '../util.js'

const models: Record<string, ModelOptions> = {
	test: {
		task: 'text-completion',
		// url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
		// sha256: 'c57380038ea85d8bec586ec2af9c91abc2f2b332d41d6cf180581d7bdffb93c1',
		// on functionary everything but the context shift test works.
		url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/raw/main/functionary-small-v2.5.Q4_0.gguf',
		sha256: '3941bf2a5d1381779c60a7ccb39e8c34241e77f918d53c7c61601679b7160c48',
		engine: 'node-llama-cpp',
		contextSize: 2048,
		engineOptions: {
			gpu: false,
		}
	},
}

suite('features', () => {
	const llms = new ModelServer({
		log: 'debug',
		models,
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('stop generation trigger', async () => {
		await runStopTriggerTest(llms)
	})

	test('system message', async () => {
		await runSystemMessageTest(llms)
	})

	test('token bias', async () => {
		await runTokenBiasTest(llms)
	})
	
	test('grammar', async () => {
		await runGrammarTest(llms)
	})
	
	test('function call', async () => {
		await runFunctionCallTest(llms)
	})
	
	test('sequential function calls', async () => {
		await runSequentialFunctionCallTest(llms)
	})
	
	test('parallel function calls', async () => {
		await runParallelFunctionCallTest(llms)
	})
})

suite('cache', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				...models.test,
				maxInstances: 2,
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('reuse context on stateless requests', async () => {
		await runContextReuseTest(llms)
	})
	test('no leak when handling multiple sessions', async () => {
		await runContextLeakTest(llms)
	})
})

suite('preload', () => {
	const preloadedMessages: ChatMessage[] = [
		{
			role: 'system',
			content: 'You are an advanced mathematician.',
		},
		{
			role: 'user',
			content: 'Whats 2+2?',
		},
		{
			role: 'assistant',
			content: "It's 5!",
		},
	]
	const llms = new ModelServer({
		models: {
			test: {
				url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
				task: 'text-completion',
				sha256:
					'1977ae6185ef5bc476e27db85bb3d79ca4bd87e7b03399083c297d9c612d334c',
				engine: 'node-llama-cpp',
				prepare: 'blocking',
				maxInstances: 2,
				preload: {
					messages: preloadedMessages,
				},
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('should utilize preloaded messages', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'test',
			messages: [
				...preloadedMessages,
				{
					role: 'user',
					content: 'Are you sure?',
				},
			],
		})
		// console.debug({
		// 	response: chat.result.message.content,
		// 	inputTokens: chat.result.promptTokens,
		// })
		expect(chat.result.promptTokens).toBe(8)
	})

	test('should not utilize preloaded messages', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'test',
			messages: [
				{
					role: 'system',
					content: 'You are an advanced mathematician.',
				},
				{
					role: 'user',
					content: 'Whats 2+2?',
				},
			],
		})
		// console.debug({
		// 	response: chat.result.message.content,
		// 	inputTokens: chat.result.promptTokens,
		// })
		expect(chat.result.promptTokens).toBe(27)
	})
})

suite('context shift', () => {
	const llms = new ModelServer({
		models,
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('input that exceeds context size', async () => {
		await runContextShiftIngestionTest(llms)
	})
	test('context shift during generation', async () => {
		await runContextShiftGenerationTest(llms)
	})
})

suite('ingest', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models,
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	// works
	test('normal text', async () => {
		const res = await runFileIngestionTest(llms, 'lovecraft')
		expect(res.message.content).toMatch(/horror|lovecraft/i)
		console.debug({
			res: res.message.content,
		})
	})
	// works
	test('a small website', async () => {
		const res = await runFileIngestionTest(llms, 'hackernews')
		expect(res.message.content).toMatch(/hacker|news/i)
		console.debug({
			res: res.message.content,
		})
	})
	// errors with "max callstack exceeded"
	test('a large website', async () => {
		const res = await runFileIngestionTest(llms, 'github')
		expect(res.message.content).toMatch(/github/i)
		console.debug({
			res: res.message.content,
		})
	})
})
