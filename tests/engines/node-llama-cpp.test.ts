import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import fs from 'node:fs'
import { InferenceServer } from '#package/server.js'
import { ChatMessage, ModelOptions } from '#package/types/index.js'
import {
	runStopTriggerTest,
	runTokenBiasTest,
	runSystemMessageTest,
	runContextLeakTest,
	runContextReuseTest,
	runFileIngestionTest,
	runGenerationContextShiftTest,
	runIngestionContextShiftTest,
	runFunctionCallTest,
	runCombinedFunctionCallTest,
	runParallelFunctionCallTest,
	runBuiltInGrammarTest,
	runRawGBNFGrammarTest,
	runJsonSchemaGrammarTest,
	runTimeoutTest,
	runCancellationTest,
	runFunctionCallWithLeadingResponseTest,
	runReversedCombinedFunctionCallTest,
} from './lib/index.js'
import { createChatCompletion, createTextCompletion, parseInstanceId } from '../util/completions.js'

const testModel: ModelOptions = {
	url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
	sha256: '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff',
	engine: 'node-llama-cpp',
	task: 'text-completion',
	contextSize: 2048,
	prepare: 'blocking',
	grammars: {
		'custom-gbnf-string': fs.readFileSync('tests/fixtures/grammar/name-age-json.gbnf', 'utf-8'),
		'custom-json-schema': {
			type: 'object',
			properties: {
				name: {
					type: 'string',
				},
				age: {
					type: 'number',
				},
			},
			required: ['name', 'age'],
		},
	},
	device: {
		gpu: 'vulkan',
	},
}

suite('features', () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})

	test('stop generation trigger', async () => {
		await runStopTriggerTest(inferenceServer)
	})

	test('system message', async () => {
		await runSystemMessageTest(inferenceServer)
	})

	test('token bias', async () => {
		await runTokenBiasTest(inferenceServer)
	})
})

suite('function calling', async () => {
	const inferenceServer = new InferenceServer({
		log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})

	test('basic function call', async () => {
		await runFunctionCallTest(inferenceServer)
	})
	test('combined function calls', async () => {
		await runCombinedFunctionCallTest(inferenceServer)
		await runReversedCombinedFunctionCallTest(inferenceServer)
	})

	test('parallel function calls', async () => {
		await runParallelFunctionCallTest(inferenceServer)
	})

	test('function call with leading response', async () => {
		await runFunctionCallWithLeadingResponseTest(inferenceServer)
	})
})

suite('grammar', async () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})

	test('built-in grammar', async () => {
		await runBuiltInGrammarTest(inferenceServer)
	})

	test('gbnf string grammar', async () => {
		await runRawGBNFGrammarTest(inferenceServer)
	})

	test('json schema grammar', async () => {
		await runJsonSchemaGrammarTest(inferenceServer)
	})
})

suite('cache', () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				maxInstances: 2,
				device: { gpu: 'auto' },
			},
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})

	test('reuse existing chat context', async () => {
		await runContextReuseTest(inferenceServer)
	})
	test('no leak when handling multiple chat sessions', async () => {
		await runContextLeakTest(inferenceServer)
	})
	test('reuse existing text completion context', async () => {
		const firstPrompt = 'The opposite of red is'
		const comp1 = await createTextCompletion(inferenceServer, {
			model: 'test',
			prompt: firstPrompt,
			stop: ['.'],
		})
		const instanceId1 = parseInstanceId(comp1.task.id)
		// console.debug(comp1.result)

		const secondPrompt = '. In consequence, '
		const comp2 = await createTextCompletion(inferenceServer, {
			model: 'test',
			prompt: firstPrompt + comp1.result.text + secondPrompt,
			stop: ['.'],
		})
		const instanceId2 = parseInstanceId(comp2.task.id)
		// console.debug(comp2.result)
		expect(instanceId1).toBe(instanceId2)
		expect(comp2.result.promptTokens).toBeLessThan(5)
	})
})

suite('preload', () => {
	const initialMessages: ChatMessage[] = [
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
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				initialMessages,
			},
		},
	})

	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('should utilize preloaded messages', async () => {
		const chat = await createChatCompletion(inferenceServer, {
			model: 'test',
			messages: [
				...initialMessages,
				{
					role: 'user',
					content: 'Are you sure?',
				},
			],
		})
		// expect(chat.result.contextTokens).toBeGreaterThan(80)
		expect(chat.result.promptTokens).toBeLessThan(10)
	})

	test('should not utilize preloaded messages', async () => {
		const chat = await createChatCompletion(inferenceServer, {
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
		expect(chat.result.contextTokens).toBe(chat.result.promptTokens + chat.result.completionTokens)
	})

	test('assistant response prefill', async () => {
		const chat = await createChatCompletion(inferenceServer, {
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
				{
					role: 'assistant',
					content: 'Certainly not',
				},
			],
		})
		expect(chat.result.message.content).toMatch(/^Certainly not/)
	})
})

suite('prefix', () => {
	const prefix = 'The Secret is "koalabear"! I continuously remind myself -'
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				prefix,
			},
		},
	})

	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('should utilize prefix', async () => {
		const comp = await createTextCompletion(inferenceServer, {
			model: 'test',
			prompt: prefix + ' It is really "',
			stop: ['.'],
		})
		expect(comp.result.promptTokens).toBeLessThan(5)
	})

	test('should not utilize prefix', async () => {
		const comp = await createTextCompletion(inferenceServer, {
			model: 'test',
			prompt: 'It is really "',
			stop: ['.'],
		})
		expect(comp.result.text).not.toMatch(/koalabear/)
	})
})

// suite('context shift', () => {
// 	const inferenceServer = new InferenceServer({
// 		// log: 'debug',
// 		models: {
// 			test: testModel,
// 		},
// 	})
// 	beforeAll(async () => {
// 		await inferenceServer.start()
// 	})
// 	afterAll(async () => {
// 		await inferenceServer.stop()
// 	})
// 	test('during first user message', async () => {
// 		await runIngestionContextShiftTest(inferenceServer)
// 	})
// 	test('during assistant response', async () => {
// 		await runGenerationContextShiftTest(inferenceServer)
// 	})
// })

suite('ingest', () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('normal text', async () => {
		const res = await runFileIngestionTest(inferenceServer, 'lovecraft')
		expect(res.message.content).toMatch(/horror|lovecraft/i)
	})
	test('a small website', async () => {
		const res = await runFileIngestionTest(inferenceServer, 'hackernews')
		expect(res.message.content).toMatch(/hacker|news/i)
	})
	test('a large website', async () => {
		const res = await runFileIngestionTest(inferenceServer, 'github')
		expect(res.message.content).toMatch(/github|html/i)
	})
})

suite('timeout and cancellation', () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				minInstances: 1,
				device: { gpu: true },
			},
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('timeout', async () => {
		await runTimeoutTest(inferenceServer)
	})
	test('cancellation', async () => {
		await runCancellationTest(inferenceServer)
	})
})
