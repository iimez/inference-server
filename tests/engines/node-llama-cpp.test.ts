import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import fs from 'node:fs'
import { ModelServer } from '#package/server.js'
import {
	ChatMessage,
	ModelOptions,
} from '#package/types/index.js'
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
	runSequentialFunctionCallTest,
	runParallelFunctionCallTest,
	runBuiltInGrammarTest,
	runRawGBNFGrammarTest,
	runJsonSchemaGrammarTest,
	runTimeoutTest,
	runCancellationTest,
} from './lib/index.js'
import { createChatCompletion, createTextCompletion, parseInstanceId } from '../util.js'

const testModel: ModelOptions = {
	// url: 'https://huggingface.co/mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf',
	// sha256: '56e1a31ac6e5037174344ac2153c33d873f301f2a312ef2619775190aade51c7',
	url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
	sha256: '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff',
	engine: 'node-llama-cpp',
	task: 'text-completion',
	contextSize: 2048,
	prepare: 'blocking',
	grammars: {
		'custom-gbnf-string': fs.readFileSync(
			'tests/fixtures/grammar/name-age-json.gbnf',
			'utf-8',
		),
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
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
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
})

suite('function calling', async () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				task: 'text-completion',
				engine: 'node-llama-cpp',
				url: 'https://huggingface.co/meetkai/functionary-small-v3.2-GGUF/blob/main/functionary-small-v3.2.Q4_0.gguf',
				sha256:
					'c0afdbbffa498a8490dea3401e34034ac0f2c6e337646513a7dbc04fcef1c3a4',
				// url: 'https://huggingface.co/mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf',
				// sha256:
				// 	'56e1a31ac6e5037174344ac2153c33d873f301f2a312ef2619775190aade51c7',
				// tools: {
				// 	definitions: {},
				// 	includeParamsDocumentation: true,
				// 	parallelism: 1,
				// },
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('basic function call', async () => {
		await runFunctionCallTest(llms)
	})
	test('sequential function calls', async () => {
		await runSequentialFunctionCallTest(llms)
	})
	test('parallel function calls', async () => {
		await runParallelFunctionCallTest(llms)
	})
})

suite('grammar', async () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('built-in grammar', async () => {
		await runBuiltInGrammarTest(llms)
	})

	test('gbnf string grammar', async () => {
		await runRawGBNFGrammarTest(llms)
	})

	test('json schema grammar', async () => {
		await runJsonSchemaGrammarTest(llms)
	})
})

suite('cache', () => {
	const llms = new ModelServer({
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
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('reuse existing chat context', async () => {
		await runContextReuseTest(llms)
	})
	test('no leak when handling multiple chat sessions', async () => {
		await runContextLeakTest(llms)
	})
	test('reuse existing text completion context', async () => {
		const firstPrompt = 'The opposite of red is'
		const comp1 = await createTextCompletion(llms, {
			model: 'test',
			prompt: firstPrompt,
			stop: ['.'],
		})
		const instanceId1 = parseInstanceId(comp1.task.id)
		// console.debug(comp1.result)
		
		const secondPrompt = '. In consequence, '
		const comp2 = await createTextCompletion(llms, {
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
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				initialMessages,
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
		expect(chat.result.contextTokens).toBe(
			chat.result.promptTokens + chat.result.completionTokens,
		)
	})

	test('assistant response prefill', async () => {
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
				{
					role: 'assistant',
					content: 'Certainly not',
				}
			],
		})
		expect(chat.result.message.content).toMatch(/^Certainly not/)
	})
})


suite('prefix', () => {
	const prefix = 'The Secret is "koalabear"! I continuously remind myself -'
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				prefix,
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('should utilize prefix', async () => {
		const comp = await createTextCompletion(llms, {
			model: 'test',
			prompt: prefix + ' It is really "',
			stop: ['.'],
		})
		expect(comp.result.promptTokens).toBeLessThan(5)
	})

	test('should not utilize prefix', async () => {
		const comp = await createTextCompletion(llms, {
			model: 'test',
			prompt: 'It is really "',
			stop: ['.'],
		})
		expect(comp.result.text).not.toMatch(/koalabear/)
	})
})

// suite('context shift', () => {
// 	const llms = new ModelServer({
// 		// log: 'debug',
// 		models: {
// 			test: testModel,
// 		},
// 	})
// 	beforeAll(async () => {
// 		await llms.start()
// 	})
// 	afterAll(async () => {
// 		await llms.stop()
// 	})
// 	test('during first user message', async () => {
// 		await runIngestionContextShiftTest(llms)
// 	})
// 	test('during assistant response', async () => {
// 		await runGenerationContextShiftTest(llms)
// 	})
// })

suite('ingest', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('normal text', async () => {
		const res = await runFileIngestionTest(llms, 'lovecraft')
		expect(res.message.content).toMatch(/horror|lovecraft/i)
	})
	test('a small website', async () => {
		const res = await runFileIngestionTest(llms, 'hackernews')
		expect(res.message.content).toMatch(/hacker|news/i)
	})
	test('a large website', async () => {
		const res = await runFileIngestionTest(llms, 'github')
		expect(res.message.content).toMatch(/github|html/i)
	})
})

suite('timeout and cancellation', () => {
	const llms = new ModelServer({
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
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('timeout', async () => {
		await runTimeoutTest(llms)
	})
	test('cancellation', async () => {
		await runCancellationTest(llms)
	})
})
