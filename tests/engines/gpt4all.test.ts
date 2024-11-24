import { suite, it, test, beforeAll, afterAll, expect } from 'vitest'
import { InferenceServer } from '#package/server.js'
import {
	ChatCompletionRequest,
	ChatMessage,
	ModelOptions,
} from '#package/types/index.js'
import {
	runContextLeakTest,
	runContextReuseTest,
	runStopTriggerTest,
	runSystemMessageTest,
	runTimeoutTest,
	runCancellationTest,
} from './lib/index.js'

// const models: Record<string, ModelOptions> = {
// 	test: {
// 		task: 'text-completion',
// 		url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
// 		md5: 'f8347badde9bfc2efbe89124d78ddaf5',
// 		engine: 'gpt4all',
// 		prepare: 'blocking',
// 		maxInstances: 2,
// 	},
// }

const testModel: ModelOptions = {
	task: 'text-completion',
	url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
	md5: 'f8347badde9bfc2efbe89124d78ddaf5',
	engine: 'gpt4all',
	prepare: 'blocking',
	maxInstances: 2,
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
})

suite('cache', () => {
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

	it('should reuse context on stateless requests', async () => {
		await runContextReuseTest(inferenceServer)
	})

	it('should not leak when handling multiple sessions', async () => {
		await runContextLeakTest(inferenceServer)
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
	const inferenceServer = new InferenceServer({
		models: {
			test: {
				task: 'text-completion',
				url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
				md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
				engine: 'gpt4all',
				prepare: 'blocking',
				maxInstances: 2,
				initialMessages: preloadedMessages,
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
		const args: ChatCompletionRequest = {
			model: 'test',
			messages: [
				...preloadedMessages,
				{
					role: 'user',
					content: 'Are you sure?',
				},
			],
		}

		const lock = await inferenceServer.pool.requestInstance(args)

		// @ts-ignore
		const activeSession = lock.instance.engineInstance.activeChatSession
		const internalMessages = activeSession.messages
		expect(internalMessages.length).toBe(2)
		await lock.release()
	})

	test('should not utilize preloaded messages', async () => {
		const args: ChatCompletionRequest = {
			model: 'test',
			messages: [
				{
					role: 'user',
					content: 'Whats 2+2?',
				},
			],
		}

		const lock = await inferenceServer.pool.requestInstance(args)

		// const internalMessagesBefore = lock.instance.llm.activeChatSession.messages
		// console.debug({
		// 	internalMessagesBefore,
		// })
		const handle = lock.instance.processChatCompletionTask(args)
		// await handle.process()
		await handle.result
		await lock.release()
		// @ts-ignore
		const activeSession = lock.instance.engineInstance.activeChatSession
		const internalMessagesAfter = activeSession.messages
		// console.debug({
		// 	internalMessagesAfter,
		// })
		expect(internalMessagesAfter[1].content).not.toBe("It's 5!")
	})
})

suite('timeout and cancellation', () => {
	const inferenceServer = new InferenceServer({
		log: 'debug',
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
