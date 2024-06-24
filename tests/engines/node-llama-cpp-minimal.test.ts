import path from 'node:path'
import os from 'node:os'
import fs from 'node:fs'
import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import {
	getLlama,
	Llama,
	defineChatSessionFunction,
	LlamaChatSession,
} from 'node-llama-cpp'

suite('function calls', () => {
	let session: LlamaChatSession
	let llama: Llama

	beforeAll(async () => {
		llama = await getLlama()
		const model = await llama.loadModel({
			modelPath: path.resolve(
				os.homedir(),
				'.cache/lllms/huggingface/meetkai/functionary-small-v2.4-GGUF-main/functionary-small-v2.4.Q4_0.gguf',
			),
		})
		const context = await model.createContext()
		session = new LlamaChatSession({
			contextSequence: context.getSequence(),
		})
	})
	
	afterAll(async () => {
		await llama.dispose()
	})

	test('Requesting two function calls that could be run in parallel', async () => {
		const generatedNumbers: number[] = []
		const functions = {
			getRandomNumber: defineChatSessionFunction({
				description: 'Generate a random integer in given range',
				params: {
					type: 'object',
					properties: {
						min: {
							type: 'number',
						},
						max: {
							type: 'number',
						},
					},
				},
				handler: async (params) => {
					const num =
						Math.floor(Math.random() * (params.max - params.min + 1)) +
						params.min
					generatedNumbers.push(num)
					console.debug('Handler called', {
						params,
						result: num,
					})
					return num.toString()
				},
			}),
		}
		const a1 = await session.prompt(
			'Roll the dice twice, then tell me the sum.',
			{ functions },
		)
		console.debug({
			a1,
		})
		expect(generatedNumbers.length).toBe(2)
	})
})

suite('long prompts', () => {
	let session: LlamaChatSession
	let llama: Llama

	beforeAll(async () => {
		llama = await getLlama()
		const model = await llama.loadModel({
			modelPath: path.resolve(
				os.homedir(),
				'.cache/lllms/huggingface/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			),
		})
		const context = await model.createContext()
		session = new LlamaChatSession({
			contextSequence: context.getSequence(),
		})
	})
	
	afterAll(async () => {
		await llama.dispose()
	})

	test('large html', async () => {
		// const text = fs.readFileSync(`tests/fixtures/hackernews.txt`, 'utf-8')
		const text = fs.readFileSync(`tests/fixtures/github.txt`, 'utf-8')
		const a1 = await session.prompt(
			text + '\n---\n\n' + 'Whats this?',
		)
		console.debug({
			a1,
		})
		expect(a1).toMatch(/github/i)
	})
	
	test('large text', async () => {
		const text = fs.readFileSync(`tests/fixtures/lovecraft.txt`, 'utf-8')
		const a1 = await session.prompt(
			text + text + text + '\n---\n\n' + 'Whats this?',
		)
		console.debug({
			a1,
		})
		expect(a1).toMatch(/lovecraft/i)
	})
})
