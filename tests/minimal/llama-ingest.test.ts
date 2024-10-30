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


suite('ingest', () => {
	let session: LlamaChatSession
	let llama: Llama

	beforeAll(async () => {
		llama = await getLlama()
		const model = await llama.loadModel({
			modelPath: path.resolve(
				os.homedir(),
				// '.cache/lmodelServer/huggingface/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
				'.cache/inference-server/huggingface.co/mradermacher/Meta-Llama-3-8B-Instruct-GGUF-main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
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
		// const text = fs.readFileSync(`tests/fixtures/github.txt`, 'utf-8').slice(-1000)
		const text = fs.readFileSync(`tests/fixtures/github.txt`, 'utf-8').slice(-2000)
		const a1 = await session.prompt(
			text + '\n---\n\n' + 'Whats this?',
		)
		console.debug({
			a1,
		})
		expect(a1).toMatch(/github|html/i)
	})

	test('large text', async () => {
		const text = fs.readFileSync(`tests/fixtures/lovecraft.txt`, 'utf-8').slice(-2000)
		const a1 = await session.prompt(
			text + '\n---\n\n' + 'Whats this?',
		)
		console.debug({
			a1,
		})
		expect(a1).toMatch(/lovecraft/i)
	})
})
