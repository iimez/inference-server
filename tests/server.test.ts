import { describe, it, expect, beforeAll } from 'vitest'
import request from 'supertest'
import express, { Express } from 'express'
import { ModelServer, ModelServerOptions } from '#package/server.js'
import { createExpressMiddleware } from '#package/http.js'

const testModel = 'phi3-mini-4k'

const testConfig: ModelServerOptions = {
	concurrency: 1,
	models: {
		[testModel]: {
			url: 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/blob/main/Phi-3.5-mini-instruct-Q4_K_M.gguf',
			sha256:
				'e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5',
			task: 'text-completion',
			device: { gpu: false },
			engine: 'node-llama-cpp',
			minInstances: 1,
		},
	},
}

describe('Express App', () => {
	let app: Express
	let modelServer: ModelServer

	beforeAll(async () => {
		modelServer = new ModelServer(testConfig)
		app = express()
		app.use(express.json(), createExpressMiddleware(modelServer))
	})

	it('Starts up without errors', async () => {
		await modelServer.start()
	})

	it('Responds to requests', async () => {
		const res = await request(app).get('/')
		expect(res.status).toBe(200)
		expect(res.body).toMatchObject({
			downloads: { queue: 0, pending: 0, tasks: [] },
			pool: { processing: 0, waiting: 0, instances: {} },
		})
	})

	it('Has an instance of the model ready', async () => {
		const res = await request(app).get('/')
		expect(res.status).toBe(200)
		expect(Object.keys(res.body.pool.instances).length).toBe(1)
		const instanceKey = Object.keys(res.body.pool.instances)[0] as string
		const instance = res.body.pool.instances[instanceKey]
		expect(instance.model).toBe(testModel)
	})
})
