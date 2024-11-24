import http from 'node:http'
import express from 'express'
import cors from 'cors'
import { createOpenAIRequestHandlers } from '#package/api/openai/index.js'
import { InferenceServer } from '#package/server.js'

export function createOpenAIMiddleware(inferenceServer: InferenceServer) {
	const router = express.Router()
	const requestHandlers = createOpenAIRequestHandlers(inferenceServer)
	router.get('/v1/models', requestHandlers.models)
	router.post('/v1/completions', requestHandlers.completions)
	router.post('/v1/chat/completions', requestHandlers.chatCompletions)
	router.post('/v1/embeddings', requestHandlers.embeddings)
	return router
}

export function createExpressMiddleware(inferenceServer: InferenceServer) {
	const router = express.Router()
	router.get('/', (req, res) => {
		res.json(inferenceServer.getStatus())
	})
	router.use('/openai', createOpenAIMiddleware(inferenceServer))
	return router
}

export function createExpressServer(inferenceServer: InferenceServer) {	
	const app = express()
	app.use(
		cors(),
		express.json({ limit: '50mb' }),
		createExpressMiddleware(inferenceServer),
	)
	app.set('json spaces', 2)
	return http.createServer(app)
}