import { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import { EmbeddingRequest } from '#package/types/index.js'
import { parseJSONRequestBody } from '#package/api/parseJSONRequestBody.js'
import { omitEmptyValues } from '#package/lib/util.js'
import { InferenceServer } from '#package/server.js'

// handler for v1/embeddings
// https://platform.openai.com/docs/api-reference/embeddings

type OpenAIEmbeddingsParams = OpenAI.EmbeddingCreateParams

export function createEmbeddingsHandler(inferenceServer: InferenceServer) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args: OpenAIEmbeddingsParams

		try {
			const body = await parseJSONRequestBody(req)
			args = body
		} catch (e) {
			console.error(e)
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}
		
		// TODO ajv schema validation?
		if (!args.model || !args.input) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}
		if (!inferenceServer.modelExists(args.model)) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid model' }))
			return
		}

		const controller = new AbortController()
		req.on('close', () => {
			console.debug('Client closed connection')
			controller.abort()
		})
		req.on('end', () => {
			console.debug('Client ended connection')
			controller.abort()
		})

		try {


			let input = args.input

			if (typeof input !== 'string') {
				throw new Error('Input must be a string')
			}

			const embeddingsReq = omitEmptyValues<EmbeddingRequest>({
				model: args.model,
				input: args.input as string,
			})

			const { instance, release } = await inferenceServer.requestInstance(
				embeddingsReq,
				controller.signal,
			)
			const task = instance.processEmbeddingTask(embeddingsReq)
			const result = await task.result
			release()

			const response: OpenAI.CreateEmbeddingResponse = {
				model: instance.modelId,
				object: 'list',
				data: result.embeddings.map((embedding, index) => ({
					embedding: Array.from(embedding),
					index,
					object: 'embedding',
				})),
				usage: {
					prompt_tokens: result.inputTokens,
					total_tokens: result.inputTokens,
				},
			}
			res.writeHead(200, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify(response, null, 2))

		} catch (err) {
			console.error(err)
			res.writeHead(500, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Internal server error' }))
		}
	}
}
