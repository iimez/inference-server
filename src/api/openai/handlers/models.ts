import type { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import type { ModelServer } from '#package/server'

// handler for v1/models
// https://platform.openai.com/docs/api-reference/models/list
export function createModelsHandler(modelServer: ModelServer) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		
		const models = modelServer.store.getStatus()
		const data: OpenAI.Model[] = Object.entries(models).map(
			([id, info]) => {
				return {
					object: 'model',
					id,
					created: 0,
					owned_by: info.engine,
				}
			},
		)

		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ object: 'list', data }, null, 2))
	}
}
