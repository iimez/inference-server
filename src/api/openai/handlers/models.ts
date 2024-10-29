import type { IncomingMessage, ServerResponse } from 'node:http'
import path from 'node:path'
import type { OpenAI } from 'openai'
import type { ModelServer } from '#package/server'

// https://platform.openai.com/docs/api-reference/models/list
export function createModelsHandler(modelServer: ModelServer) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		
		const models = modelServer.store.getStatus()
		const data: OpenAI.Model[] = Object.entries(models).map(
			([id, info]) => {
				// const lastModDate = new Date(info.source.lastModified)
				// const created = Math.floor(lastModDate.getTime() / 1000)
				
				// const dirPath = path.dirname(info.source.file);
				// const lastDir = path.basename(dirPath);
				// const baseName = path.basename(info.source.file);
				const owned_by = info.engine// + ':' + path.join(lastDir, baseName);

				return {
					object: 'model',
					id,
					created: 0,
					owned_by,
				}
			},
		)

		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ object: 'list', data }, null, 2))
	}
}
