import type { ModelServer } from '#package/server.js'
import { createChatCompletionHandler } from './handlers/chat.js'
import { createCompletionHandler } from './handlers/completions.js'
import { createModelsHandler } from './handlers/models.js'
import { createEmbeddingsHandler } from './handlers/embeddings.js'


// See OpenAI API specs at https://github.com/openai/openai-openapi/blob/master/openapi.yaml
export function createOpenAIRequestHandlers(modelServer: ModelServer) {
	return {
		chatCompletions: createChatCompletionHandler(modelServer),
		completions: createCompletionHandler(modelServer),
		models: createModelsHandler(modelServer),
		embeddings: createEmbeddingsHandler(modelServer),
	}
}
