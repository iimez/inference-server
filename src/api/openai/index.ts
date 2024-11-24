import type { InferenceServer } from '#package/server.js'
import { createChatCompletionHandler } from './handlers/chat.js'
import { createCompletionHandler } from './handlers/completions.js'
import { createModelsHandler } from './handlers/models.js'
import { createEmbeddingsHandler } from './handlers/embeddings.js'


// See OpenAI API specs at https://github.com/openai/openai-openapi/blob/master/openapi.yaml
export function createOpenAIRequestHandlers(inferenceServer: InferenceServer) {
	return {
		chatCompletions: createChatCompletionHandler(inferenceServer),
		completions: createCompletionHandler(inferenceServer),
		models: createModelsHandler(inferenceServer),
		embeddings: createEmbeddingsHandler(inferenceServer),
	}
}
