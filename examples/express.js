import http from 'node:http'
import express from 'express'
import OpenAI from 'openai'
import { InferenceServer } from '#package/server.js'
import { createExpressMiddleware } from '#package/http.js'

// Demonstration of integrating with existing express server.

// Create a server with a single model, limiting to 2 instances that can run concurrently.
// Models will be downloaded on-demand or during ModelServer.start() if minInstances > 0.
const localModels = new InferenceServer({
	concurrency: 2,
	models: {
		'my-model': {
			task: 'text-completion',
			url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
			sha256: 'a98d3857b95b96c156d954780d28f39dcb35b642e72892ee08ddff70719e6220',
			engine: 'node-llama-cpp',
			maxInstances: 2,
		},
	},
})

await localModels.start()

const app = express()
app.use(express.json(), createExpressMiddleware(localModels))
const server = http.createServer(app)
server.listen(3001)

console.log('Server up, sending chat completion request...')

const openai = new OpenAI({
	baseURL: 'http://localhost:3001/openai/v1/',
	apiKey: '123',
})

const completion = await openai.chat.completions.create({
	model: 'my-model',
	messages: [{ role: 'user', content: 'Lets count to three!' }],
	stop: ['Two'],
})

console.log(JSON.stringify(completion, null, 2))

/*
{
  "id": "my-model:pU2BHWUv-kHdAeVn8",
  "model": "my-model",
  "object": "chat.completion",
  "created": 1714431837,
  "system_fingerprint": "0159c68a067a360e4be3e285d3e309440c070734",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Sure, let's count together: 1 (one), 2 (two), and 3 (three). If you have any other questions or need further assistance, feel free to ask!"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 41,
    "total_tokens": 47
  }
}
*/
