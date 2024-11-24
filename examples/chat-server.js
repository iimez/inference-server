import http from 'node:http'
import { InferenceServer } from '#package/index.js'

// A minimal chat server using the ModelServer.

const localModels = new InferenceServer({
	log: 'info',
	concurrency: 2,
	models: {
		'phi3-mini-4k': {
			task: 'text-completion',
			url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
			engine: 'gpt4all',
			maxInstances: 2,
		},
	},
})
await localModels.start()

const httpServer = http.createServer((req, res) => {
	if (req.url === '/chat' && req.method === 'POST') {
		let body = ''
		req.on('data', (chunk) => {
			body += chunk.toString()
		})
		req.on('end', async () => {
			const req = JSON.parse(body)
			const completion = await modelServer.processChatCompletionTask(req)
			res.writeHead(200, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify(completion, null, 2))
		})
	} else {
		res.writeHead(404, { 'Content-Type': 'text/plain' })
		res.end('Not found')
	}
})
httpServer.listen(3000).on('listening', () => {
	console.log('HTTP Server up')
})

/*
curl http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
      "model": "phi3-mini-4k",
      "messages": [
          {
              "role": "user",
              "content": "how to find my kernel version on linux=?"
          }
      ]
  }'
*/

/*
{
  "finishReason": "eogToken",
  "message": {
    "role": "assistant",
    "content": "To find your kernel version on Linux, you can use the following methods: [...]"
  },
  "promptTokens": 10,
  "completionTokens": 344,
  "totalTokens": 354
}
*/
