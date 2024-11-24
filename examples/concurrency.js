import { createExpressServer } from '#package/http.js'
import OpenAI from 'openai'
import readline from 'node:readline'

// Printing two parallel completion processes to the console.
const localModels = new InferenceServer({
	log: 'info',
	concurrency: 2, // two clients may process chat completions at the same time.
	models: {
		'my-model': {
			task: 'text-completion',
			engine: 'node-llama-cpp',
			url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
			sha256: 'a98d3857b95b96c156d954780d28f39dcb35b642e72892ee08ddff70719e6220',
			minInstances: 1, // one instance / session will always be ready
			maxInstances: 2, // up to two may be spawned
			device: { gpu: false, cpuThreads: 4 }, // configure so they're roughly the same speed
		},
	},
})
await localModels.start()
const httpServer = createExpressServer(localModels)
httpServer.listen(3000)

const openai = new OpenAI({
	baseURL: 'http://localhost:3000/openai/v1/',
	apiKey: 'yes',
})
let sentence1 = 'Sometimes I feel like'
let sentence2 = 'The locality of'
const clearLine = () => {
	readline.cursorTo(process.stdout, 0)
	readline.clearLine(process.stdout, 0)
}
const updateOutputs = () => {
	const truncateLine = (line) => {
		return line.length > process.stdout.columns
			? '...' + line.slice(line.length - process.stdout.columns + 3)
			: line
	}
	readline.moveCursor(process.stdout, 0, -2)
	clearLine()
	process.stdout.write(truncateLine(sentence1) + '\n')
	clearLine()
	process.stdout.write(truncateLine(sentence2) + '\n')
}
const completeSentence = async (prompt, onTokens) => {
	const completion = await openai.completions.create({
		stream_options: { include_usage: true },
		model: 'my-model',
		stream: true,
		temperature: 1,
		stop: ['.'],
		prompt,
	})
	for await (const chunk of completion) {
		if (chunk.choices[0].text) {
			onTokens(chunk.choices[0].text.replaceAll('\n', '\\n'))
		}
	}
	onTokens('.')
}
setInterval(updateOutputs, 200)
console.log(sentence1)
console.log(sentence2)
while (true) {
	await Promise.all([
		completeSentence(sentence1, (text) => (sentence1 += text)),
		completeSentence(sentence2, (text) => (sentence2 += text)),
	])
}
httpServer.close()
clearInterval(updateOutputs)