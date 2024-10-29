import readline from 'node:readline'
import chalk from 'chalk'
import { ModelServer } from '#package/index.js'

// A command-line chat example using the ModelServer.

const modelServer = new ModelServer({
	// log: 'info',
	models: {
		'my-model': {
			task: 'text-completion',
			minInstances: 1,
			url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
			sha256: 'a98d3857b95b96c156d954780d28f39dcb35b642e72892ee08ddff70719e6220',
			engine: 'node-llama-cpp',
			// device: { gpu: false },
		},
	},
})

console.log('Initializing models...')

await modelServer.start()

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
})

const messages = []

while (true) {
	const input = await new Promise((resolve) => {
		rl.question(chalk.bold(chalk.dim('user > ')), (input) => {
			resolve(input)
		})
	})
	messages.push({
		role: 'user',
		content: input,
	})
	process.stdout.write(chalk.bold(chalk.dim('model > ')))
	const result = await modelServer.processChatCompletionTask(
		{
			model: 'my-model',
			messages,
		},
		{
			onChunk: (chunk) => {
				process.stdout.write(chunk.text)
			},
		},
	)
	messages.push(result.message)
	process.stdout.write(' ' + chalk.dim(`[${result.finishReason}]`) + '\n')
}
