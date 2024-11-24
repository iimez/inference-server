import os from 'node:os'
import path from 'node:path'
import chalk from 'chalk'
import { ModelPool } from '#package/index.js'
import { elapsedMillis } from '#package/lib/util.js'
import * as LlamaCppEngine from '#package/engines/node-llama-cpp/engine.js'

// Complete multiple prompts concurrently using ModelPool only.

async function onPrepareInstance(instance) {
	// can be used to set up the instance before it's used.
	// the model will not be loaded until this promise resolves.
	// console.log('Instance about to load:', instance)
	// throwing here will put the instance in an error state
}

const pool = new ModelPool(
	{
		// to see what's going on, set the log level to 'debug'
		// log: 'debug',
		// global processing concurrency limit, across all instances of all models
		concurrency: 2,
		models: {
			'my-model': {
				task: 'text-completion',
				// note that this path needs to be absolute and the file needs to be downloaded beforehand.
				location: path.resolve(
					os.homedir(),
					'.cache/inference-server/huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF-main/smollm-135m-instruct-add-basics-q8_0.gguf',
				),
				engine: 'node-llama-cpp',
				minInstances: 1, // setting this to something greater 0 will load the model on pool.init()
				maxInstances: 2, // allow the pool to spawn additional instances of this model
			},
		},
	},
	onPrepareInstance,
)

process.on('exit', () => {
	pool.dispose()
})

console.log('Initializing pool...')
await pool.init({
	'node-llama-cpp': LlamaCppEngine,
})

async function createCompletion(prompt) {
	const req = {
		model: 'my-model',
		prompt,
		temperature: 3,
		maxTokens: 200,
	}
	const completionModel = await pool.requestInstance(req)
	const completionBegin = process.hrtime.bigint()
	const task = completionModel.instance.processTextCompletionTask(req)
	const result = await task.result
	completionModel.release()
	const elapsed = Math.max(elapsedMillis(completionBegin), 1000)
	return {
		text: result.text,
		instance: completionModel.instance.id,
		device: completionModel.instance.gpu ? 'GPU' : 'CPU',
		speed: Math.round(result.completionTokens / (elapsed / 1000)),
	}
}

const printResult = (title) => (result) => {
	console.log(
		chalk.yellow(title) +
			' ' +
			chalk.bold(result.instance) +
			' ' +
			chalk.dim(`generated ${result.speed} tokens/s on ${result.device}`),
	)
	console.log(chalk.dim(prompt) + result.text)
}

const completionCount = 20
const prompt = 'Locality of '

const res = await createCompletion(prompt)
printResult('Initial completion')(res)

console.log(`Processing ${completionCount} completions...`)

for (let i = 1; i <= completionCount; i++) {
	createCompletion(prompt).then(printResult(`#${i}`))
}
