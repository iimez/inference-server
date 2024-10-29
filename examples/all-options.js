import { startHTTPServer } from '../dist/http.js'

// Starts a http server for up to two instances of phi3 and serves them via openai API.
// startHTTPServer is only a thin wrapper around the ModelServer class that spawns a web server.
startHTTPServer({
	log: 'info', // 'debug', 'info', 'warn', 'error' - or pass a function as custom logger.
	// Limit how many instances may be handed out concurrently for processing.
	// If its exceeded, requests will be queued up and stall until a model becomes available.
	// Defaults to 1 = process one request at a time.
	concurrency: 2,
	// Where to cache to disk. Defaults to `~/.cache/node/inference-server`
	// cachePath: '/path/to/cache',
	models: {
		// Specify as many models as you want. Identifiers can use a-zA-Z0-9_:\-\.
		// Required are `task`, `engine`, `url` and/or `file`.
		'my-model': {
			task: 'text-completion', // 'text-completion', 'embedding', 'image-to-text', 'speech-to-text'
			engine: 'node-llama-cpp', // 'node-llama-cpp', 'transformers-js', 'gpt4all'
			// Model weights may be specified by file and/or url.
			url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
			// specify sha256 hash to verify the downloaded file.
			sha256: 'a98d3857b95b96c156d954780d28f39dcb35b642e72892ee08ddff70719e6220',
			// The preparation process downloads and verifies model files before instantiating the model.
			// Use this to control when that happens. Options are:
			// - 'on-demand' = prepare on first request. This is the default.
			// - 'blocking' = prepare immediately on startup
			// - 'async' = prepare in background but don't block startup. Requests to the model during the preparation process will resolve once its ready.
			// Note that if minInstances > 0 then this is effectively always "blocking" because the model preparation will happen immediately.
			prepare: 'on-demand',
			// What should be preloaded in context, for text completion / chat models.
			preload: {
				// Note that for preloading to be utilized, requests must
				// also have these leading messages before the user message.
				messages: [
					{
						role: 'system',
						content: 'You are a helpful assistant.',
					},
				],
				// toolDocumentation: true, // Tool docs may also be preloaded. See `tools` below.
			},
			// Options to control resource usage.
			contextSize: 2046, // Maximum context size. Will be determined automatically if not set.
			maxInstances: 2, // How many active sessions you wanna be able to cache at the same time.
			minInstances: 1, // To always keep at least one instance ready. Defaults to 0.
			// Idle instances will be disposed after this many seconds.
			ttl: 300, // Defaults to 5min. Set it to zero to immediately dispose of instances after use.
			// Set defaults for completions. These can be overridden per request.
			// If unset, default values depend on the engine.
			completionDefaults: {
				temperature: 1,
			},
			// Configure hardware / device to use.
			device: {
				// GPU will be used automatically if left unset.
				// Only one model can use the gpu at a time.
				// gpu: true, // Force gpu use for instance of this model. (This effectively limits maxInstance to 1.)
				// cpuThreads: 4, // Only gpt4all and node-llama-cpp
				// memLock: true, // Only node-llama-cpp.
			},
			// node-llama-cpp text-completion models may have GBNF grammars and tools configured.
			// You can define multiple grammars for a model. `json` grammar will alway be available.
			// Key is the grammar name (that later can be used as value for `grammar` in a request). Value is a string containing the GBNF grammar.
			grammars: {
				// For example:
				// 'custom-grammar': fs.readFileSync('custom-grammar.gbnf', 'utf8'), // Supply your own grammar
				// 'chess': await LlamaGrammar.getFor(llama, 'chess') // Or reuse a grammar shipped with (node-)llama-cpp
			},
			// Avilable tools may be defined on the model or during requests.
			// Note that for using `preload` with `toolDocumentation` they _must_ be defined here (on the model).
			tools: {
				includeParamsDocumentation: true, // Include parameter documentation in tool documentation.
				parallelism: 2, // How many tools may be executed in parallel. Defaults to 1.
				definitions: {
					getLocationWeather: {
						description: 'Get the weather in a location',
						parameters: {
							type: 'object',
							properties: {
								location: {
									type: 'string',
									description: 'The city and state, e.g. San Francisco, CA',
								},
								unit: {
									type: 'string',
									enum: ['celsius', 'fahrenheit'],
								},
							},
							required: ['location'],
						},
						// Handler is optional. If its set, the model will ingest the return value and respond with the final assistant message.
						// If unset the model will respond with a tool call message instead. In this case you need to push tool call results into the message array.
						handler: async (parameters) => {
							const { location, unit } = parameters
							// Call a weather API or something
							return `The temperature in ${location} is 23°C`
						},
					}
				}
			}
		},
	},
	// HTTP listen options. If you don't need a web server, use `startModelServer` or `new ModelServer()`.
	// Accepted arguments are identical, apart from `listen`.
	listen: {
		port: 3000,
	},
})
