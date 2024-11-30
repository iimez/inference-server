## inference-server

Libraries and server to build AI applications doing local inference in node. Use it within your application, or as a microservice. Adapters to [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all), [transformers.js](https://github.com/xenova/transformers.js/) and [@lmagder/node-stable-diffusion-cpp](https://github.com/lmagder/node-stable-diffusion.cpp).

The project includes a model resource pool, an inference queue and a HTTP API server. Model file management is abstracted away as much as possible - configure a URL and go. This package is useful for quick model evaluations and experiments (in JavaScript), small-scale chatbots, resource efficient assistants on edge devices, or any applications where private & offline are interesting criteria. For other - not node-based solutions - check out the [related solutions](#related-solutions) section.

⚠️ This package is currently in beta. APIs may change. Things may break. [Help is welcome](#contributing).

### Features

- Configure as many models as you want, they will be downloaded and cached to disk. You may provide them as abs file paths if you already have models downloaded.
- Adjust the pool `concurrency`, and the models `maxInstances`, `ttl` and `contextSize` to fit your usecase. Combine multiple pools for more complex setups.
- Can be tuned to either use no resources when idle or to always keep a model ready with context preloaded.
- A chat session cache that will effectively reuse context across multiple turns or stateless requests.
- OpenAI spec API endpoints. See [HTTP API docs](./docs/http-api.md) for details. A "native" HTTP API is not yet implemented.
- BYO web server or use the provided express server and middleware. Incoming requests are queued - stall, if needed - and processed as soon as resources are available.
- Have as many ModelServers running as you want, they can share the same cache directory. (Multiple processes can as well)
- Use the [ModelPool](./examples/pool.js) class directly for a lowerlevel transaction-like API to aquire/release model instances.
- Use [custom engines](./docs/engines.md#custom-engines) to combine multiple models (or do RAG) behind the scenes.

### Usage

Example with minimal configuration:

```ts basic.ts
import { InferenceServer } from 'inference-server'

const localModels = new InferenceServer({
  log: 'info', // default is 'warn'
  models: {
    'my-model': { // Identifiers can use a-zA-Z0-9_:\-\.
      // Required are `task`, `engine`, `url` and/or `file`.
      task: 'text-completion', // text-completion models can be used for chat and text generation tasks
      engine: 'node-llama-cpp', // each engine comes with a peer dep. `npm install node-llama-cpp@3`
      url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/blob/main/smollm2-1.7b-instruct-q4_k_m.gguf',
    },
  },
})
await localModels.start()
const result = await localModels.processChatCompletionTask({
  model: 'my-model',
  messages: [
    {
      role: 'user',
      content: 'Why are bananas rather blue than bread at night?',
    },
  ],
})
console.debug(result)
localModels.stop()
```

Or, to launch an OpenAI-compatible HTTP server with two concurrent instances of the same model:


```ts http-api.ts
import { InferenceServer, createExpressServer } from 'inference-server'
import OpenAI from 'openai'

const localModels = new InferenceServer({
  log: 'info', // default is 'warn'
  concurrency: 2, // two inference processes may run at the same time
  models: {
    'smollm': {
      task: 'text-completion',
      engine: 'node-llama-cpp',
      url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/blob/main/smollm2-1.7b-instruct-q4_k_m.gguf',
      maxInstances: 2, // two instances of this model may be loaded into memory
      device: {
        cpuThreads: 4, // limit cpu threads so we dont occupy all cores
      }
    },
  },
})
localModels.start()
const httpServer = createExpressServer(localModels)
httpServer.listen(3000)

const client = new OpenAI({
  baseURL: 'http://localhost:3000/openai/v1/',
  apiKey: 'yes',
})
const completion = await client.beta.chat.completions.stream({
  stream_options: { include_usage: true },
  model: 'smollm',
  messages: [
    { role: 'user', content: 'lets count to 10, but only whisper every second number' },
  ],
})
for await (const chunk of completion) {
  if (chunk.choices[0]?.delta?.content) {
    process.stdout.write(chunk.choices[0].delta.content)
  }
}
localModels.stop()
httpServer.close()
```

In addition to the OpenAI endpoints, the server created by `createExpressServer` will also display some internal state at `/`. The endpoint implementations are also exported from the package and can be used with any http server.

More usage examples:
- Using all available options / model options API doc [./examples/all-options](./examples/all-options.js).
- Custom engines [./tests/engines/experiments.test.ts](./tests/engines/experiments.test.ts).
- A chat cli [./examples/chat-cli](./examples/chat-cli.js).
- `concurrency` behavior [./examples/concurrency](./examples/concurrency.js).
- Using the ModelPool directly [./examples/pool](./examples/pool.js).
- Using the express middleware [./examples/express](./examples/express.js).

Currently supported inference engines are:

| Engine | Peer Dependency |
| --- | --- |
| node-llama-cpp | `node-llama-cpp >= 3.0.0` |
| gpt4all | `gpt4all >= 4.0.0` |
| transformers-js | `@huggingface/transformers >= 3.0.0` |
| stable-diffusion-cpp | `@lmagder/node-stable-diffusion-cpp >= 0.1.6` |

See [engine docs](./docs/engines.md) for more information on each.

#### Limitations and Known Issues

##### Only one model can run on GPU at a time
Llama.cpp bindings currently do not support running multiple models on gpu at the same time. This can/will likely be improved in the future. See [GPU docs](./docs/gpu.md) for more information on how to work around that.

##### System Messages
System role messages are supported only as the first message in a chat completion session. All other system messages will be ignored. This is only for simplicity reasons and might change in the future.

##### Chat Context Cache / Reusing the correct instance on stateless requests
Note that the current context cache implementation only works if (apart from the final user message) the _same messages_ are resent in the _same order_. This is because the messages will be hashed to be compared during follow up turns, to match requests to the correct session. If no hash matches everything will still work, but slower. Because a fresh context will be used and the whole input conversation will be reingested, instead of just the new user message.

##### Function Calling
Only available when using node-llama-cpp and a model that supports function calling. `tool_choice` can currently not be controlled and will always be `auto`. GBNF grammars cannot be used together with function calling.

##### Huge node_modules when installing all engines
CUDA binaries are distributed with each engine seperately, which leads to an extra 0.5-1GB of disk use. Unfortunately there is nothing I can do about that.

##### Can't use larger models with transformers.js
Models using the ONNX external data format are currently ([not supported in node.js+transformers.js](https://github.com/huggingface/transformers.js/blob/e129c47c65a049173f35e6263fd8d9f660dfc1a7/src/models.js#L240-L242)).

#### TODO / Roadmap

Not in any particular order:

- [x] transformer.js text + image embeddings
- [x] transformer.js multimodal image/text embeddings (see [jina-clip-v1](https://github.com/xenova/transformers.js/issues/793) and [nomic-embed-vision](https://github.com/xenova/transformers.js/issues/848) issues.)
- [x] Allow "prefilling" (partial) assistant responses like outlined [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#how-to-prefill-claudes-response)
- [x] non-chat text completions: Allow reuse of context
- [x] non-chat text completions: Support preloading of prefixes
- [x] Add stable-diffusion engine
- [x] Implement more transformer.js tasks
- [x] [CLI](https://github.com/iimez/inference-server/discussions/7)
- [x] Rename ModelServer -> InferenceServer and refactor task types
- [x] Support for transformer.js text-completion and chat-completion tasks
- [ ] Add some light jsdoc for server/pool/store methods
- [ ] Should support adding a json schema grammar on completion requests (not just a grammar name)
- [ ] utilize node-llama-cpp's support to reuse LlamaModel weights with multiple contexts (across instances)
- [ ] Add `n` parameter support to node-llama-cpp+transformers chat completions
- [ ] Infill completion support https://github.com/withcatai/node-llama-cpp/blob/beta/src/evaluator/LlamaCompletion.ts#L322-L336
- [ ] Support HF auth and support HF-like model repositories on other domains
- [ ] Rework GPU+device usage / lock (Support multiple models on gpu in cases where its possible)
- [ ] Add engine interfaces for resource use (and estimates, see https://github.com/ggerganov/llama.cpp/issues/4315 and https://github.com/withcatai/node-llama-cpp/blob/beta/src/gguf/insights/utils/resolveContextContextSizeOption.ts)
- [ ] Allow configuring a pools max memory usage
- [ ] Support logits with transformers and node-llama-cpp
- [ ] Test deno/bun support
- [ ] Add image generation endpoint in oai api
- [ ] Add transcript endpoint in oai api
- [ ] Add createSpeech endpoint in oai api
- [ ] Read more model metadata and expose it in API+CLI
- [ ] Replace express with tinyhttp
- [ ] Find a way to type available custom engines (and their options?)

### Contributing

If you are using this package - let me know where [you would like this to go](https://github.com/iimez/inference-server/discussions). Code also welcome. You find things im planning to do (eventually) above, and the wishlist below.

#### Possible Future Goals

- Create a separate HTTP API thats independent of the OpenAI spec and stateful. See [discussion](https://github.com/iimez/inference-server/discussions/8).
- Add a clientside library (React hooks?) for use of above API.
- Provide a Docker image. And maybe a Prometheus endpoint.

#### Currently not the Goals

- A facade to LLM cloud hoster HTTP API's. The strengths here are local/private/offline use.
- Worry too much about authentication or rate limiting or misuse. Host this with caution, it's likely DDoS-able.
- Some kind of distributed or multi-node setup. That should probably be something designed for this purpose from the ground up.
- Other common related tooling like vector stores, Chat GUIs, etc. Scope would probably get out of hand.

### Related Solutions

If you look at this package, you might also want to take a look at these other solutions:

- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Uses llama.cpp and provides a HTTP API. Also has experimental OpenAI API compatibility.
- [llama.cpp Server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server) - The official llama.cpp HTTP API.
- [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - A more production ready solution for hosting large language models.
- [LM Studio](https://lmstudio.ai/docs/local-server) - Also has a local server.
- [LocalAI](https://github.com/mudler/LocalAI) - Similar project in go.
- [Petals](https://github.com/bigscience-workshop/petals) - Local (and distributed!) inference in python.
- [cortex.cpp](https://github.com/janhq/cortex.cpp/)
