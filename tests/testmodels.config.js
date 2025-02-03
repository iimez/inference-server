// import { CLIPTextModelWithProjection, CLIPVisionModelWithProjection } from '@huggingface/transformers'

/** @type {import('#package/server.js').InferenceServerOptions} */
export const config = {
	models: {
		'nomic-text-embed': {
			url: 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/blob/main/nomic-embed-text-v1.5.Q8_0.gguf',
			sha256: '3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7',
			engine: 'node-llama-cpp',
			task: 'embedding',
		},
		'llama-3.2-3b': {
			url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
			sha256: '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff',
			engine: 'node-llama-cpp',
			task: 'text-completion',
		},
		'gpt4all-nomic-text-embed': {
			url: 'https://gpt4all.io/models/gguf/nomic-embed-text-v1.f16.gguf',
			engine: 'gpt4all',
			task: 'embedding',
		},
		'gpt4all-phi3': {
			url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
			engine: 'gpt4all',
			task: 'text-completion',
		},
		'mxbai-embed-large-v1': {
			url: 'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1',
			engine: 'transformers-js',
			task: 'embedding',
		},
		'jina-clip-v1': {
			url: 'https://huggingface.co/jinaai/jina-clip-v1',
			engine: 'transformers-js',
			task: 'embedding',
			modelClass: 'CLIPTextModelWithProjection',
			visionModel: {
				processor: {
					url: 'https://huggingface.co/Xenova/clip-vit-base-patch32',
				},
				modelClass: 'CLIPVisionModelWithProjection',
			},
		},
		'trocr-printed': {
			url: 'https://huggingface.co/Xenova/trocr-small-printed',
			engine: 'transformers-js',
			task: 'image-to-text',
		},
		'florence2-large': {
			url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
			engine: 'transformers-js',
			task: 'image-to-text',
			dtype: {
				embed_tokens: 'fp16',
				vision_encoder: 'fp32',
				encoder_model: 'fp16',
				decoder_model_merged: 'q4',
			},
		},
		speecht5_tts: {
			url: 'https://huggingface.co/Xenova/speecht5_tts',
			engine: 'transformers-js',
			task: 'text-to-speech',
			vocoder: {
				url: 'https://huggingface.co/Xenova/speecht5_hifigan',
			},
			speakerEmbeddings: {
				defaultVoice: {
					url: 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin',
				},
			},
		},
		'whisper-base': {
			url: 'https://huggingface.co/onnx-community/whisper-base',
			engine: 'transformers-js',
			task: 'speech-to-text',
			dtype: {
				encoder_model: 'fp16',
				decoder_model_merged: 'q4',
			},
		},
		'table-transformer-detection': {
			url: 'https://huggingface.co/Xenova/table-transformer-detection',
			engine: 'transformers-js',
			task: 'object-detection',
		},
		'table-transformer-structure-recognition': {
			url: 'https://huggingface.co/Xenova/table-transformer-structure-recognition',
			engine: 'transformers-js',
			task: 'object-detection',
		},
		'owlv2-base': {
			url: 'https://huggingface.co/Xenova/owlv2-base-patch16-finetuned',
			engine: 'transformers-js',
			task: 'object-detection',
			dtype: 'fp16',
		},
		'smollm2': {
			url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct',
			// url: 'https://huggingface.co/onnx-community/Janus-1.3B-ONNX',
			// modelClass: 'MultiModalityCausalLM',
			engine: 'transformers-js',
			task: 'chat-completion',
			dtype: 'int8',
		}
	},
}
