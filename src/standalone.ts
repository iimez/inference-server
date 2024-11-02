import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { ModelHTTPServer, ModelHTTPServerOptions } from '#package/http.js'
import { ChatWithVisionEngine } from '#package/experiments/ChatWithVision.js'
import { VoiceFunctionCallEngine } from '#package/experiments/VoiceFunctionCall.js'

import {
	Florence2ForConditionalGeneration,
	WhisperForConditionalGeneration,
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
	AutoModelForCausalLM,
} from '@huggingface/transformers'

// Currently only used for debugging. Do not use.
const serverOptions: ModelHTTPServerOptions = {
	listen: {
		port: 3000,
	},
	log: 'debug',
	concurrency: 2,
	engines: {
		// 'chat-with-vision': new ChatWithVisionEngine({
		// 	imageToTextModel: 'florence2',
		// 	chatModel: 'llama3-8b',
		// }),
		// 'voice-function-calling': new VoiceFunctionCallEngine({
		// 	speechToTextModel: 'whisper-base',
		// 	chatModel: 'functionary',
		// }),
	},
	models: {
		// 'sciphi-triplex': {
		// 	url: 'https://huggingface.co/SciPhi/Triplex/blob/main/quantized_model-Q4_K_M.gguf',
		// 	sha256: '6f8f6f1fca005640a1282dd0bd12512dedf22957d0c2135ba5e71583d33754fc',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// },
		// 'lite-mistral': {
		// 	url: 'https://huggingface.co/bartowski/Lite-Mistral-150M-v2-Instruct-GGUF/resolve/main/Lite-Mistral-150M-v2-Instruct-Q8_0.gguf',
		// 	sha256: 'b369c9b1ac20b66b2f94117d5cdc71d029a47a33948cefef9fe104615dcddfbd',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// },
		// 'gemma-9b': {
		// 	url: 'https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/blob/main/gemma-2-9b-it-Q4_K_M.gguf',
		// 	sha256:
		// 		'13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// },
		// 'llama3.1-8b': {
		// 	url: 'https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
		// 	sha256:
		// 		'2a4ca64e02e7126436cfdb066dd7311f2486eb487191910d3d000fde13826a4d',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// },
		// 'dolphin-nemo-12b': {
		// 	url: 'https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf/blob/main/dolphin-2.9.3-mistral-nemo-Q4_K_M.gguf',
		// 	sha256: '09f9114e06d88b791e322586cf28a844d2d0a3876d04d6deffe2dfb26616dd83',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// },
		// 'phi3-mini-4k': {
		// 	task: 'text-completion',
		// 	url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		// 	engine: 'gpt4all',
		// 	maxInstances: 2,
		// 	prepare: 'async',
		// },
		// 'mxbai-embed-large-v1': {
		// 	url: 'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1',
		// 	engine: 'transformers-js',
		// 	task: 'embedding',
		// 	prepare: 'blocking',
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'jina-clip-v1': {
		// 	url: 'https://huggingface.co/jinaai/jina-clip-v1',
		// 	engine: 'transformers-js',
		// 	task: 'embedding',
		// 	textModel: {
		// 		modelClass: CLIPTextModelWithProjection,
		// 	},
		// 	visionModel: {
		// 		processor: {
		// 			url: 'https://huggingface.co/Xenova/clip-vit-base-patch32',
		// 			// url: 'https://huggingface.co/Xenova/vit-base-patch16-224-in21k',
		// 		},
		// 		modelClass: CLIPVisionModelWithProjection,
		// 	},
		// 	prepare: 'blocking',
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'florence2-large': {
		// 	url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
		// 	engine: 'transformers-js',
		// 	task: 'image-to-text',
		// 	prepare: 'blocking',
		// 	visionModel: {
		// 		modelClass: Florence2ForConditionalGeneration,
		// 		dtype: {
		// 			embed_tokens: 'fp16',
		// 			vision_encoder: 'fp32',
		// 			encoder_model: 'fp16',
		// 			decoder_model_merged: 'q4',
		// 		},
		// 	},
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'whisper-base': {
		// 	url: 'https://huggingface.co/onnx-community/whisper-base',
		// 	engine: 'transformers-js',
		// 	task: 'speech-to-text',
		// 	prepare: 'async',
		// 	minInstances: 1,
		// 	speechModel: {
		// 		modelClass: WhisperForConditionalGeneration,
		// 		dtype: {
		// 			encoder_model: 'fp32', // 'fp16' works too
		// 			decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
		// 		},
		// 	},
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'mistral-nemo-12b': {
		// 	'url': 'https://huggingface.co/mradermacher/Mistral-Nemo-Instruct-2407-GGUF/blob/main/Mistral-Nemo-Instruct-2407.Q4_K_M.gguf',
		// 	'sha256': '1ac4b6cdf0eeb1e2145f0097c6fd0a75df541e143f226a8ff25c8ae0e8dfff6f',
		// 	'engine': 'node-llama-cpp',
		// 	'task': 'text-completion',
		// 	'prepare': 'async',
		// },
		// 'phi-3.5-mini': {
		// 	url: 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/blob/main/Phi-3.5-mini-instruct-Q4_K_M.gguf',
		// 	sha256:
		// 		'e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// 	minInstances: 1,
		// 	device: {
		// 		gpu: 'vulkan',
		// 	},
		// },
		// 'falcon-mamba-7b': {
		// 	url: 'https://huggingface.co/mradermacher/falcon-mamba-7b-instruct-GGUF/blob/main/falcon-mamba-7b-instruct.Q4_K_M.gguf',
		// 	sha256: 'f3357486034d89dd91fcefdb91bb1dfadfe0fd2969349a8a404e59d2bd3ad1b8',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// },
		// 'florence2-large': {
		// 	url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
		// 	engine: 'transformers-js',
		// 	task: 'image-to-text',
		// 	minInstances: 1,
		// 	visionModel: {
		// 		modelClass: Florence2ForConditionalGeneration,
		// 		dtype: {
		// 			embed_tokens: 'fp16',
		// 			vision_encoder: 'fp32',
		// 			encoder_model: 'fp16',
		// 			decoder_model_merged: 'q4',
		// 		},
		// 	},
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'mxbai-embed-large-v1': {
		// 	url: 'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1',
		// 	engine: 'transformers-js',
		// 	task: 'embedding',
		// 	prepare: 'blocking',
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'functionary-3.2-small': {
		// 	url: 'https://huggingface.co/meetkai/functionary-small-v3.2-GGUF/blob/main/functionary-small-v3.2.Q4_0.gguf',
		// 	sha256: 'c0afdbbffa498a8490dea3401e34034ac0f2c6e337646513a7dbc04fcef1c3a4',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'blocking',
		// },
		// 'flux-schnell': {
		// 	url: 'https://huggingface.co/leejet/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-q4_0.gguf',
		// 	task: 'text-to-image',
		// 	sha256: '4f30741d2bfc786c92934ce925fcb0a43df3441e76504b797c3d5d5f0878fa6f',
		// 	engine: 'stable-diffusion-cpp',
		// 	prepare: 'blocking',
		// 	diffusionModel: true,
		// 	samplingMethod: 'euler_a',
		// 	vae: {
		// 		url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/ae.safetensors',
		// 	},
		// 	clipL: {
		// 		url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/clip_l.safetensors',
		// 	},
		// 	t5xxl: {
		// 		// url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl_fp16.safetensors',
		// 		url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q8_0.gguf',
		// 	},
		// },
		// 'sd-3.5-turbo': {
		// 	url: 'https://huggingface.co/stduhpf/SD3.5-Large-Turbo-GGUF-mixed-sdcpp/blob/main/legacy/sd3.5_large_turbo-q4_0.gguf',
		// 	sha256: '52495d9c4356065a1378a93c9556a9eb465e10014ba9ce364512674267405bb2',
		// 	engine: 'stable-diffusion-cpp',
		// 	task: 'text-to-image',
		// 	prepare: 'blocking',
		// 	samplingMethod: 'euler',
		// 	clipG: {
		// 		url: 'https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors',
		// 		sha256: 'ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4',
		// 	},
		// 	clipL: {
		// 		url: 'https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors',
		// 		sha256: '660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd',
		// 	},
		// 	t5xxl: {
		// 		url: 'https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors',
		// 		sha256: '7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09',
		// 	},
		// },
		// 'trocr-printed': {
		// 	url: 'https://huggingface.co/Xenova/trocr-small-printed',
		// 	engine: 'transformers-js',
		// 	task: 'image-to-text',
		// 	prepare: 'blocking',
		// 	minInstances: 1,
		// 	// textModel: {
		// 	//   modelClass: TrOCRPreTrainedModel,
		// 	// 	processorClass: DeiTFeatureExtractor,
		// 	// },
		// 	device: {
		// 		gpu: false,
		// 	},
		// },
		// 'sdxl-turbo': {
		// 	url: 'https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors',
		// 	sha256:
		// 		'e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26',
		// 	engine: 'stable-diffusion-cpp',
		// 	task: 'image-to-image',
		// 	prepare: 'blocking',
		// 	samplingMethod: 'euler',
		// 	vae: {
		// 		url: 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl.vae.safetensors',
		// 		sha256:
		// 			'235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1',
		// 	},
		// },
		// 'flux-light': {
		// 	url: 'https://huggingface.co/city96/flux.1-lite-8B-alpha-gguf/blob/main/flux.1-lite-8B-alpha-Q8_0.gguf',
		// 	sha256:
		// 		'efc598d62123f2fdfd682948f533fee081f7fb1295b14d002ac1e66cae5f01a5',
		// 	engine: 'stable-diffusion-cpp',
		// 	task: 'image-to-image',
		// 	prepare: 'blocking',
		// },
		// 'sd-3-medium': {
		// 	url: 'https://huggingface.co/second-state/stable-diffusion-3-medium-GGUF/blob/main/sd3-medium-Q8_0.gguf',
		// 	sha256: '7e34dfeb71f8cdbc8338677b63a444897cf4c5692ab4c1d98f04cbba6751885a',
		// 	engine: 'stable-diffusion-cpp',
		// 	task: 'text-to-image',
		// 	prepare: 'async',
		// },
		// 'sd-1.5': {
		// 	url: 'https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/blob/main/stable-diffusion-v1-5-pruned-emaonly-f32.gguf',
		// 	sha256: '52c7ca39d8d48d6f44fa4ff2c44569f3c924d92311108cb38492958350d48ff8',
		// 	engine: 'stable-diffusion-cpp',
		// 	task: 'text-to-image',
		// 	prepare: 'async',
		// },
		// 'llama-3.2-3b': {
		// 	url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf',
		// 	sha256:
		// 		'c542b14ec07b8b3cb8d777e1a68ee5aabb964167719466d4c685c29fcfd04900',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'blocking',
		// },
		speecht5: {
			url: 'https://huggingface.co/Xenova/speecht5_tts',
			engine: 'transformers-js',
			task: 'text-to-speech',
			prepare: 'async',
			minInstances: 1,
			speechModel: {
				speakerEmbeddings: {
					voice: {
						url: 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin',
					},
				},
			},
		},
	},
}

async function main() {
	const server = new ModelHTTPServer(serverOptions)
	await server.start()
	const { address, port } = server.httpServer.address() as AddressInfo
	const hostname = address === '' || address === '::' ? 'localhost' : address
	const url = formatURL({
		protocol: 'http',
		hostname,
		port,
		pathname: '/',
	})
	console.log(`Server listening at ${url}`)
}

main().catch((err: Error) => {
	console.error(err)
	process.exit(1)
})

process.on('unhandledRejection', (err) => {
	console.error('Unhandled rejection:', err)
})
