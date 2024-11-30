import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { InferenceServer } from '#package/server.js'
import { cosineSimilarity } from '#package/lib/math.js'
import { loadImageFromFile, saveImageToFile } from '#package/lib/loadImage.js'
import { saveAudioToFile } from '#package/lib/loadAudio.js'
import { drawBoundingBoxes, createPaddedCrop } from '../util/images.js'
import {
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
	Florence2ForConditionalGeneration,
	SpeechT5ForTextToSpeech,
	SpeechT5Model,
	MobileLLMForCausalLM,
	WhisperForConditionalGeneration,
} from '@huggingface/transformers'
import {
	runStopTriggerTest,
	runTokenBiasTest,
	runSystemMessageTest,
	runContextLeakTest,
	runContextReuseTest,
	// runFileIngestionTest,
	// runGenerationContextShiftTest,
	// runIngestionContextShiftTest,
	// runFunctionCallTest,
	// runSequentialFunctionCallTest,
	// runParallelFunctionCallTest,
	// runBuiltInGrammarTest,
	// runRawGBNFGrammarTest,
	// runJsonSchemaGrammarTest,
	// runTimeoutTest,
	// runCancellationTest,
} from './lib/index.js'

suite('embeddings', () => {
	const inferenceServer = new InferenceServer({
		log: 'debug',
		models: {
			'mxbai-embed-large-v1': {
				url: 'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1',
				engine: 'transformers-js',
				task: 'embedding',
				device: {
					gpu: false,
				},
			},
			'jina-clip-v1': {
				url: 'https://huggingface.co/jinaai/jina-clip-v1',
				engine: 'transformers-js',
				task: 'embedding',
				modelClass: CLIPTextModelWithProjection,
				visionModel: {
					processor: {
						url: 'https://huggingface.co/Xenova/clip-vit-base-patch32',
						// url: 'https://huggingface.co/Xenova/vit-base-patch16-224-in21k',
					},
					modelClass: CLIPVisionModelWithProjection,
				},
				device: {
					gpu: false,
				},
			},
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})

	test('multimodal embedding', async () => {
		const [blueCatImage, redCatImage] = await Promise.all([
			loadImageFromFile('tests/fixtures/blue-cat.jpg'),
			loadImageFromFile('tests/fixtures/red-cat.jpg'),
		])
		const res = await inferenceServer.processEmbeddingTask({
			model: 'jina-clip-v1',
			input: [
				{
					type: 'image',
					content: blueCatImage,
				},
				{
					type: 'image',
					content: redCatImage,
				},
				'A blue cat',
				'A red cat',
			],
		})

		const blueCatImageEmbedding = Array.from(res.embeddings[0])
		const redCatImageEmbedding = Array.from(res.embeddings[1])
		const blueCatTextEmbedding = Array.from(res.embeddings[2])
		const redCatTextEmbedding = Array.from(res.embeddings[3])
		const textSimilarity = cosineSimilarity(blueCatTextEmbedding, redCatTextEmbedding)
		expect(textSimilarity.toFixed(2)).toBe('0.56')
		const textImageSimilarity = cosineSimilarity(blueCatTextEmbedding, blueCatImageEmbedding)
		expect(textImageSimilarity.toFixed(2)).toBe('0.29')
		const textImageSimilarity2 = cosineSimilarity(redCatTextEmbedding, blueCatImageEmbedding)
		expect(textImageSimilarity2.toFixed(2)).toBe('0.12')
		const textImageSimilarity3 = cosineSimilarity(blueCatTextEmbedding, redCatImageEmbedding)
		expect(textImageSimilarity3.toFixed(2)).toBe('0.05')
		const textImageSimilarity4 = cosineSimilarity(redCatTextEmbedding, redCatImageEmbedding)
		expect(textImageSimilarity4.toFixed(2)).toBe('0.29')
	})

	test('text embedding', async () => {
		const res = await inferenceServer.processEmbeddingTask({
			model: 'mxbai-embed-large-v1',
			input: [
				'Represent this sentence for searching relevant passages: A man is eating a piece of bread',
				'A man is eating food.',
				'A man is eating pasta.',
				'The girl is carrying a baby.',
				'A man is riding a horse.',
			],
			// dimensions: 1024,
			pooling: 'cls',
		})
		const searchEmbedding = res.embeddings[0]
		const sentenceEmbeddings = res.embeddings.slice(1)
		const similarities = sentenceEmbeddings.map((x) => cosineSimilarity(Array.from(searchEmbedding), Array.from(x)))
		expect(similarities[0].toFixed(2)).toBe('0.79')
	})
})

suite('image recognition', () => {
	const inferenceServer = new InferenceServer({
		log: 'debug',
		models: {
			'trocr-printed': {
				url: 'https://huggingface.co/Xenova/trocr-small-printed',
				engine: 'transformers-js',
				task: 'image-to-text',
				device: {
					gpu: false,
				},
			},
			'florence2-large': {
				url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
				engine: 'transformers-js',
				task: 'image-to-text',
				modelClass: Florence2ForConditionalGeneration,
				dtype: {
					embed_tokens: 'fp16',
					vision_encoder: 'fp32',
					encoder_model: 'fp16',
					decoder_model_merged: 'q4',
				},
				device: {
					gpu: false,
				},
			},
			janus: {
				url: 'https://huggingface.co/onnx-community/Janus-1.3B-ONNX',
				modelClass: 'MultiModalityCausalLM',
				engine: 'transformers-js',
				task: 'chat-completion',
				prepare: 'blocking',
				dtype: {
					prepare_inputs_embeds: "q4",
					language_model: "q4f16",
					lm_head: "fp16",
					gen_head: "fp16",
					gen_img_embeds: "fp16",
					image_decode: "fp32",
				}
			},
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('ocr single line (trocr)', async () => {
		const ocrImage = await loadImageFromFile('tests/fixtures/ocr-line.png')
		const res = await inferenceServer.processImageToTextTask({
			model: 'trocr-printed',
			image: ocrImage,
		})
		expect(res.text.match(/OVER THE \$43,456.78 <LAZY> #90 DOG/)).toBeTruthy()
	})

	test('ocr multiline (florence2)', async () => {
		const ocrImage = await loadImageFromFile('tests/fixtures/ocr-multiline.png')
		const res = await inferenceServer.processImageToTextTask({
			model: 'florence2-large',
			image: ocrImage,
			// see doc here for prompts: https://huggingface.co/microsoft/Florence-2-base#tasks
			prompt: 'What is the text in the image?',
		})
		expect(res.text.startsWith('The (quick) [brown] {fox} jumps!')).toBe(true)
	})

	test('cat recognition (janus)', async () => {
		// see examples here: https://github.com/huggingface/transformers.js/releases/tag/3.1.0
		// const ocrImage = await loadImageFromFile('tests/fixtures/quadratic_formula.png')
		const ocrImage = await loadImageFromFile('tests/fixtures/red-cat.jpg')
		const res = await inferenceServer.processChatCompletionTask({
			model: 'janus',
			maxTokens: 64,
			messages: [
				{
					role: 'system',
					content: 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.',
				},
				{
					role: 'user',
					content: [
						{
							type: 'image',
							image: ocrImage,
						},
						{
							type: 'text',
							text: 'Whats in the image?',
						},
					],
				},
			],
		})
		// console.debug(res)
		expect(res.message.content).toMatch(/red/)
		expect(res.message.content).toMatch(/cat/)
	})
})

suite('object detection', () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
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
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('cat detection (owlv2)', async () => {
		const image = await loadImageFromFile('tests/fixtures/blue-cat.jpg')
		const res = await inferenceServer.processObjectDetectionTask({
			model: 'owlv2-base',
			image,
			labels: ['cat', 'smurf'],
		})
		expect(res.objects).toBeTruthy()
		expect(res.objects[0].label).toBe('cat')
		// const debugImage = await drawBoundingBoxes(image, res.objects)
		// await saveImageToFile(debugImage, 'tests/fixtures/blue-cat-detected.png')
	})

	test('table recognition (table-transformer)', async () => {
		const image = await loadImageFromFile('tests/fixtures/table.png')
		const tableRes = await inferenceServer.processObjectDetectionTask({
			model: 'table-transformer-detection',
			image,
		})
		const tableObject = tableRes.objects[0]
		expect(tableObject).toBeTruthy()
		// padding because https://github.com/microsoft/table-transformer/issues/21
		const paddedCrop = await createPaddedCrop(image, tableObject.box, 40)
		// await saveImageToFile(paddedCrop, 'tests/fixtures/table-detected.png')
		const tableStructureRes = await inferenceServer.processObjectDetectionTask({
			model: 'table-transformer-structure-recognition',
			image: paddedCrop,
		})
		expect(tableStructureRes.objects).toBeTruthy()
		const tableRows = tableStructureRes.objects.filter((x) => x.label === 'table row')
		expect(tableRows.length).toEqual(8)
		// const imageWithBoundingBoxes = await drawBoundingBoxes(paddedCrop, tableRows)
		// await saveImageToFile(imageWithBoundingBoxes, 'tests/fixtures/table-detected-rows.png')
	})
})

suite('speech syntesis and transcription', () => {
	const inferenceServer = new InferenceServer({
		// log: 'debug',
		models: {
			speecht5: {
				url: 'https://huggingface.co/Xenova/speecht5_tts',
				engine: 'transformers-js',
				task: 'text-to-speech',
				modelClass: SpeechT5ForTextToSpeech,
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
				prepare: 'blocking',
				modelClass: WhisperForConditionalGeneration,
				dtype: {
					encoder_model: 'fp16',
					decoder_model_merged: 'q4',
				},
				device: {
					gpu: false,
				},
			},
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})
	test('text to speech to text', async () => {
		const speechRes = await inferenceServer.processTextToSpeechTask({
			model: 'speecht5',
			text: 'Hello world, this is a test synthesizing speech.',
		})
		expect(speechRes.audio).toBeTruthy()
		// await saveAudioToFile(speechRes.audio, 'tests/fixtures/speecht5.wav')
		const transcriptionRes = await inferenceServer.processSpeechToTextTask({
			model: 'whisper-base',
			audio: speechRes.audio,
		})
		expect(transcriptionRes.text.trim()).toEqual('Hello world, this is a test synthesizing speech.')
	})
})

suite('text and chat', () => {
	const inferenceServer = new InferenceServer({
		log: 'debug',
		models: {
			smollm2: {
				url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct',
				engine: 'transformers-js',
				task: 'chat-completion',
				prepare: 'blocking',
				dtype: 'int8',
			},
		},
	})
	beforeAll(async () => {
		await inferenceServer.start()
	})
	afterAll(async () => {
		await inferenceServer.stop()
	})

	test('text completion', async () => {
		const res = await inferenceServer.processTextCompletionTask({
			model: 'smollm2',
			prompt: 'The opposite of orange is',
			maxTokens: 32,
			temperature: 2,
			stop: ['\n'],
		})
		console.debug(res)
		expect(res.text).toBeTruthy()
	})

	test('chat completion', async () => {
		const res = await inferenceServer.processChatCompletionTask({
			model: 'smollm2',
			messages: [
				{ role: 'system', content: 'You are a helpful assistant.' },
				{ role: 'user', content: 'What is the capital of France?' },
			],
			maxTokens: 32,
		})
		console.debug(res)
		expect(res.message.content).toMatch(/Paris/)
	})
})
