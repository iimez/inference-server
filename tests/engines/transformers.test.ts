import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { WaveFile } from 'wavefile'
import { readFileSync, writeFileSync } from 'fs'
import sharp from 'sharp'
import { ModelServer } from '#package/server.js'
import { cosineSimilarity } from '#package/lib/math.js'
import { loadImageFromFile } from '#package/lib/loadImage.js'
import { drawBoundingBoxes } from '#package/lib/drawBoundingBoxes.js'
import { createPaddedCrop } from '#package/lib/createPaddedCrop.js'
import {
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
	Florence2ForConditionalGeneration,
	SpeechT5ForTextToSpeech,
	SpeechT5Model,
	WhisperForConditionalGeneration,
} from '@huggingface/transformers'

suite('basic', () => {
	const modelServer = new ModelServer({
		log: 'debug',
		models: {
			'mxbai-embed-large-v1': {
				url: 'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1',
				engine: 'transformers-js',
				task: 'embedding',
				prepare: 'blocking',
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
				prepare: 'blocking',
				device: {
					gpu: false,
				},
			},
			'trocr-printed': {
				url: 'https://huggingface.co/Xenova/trocr-small-printed',
				engine: 'transformers-js',
				task: 'image-to-text',
				prepare: 'blocking',
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
				prepare: 'async',
				minInstances: 1,
				modelClass: WhisperForConditionalGeneration,
				dtype: {
					encoder_model: 'fp16',
					decoder_model_merged: 'q4',
				},
				device: {
					gpu: false,
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
		},
	})
	beforeAll(async () => {
		await modelServer.start()
	})
	afterAll(async () => {
		await modelServer.stop()
	})

	test('cat recognition', async () => {
		const image = await loadImageFromFile('tests/fixtures/blue-cat.jpg')
		const res = await modelServer.processObjectRecognitionTask({
			model: 'owlv2-base',
			image,
			labels: ['cat', 'smurf'],
		})
		expect(res.objects).toBeTruthy()
		// console.debug(res.objects)
		expect(res.objects[0].label).toBe('cat')
		// const testImage = await drawBoundingBoxes(image.handle, res.objects)
		// await testImage.toFile('tests/fixtures/blue-cat-detected.png')
	})

	test('table recognition', async () => {
		const image = await loadImageFromFile('tests/fixtures/table.png')
		const tableRes = await modelServer.processObjectRecognitionTask({
			model: 'table-transformer-detection',
			image,
		})
		const tableObject = tableRes.objects[0]
		expect(tableObject).toBeTruthy()
		// padding because https://github.com/microsoft/table-transformer/issues/21
		const paddedCrop = await createPaddedCrop(image.handle, tableObject.box, 40)
		// await paddedCrop.toFile('tests/fixtures/table-detected.png')
		const tableStructureRes = await modelServer.processObjectRecognitionTask({
			model: 'table-transformer-structure-recognition',
			image: {
				handle: paddedCrop,
				width: Math.round(tableObject.box.width) + 80,
				height: Math.round(tableObject.box.height) + 80,
				channels: image.channels,
			},
		})
		expect(tableStructureRes.objects).toBeTruthy()
		const tableRows = tableStructureRes.objects.filter((x) => x.label === 'table row')
		expect(tableRows.length).toEqual(8)
		// const imageWithBoundingBoxed = await drawBoundingBoxes(paddedCrop, tableRows)
		// await imageWithBoundingBoxed.toFile('tests/fixtures/table-detected-rows.png')
	})

	test('text to speech to text', async () => {
		const speechRes = await modelServer.processTextToSpeechTask({
			model: 'speecht5',
			text: 'Hello world, this is a test synthesizing speech.',
		})
		expect(speechRes.audio).toBeTruthy()
		// const wav = new WaveFile();
		// wav.fromScratch(speechRes.audio.channels, speechRes.audio.sampleRate, '32f', speechRes.audio.samples);
		// writeFileSync('tests/fixtures/speecht5.wav', wav.toBuffer());
		const transcriptionRes = await modelServer.processSpeechToTextTask({
			model: 'whisper-base',
			audio: speechRes.audio,
		})
		expect(transcriptionRes.text.trim()).toEqual('Hello world, this is a test synthesizing speech.')
	})

	test('ocr single line', async () => {
		const ocrImage = await loadImageFromFile('tests/fixtures/ocr-line.png')
		const res = await modelServer.processImageToTextTask({
			model: 'trocr-printed',
			image: ocrImage,
		})
		expect(res.text.match(/OVER THE \$43,456.78 <LAZY> #90 DOG/)).toBeTruthy()
	})

	test('ocr multiline', async () => {
		const ocrImage = await loadImageFromFile('tests/fixtures/ocr-multiline.png')
		const res = await modelServer.processImageToTextTask({
			model: 'florence2-large',
			image: ocrImage,
			// see doc here for prompts: https://huggingface.co/microsoft/Florence-2-base#tasks
			prompt: 'What is the text in the image?',
		})
		expect(res.text.startsWith('The (quick) [brown] {fox} jumps!')).toBe(true)
	})

	test('multimodal embedding', async () => {
		const [blueCatImage, redCatImage] = await Promise.all([
			loadImageFromFile('tests/fixtures/blue-cat.jpg'),
			loadImageFromFile('tests/fixtures/red-cat.jpg'),
		])
		const res = await modelServer.processEmbeddingTask({
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
		const textSimilarity = cosineSimilarity(
			blueCatTextEmbedding,
			redCatTextEmbedding,
		)
		expect(textSimilarity.toFixed(2)).toBe('0.56')
		const textImageSimilarity = cosineSimilarity(
			blueCatTextEmbedding,
			blueCatImageEmbedding,
		)
		expect(textImageSimilarity.toFixed(2)).toBe('0.29')
		const textImageSimilarity2 = cosineSimilarity(
			redCatTextEmbedding,
			blueCatImageEmbedding,
		)
		expect(textImageSimilarity2.toFixed(2)).toBe('0.12')
		const textImageSimilarity3 = cosineSimilarity(
			blueCatTextEmbedding,
			redCatImageEmbedding,
		)
		expect(textImageSimilarity3.toFixed(2)).toBe('0.05')
		const textImageSimilarity4 = cosineSimilarity(
			redCatTextEmbedding,
			redCatImageEmbedding,
		)
		expect(textImageSimilarity4.toFixed(2)).toBe('0.29')
	})

	test('text embedding', async () => {
		const res = await modelServer.processEmbeddingTask({
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
		const similarities = sentenceEmbeddings.map((x) =>
			cosineSimilarity(Array.from(searchEmbedding), Array.from(x)),
		)
		expect(similarities[0].toFixed(2)).toBe('0.79')
	})
})
