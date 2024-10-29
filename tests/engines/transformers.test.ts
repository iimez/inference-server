import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { ModelServer } from '#package/server.js'
import { cosineSimilarity } from '#package/lib/math.js'
import { loadImageFromFile } from '#package/lib/loadImage.js'
import {
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
	Florence2ForConditionalGeneration,
	Florence2Processor,
}	from '@huggingface/transformers'

suite('basic', () => {
	const llms = new ModelServer({
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
				textModel: {
					modelClass: CLIPTextModelWithProjection,
				},
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
				visionModel: {
					modelClass: Florence2ForConditionalGeneration,
					dtype: {
						embed_tokens: 'fp16',
						vision_encoder: 'fp32',
						encoder_model: 'fp16',
						decoder_model_merged: 'q4',
					},
				},
				device: {
					gpu: false,
				},
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('ocr single line', async () => {
		const ocrImage = await loadImageFromFile('tests/fixtures/ocr-line.png')
		const res = await llms.processImageToTextTask({
			model: 'trocr-printed',
			image: ocrImage,
		})
		expect(res.text.match(/OVER THE \$43,456.78 <LAZY> #90 DOG/)).toBeTruthy()
	})

	test('ocr multiline', async () => {
		const ocrImage = await loadImageFromFile('tests/fixtures/ocr-multiline.png')
		const res = await llms.processImageToTextTask({
			model: 'florence2-large',
			image: ocrImage,
			prompt: 'What is the text in the image?',
		})
		expect(res.text.startsWith('The (quick) [brown] {fox} jumps!')).toBe(true)
	})

	test('multimodal embedding', async () => {
		const [blueCatImage, redCatImage] = await Promise.all([
			loadImageFromFile('tests/fixtures/blue-cat.jpg'),
			loadImageFromFile('tests/fixtures/red-cat.jpg'),
		])
		const res = await llms.processEmbeddingTask({
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
		const res = await llms.processEmbeddingTask({
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
