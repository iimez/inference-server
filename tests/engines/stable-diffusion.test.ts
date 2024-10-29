import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import sharp from 'sharp'
import { CLIPTextModelWithProjection, CLIPVisionModelWithProjection } from '@huggingface/transformers'
import { ModelServer } from '#package/server.js'
import { loadImageFromFile } from '#package/lib/loadImage.js'
import { cosineSimilarity } from '#package/lib/math.js'

suite('basic', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			'flux-schnell': {
				url: 'https://huggingface.co/leejet/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-q4_0.gguf',
				task: 'text-to-image',
				sha256: '4f30741d2bfc786c92934ce925fcb0a43df3441e76504b797c3d5d5f0878fa6f',
				engine: 'stable-diffusion-cpp',
				prepare: 'blocking',
				diffusionModel: true,
				samplingMethod: 'euler_a',
				vae: {
					url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/ae.safetensors',
				},
				clipL: {
					url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/clip_l.safetensors',
				},
				t5xxl: {
					// url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl_fp16.safetensors',
					url: 'https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q8_0.gguf',
				},
			},
			'sd-3.5-turbo': {
				url: 'https://huggingface.co/stduhpf/SD3.5-Large-Turbo-GGUF-mixed-sdcpp/blob/main/legacy/sd3.5_large_turbo-q4_0.gguf',
				sha256: '52495d9c4356065a1378a93c9556a9eb465e10014ba9ce364512674267405bb2',
				engine: 'stable-diffusion-cpp',
				task: 'text-to-image',
				prepare: 'blocking',
				samplingMethod: 'euler',
				clipG: {
					url: 'https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors',
					sha256: 'ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4',
				},
				clipL: {
					url: 'https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors',
					sha256: '660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd',
				},
				t5xxl: {
					url: 'https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors',
					sha256: '7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09',
				},
			},
			'sdxl-turbo': {
				url: 'https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors',
				sha256: 'e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26',
				engine: 'stable-diffusion-cpp',
				task: 'image-to-image',
				prepare: 'blocking',
				samplingMethod: 'euler',
				vae: {
					url: 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl.vae.safetensors',
					sha256: '235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1',
				},
			},
			'jina-clip-v1': {
				url: 'https://huggingface.co/jinaai/jina-clip-v1',
				engine: 'transformers-js',
				task: 'embedding',
				prepare: 'blocking',
				minInstances: 1,
				maxInstances: 1,
				device: {
					gpu: false,
				},
				textModel: {
					modelClass: CLIPTextModelWithProjection,
				},
				visionModel: {
					processor: {
						url: 'https://huggingface.co/Xenova/clip-vit-base-patch32',
					},
					modelClass: CLIPVisionModelWithProjection,
				},
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	}, 180000)
	afterAll(async () => {
		await llms.stop()
	})

	test('text to image (sd-3.5-turbo)', async () => {
		const imageRes = await llms.processTextToImageTask({
			model: 'sd-3.5-turbo',
			width: 512,
			height: 512,
			cfgScale: 1,
			sampleSteps: 4,
			guidance: 3.5,
			prompt:
				'Comic panel of a lime green velociraptor with a red cape and a blue top hat, riding a unicycle on a tightrope over a pit of lava.',
		})
		// for (let i = 0; i < imageRes.images.length; i++) {
		// 	const image = imageRes.images[i]
		// 	await image.handle.toFile(`tests/engines/output-tti-sd-${i}.png`)
		// }

		const embeddingRes = await llms.processEmbeddingTask({
			model: 'jina-clip-v1',
			input: [
				{
					type: 'image',
					content: imageRes.images[0],
				},
				{
					type: 'image',
					content: await loadImageFromFile('tests/fixtures/blue-cat.jpg'),
				},
				'Comic panel of a lime green velociraptor with a red cape and a blue top hat, riding a unicycle on a tightrope over a pit of lava.',
			],
		})
		const imageEmbedding = Array.from(embeddingRes.embeddings[0])
		const compareImageEmbedding = Array.from(embeddingRes.embeddings[1])
		const promptEmbedding = Array.from(embeddingRes.embeddings[2])
		const imageSimilarity = cosineSimilarity(imageEmbedding, promptEmbedding)
		const compareSimilarity = cosineSimilarity(compareImageEmbedding, promptEmbedding)
		expect(imageSimilarity).toBeGreaterThan(compareSimilarity)
	}, 360000)

	test('text to image (flux-schnell)', async () => {
		const imageRes = await llms.processTextToImageTask({
			model: 'flux-schnell',
			width: 512,
			height: 512,
			cfgScale: 1,
			sampleSteps: 4,
			prompt:
				'comic panel of a lime green velociraptor with a red cape and a blue top hat, riding a unicycle on a tightrope over a pit of lava.',
		})
		// for (let i = 0; i < imageRes.images.length; i++) {
		// 	const image = imageRes.images[i]
		// 	await image.handle.toFile(`tests/engines/output-tti-flux-${i}.png`)
		// }

		const embeddingRes = await llms.processEmbeddingTask({
			model: 'jina-clip-v1',
			input: [
				{
					type: 'image',
					content: imageRes.images[0],
				},
				{
					type: 'image',
					content: await loadImageFromFile('tests/fixtures/blue-cat.jpg'),
				},
				'Comic panel of a lime green velociraptor with a red cape and a blue top hat, riding a unicycle on a tightrope over a pit of lava.',
			],
		})
		const imageEmbedding = Array.from(embeddingRes.embeddings[0])
		const compareImageEmbedding = Array.from(embeddingRes.embeddings[1])
		const promptEmbedding = Array.from(embeddingRes.embeddings[2])
		const imageSimilarity = cosineSimilarity(imageEmbedding, promptEmbedding)
		const compareSimilarity = cosineSimilarity(compareImageEmbedding, promptEmbedding)
		expect(imageSimilarity).toBeGreaterThan(compareSimilarity)
	}, 360000)

	test('image to image (sdxl-turbo)', async () => {
		const inputImage = await loadImageFromFile('tests/fixtures/blue-cat.jpg', {
			resize: {
				width: 512,
				height: 512,
				fit: 'cover',
			},
		})
		// await inputImage.handle.toFile('tests/engines/input-iti.png')
		const imageRes = await llms.processImageToImageTask({
			model: 'sdxl-turbo',
			image: inputImage,
			sampleSteps: 8,
			cfgScale: 1.0,
			width: 512,
			height: 512,
			prompt: 'renaissance painting of a cat in a spacesuite',
			strength: 0.65,
		})
		// for (let i = 0; i < imageRes.images.length; i++) {
		// 	const image = imageRes.images[i]
		// 	await image.handle.toFile(`tests/engines/output-iti-${i}.png`)
		// }
		const embeddingRes = await llms.processEmbeddingTask({
			model: 'jina-clip-v1',
			input: [
				{
					type: 'image',
					content: imageRes.images[0],
				},
				{
					type: 'image',
					content: await loadImageFromFile('tests/fixtures/red-cat.jpg'),
				},
				'A renaissance painting of a cat in a spacesuite.',
			],
		})
		const spaceCatImageEmbedding = Array.from(embeddingRes.embeddings[0])
		const redCatImageEmbedding = Array.from(embeddingRes.embeddings[1])
		const promptEmbedding = Array.from(embeddingRes.embeddings[2])
		const spaceCatSimilarity = cosineSimilarity(spaceCatImageEmbedding, promptEmbedding)
		const redCatSimilarity = cosineSimilarity(redCatImageEmbedding, promptEmbedding)
		expect(spaceCatSimilarity).toBeGreaterThan(redCatSimilarity)
	}, 360000)
})
