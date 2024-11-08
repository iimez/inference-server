import path from 'node:path'
import sharp, { ResizeOptions } from 'sharp'
import { Image } from '#package/types/index.js'

interface ImageTransformationOptions {
	resize?: {
		width: number
		height: number
		fit?: ResizeOptions['fit']
	}
}

export async function loadImageFromUrl(url: string, opts: ImageTransformationOptions = {}): Promise<Image> {
	const imageBuffer = await fetch(url).then((res) => res.arrayBuffer())
	const buffer = await Buffer.from(imageBuffer)
	const sharpHandle = sharp(buffer).rotate()
	if (opts.resize) {
		sharpHandle.resize(opts.resize)
	}
	const { data, info } = await sharpHandle.raw().toBuffer({ resolveWithObject: true })
	return {
		data,
		height: opts.resize?.height ?? info.height,
		width: opts.resize?.width ?? info.width,
		channels: info.channels,
	}
}

export async function loadImageFromFile(filePath: string, opts: ImageTransformationOptions = {}): Promise<Image> {
	const sharpHandle = sharp(filePath).rotate()
	if (opts.resize) {
		sharpHandle.resize(opts.resize)
	}
	const { data, info } = await sharpHandle.raw().toBuffer({ resolveWithObject: true })
	return {
		data,
		height: info.height,
		width: info.width,
		channels: info.channels,
	}
}

export async function saveImageToFile(image: Image, destPath: string): Promise<void> {
	// Derive the format from the file extension in `destPath`
	const format = path.extname(destPath).toLowerCase().replace('.', '')
	let sharpHandle = sharp(image.data, {
		raw: {
			width: image.width,
			height: image.height,
			channels: image.channels,
		},
	})

	// Apply format based on extension
	if (format === 'jpg' || format === 'jpeg') {
		sharpHandle = sharpHandle.jpeg()
	} else if (format === 'png') {
		sharpHandle = sharpHandle.png()
	} else if (format === 'webp') {
		sharpHandle = sharpHandle.webp()
	} else if (format === 'tiff') {
		sharpHandle = sharpHandle.tiff()
	} else {
		throw new Error(`Unsupported image format: ${format}`)
	}

	await sharpHandle.toFile(destPath)
}
