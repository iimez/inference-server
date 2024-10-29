import { Image } from '#package/types/index.js'
import sharp, { ResizeOptions } from 'sharp'

interface ImageTransformationOptions {
	resize?: {
		width: number
		height: number
		fit?: ResizeOptions['fit']
	}
}

export async function loadImageFromUrl(
	url: string,
	opts: ImageTransformationOptions = {},
): Promise<Image> {
	const imageBuffer = await fetch(url).then((res) => res.arrayBuffer())
	const buffer = await Buffer.from(imageBuffer)
	const sharpHandle = sharp(buffer).rotate()
	if (opts.resize) {
		sharpHandle.resize(opts.resize)
	}
	const { info } = await sharpHandle.toBuffer({ resolveWithObject: true })
	return {
		handle: sharpHandle,
		height: opts.resize?.height ?? info.height,
		width: opts.resize?.width ?? info.width,
		channels: info.channels,
	}
}

export async function loadImageFromFile(
	filePath: string,
	opts: ImageTransformationOptions = {},
): Promise<Image> {
	let sharpHandle = sharp(filePath).rotate()
	if (opts.resize) {
		sharpHandle.resize(opts.resize)
	}
	const { info } = await sharpHandle.toBuffer({ resolveWithObject: true })
	return {
		handle: sharpHandle,
		height: info.height,
		width: info.width,
		channels: info.channels,
	}
}
