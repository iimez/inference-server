import sharp, { Sharp } from 'sharp'
import { Image } from '#package/types/index.js'

export interface BoundingBox {
	x: number
	y: number
	width: number
	height: number
}

interface DetectedObject {
	score?: number
	label?: string
	box: BoundingBox
}

interface DrawOptions {
	boxColor: { r: number; g: number; b: number; alpha: number }
	boxWidth: number
	drawLabel: boolean
	labelColor: { r: number; g: number; b: number; alpha: number }
	labelBgColor: { r: number; g: number; b: number; alpha: number }
}

function createDetectionSVG(detection: DetectedObject, options: DrawOptions): string {
	const { box, label, score } = detection
	const boxStrokeColor = `rgba(${options.boxColor.r}, ${options.boxColor.g}, ${options.boxColor.b}, ${options.boxColor.alpha})`
	const boxSvg = `
    <rect
      x="${box.x}"
      y="${box.y}"
      width="${box.width}"
      height="${box.height}"
      fill="none"
      stroke="${boxStrokeColor}"
      stroke-width="${options.boxWidth}"
    />`

	if (!options.drawLabel || !label) {
		return boxSvg
	}

	const labelWidth = label.length * 12 + (score ? 55 : 10)
	const labelText = score ? `${label} (${(score * 100).toFixed(1)}%)` : label
	const labelBgColor = `rgba(${options.labelBgColor.r}, ${options.labelBgColor.g}, ${options.labelBgColor.b}, ${options.labelBgColor.alpha})`
	const labelColor = `rgba(${options.labelColor.r}, ${options.labelColor.g}, ${options.labelColor.b}, ${options.labelColor.alpha})`

	const labelSvg = `
    <rect
      x="${box.x}"
      y="${box.y}"
      width="${labelWidth}"
      height="22"
      fill="${labelBgColor}"
    />
    <text
      x="${box.x + 5}"
      y="${box.y + 15}"
      font-family="Arial"
      font-size="16"
      fill="${labelColor}"
    >${labelText}</text>`

	return boxSvg + labelSvg
}

function createSVGString(width: number, height: number, detections: DetectedObject[], options: DrawOptions): string {
	const detectionsSvg = detections.map((detection) => createDetectionSVG(detection, options)).join('')

	return `
    <svg 
      width="${width}" 
      height="${height}" 
      viewBox="0 0 ${width} ${height}"
      xmlns="http://www.w3.org/2000/svg"
    >
      ${detectionsSvg}
    </svg>`
}

const defaultdrawOptions = {
	boxWidth: 2,
	drawLabel: true,
	boxColor: { r: 255, g: 0, b: 0, alpha: 0.5 }, // Red by default
	labelColor: { r: 255, g: 255, b: 255, alpha: 1 }, // White by default
	labelBgColor: { r: 255, g: 0, b: 0, alpha: 0.7 }, // Semi-transparent red
}

export async function drawBoundingBoxes(
	image: Image,
	detections: DetectedObject[],
	options: Partial<DrawOptions> = defaultdrawOptions,
): Promise<Image> {
	const sharpHandle = sharp(image.data, {
		raw: {
			width: image.width,
			height: image.height,
			channels: image.channels as 1 | 2 | 3 | 4,
		},
	})
	const svg = createSVGString(image.width, image.height, detections, {
		...defaultdrawOptions,
		...options,
	})

	const svgBuffer = await sharp(Buffer.from(svg)).resize(image.width, image.height).toBuffer()

	const result = await sharpHandle
		.composite([
			{
				input: svgBuffer,
				top: 0,
				left: 0,
			},
		])
		.raw()
		.toBuffer({ resolveWithObject: true })
	return {
		data: result.data,
		width: result.info.width,
		height: result.info.height,
		channels: result.info.channels,
	}
}

export async function createPaddedCrop(image: Image, area: BoundingBox, padding: number): Promise<Image> {
	// Create a new image with padding added to the original dimensions
	const paddedWidth = image.width + padding * 2
	const paddedHeight = image.height + padding * 2

	// Create a new image with the extended canvas and fill color
	let sharpHandle = sharp({
		create: {
			width: paddedWidth,
			height: paddedHeight,
			channels: 3,
			background: { r: 255, g: 255, b: 255 }, // Set your desired fill color (white in this example)
		},
	})

	// Composite the original image onto the padded canvas, centering it
	sharpHandle = sharp(await sharpHandle.composite([
		{
			input: image.data,
			raw: { height: image.height, width: image.width, channels: image.channels },
			left: padding,
			top: padding,
		},
	]).png().toBuffer())

	// Get the original bounding box coordinates
	const left = Math.round(area.x)
	const top = Math.round(area.y)
	const width = Math.round(area.width)
	const height = Math.round(area.height)

	// Adjust the extraction coordinates to include padding
	const adjustedLeft = Math.max(0, left)
	const adjustedTop = Math.max(0, top)

	// Extract the desired area from the padded image including the padding
	const result = await sharpHandle
		.extract({
			left: adjustedLeft,
			top: adjustedTop,
			width: width + padding * 2, // Include padding on both sides
			height: height + padding * 2, // Include padding on both sides
		})
		.raw()
		.toBuffer({ resolveWithObject: true })

	return {
		data: result.data,
		width: result.info.width,
		height: result.info.height,
		channels: result.info.channels,
	}
}
