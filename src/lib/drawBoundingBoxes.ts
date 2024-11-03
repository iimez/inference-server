import sharp, { Sharp } from 'sharp'

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
	const labelBgColor = `rgba(${options.labelBgColor.r}, ${options.labelBgColor.g}, ${options.labelBgColor.b}, ${
		options.labelBgColor.alpha
	})`
	const labelColor = `rgba(${options.labelColor.r}, ${options.labelColor.g}, ${options.labelColor.b}, ${
		options.labelColor.alpha
	})`

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
	input: Sharp,
	detections: DetectedObject[],
	options: Partial<DrawOptions> = defaultdrawOptions,
): Promise<Sharp> {
	try {
		const inputCopy = await sharp(await input.png().toBuffer())
		const metadata = await inputCopy.metadata()

		if (!metadata.width || !metadata.height) {
			throw new Error('Unable to get image dimensions')
		}

		const svg = createSVGString(metadata.width, metadata.height, detections, {
			...defaultdrawOptions,
			...options,
		})

		const svgBuffer = await sharp(Buffer.from(svg)).resize(metadata.width, metadata.height).toBuffer()

		return input.composite([
			{
				input: svgBuffer,
				top: 0,
				left: 0,
			},
		])
	} catch (error) {
		console.error('Error drawing bounding boxes:', error)
		throw error
	}
}
