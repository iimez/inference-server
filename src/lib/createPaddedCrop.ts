import sharp, { Sharp } from 'sharp'

interface BoundingBox {
	x: number
	y: number
	width: number
	height: number
}

export async function createPaddedCrop(image: Sharp, area: BoundingBox, padding: number): Promise<Sharp> {
	// Get the original bounding box coordinates
	const left = Math.round(area.x)
	const top = Math.round(area.y)
	const width = Math.round(area.width)
	const height = Math.round(area.height)

	// Get the image metadata to determine its dimensions
	const { width: imgWidth, height: imgHeight } = await image.metadata()

	if (!imgWidth || !imgHeight) {
		throw new Error('Unable to get image dimensions')
	}

	// Create a new image with padding added to the original dimensions
	const paddedWidth = imgWidth + padding * 2
	const paddedHeight = imgHeight + padding * 2

	// Create a new image with the extended canvas and fill color
	const paddedCanvas = sharp({
		create: {
			width: paddedWidth,
			height: paddedHeight,
			channels: 3,
			background: { r: 255, g: 255, b: 255 }, // Set your desired fill color (white in this example)
		},
	})

	// Composite the original image onto the padded canvas, centering it
	const paddedImage = await paddedCanvas
		.composite([{ input: await image.png().toBuffer(), left: padding, top: padding }])
		.png()
		.toBuffer()

	// Adjust the extraction coordinates to include padding
	const adjustedLeft = Math.max(0, left)
	const adjustedTop = Math.max(0, top)

	// Extract the desired area from the padded image including the padding
	const paddedCrop = await sharp(paddedImage)
		.extract({
			left: adjustedLeft,
			top: adjustedTop,
			width: width + padding * 2, // Include padding on both sides
			height: height + padding * 2, // Include padding on both sides
		})
		.toBuffer()

	return sharp(paddedCrop)
}
