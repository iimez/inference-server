import StableDiffusion from '@lmagder/node-stable-diffusion-cpp'

export function parseQuantization(filename: string): string | null {
	// Regular expressions to match different quantization patterns
	const regexPatterns = [
		/q(\d+)_(\d+)/i, // q4_0
		/[-_\.](f16|f32|int8|int4)/i, // f16, f32, int8, int4
		/[-_\.](fp16|fp32)/i, // fp16, fp32
	]

	for (const regex of regexPatterns) {
		const match = filename.match(regex)
		if (match) {
			// If there's a match, return the full matched quantization string
			// Remove leading dash if present, convert to uppercase
			return match[0].replace(/^[-_]/, '').replace(/fp/i, 'f').toLowerCase()
		}
	}
	return null
}

export function getWeightType(key: string): number | undefined {
	const weightKey = key.toUpperCase() as keyof typeof StableDiffusion.Type
	if (weightKey in StableDiffusion.Type) {
		return StableDiffusion.Type[weightKey]
	}
	console.warn('Unknown weight type', weightKey)
	return undefined
}

export function getSamplingMethod(method?: string): StableDiffusion.SampleMethod | undefined {
	switch (method) {
		case 'euler':
			return StableDiffusion.SampleMethod.Euler
		case 'euler_a':
			return StableDiffusion.SampleMethod.EulerA
		case 'lcm':
			return StableDiffusion.SampleMethod.LCM
		case 'heun':
			return StableDiffusion.SampleMethod.Heun
		case 'dpm2':
			return StableDiffusion.SampleMethod.DPM2
		case 'dpm++2s_a':
			return StableDiffusion.SampleMethod.DPMPP2SA
		case 'dpm++2m':
			return StableDiffusion.SampleMethod.DPMPP2M
		case 'dpm++2mv2':
			return StableDiffusion.SampleMethod.DPMPP2Mv2
		case 'ipndm':
			// @ts-ignore
			return StableDiffusion.SampleMethod.IPNDM
		case 'ipndm_v':
			// @ts-ignore
			return StableDiffusion.SampleMethod.IPNDMV
	}
	console.warn('Unknown sampling method', method)
	return undefined
}
