import fs from 'node:fs'
import { Audio } from '#package/types/index.js'
import decode from 'audio-decode'
import libSampleRate from '@alexanderolsen/libsamplerate-js'

interface ResampleOptions {
	inputSampleRate?: number
	outputSampleRate: number
	nChannels?: number
}

export async function resampleAudioBuffer(input: Float32Array, opts: ResampleOptions) {
	const nChannels = opts.nChannels ?? 2
	const inputSampleRate = opts.inputSampleRate ?? 44100

	const resampler = await libSampleRate.create(nChannels, inputSampleRate, opts.outputSampleRate, {
		// http://www.mega-nerd.com/SRC/api_full.html http://www.mega-nerd.com/SRC/api_simple.html
		converterType: libSampleRate.ConverterType.SRC_SINC_BEST_QUALITY, // default SRC_SINC_FASTEST. see API for more
	})
	const resampledData = resampler.simple(input)
	resampler.destroy()
	return resampledData
}

export async function decodeAudio(fileBuffer: ArrayBuffer | Uint8Array, sampleRate: number = 44100) {
	const decodedAudio = await decode(fileBuffer)
	let audio = decodedAudio.getChannelData(0)
	
	if (decodedAudio.sampleRate !== sampleRate) {
		audio = await resampleAudioBuffer(audio, {
			inputSampleRate: decodedAudio.sampleRate,
			outputSampleRate: sampleRate,
			nChannels: 1,
		})
	}
	
	return audio

}

interface LoadAudioOptions {
	sampleRate?: number
	// maxSamples?: number // TODO
}

export async function loadAudioFromFile(
	filePath: string,
	opts: LoadAudioOptions = {},
): Promise<Audio> {
	const buffer = fs.readFileSync(filePath)
	const sampleRate = opts.sampleRate ?? 16000
	const audio = await decodeAudio(buffer, sampleRate)
	return {
		sampleRate,
		channels: 1,
		samples: audio,
	}
}

export async function loadAudioFromUrl(
	url: string,
	opts: LoadAudioOptions = {},
): Promise<Audio> {
	const arrayBuffer = await fetch(url).then((res) => res.arrayBuffer())
	const buffer = await Buffer.from(arrayBuffer)
	const audio = await decodeAudio(buffer)
	return {
		sampleRate: opts.sampleRate ?? 16000,
		channels: 1,
		samples: audio,
	}
}