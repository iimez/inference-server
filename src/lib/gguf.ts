import { promises as fs } from 'fs'
import { ggufMetadata } from 'hyllama'

// only typing the properties we interact with
export interface GGUFMeta {
	version: number
	general: {
		architecture: string
		name: string
		file_type: number
		quantization_version: number
	}
	llama?: unknown
	tokenizer?: {
		ggml?: {
			tokens?: string[]
			scores?: number[]
			token_type?: number[]
			merges?: string[]
		}
	}
}

// Creates a nested object from the metadata key/value pairs
function structureGGUFMeta(metadata: Record<string, any>): GGUFMeta {
	const structuredMeta: any = {}

	for (const key in metadata) {
		const parts = key.split('.')
		let current: any = structuredMeta

		for (let i = 0; i < parts.length - 1; i++) {
			const part = parts[i]
			current[part] = current[part] || {}
			current = current[part]
		}

		current[parts[parts.length - 1]] = metadata[key]
	}

	return structuredMeta
}

export async function readGGUFMetaFromFile(file: string) {
	// Read first 10mb of gguf file
	const fd = await fs.open(file, 'r')
	const buffer = Buffer.alloc(10_000_000)
	await fd.read(buffer, 0, 10_000_000, 0)
	await fd.close()
	const { metadata, tensorInfos } = ggufMetadata(buffer.buffer)
	return structureGGUFMeta(metadata)
}

export async function readGGUFMetaFromURL(url: string) {
	const headers = new Headers({ Range: 'bytes=0-10000000' })
	const res = await fetch(url, { headers })
	const arrayBuffer = await res.arrayBuffer()
	const { metadata, tensorInfos } = ggufMetadata(arrayBuffer)
	return structureGGUFMeta(metadata)
}

// see node-llama-cpp src/gguf/utils/normalizeGgufDownloadUrl.ts
export function normalizeGGUFDownloadUrl(url: string) {
	const parsedUrl = new URL(url)
	if (parsedUrl.hostname === 'huggingface.co') {
		const pathnameParts = parsedUrl.pathname.split('/')
		if (pathnameParts.length > 3) {
			const newUrl = new URL(url)
			if (pathnameParts[3] === 'blob' || pathnameParts[3] === 'raw') {
				pathnameParts[3] = 'resolve'
			}
			newUrl.pathname = pathnameParts.join('/')
			if (newUrl.searchParams.get('download') !== 'true') {
				newUrl.searchParams.set('download', 'true')
			}
			return newUrl.href
		}
	}
	return url
}