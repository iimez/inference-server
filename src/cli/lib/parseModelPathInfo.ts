import path from 'node:path'

interface ModelPathInfo {
	domain?: string
	org?: string
	name: string
	branch?: string
}

export function parseModelPathInfo(modelPath: string): ModelPathInfo {
	const parts = modelPath.split(path.sep).filter(Boolean) // Remove empty parts
	if (parts.length === 0) {
		throw new Error('Invalid model path: empty path')
	}

	// Special handling for huggingface.co paths with branch
	if (parts[0] === 'huggingface.co') {
		const lastPart = parts[parts.length - 1]
		if (lastPart.includes('-')) {
			const lastMinusIndex = lastPart.lastIndexOf('-')
			const repoName = lastPart.slice(0, lastMinusIndex)
			const branch = lastPart.slice(lastMinusIndex + 1)
			return {
				domain: parts[0],
				org: parts[1],
				name: repoName,
				branch: branch,
			}
		}
	}
	const name = parts[parts.length - 1]
	const domain = parts[0]
	return {
		domain,
		name,
	}
}