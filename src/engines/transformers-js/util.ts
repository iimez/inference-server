export async function fetchBuffer(url: string): Promise<Buffer> {
	const response = await fetch(url)
	if (!response.ok) {
		throw new Error(`Failed to fetch ${url}: ${response.statusText}`)
	}
	return Buffer.from(await response.arrayBuffer())
}


export async function remoteFileExists(url: string): Promise<boolean> {
	try {
		const response = await fetch(url, { method: 'HEAD' })
		return response.ok
	} catch (error) {
		console.error('Error checking remote file:', error)
		return false
	}
}

interface HuggingfaceModelInfo {
	modelId: string
	branch: string
}

export function parseHuggingfaceModelIdAndBranch(url: string): HuggingfaceModelInfo {
	// url to the hub model, like https://huggingface.co/jinaai/jina-clip-v1
	const parsedUrl = new URL(url)
	const urlSegments = parsedUrl.pathname.split('/')
	const repoOrg = urlSegments[1]
	const repoName = urlSegments[2]
	const branch = urlSegments[4] || 'main'
	return {
		modelId: `${repoOrg}/${repoName}`,
		branch,
	}
}