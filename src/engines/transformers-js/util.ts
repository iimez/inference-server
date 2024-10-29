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
	// if (filePath) { // path to the cached model, like /path/to/huggingface/jinaai/jina-clip-v1-main
	// 	const filePathSegments = filePath.split('/')
	// 	const modelDir = filePathSegments[filePathSegments.length - 1]
	// 	const branch = modelDir.split('-').pop() || 'main'
	// 	const repoName = modelDir.replace(new RegExp(`-${branch}$`), '')
	// 	const repoOrg = filePathSegments[filePathSegments.length - 2]
	// 	const modelId = `${repoOrg}/${repoName}`
	// 	return {
	// 		modelId,
	// 		branch,
	// 	}
	// }
	// throw new Error('Either url or filePath must be provided')
}
