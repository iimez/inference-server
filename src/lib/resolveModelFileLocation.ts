import path from 'node:path'

interface ResolveModelFileLocationArgs {
	url?: string
	filePath?: string
	modelsCachePath: string
}

/**
 * Resolve a model file/url to an absolute path to either a file or directory.
 * @param url - Optional URL to the model file. Location will be derived from it.
 * @param filePath - Optional relative (to modelsCachePath) or absolute file path that short-circuits resolution.
 * @param modelsCachePath - The path to the models cache directory.
 * @returns The abs file path on the local filesystem.
 * @throws If the model location could not be resolved.
 */
export function resolveModelFileLocation({ url, filePath, modelsCachePath }: ResolveModelFileLocationArgs) {

	if (filePath) {
		// immediately return if an absolute path is provided
		if (path.isAbsolute(filePath)) {
			return filePath
		} else {
			return path.join(modelsCachePath, filePath)
		}
	}

	if (url) {
		const parsedUrl = new URL(url)
		let destinationPath = filePath
		// support branches for huggingface URLs
		if (parsedUrl.hostname === 'huggingface.co' && !destinationPath) {
			const pathnameSegments = parsedUrl.pathname.split('/')
			const repoOrg = pathnameSegments[1]
			const repoName = pathnameSegments[2]
			const branch = pathnameSegments[4] || 'main'
			const trailingPath = pathnameSegments.slice(5).join('/')
			destinationPath = path.join(modelsCachePath, parsedUrl.hostname, repoOrg, `${repoName}-${branch}`, trailingPath)
		}
		// otherwise, use the hostname and last path segment
		if (!destinationPath) {
			const fileName = parsedUrl.pathname.split('/').pop()
			destinationPath = path.join(modelsCachePath, parsedUrl.hostname, fileName || '')
		}
		return destinationPath
	}

	throw new Error('Failed to resolve model location')
}
