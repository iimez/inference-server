import path from 'node:path'

interface ResolveModelLocationArgs {
	url?: string
	filePath?: string
	modelsPath: string
}

/**
 * Resolve a model file to an absolute file path.
 * @param url - Optional URL to the model file. Location will be derived from it.
 * @param filePath - Optional relative (to modelsPath) or absolute file path. Takes precedence over URL.
 * @param modelsPath - The path to the models directory.
 * @returns The abs file path on the local filesystem.
 * @throws If the model location could not be resolved.
 */
export function resolveModelFileLocation({ url, filePath, modelsPath }: ResolveModelLocationArgs) {
	if (filePath) {
		if (path.isAbsolute(filePath)) {
			return filePath
		} else {
			return path.join(modelsPath, filePath)
		}
	}

	if (url) {
		const parsedUrl = new URL(url)
		let destinationPath = filePath
		if (parsedUrl.hostname === 'huggingface.co') {
			const pathnameParts = parsedUrl.pathname.split('/')
			if (pathnameParts.length > 3 && pathnameParts[3] === 'blob') {
				const newUrl = new URL(url)
				pathnameParts[3] = 'resolve'
				newUrl.pathname = pathnameParts.join('/')
				if (newUrl.searchParams.get('download') !== 'true') {
					newUrl.searchParams.set('download', 'true')
				}
				url = newUrl.href
			}
			if (!destinationPath) {
				const repoOrg = pathnameParts[1]
				const repoName = pathnameParts[2]
				const branch = pathnameParts[4] || 'main'
				const filePath = pathnameParts.slice(5).join('/')
				destinationPath = path.join(modelsPath, 'huggingface', repoOrg, `${repoName}-${branch}`, filePath)
			}
		}
		if (!destinationPath) {
			const fileName = parsedUrl.pathname.split('/').pop()
			destinationPath = `${modelsPath}/${parsedUrl.hostname}/${fileName}`
		}
		return destinationPath
	}

	throw new Error('Could not resolve model location')
}
