import { downloadFile as createFileDownload } from 'ipull'
import fs from 'node:fs'
import path from 'node:path'
import { FileDownloadProgress } from '#package/types/index.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'

interface DownloadArgs {
	url: string
	filePath?: string
	modelsPath: string
	onProgress?: (progress: FileDownloadProgress) => void
	signal?: AbortSignal
}

export async function downloadModelFile({ url, filePath, modelsPath, onProgress, signal }: DownloadArgs) {
	let downloadUrl = url
	const parsedUrl = new URL(url)
	if (parsedUrl.hostname === 'huggingface.co') {
		const pathnameParts = parsedUrl.pathname.split('/')
		if (pathnameParts.length > 3 && pathnameParts[3] === 'blob') {
			const newUrl = new URL(url)
			pathnameParts[3] = 'resolve'
			newUrl.pathname = pathnameParts.join('/')
			if (newUrl.searchParams.get('download') !== 'true') {
				newUrl.searchParams.set('download', 'true')
			}
			downloadUrl = newUrl.href
		}
	}

	const destinationFile = resolveModelFileLocation({
		url: downloadUrl,
		filePath,
		modelsPath,
	})

	fs.mkdirSync(path.dirname(destinationFile), { recursive: true })
	const controller = await createFileDownload({
		url: downloadUrl,
		savePath: destinationFile,
		skipExisting: true,
	})

	let partialBytes = 0
	if (fs.existsSync(destinationFile)) {
		partialBytes = fs.statSync(destinationFile).size
	}
	const progressInterval = setInterval(() => {
		if (onProgress) {
			onProgress({
				file: destinationFile,
				loadedBytes: controller.status.transferredBytes + partialBytes,
				totalBytes: controller.status.totalBytes,
			})
		}
	}, 3000)
	if (signal) {
		signal.addEventListener('abort', () => {
			controller.close()
		})
	}
	await controller.download()
	if (progressInterval) {
		clearInterval(progressInterval)
	}
}
