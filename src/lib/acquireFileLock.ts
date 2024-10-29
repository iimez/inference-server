import fs from 'node:fs'
import path from 'node:path'
import lockfile from 'proper-lockfile'
import { touchFileSync } from '#package/lib/util.js'

/**
 * Acquire a lock on a file or directory, creating the resource if it does not exist.
 * @param fileOrDirPath Path to the file or directory to lock.
 * @param signal Abort signal to release the lock.
 * @returns A function that releases the lock.
 */
export function acquireFileLock(fileOrDirPath: string, signal?: AbortSignal): Promise<() => void> {
	const pathInfo = path.parse(fileOrDirPath)
	const looksLikeFile = !!pathInfo.ext
	const doesExist = fs.existsSync(fileOrDirPath)
	if (looksLikeFile && !doesExist) {
		touchFileSync(fileOrDirPath)
	}
	if (!looksLikeFile && !doesExist) {
		fs.mkdirSync(fileOrDirPath, { recursive: true })
	}
	const isLocked = lockfile.checkSync(fileOrDirPath)
	const lockExists = fs.existsSync(`${fileOrDirPath}.lock`)
	if (!isLocked && lockExists) {
		fs.unlinkSync(`${fileOrDirPath}.lock`)
	}
	return new Promise((resolve, reject) => {
		lockfile.lock(fileOrDirPath, { retries: { forever: true } })
		.then((release) => {
			signal?.addEventListener('abort', release)
			resolve(() => {
				release()
				signal?.removeEventListener('abort', release)
			})
		})
		.catch(reject)
	})
}
