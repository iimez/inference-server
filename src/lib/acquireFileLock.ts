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

	// Check write permissions on parent directory for creation
	try {
		if (!doesExist) {
			fs.accessSync(path.dirname(fileOrDirPath), fs.constants.W_OK)
		}
		// Check permissions on the actual path
		if (doesExist) {
			fs.accessSync(fileOrDirPath, fs.constants.R_OK | fs.constants.W_OK)
		}
	} catch (err) {
		const message = err instanceof Error ? err.message : 'unknown error'
		return Promise.reject(new Error(`Insufficient permissions for ${fileOrDirPath}: ${message}`))
	}

	// Create the file or directory if it does not exist
	if (looksLikeFile && !doesExist) {
		touchFileSync(fileOrDirPath)
	}
	if (!looksLikeFile && !doesExist) {
		fs.mkdirSync(fileOrDirPath, { recursive: true })
	}

	// If the lockfile exists but the lock is not active, it's likely a stale lock
	const isLocked = lockfile.checkSync(fileOrDirPath)
	const lockExists = fs.existsSync(`${fileOrDirPath}.lock`)
	if (!isLocked && lockExists) {
		fs.rmSync(`${fileOrDirPath}.lock`, { recursive: true, force: true })
	}

	return new Promise((resolve, reject) => {
		// Note that on linux this retries forever if we don't have permissions on fileOrDirPath
		lockfile
			.lock(fileOrDirPath, { retries: { forever: true } })
			.then((release) => {
				signal?.addEventListener('abort', release)
				resolve(() => {
					release()
					signal?.removeEventListener('abort', release)
				})
			})
			.catch((err) => reject(new Error(`Failed to acquire lock on ${fileOrDirPath}: ${err.message}`)))
	})
}
