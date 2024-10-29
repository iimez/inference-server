import lockfile from 'proper-lockfile'

export function acquireFileLock(file: string, signal?: AbortSignal): Promise<() => void> {
	signal?.addEventListener('abort', () => {
		lockfile.unlock(file)
	})
	return lockfile.lock(file, { retries: { forever: true } }).then((release) => {
		return () => {
			release()
		}
	})
}
