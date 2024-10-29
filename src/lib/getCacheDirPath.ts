import os from 'node:os'
import path from 'node:path'

/**
 * Determines the appropriate cache directory for the current platform.
 *
 * Follows platform-specific conventions:
 * - Windows: %LOCALAPPDATA%\Cache\inference-server
 * - macOS: ~/Library/Caches/inference-server
 * - Linux: $XDG_CACHE_HOME/inference-server or ~/.cache/inference-server
 *
 * @param {string} subDir - The name of the cache subdirectory
 * @returns {string} The absolute path to the cache directory
 */
export function getCacheDirPath(subDir: string = ''): string {
	const platform = process.platform
	let basePath: string

	switch (platform) {
		case 'win32':
			// Windows: Use %LOCALAPPDATA%\Cache
			basePath = process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local')
			return path.join(basePath, 'Cache', 'inference-server', subDir)

		case 'darwin':
			// macOS: Use ~/Library/Caches
			return path.join(os.homedir(), 'Library', 'Caches', 'inference-server', subDir)

		default:
			// Linux/Unix: Use $XDG_CACHE_HOME or ~/.cache
			basePath = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache')
			return path.join(basePath, 'inference-server', subDir)
	}
}
