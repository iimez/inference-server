import { promises as fs } from 'node:fs'
import path from 'node:path'

/**
 * Moves all contents from source directory into destination directory,
 * overwriting any existing files/directories with the same name in destination.
 * Files in destination that don't exist in source remain untouched.
 * @param src - Source directory
 * @param dest - Destination directory
 */
export async function moveDirectoryContents(src: string, dest: string): Promise<void> {
	// Ensure the destination directory exists
	await fs.mkdir(dest, { recursive: true })

	// Read the contents of the source directory
	const entries = await fs.readdir(src, { withFileTypes: true })

	// Iterate through all entries (files and directories)
	for (const entry of entries) {
		const srcPath = path.join(src, entry.name)
		const destPath = path.join(dest, entry.name)

		if (entry.isDirectory()) {
			// For directories: recursively move contents
			await moveDirectoryContents(srcPath, destPath)
			// Remove the now-empty source directory
			await fs.rmdir(srcPath)
		} else {
			try {
				// For files: try to move directly first (more efficient)
				await fs.rename(srcPath, destPath)
			} catch (error: any) {
				if (error.code === 'EXDEV') {
					// If cross-device move is not supported:
					// 1. Copy the file
					await fs.copyFile(srcPath, destPath)
					// 2. Remove the original
					await fs.unlink(srcPath)
				} else {
					throw error
				}
			}
		}
	}
}
