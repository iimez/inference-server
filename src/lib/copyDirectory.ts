import { promises as fs } from 'node:fs'
import path from 'node:path'

/**
 * Recursively copy a directory and its contents
 * @param src - Source directory
 * @param dest - Destination directory
 */
export async function copyDirectory(src: string, dest: string): Promise<void> {
	// Ensure the destination directory exists, create it if it doesn't
	await fs.mkdir(dest, { recursive: true })

	// Read the contents of the source directory
	const entries = await fs.readdir(src, { withFileTypes: true })

	// Iterate through all the entries (files and directories)
	for (const entry of entries) {
		const srcPath = path.join(src, entry.name)
		const destPath = path.join(dest, entry.name)

		if (entry.isDirectory()) {
			// Recursively copy directories
			await copyDirectory(srcPath, destPath)
		} else {
			// Copy files
			await fs.copyFile(srcPath, destPath)
		}
	}
}