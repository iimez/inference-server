import fs from 'node:fs/promises'
import path from 'node:path'
import prettyBytes from 'pretty-bytes'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { ModelOptions } from '#package/types/index.js'

// Helper to get total size of a directory
async function getDirSize(dir: string): Promise<number> {
	let size = 0
	const items = await fs.readdir(dir, { withFileTypes: true })

	for (const item of items) {
		const itemPath = path.join(dir, item.name)
		if (item.isDirectory()) {
			size += await getDirSize(itemPath)
		} else {
			const stats = await fs.stat(itemPath)
			size += stats.size
		}
	}

	return size
}

async function isLikelyModelLocation(absPath: string, relPath: string): Promise<boolean> {
	const parts = relPath.split(path.sep)
	const stat = await fs.stat(absPath).catch(() => null)
	if (!stat) {
		return false
	}
	const isJson = parts[parts.length - 1].endsWith('.json')
	const isSmallFile = stat.size < 1024 * 1024
	if (parts.length < 2) {
		return stat.isFile() && !isJson && !isSmallFile
	}
	if (parts[0] === 'huggingface.co') {
		return parts.length === 3
	} else {
		return parts.length === 2 && !isJson && !isSmallFile
	}
}

interface File {
	name: string
	absPath: string
	relPath: string
	type: 'file'
	size: number
	sizeFormatted: string
	isModelLocation: boolean
}

interface Directory {
	name: string
	absPath: string
	relPath: string
	type: 'directory'
	size: number
	sizeFormatted: string
	children: FileTreeItem[]
	isModelLocation: boolean
	fileCount: number
}

export type FileTreeItem = File | Directory

interface BuildFileTreeOptions {
	cacheRoot?: string
}

async function buildFileTree(dir: string, opts: BuildFileTreeOptions = {}): Promise<FileTreeItem[]> {
	const items = await fs.readdir(dir, { withFileTypes: true })
	const result: FileTreeItem[] = []

	for (const item of items) {
		if (item.name.startsWith('.')) continue

		const absPath = path.join(dir, item.name)
		const relPath = path.relative(opts.cacheRoot ?? dir, absPath)
		const isModelLocation = await isLikelyModelLocation(absPath, relPath)
		if (item.isDirectory()) {
			const size = await getDirSize(absPath)
			const children = await buildFileTree(absPath, opts)
			const fileCount = children.reduce((count, child) => {
				if (child.type === 'file') {
					return count + 1
				} else {
					return count + (child.fileCount || 0)
				}
			}, 0)

			result.push({
				name: item.name,
				absPath: absPath,
				relPath,
				type: 'directory',
				size,
				sizeFormatted: prettyBytes(size),
				isModelLocation,
				fileCount,
				children,
			})
		} else {
			const stats = await fs.stat(absPath)
			result.push({
				name: item.name,
				absPath: absPath,
				relPath,
				type: 'file',
				size: stats.size,
				sizeFormatted: prettyBytes(stats.size),
				isModelLocation,
			})
		}
	}

	return result
}

function filterFileTree(tree: FileTreeItem[], predicate: (item: FileTreeItem) => boolean): FileTreeItem[] {
	const result: FileTreeItem[] = []
	for (const item of tree) {
		if (predicate(item)) {
			if (item.type === 'directory') {
				const children = filterFileTree(item.children, predicate)
				result.push({
					...item,
					children,
				})
			} else {
				result.push(item)
			}
		}
	}
	return result
}

function calculateFileCount(tree: FileTreeItem[]): number {
	return tree.reduce((count, child) => {
		if (child.type === 'file') {
			return count + 1
		} else {
			return count + calculateFileCount(child.children)
		}
	}, 0)
}

export interface ModelCacheInfo {
	fileTree: FileTreeItem[]
	fileCount: number
}

interface IndexModelCacheOptions {
	includeFiles?: boolean
	includeUnused?: boolean
	usedModels?: Record<string, ModelOptions>
}

export async function indexModelCache(dir: string, opts: IndexModelCacheOptions = {}): Promise<ModelCacheInfo> {
	// keep track of which models paths are used in the config so we can determine which files belong to which model
	const usedModelPaths: string[] = []
	if (opts.usedModels) {
		for (const model of Object.values(opts.usedModels)) {
			const location = resolveModelFileLocation({
				// @ts-ignore
				url: model.url,
				filePath: model.location,
				modelsCachePath: dir,
			})
			// normalize to dir if its a file
			const stat = await fs.stat(location).catch(() => null)
			if (stat) {
				const isDir = stat.isDirectory()
				const subPath = path.relative(dir, location)
				const subPathParts = subPath.split(path.sep)

				if (subPathParts.length > 2) {
					// assume hf like structure, pick first three segments
					usedModelPaths.push(path.join(dir, subPathParts.slice(0, 3).join(path.sep)))
				} else {
					// any other model source
					let usedPath = location
					if (!isDir) {
						// remove file extension
						usedPath = location.slice(0, -path.extname(location).length)
					}
					usedModelPaths.push(usedPath)
				}
			}
		}
	}

	let fileTree = await buildFileTree(dir, {
		cacheRoot: dir,
	})

	const filterUnused = !opts.includeUnused && usedModelPaths.length > 0
	const filterFiles = !opts.includeFiles
	const isUsedModelPath = (modelPath: string) => {
		// its "used", if:
		// - any usedModelPaths is a prefix of the given modelPath, or
		// - any usedModelPaths is a prefix of the parent directory of the given modelPath
		return usedModelPaths.some((usedPath) => {
			return modelPath.startsWith(usedPath) || usedPath.startsWith(modelPath)
		})
	}

	if (filterUnused) {
		fileTree = filterFileTree(fileTree, (item) => {
			const isUsed = isUsedModelPath(item.absPath)
			if (filterUnused && !isUsed) {
				return false
			}
			return true
		})
	}

	const fileCount = calculateFileCount(fileTree)

	if (filterFiles) {
		fileTree = filterFileTree(fileTree, (item) => {
			return item.type === 'directory' || item.isModelLocation
		})
	}
	return {
		fileTree,
		fileCount,
	}
}
