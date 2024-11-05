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

interface File {
	name: string
	type: 'file'
	size: number
	sizeFormatted: string
	isUsedByConfig: boolean
}

interface Directory {
	name: string
	type: 'directory'
	size: number
	sizeFormatted: string
	children: FileTreeItem[]
	isUsedByConfig: boolean
}

export type FileTreeItem = File | Directory

interface BuildFileTreeOptions {
	includeFiles: boolean
	includeUnused?: boolean
	usedModelPaths?: string[]
}

const defaultOptions: BuildFileTreeOptions = {
	includeFiles: false,
}

async function buildFileTree(dir: string, opts: BuildFileTreeOptions = defaultOptions): Promise<FileTreeItem[]> {
	const items = await fs.readdir(dir, { withFileTypes: true })
	const result: FileTreeItem[] = []
	const includeUnused = opts.includeUnused ?? defaultOptions.includeUnused
	const usedModelPaths = opts.usedModelPaths ?? []
	const isUsedModelPath = (modelPath: string) => {
		// - any used model paths are a prefix of the given path
		// - any used model paths are a prefix of the parent directory of the given path
		return usedModelPaths.some((usedPath) => {
			return modelPath.startsWith(usedPath) || usedPath.startsWith(modelPath)
		})
	}

	for (const item of items) {
		if (item.name.startsWith('.')) continue

		const itemPath = path.join(dir, item.name)

		if (item.isDirectory()) {
			const size = await getDirSize(itemPath)
			const children = await buildFileTree(itemPath, opts)
			const isUsed = isUsedModelPath(itemPath)
			if (!includeUnused && !isUsed) continue
			result.push({
				name: item.name,
				type: 'directory',
				size,
				sizeFormatted: prettyBytes(size),
				isUsedByConfig: isUsed,
				children,
			})
		} else if (opts.includeFiles) {
			const isUsed = isUsedModelPath(itemPath)
			if (!includeUnused && !isUsed) continue
			const stats = await fs.stat(itemPath)
			result.push({
				name: item.name,
				type: 'file',
				size: stats.size,
				sizeFormatted: prettyBytes(stats.size),
				isUsedByConfig: isUsed,
			})
		}
	}

	return result
}

export interface ModelCacheInfo {
	inventory: FileTreeItem[]
}

interface IndexModelCacheOptions {
	includeFiles?: boolean
	includeUnused?: boolean
	usedModels?: Record<string, ModelOptions>
}

export async function indexModelCache(dir: string, opts: IndexModelCacheOptions = {}): Promise<ModelCacheInfo> {
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
				usedModelPaths.push(stat.isDirectory() ? location : path.dirname(location))
			}
		}
	}

	const cacheInventory = await buildFileTree(dir, {
		includeFiles: opts?.includeFiles ?? defaultOptions.includeFiles,
		includeUnused: opts?.includeUnused,
		usedModelPaths,
	})
	return {
		inventory: cacheInventory,
	}
}
