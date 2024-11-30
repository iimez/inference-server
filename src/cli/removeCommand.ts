import path from 'node:path'
import fs from 'node:fs/promises'
import readline from 'node:readline/promises'
import type { CommandModule } from 'yargs'
import chalk from 'chalk'
import micromatch from 'micromatch'
import { indexModelCache, ModelCacheInfo, FileTreeItem } from '#package/cli/lib/indexModelCache.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'
import { renderTreeView } from '#package/cli/lib/renderTreeView.js'
import prettyBytes from 'pretty-bytes'

interface RemoveCommandArgs {
	pattern?: string
}

function matchCachedModels(cacheInfo: ModelCacheInfo, pattern: string): FileTreeItem[] {
	const matches: FileTreeItem[] = []
	const isMatch = micromatch.matcher(pattern)
	const visit = (node: FileTreeItem, nodePath: string) => {
		if (isMatch(nodePath)) {
			matches.push(node)
		}
		if (node.type === 'directory') {
			for (const child of node.children) {
				visit(child, nodePath + '/' + child.name)
			}
		}
	}
	for (const node of cacheInfo.fileTree) {
		visit(node, node.name)
	}
	return matches
}

function calculateTreeAggs(models: FileTreeItem[]): { totalSize: number; fileCount: number } {
	let totalSize = 0
	let fileCount = 0
	const visit = (node: FileTreeItem) => {
		if (node.type === 'directory') {
			for (const child of node.children) {
				visit(child)
			}
		} else {
			fileCount++
			totalSize += node.size
		}
	}
	for (const node of models) {
		visit(node)
	}
	return { totalSize, fileCount }
}

async function deleteFiles(files: FileTreeItem[]): Promise<void> {
	for (const file of files) {
		if (file.type === 'directory') {
			await fs.rm(file.absPath, { recursive: true })
		}
		if (file.type === 'file') {
			await fs.unlink(file.absPath)
		}
	}
}

async function promptConfirmation(message: string): Promise<boolean> {
	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	})

	try {
		const answer = await rl.question(`${message} (y/N): `)
		return answer.toLowerCase() === 'y'
	} finally {
		rl.close()
	}
}

async function removeModels(pattern: string, skipPrompt?: boolean): Promise<void> {
	const config = await loadConfig()
	let modelsCachePath = getCacheDirPath('models')
	if (config?.options.cachePath) {
		modelsCachePath = path.join(config.options.cachePath, 'models')
	}
	const cacheInfo = await indexModelCache(modelsCachePath, {
		includeFiles: true,
		includeUnused: true,
	})
	const matches = matchCachedModels(cacheInfo, pattern)
	if (matches.length === 0) {
		console.error(chalk.red(`No models found matching the given pattern`))
		return
	}
	if (!skipPrompt) {
		const treeLines = renderTreeView(matches)
		console.log(treeLines.join('\n'))
		const { totalSize, fileCount } = calculateTreeAggs(matches)
		// console.log(chalk.cyan(`Total size: ${prettyBytes(totalSize)} bytes`))
		const fileText = fileCount === 1 ? `one file` : `${fileCount} files`
		const modelText = matches.length === 1 ? chalk.bold(matches[0].relPath) : `${matches.length} models`
		console.log(`\nThis will remove ${fileText} freeing ${prettyBytes(totalSize)} total.`)
		const confirmed = await promptConfirmation(`Delete ${modelText} from disk?`)
		if (!confirmed) {
			console.log('Aborted')
			return
		}
	}

	try {
		// const subject = matches.length === 1 ? 'model' : 'models'
		const modelText = matches.length === 1 ? matches[0].name : `${matches.length} models`
		console.log(`Deleting ${modelText} ...`)
		await deleteFiles(matches)
		console.log(chalk.green('Done'))
	} catch (error) {
		console.error(chalk.red(`Error during removal: ${(error as Error).message}`))
	}
}

export const removeCommand: CommandModule<{}, RemoveCommandArgs> = {
	command: 'remove <pattern>',
	aliases: ['rm', 'del'],
	describe: 'Delete models matching the pattern',
	builder: (yargs) => {
		return yargs
			.positional('pattern', {
				type: 'string',
				describe: 'Glob pattern to match model paths',
			})
			.demandOption('pattern')
			.option('yes', {
				type: 'boolean',
				alias: 'y',
				describe: 'Skip confirmation prompt',
				default: false,
			})
	},
	handler: async (argv) => {
		await removeModels(argv.pattern as string, argv.yes as boolean)
	},
}
