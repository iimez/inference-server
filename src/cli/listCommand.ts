import type { CommandModule } from 'yargs'
import prettyBytes from 'pretty-bytes'
import chalk from 'chalk'
import path from 'node:path'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'
import { FileTreeItem, indexModelCache } from '#package/cli/lib/indexModelCache.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'

interface ListCommandArgs {
	files: boolean
	json: boolean
	list: boolean
	all?: boolean
}

function renderTreeView(tree: FileTreeItem[], prefix = '', isLast = true): string[] {
	const output: string[] = []

	for (let i = 0; i < tree.length; i++) {
		const item = tree[i]
		const isLastItem = i === tree.length - 1
		const branch = isLastItem ? '└── ' : '├── '
		const childPrefix = prefix + (isLastItem ? '    ' : '│   ')

		if (item.type === 'directory') {
			output.push(`${prefix}${branch}${chalk.blue(item.name)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
			if (item.children) {
				const childLines = renderTreeView(item.children, childPrefix, isLastItem)
				output.push(...childLines)
			}
		} else {
			output.push(`${prefix}${branch}${chalk.gray(item.name)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
		}
	}

	return output
}

function renderListView(tree: FileTreeItem[], parentPath = ''): string[] {
	const output: string[] = []

	for (const item of tree) {
		const currentPath = parentPath ? path.join(parentPath, item.name) : item.name
		if (item.type === 'directory') {
			const pathSegments = path.posix.normalize(currentPath).split('/')
			if (pathSegments[0] === 'huggingface.co') {
				if (pathSegments.length === 3) {
					output.push(`${chalk.blue(currentPath)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
				}
			} else {
				output.push(`${chalk.blue(currentPath)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
			}
			output.push(...renderListView(item.children, currentPath))
		} else {
			output.push(`${chalk.gray(currentPath)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
		}
	}

	return output
}

async function listModels({
	showFiles = false,
	json = false,
	list = false,
	showAll,
	configPath,
}: {
	showFiles: boolean
	json: boolean
	list: boolean
	showAll?: boolean
	configPath?: string
}): Promise<void> {
	const config = await loadConfig(configPath)
	let modelsCachePath = getCacheDirPath('models')
	if (config?.options.cachePath) {
		modelsCachePath = path.join(config.options.cachePath, 'models')
	}

	try {
		const cacheInfo = await indexModelCache(modelsCachePath, {
			includeFiles: showFiles,
			includeUnused: showAll ?? (config ? false : true),
			usedModels: config?.options.models,
		})
		const totalSize = cacheInfo.inventory.reduce((acc, model) => acc + model.size, 0)

		if (json) {
			console.log(JSON.stringify(cacheInfo.inventory, null, 2))
		} else {
			if (configPath) {
				console.log(chalk.cyanBright(`Loaded config from: ${configPath}`))
			}
			console.log(chalk.cyan(`Models cache path:  ${modelsCachePath}`))
			console.log(chalk.cyan(`Total cache size:   ${prettyBytes(totalSize)}`))
			console.log(chalk.green(`\nModels in cache:`))

			// Render either as tree or list
			const rendered = list ? renderListView(cacheInfo.inventory) : renderTreeView(cacheInfo.inventory)
			console.log(rendered.join('\n'))
		}
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
			if (json) {
				console.log(JSON.stringify({ error: `No cached models found in ${modelsCachePath}` }))
			} else {
				console.log(chalk.yellow(`\nNo cached models found in ${modelsCachePath}`))
			}
		} else {
			if (json) {
				console.log(JSON.stringify({ error: (error as Error).message }))
			} else {
				console.error(chalk.red(`Error: ${(error as Error).message}`))
			}
		}
	}
}

export const listCommand: CommandModule<{}, ListCommandArgs> = {
	command: 'list [configPath]',
	aliases: ['ls', 'dir'],
	describe: 'List stored models',
	builder: {
		all: {
			alias: 'a',
			type: 'boolean',
			description: 'Show all models in the cache',
		},
		files: {
			alias: 'f',
			type: 'boolean',
			description: 'Include individual files',
			default: false,
		},
		json: {
			alias: 'j',
			type: 'boolean',
			description: 'Output as JSON',
			default: false,
		},
		list: {
			alias: 'l',
			type: 'boolean',
			description: 'Show as flat list instead of tree',
			default: false,
		},
	},
	handler: async (argv) => {
		await listModels({
			configPath: argv.configPath as string,
			showFiles: argv.files,
			json: argv.json,
			list: argv.list,
			showAll: argv.all,
		})
	},
}
