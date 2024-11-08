import type { CommandModule } from 'yargs'
import prettyBytes from 'pretty-bytes'
import chalk from 'chalk'
import path from 'node:path'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'
import { indexModelCache } from '#package/cli/lib/indexModelCache.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'
import { renderListView } from '#package/cli/lib/renderListView.js'
import { renderTreeView } from '#package/cli/lib/renderTreeView.js'

interface ListCommandArgs {
	files: boolean
	json: boolean
	list: boolean
	all?: boolean
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
		const totalSize = cacheInfo.fileTree.reduce((acc, model) => acc + model.size, 0)

		if (json) {
			console.log(JSON.stringify(cacheInfo.fileTree, null, 2))
		} else {
			if (configPath) {
				console.log(chalk.cyanBright(`Loaded config from: ${configPath}`))
			}
			console.log(chalk.cyan(`Models cache path:  ${modelsCachePath}`))
			console.log(chalk.cyan(`Total cache size:   ${prettyBytes(totalSize)}`))
			console.log(chalk.green(`\nModels in cache:`))

			// Render either as tree or list
			const rendered = list ? renderListView(cacheInfo.fileTree) : renderTreeView(cacheInfo.fileTree)
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
