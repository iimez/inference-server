import path from 'node:path'
import type { CommandModule } from 'yargs'
import micromatch from 'micromatch'
import { indexModelCache, ModelCacheInfo, FileTreeItem } from '#package/cli/lib/indexModelCache.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'
import { renderTreeView } from '#package/cli/lib/renderTreeView.js'

interface ShowCommandArgs {
	modelName?: string
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

async function showModels(pattern: string): Promise<void> {
	const config = await loadConfig()
	let modelsCachePath = getCacheDirPath('models')
	if (config?.options.cachePath) {
		modelsCachePath = path.join(config.options.cachePath, 'models')
	}
	const cacheInfo = await indexModelCache(modelsCachePath, {
		includeFiles: true,
		includeUnused: true,
	})
	const models = matchCachedModels(cacheInfo, pattern)
	
	if (models.length === 0) {
		console.log('No models found matching the name')
		return
	}
	
	if (models.length > 1) {
		console.log('Found multiple models matching the name:')
		const treeLines = renderTreeView(models)
		console.log(treeLines.join('\n'))
		return
	}
	const matchedModel = models[0]
	const treeLines = renderTreeView([matchedModel])
	// console.debug(matchedModel)
	// TODO: read metadata from gguf, onnx and safetensor files
	console.log('Model files:')
	console.log(treeLines.join('\n'))
}

export const showCommand: CommandModule<{}, ShowCommandArgs> = {
	command: 'show <modelName>',
	aliases: ['info', 'details'],
	describe: 'Print details of a model',
	builder: (yargs) => {
		return yargs
			.positional('modelName', {
				type: 'string',
				describe: 'Name of the model to show details for',
			})
			.demandOption('modelName')
	},
	handler: async (argv) => {
		await showModels(argv.modelName as string)
	},
}
