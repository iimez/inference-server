import path from 'node:path'
import type { CommandModule } from 'yargs'
import { indexModelCache, ModelCacheInfo, FileTreeItem } from '#package/cli/lib/indexModelCache.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'
import { renderTreeView } from '#package/cli/lib/renderTreeView.js'

interface ShowCommandArgs {
	modelName?: string
}

interface ModelPathInfo {
	domain?: string
	org?: string
	name: string
	branch?: string
}

function parseModelPathInfo(modelPath: string): ModelPathInfo {
	const parts = modelPath.split(path.sep).filter(Boolean) // Remove empty parts
	if (parts.length === 0) {
		throw new Error('Invalid model path: empty path')
	}

	// Special handling for huggingface.co paths with branch
	if (parts[0] === 'huggingface.co') {
		const lastPart = parts[parts.length - 1]
		if (lastPart.includes('-')) {
			const [repoName, ...branchParts] = lastPart.split('-')
			return {
				domain: parts[0],
				org: parts[1],
				name: repoName,
				branch: branchParts.join('-'), // Handle branches that might contain hyphens
			}
		}
	}
	const name = parts[parts.length - 1]
	const domain = parts[0]
	return {
		domain,
		name,
	}
}

function matchModelName(node: FileTreeItem, modelName: string): boolean {
	if (!node.isModelLocation) {
		return false
	}
	const info = parseModelPathInfo(node.relPath)
	if (modelName.includes(path.sep)) {
		const modelNameParts = modelName.split(path.sep)
		const lastPart = modelNameParts[modelNameParts.length - 1].toLowerCase() // TODO should strip file extension
		const firstPart = modelNameParts[0].toLowerCase()
		const prefixMatches = firstPart === info.domain?.toLowerCase() || firstPart === info.org?.toLowerCase()
		const nameMatches = lastPart === info.name.toLowerCase()
		return nameMatches && prefixMatches
	}
	const nameMatches = modelName.toLowerCase() === info.name.toLowerCase()
	return nameMatches
}

function searchCachedModels(cacheInfo: ModelCacheInfo, modelName: string): FileTreeItem[] {
	const matches: FileTreeItem[] = []
	const visit = (node: FileTreeItem) => {
		if (matchModelName(node, modelName)) {
			matches.push(node)
		}
		if (node.type === 'directory') {
			for (const child of node.children) {
				visit(child)
			}
		}
	}
	for (const node of cacheInfo.fileTree) {
		visit(node)
	}
	return matches
}

async function printModelDetails(modelName: string): Promise<void> {
	const config = await loadConfig()
	let modelsCachePath = getCacheDirPath('models')
	if (config?.options.cachePath) {
		modelsCachePath = path.join(config.options.cachePath, 'models')
	}
	const cacheInfo = await indexModelCache(modelsCachePath, {
		includeFiles: true,
		includeUnused: true,
	})
	const models = searchCachedModels(cacheInfo, modelName)
	
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
		await printModelDetails(argv.modelName as string)
	},
}
