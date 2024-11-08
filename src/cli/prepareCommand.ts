import type { CommandModule } from 'yargs'
import path from 'node:path'
import chalk from 'chalk'
import { ModelServerOptions } from '#package/server.js'
import { ModelStore } from '#package/store.js'
import { BuiltInModelOptions, ModelConfigBase, ModelEngine } from '#package/types/index.js'
import { builtInEngineNames } from '#package/engines/index.js'
import { validateModelOptions } from '#package/lib/validateModelOptions.js'
import { resolveModelFileLocation } from '#package/lib/resolveModelFileLocation.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'
import { LogLevels } from '#package/lib/logger.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'

interface PrepareCommandArgs {
	configPath?: string
	concurrency?: number
}

async function prepareAllModels(options: ModelServerOptions, concurrency?: number): Promise<void> {
	let modelsCachePath = getCacheDirPath('models')
	if (options.cachePath) {
		modelsCachePath = path.join(options.cachePath, 'models')
	}
	const modelsWithDefaults: Record<string, ModelConfigBase> = {}
	const usedEngines: Array<{ model: string; engine: string }> = []
	for (const modelId in options.models) {
		const modelOptions = options.models[modelId]
		const isBuiltIn = builtInEngineNames.includes(modelOptions.engine)
		if (isBuiltIn) {
			const builtInModelOptions = modelOptions as BuiltInModelOptions
			// can validate and resolve location of model files if a built-in engine is used
			validateModelOptions(modelId, builtInModelOptions)
			modelsWithDefaults[modelId] = {
				id: modelId,
				minInstances: 0,
				maxInstances: 1,
				modelsCachePath,
				prepare: 'blocking',
				location: resolveModelFileLocation({
					url: builtInModelOptions.url,
					filePath: builtInModelOptions.location,
					modelsCachePath,
				}),
				...builtInModelOptions,
			}
		}
	}
	const customEngines = Object.keys(options.engines ?? {})
	const loadedEngines: Record<string, ModelEngine> = {}
	for (const ref of usedEngines) {
		const isBuiltIn = builtInEngineNames.includes(ref.engine)
		const isCustom = customEngines.includes(ref.engine)
		if (!isBuiltIn && !isCustom) {
			throw new Error(`Engine "${ref.engine}" used by model "${ref.model}" does not exist`)
		}
		if (isCustom) {
			loadedEngines[ref.engine] = options.engines![ref.engine]
		}
	}
	const store = new ModelStore({
		log: LogLevels.info,
		prepareConcurrency: concurrency,
		models: modelsWithDefaults,
		modelsCachePath,
	})

	const importPromises = []
	for (const key of builtInEngineNames) {
		// skip unused engines
		const modelUsingEngine = Object.keys(store.models).find((modelId) => store.models[modelId].engine === key)
		if (!modelUsingEngine) {
			continue
		}
		importPromises.push(
			new Promise(async (resolve, reject) => {
				try {
					const engine = await import(`../engines/${key}/engine.js`)
					loadedEngines[key] = engine
					resolve({
						key,
						engine,
					})
				} catch (err) {
					reject(err)
				}
			}),
		)
	}
	await Promise.all(importPromises)
	await store.init(loadedEngines)
}

async function prepareModels(configPath?: string, concurrency?: number): Promise<void> {
	try {
		const config = await loadConfig(configPath)
		if (!config) {
			throw new Error(
				'No configuration file found. Please provide a config file path or create one of: infs.config.js, infs.config.mjs, infs.config.json, or add an "infs" key to package.json',
			)
		}

		console.log(chalk.blue(`Using config from: ${configPath}`))
		await prepareAllModels(config.options, concurrency)

		console.log(chalk.green('\nModel preparation complete!'))
	} catch (error) {
		console.error(chalk.red(`Error: ${(error as Error).message}`))
		process.exit(1)
	}
}

export const prepareCommand: CommandModule<{}, PrepareCommandArgs> = {
	command: 'prepare <configPath>',
	aliases: ['prep', 'download'],
	describe: 'Prepare models defined in configuration',
	builder: (yargs) => {
		return yargs
			.positional('configPath', {
				type: 'string',
				describe: 'Path to config file (defaults to infs.config.js)',
			})
			.option('concurrency', {
				alias: 'c',
				type: 'number',
				describe: 'Number of models to prepare concurrently',
				default: 1,
			})
	},
	handler: async (argv) => {
		await prepareModels(argv.configPath, argv.concurrency)
	},
}
