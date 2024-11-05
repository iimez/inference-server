import type { CommandModule } from 'yargs'
import path from 'node:path'
import chalk from 'chalk'
import micromatch from 'micromatch'
import { indexModelCache } from '#package/cli/lib/indexModelCache.js'
import { loadConfig } from '#package/cli/lib/loadConfig.js'
import { getCacheDirPath } from '#package/lib/getCacheDirPath.js'

interface RemoveCommandArgs {
	pattern?: string
}

async function removeModels(pattern: string): Promise<void> {
	const config = await loadConfig()
	let modelsCachePath = getCacheDirPath('models')
	if (config?.options.cachePath) {
		modelsCachePath = path.join(config.options.cachePath, 'models')
	}
	
	const cacheInfo = await indexModelCache(modelsCachePath, {
		includeFiles: true,
		includeUnused: true,
	})
	const isMatch = micromatch.matcher(pattern)
	
	console.debug('cacheInfo', cacheInfo)
}

export const removeCommand: CommandModule<{}, RemoveCommandArgs> = {
	command: 'remove <pattern>',
	aliases: ['rm', 'del'],
	describe: 'Delete models matching the pattern',
	builder: (yargs) => {
		return yargs.positional('pattern', {
			type: 'string',
			describe: 'Glob pattern to match model paths',
		})
		.demandOption('pattern')
	},
	handler: async (argv) => {
		await removeModels(argv.pattern as string)
	},
}
