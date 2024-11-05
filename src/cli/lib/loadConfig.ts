import fs from 'node:fs/promises'
import path from 'node:path'
import url from 'node:url'
import { ModelServerOptions } from '#package/server.js'

export async function loadConfig(
	customPath?: string,
): Promise<{ path: string; options: ModelServerOptions } | null> {
	// If custom path provided, try only that
	if (customPath) {
		try {
			const configPath = path.resolve(customPath)
			const options = await importConfigOptions(configPath)
			return { path: configPath, options }
		} catch (error) {
			throw new Error(`Failed to load config from ${customPath}: ${(error as Error).message}`)
		}
	}

	// Default paths to check
	const configNames = ['infs.config.js', 'infs.config.mjs', 'infs.config.json', 'package.json']

	// Try each possible config file
	for (const configName of configNames) {
		try {
			const configPath = path.resolve(process.cwd(), configName)
			const options = await importConfigOptions(configPath)

			// For package.json, we need to check for the 'infs' or 'inference-server' key
			if (configName === 'package.json') {
				const packageJson = options as any
				const modelServerOptions = packageJson.infs || packageJson['inference-server']
				if (modelServerOptions) {
					return { path: configPath, options: modelServerOptions }
				}
				continue
			}

			return { path: configPath, options }
		} catch (error) {
			// Ignore errors for default paths
			continue
		}
	}
	


	return null
}

async function importConfigOptions(configPath: string): Promise<ModelServerOptions> {
	// Handle different config formats
	if (configPath.endsWith('.json')) {
		const content = await fs.readFile(configPath, 'utf-8')
		return JSON.parse(content)
	} else {
		// For .js and .mjs files, use dynamic import
		const imported = await import(url.pathToFileURL(configPath).href)
		return imported.config || imported.default
	}
}
