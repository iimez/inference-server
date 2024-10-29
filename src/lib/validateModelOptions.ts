import { BuiltInModelOptions } from '#package/types/index.js'
import { builtInEngineNames } from '#package/engines/index.js'

const modelIdPattern = /^[a-zA-Z0-9_\-\.]+$/
export function validateModelId(id: string) {
	if (!modelIdPattern.test(id)) {
		throw new Error(
			`Model "${id}" has invalid name; requires ${modelIdPattern}`,
		)
	}
}

export function validateModelOptions(
	id: string,
	modelOptions: BuiltInModelOptions,
) {
	validateModelId(id)
	if (!modelOptions.engine) {
		throw new Error(`Model "${id}" must have an engine`)
	}
	const isSourceMissing =
		!('file' in modelOptions && modelOptions.file) &&
		!modelOptions.url &&
		!modelOptions.location
	if (builtInEngineNames.includes(modelOptions.engine) && isSourceMissing) {
		throw new Error(`Model "${id}" must have either file or url`)
	}
	if (!modelOptions.task) {
		throw new Error(`Model "${id}" must have a task`)
	}
}
