import path from 'node:path'
import chalk from 'chalk'
import { FileTreeItem } from './indexModelCache.js'

export function renderListView(tree: FileTreeItem[], parentPath = ''): string[] {
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