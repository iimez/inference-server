
import chalk from 'chalk'
import { FileTreeItem } from './indexModelCache.js'

export function renderTreeView(tree: FileTreeItem[], prefix = '', isLast = true): string[] {
	const output: string[] = []

	for (let i = 0; i < tree.length; i++) {
		const item = tree[i]
		const isLastItem = i === tree.length - 1
		const branch = isLastItem ? '└── ' : '├── '
		const childPrefix = prefix + (isLastItem ? '    ' : '│   ')
		const color = item.isModelLocation ? chalk.blue : chalk.gray
		if (item.type === 'directory') {

			output.push(`${prefix}${branch}${color(item.name)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
			if (item.children) {
				const childLines = renderTreeView(item.children, childPrefix, isLastItem)
				output.push(...childLines)
			}
		} else {
			output.push(`${prefix}${branch}${color(item.name)} ${chalk.yellow(`(${item.sizeFormatted})`)}`)
		}
	}

	return output
}
