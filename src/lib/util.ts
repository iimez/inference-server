import { inspect } from 'node:util'
import fs from 'node:fs'
import path from 'node:path'

export function elapsedMillis(since: bigint): number {
	const now = process.hrtime.bigint()
	return Number(now - BigInt(since)) / 1e6
}

export function omitEmptyValues<T extends Record<string, any>>(dict: T): T {
	return Object.fromEntries(
		Object.entries(dict).filter(([_, v]) => {
			return v !== null && v !== undefined
		}),
	) as T
}

export function touchFileSync(filePath: string) {
	fs.closeSync(fs.openSync(filePath, 'w'))
}

function isSubpath(parent: string, child: string): boolean {
	// Normalize paths to resolve .. and . segments
	const normalizedParent = path.normalize(parent)
	const normalizedChild = path.normalize(child)

	// Get relative path from parent to child
	const relativePath = path.relative(normalizedParent, normalizedChild)

	// Check if relative path:
	// 1. Doesn't start with .. (which would mean going up directories)
	// 2. Isn't an absolute path (which would start with / or C:\ etc)
	return !relativePath.startsWith('..') && !path.isAbsolute(relativePath)
}

export function mergeAbortSignals(signals: Array<AbortSignal | undefined>): AbortSignal {
	const controller = new AbortController()
	const onAbort = () => {
		controller.abort()
	}
	for (const signal of signals) {
		if (signal) {
			signal.addEventListener('abort', onAbort)
		}
	}
	return controller.signal
}

export function getRandomNumber(min: number, max: number) {
	min = Math.ceil(min)
	max = Math.floor(max)
	return Math.floor(Math.random() * (max - min)) + min
}

export function printActiveHandles() {
	//@ts-ignore
	const handles = process._getActiveHandles()
	//@ts-ignore
	const requests = process._getActiveRequests()

	console.log('Active Handles:', inspect(handles, { depth: 1 }))
	console.log('Active Requests:', inspect(requests, { depth: 1 }))
}

export function formatBytesPerSecond(speed: number) {
	const units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
	let unitIndex = 0

	while (speed >= 1024 && unitIndex < units.length - 1) {
		speed /= 1024
		unitIndex++
	}

	return `${speed.toFixed(2)} ${units[unitIndex]}`
}
