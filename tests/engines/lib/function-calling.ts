import { expect } from 'vitest'
import { InferenceServer } from '#package/server.js'
import { ChatMessage, ToolDefinition } from '#package/types/index.js'
import { createChatCompletion } from '../../util/completions.js'
import { get } from 'http'

interface GetLocationWeatherParams {
	location: string
	unit?: 'celsius' | 'fahrenheit'
}

const getLocationWeather: ToolDefinition<GetLocationWeatherParams> = {
	description: 'Get realtime weather information for a location',
	parameters: {
		type: 'object',
		properties: {
			location: {
				type: 'string',
				description: 'The city and country, e.g. "Rome, Italy"',
			},
			unit: {
				type: 'string',
				enum: ['celsius', 'fahrenheit'],
			},
		},
		required: ['location'],
	},
}

const getUserLocation = {
	description: "Get the user's location as a city and country",
	params: {
		type: 'object',
		properties: {},
	},
}

export async function runFunctionCallTest(inferenceServer: InferenceServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Where am I?',
		},
	]
	const turn1 = await createChatCompletion(inferenceServer, {
		tools: {
			definitions: {
				getUserLocation: {
					...getUserLocation,
					handler: async () => 'New York, USA',
				},
			},
		},
		messages,
	})
	expect(turn1.result.message.toolCalls).toBeUndefined()
	expect(turn1.result.message.content).toMatch(/new york/i)
}

export async function runCombinedFunctionCallTest(inferenceServer: InferenceServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Find my location, then look up the weather. Use the tools, then tell me.',
		},
	]
	const toolDefs: Record<string, ToolDefinition<any>> = {
		getUserLocation: {
			...getUserLocation,
			handler: async () => 'New York, USA',
		},
		getLocationWeather,
	}
	const turn1 = await createChatCompletion(inferenceServer, {
		tools: {
			definitions: toolDefs,
		},
		messages,
	})
	expect(turn1.result.message.toolCalls).toBeDefined()
	expect(turn1.result.message.toolCalls!.length).toBe(1)
	expect(turn1.result.message.toolCalls![0].name).toBe('getLocationWeather')
	expect(turn1.result.message.toolCalls![0].parameters!.location).toMatch(/new york/i)
	const turn1Call = turn1.result.message.toolCalls![0]
	messages.push({
		callId: turn1Call.id,
		role: 'tool',
		content: 'Weather in New York today: Cloudy, 21°, low chance of rain.',
	})
	const turn2 = await createChatCompletion(inferenceServer, {
		messages,
		tools: {
			definitions: toolDefs,
		},
	})
}

export async function runReversedCombinedFunctionCallTest(inferenceServer: InferenceServer) {
	const toolDefs: Record<string, ToolDefinition<any>> = {
		getUserLocation,
		getLocationWeather: {
			...getLocationWeather,
			handler: async () => {
				return 'Weather in New York today: Cloudy, 21°, low chance of rain.'
			},
		},
	}
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Find my location, then look up the weather. Use the tools, then tell me.',
		},
	]
	const turn1 = await createChatCompletion(inferenceServer, {
		tools: {
			definitions: toolDefs,
		},
		messages,
	})

	expect(turn1.result.message.toolCalls).toBeDefined()
	expect(turn1.result.message.toolCalls!.length).toBe(1)
	expect(turn1.result.message.toolCalls![0].name).toBe('getUserLocation')

	const turn1Call = turn1.result.message.toolCalls![0]
	messages.push({
		callId: turn1Call.id,
		role: 'tool',
		content: 'You are in New York, USA.',
	})

	const turn2 = await createChatCompletion(inferenceServer, {
		messages,
		tools: {
			definitions: toolDefs,
		},
	})
	expect(turn2.result.message.content).toContain('21°')
}

interface GetRandomNumberParams {
	min: number
	max: number
}

export async function runParallelFunctionCallTest(inferenceServer: InferenceServer) {
	const generatedNumbers: number[] = []
	const getRandomNumber: ToolDefinition<GetRandomNumberParams> = {
		description: 'Generate a random integer in given range',
		parameters: {
			type: 'object',
			required: ['min', 'max'],
			properties: {
				min: {
					type: 'number',
				},
				max: {
					type: 'number',
				},
			},
		},
		handler: async (params) => {
			const num = Math.floor(Math.random() * (params.max - params.min + 1)) + params.min
			generatedNumbers.push(num)
			return num.toString()
		},
	}

	const turn1 = await createChatCompletion(inferenceServer, {
		tools: {
			definitions: { getRandomNumber },
		},
		messages: [
			{
				role: 'user',
				content: 'Roll the dice twice, then tell me the results.',
			},
		],
	})

	expect(generatedNumbers.length).toBe(2)
	expect(turn1.result.message.content).toContain(generatedNumbers[0].toString())
	expect(turn1.result.message.content).toContain(generatedNumbers[1].toString())
}

export async function runFunctionCallWithLeadingResponseTest(inferenceServer: InferenceServer) {
	const generatedNumbers: number[] = []
	const leadingResponses: Array<string | undefined> = []
	const getRandomNumber: ToolDefinition<GetRandomNumberParams> = {
		description: 'Generate a random integer in given range',
		parameters: {
			type: 'object',
			required: ['min', 'max'],
			properties: {
				min: {
					type: 'number',
				},
				max: {
					type: 'number',
				},
			},
		},
		handler: async (params, leadingResponse) => {
			const num = Math.floor(Math.random() * (params.max - params.min + 1)) + params.min
			generatedNumbers.push(num)
			leadingResponses.push(leadingResponse)
			return num.toString()
		},
	}

	const turn1 = await createChatCompletion(inferenceServer, {
		tools: {
			definitions: { getRandomNumber },
		},
		messages: [
			{
				role: 'user',
				content:
					'First tell me what is the capitol of france? After you have told me that I want you to roll the dice twice.',
			},
		],
	})

	expect(generatedNumbers.length).toBe(2)
	expect(turn1.result.message.content).toContain(generatedNumbers[0].toString())
	expect(turn1.result.message.content).toContain(generatedNumbers[1].toString())
	expect(leadingResponses[0]).toMatch(/paris/i)
	// expect(leadingResponses[1]).toBeUndefined() // this will be defined for parallel calling.
}
