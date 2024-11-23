import { expect } from 'vitest'
import { ModelServer } from '#package/server.js'
import { ChatMessage, ToolDefinition } from '#package/types/index.js'
import { createChatCompletion } from '../../util/completions.js'

interface GetLocationWeatherParams {
	location: string
	unit?: 'celsius' | 'fahrenheit'
}

const getLocationWeather: ToolDefinition<GetLocationWeatherParams> = {
	description: 'Get the weather in a location',
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
	description: 'Get the user\'s location as a city and country',
	params: {
		type: 'object',
		properties: {},
	},
	handler: async () => {
		return 'New York, USA'
	},
}

export async function runFunctionCallTest(modelServer: ModelServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: 'Where am I?',
		},
	]
	const turn1 = await createChatCompletion(modelServer, {
		tools: {
			definitions: { getUserLocation },
		},
		messages,
	})
	expect(turn1.result.message.toolCalls).toBeUndefined()
	expect(turn1.result.message.content).toMatch(/new york/i)
}

export async function runSequentialFunctionCallTest(modelServer: ModelServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content:
				// "What's the weather like today? (Hint: Use getUserLocation, then check the weather for the resulting location.)",
				"What's the weather like today?",
		},
	]
	const turn1 = await createChatCompletion(modelServer, {
		tools: {
			definitions: {
				getUserLocation,
				getLocationWeather,
			},
		},
		messages,
	})
	expect(turn1.result.message.toolCalls).toBeDefined()
	expect(turn1.result.message.toolCalls!.length).toBe(1)

	const turn1FunctionCall = turn1.result.message.toolCalls![0]
	messages.push({
		callId: turn1FunctionCall.id,
		role: 'tool',
		// name: turn1FunctionCall.name,
		content: 'New York today: Cloudy, 21°, low chance of rain.',
	})
	const turn2 = await createChatCompletion(modelServer, {
		messages,
	})
	expect(turn2.result.message.content).toMatch(/cloudy/)
}

interface GetRandomNumberParams {
	min: number
	max: number
}

export async function runParallelFunctionCallTest(modelServer: ModelServer) {
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

	const turn1 = await createChatCompletion(modelServer, {
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

	// console.debug({
	// 	turn1: turn1.result.message,
	// })
	expect(generatedNumbers.length).toBe(2)
	expect(turn1.result.message.content).toContain(generatedNumbers[0].toString())
	expect(turn1.result.message.content).toContain(generatedNumbers[1].toString())
}
