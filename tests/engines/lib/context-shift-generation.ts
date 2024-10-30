import { expect } from 'vitest'
import { ModelServer } from '#package/server.js'
import { ChatMessage } from '#package/types/index.js'
import { createChatCompletion, parseInstanceId } from '../../util.js'

// conversation that tests behavior when context window is exceeded while the model is generating text
export async function runGenerationContextShiftTest(
	modelServer: ModelServer,
	model: string = 'test',
) {
	// Turn 1: Tell the model a fact to remember so we can later make sure that a shift occured.
	const messages: ChatMessage[] = [
		{
			role: 'system',
			content: 'Be a helpful biologist.',
		},
		{
			role: 'user',
			content:
				"Please remember this fact for later: Platypuses have venomous spurs on their hind legs. Don't forget! Just answer with 'OK'.",
		},
	]
	const response1 = await createChatCompletion(modelServer, {
		model,
		temperature: 0,
		messages,
		stop: ['OK'],
	})
	const instanceId1 = parseInstanceId(response1.task.id)

	// Turn 2: Ask the model to start generating text about an animal with a specific name
	// The animal name will later be used to check if the context shift messed up the output.
	const animalName = 'Nedwick'
	messages.push(response1.result.message, {
		role: 'user',
		content: [
			`Now, I\'d like you to create a concept for a new animal called "${animalName}".`,
			'Please provide a realistic outline of a made up animal, including its social structures, appearance, habitat, origins and diet.',
		].join(' '),
	})
	const response2 = await createChatCompletion(modelServer, {
		temperature: 1,
		messages,
		maxTokens: 1024,
	})
	// console.debug({ turn2: response2.result.message.content })
	const instanceId2 = parseInstanceId(response2.task.id)
	expect(instanceId1).toBe(instanceId2)

	messages.push(response2.result.message)

	const elaborateOn = async (field: string) => {
		messages.push({
			role: 'user',
			content: `Elaborate a bit more on its ${field}. End your elaboration with 'OK'.`,
		})
		const response = await createChatCompletion(
			modelServer,
			{
				temperature: 1,
				messages,
				maxTokens: 2048,
			},
			60000,
		)
		// console.debug({ field, response: response.result.message.content })
		const instanceId = parseInstanceId(response.task.id)
		expect(instanceId1).toBe(instanceId)
		// make sure the model gave proper output throughout its elaboration
		expect(response.result.message.content?.substring(-6)).toMatch(/OK/i)
		// make sure its still about the same animal
		expect(response.result.message.content).toMatch(new RegExp(animalName, 'i'))
		messages.push(response.result.message)
	}

	await elaborateOn('social structures')
	await elaborateOn('origins')
	await elaborateOn('habitat')
	await elaborateOn('appearance')
	// await elaborateOn('diet')

	messages.push({
		role: 'user',
		content: 'What was the animal fact I asked you to remember earlier?',
	})
	const response3 = await createChatCompletion(modelServer, {
		messages,
		maxTokens: 1024,
	})
	console.debug({ shiftCheck: response3.result.message.content })
	expect(response3.result.message.content).not.toMatch(/platypus/i)
}
