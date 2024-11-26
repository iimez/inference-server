import {
	ModelEngine,
	SpeechToTextTaskArgs,
	ToolDefinition,
} from '#package/types/index.js'
import { CustomEngine } from '#package/engines/index.js'

type EngineArgs = {
	speechToTextModel: string
	chatModel: string
	tools: Record<string, ToolDefinition>
}

// an experimental engine that forwards a transcription to a (function calling) chat model

export class VoiceFunctionCallEngine
	extends CustomEngine
	implements ModelEngine
{
	speechToTextModel: string
	chatModel: string
	tools: Record<string, ToolDefinition>

	constructor({ speechToTextModel, chatModel, tools }: EngineArgs) {
		super()
		this.speechToTextModel = speechToTextModel
		this.chatModel = chatModel
		this.tools = tools
	}
	
	async createTranscription(task: SpeechToTextTaskArgs) {
		const speechToTextModel = await this.pool.requestInstance({
			model: this.speechToTextModel,
		})
		const transcriptionTask = speechToTextModel.instance.processSpeechToTextTask(
			{
				...task,
				model: this.speechToTextModel,
			},
		)
		const transcription = await transcriptionTask.result
		speechToTextModel.release()
		return transcription.text
	}

	async processSpeechToTextTask(
		task: SpeechToTextTaskArgs,
	) {
		const [transcription, chatModel] = await Promise.all([
			this.createTranscription(task),
			this.pool.requestInstance({
				model: this.chatModel,
			}),
		])
		const chatTask = chatModel.instance.processChatCompletionTask({
			onChunk: task.onChunk,
			model: this.chatModel,
			tools: this.tools ? { definitions: this.tools } : undefined,
			messages: [
				{
					role: 'user',
					content: transcription,
				},
			],
		})
		const chatResponse = await chatTask.result
		chatModel.release()
		return {
			text: chatResponse.message.content,
		}
	}
}
