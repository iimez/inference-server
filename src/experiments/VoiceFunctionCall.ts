import {
	EngineSpeechToTextArgs,
	ModelEngine,
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
	
	async createTranscription(args: EngineSpeechToTextArgs) {
		const speechToTextModel = await this.pool.requestInstance({
			model: this.speechToTextModel,
		})
		const transcriptionTask = speechToTextModel.instance.processSpeechToTextTask(
			{
				...args.request,
				model: this.speechToTextModel,
			},
		)
		const transcription = await transcriptionTask.result
		speechToTextModel.release()
		return transcription.text
	}

	async processSpeechToTextTask(args: EngineSpeechToTextArgs) {
		const [transcription, chatModel] = await Promise.all([
			this.createTranscription(args),
			this.pool.requestInstance({
				model: this.chatModel,
			}),
		])
		const chatTask = chatModel.instance.processChatCompletionTask({
			model: this.chatModel,
			tools: this.tools ? { definitions: this.tools } : undefined,
			messages: [
				{
					role: 'user',
					content: transcription,
				},
			],
		}, {
			onChunk: args.onChunk,
		})
		const chatResponse = await chatTask.result
		chatModel.release()
		return {
			text: chatResponse.message.content,
		}
	}
}
