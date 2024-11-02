
import { EngineChatCompletionArgs, ImageToTextRequest, ModelEngine } from '#package/types/index.js'
import { CustomEngine } from '#package/engines/index.js'

// an experimental engine that replaces images with their descriptions before passing them to a chat model

export class ChatWithVisionEngine extends CustomEngine implements ModelEngine {
	imageToTextModel: string
	chatModel: string
	
	constructor({ imageToTextModel, chatModel }: { imageToTextModel: string, chatModel: string }) {
		super()
		this.imageToTextModel = imageToTextModel
		this.chatModel = chatModel
	}

	async processChatCompletionTask (
		args: EngineChatCompletionArgs,
	) {

		const imageTextPromises: any[] = []
		const imageToTextModel = await this.pool.requestInstance({
			model: this.imageToTextModel,
		})
		
		const messagesWithImageDescriptions = [... args.request.messages]

		for (let m = 0; m < messagesWithImageDescriptions.length; m++) {
			const message = messagesWithImageDescriptions[m]
			if (!Array.isArray(message.content)) {
				continue
			}
			for (let p = 0; p < message.content.length; p++) {
				const contentPart = message.content[p]
				if (contentPart.type !== 'image') {
					continue
				}
				imageTextPromises.push(new Promise(async (resolve, reject) => {
					// all florence2 task prompts: https://huggingface.co/microsoft/Florence-2-base#tasks
					// "task_prompts_without_inputs": {
					// 	"<OCR>": "What is the text in the image?",
					// 	"<OCR_WITH_REGION>": "What is the text in the image, with regions?",
					// 	"<CAPTION>": "What does the image describe?",
					// 	"<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
					// 	"<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
					// 	"<OD>": "Locate the objects with category name in the image.",
					// 	"<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
					// 	"<REGION_PROPOSAL>": "Locate the region proposals in the image."
					// },
					// "task_prompts_with_input": {
					// 	"<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
					// 	"<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
					// 	"<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
					// 	"<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
					// 	"<REGION_TO_CATEGORY>": "What is the region {input}?",
					// 	"<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
					// 	"<REGION_TO_OCR>": "What text is in the region {input}?"
					// }
					// const imageData = await fetch(contentPart.image.url).then((res) => res.arrayBuffer())
					const task = imageToTextModel.instance.processImageToTextTask({
						model: this.imageToTextModel,
						// url: contentPart.url,
						image: contentPart.image,
						prompt: 'What does the image describe?',
					})
					const result = await task.result
					resolve({
						text: result.text,
						messageIndex: m,
						contentPartIndex: p,
					})
				}))
			}
		}

		const imageTextResults = await Promise.all(imageTextPromises)
		imageToTextModel.release()
		console.debug('Image text results', imageTextResults)
		
		for (const imageTextResult of imageTextResults) {
			const { text, messageIndex, contentPartIndex } = imageTextResult
			const message = messagesWithImageDescriptions[messageIndex]
			// if ('type' in message.content[contentPartIndex]) {
			// message.content[contentPartIndex].type = 'text'
			// @ts-ignore
			message.content[contentPartIndex] = {
				type: 'text',
				text: `User uploaded image: ${text}`,
			}
		}

		const chatRequest = { ...args.request, messages: messagesWithImageDescriptions, model: this.chatModel }
		const chatModel = await this.pool.requestInstance(chatRequest)
		const task = chatModel.instance.processChatCompletionTask(chatRequest, {
			onChunk: (chunk) => {
				if (args.onChunk) args.onChunk(chunk)
			},
		})
		const result = await task.result
		chatModel.release()
		return result
	}
}