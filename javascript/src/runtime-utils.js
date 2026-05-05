import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ALL_CLASSES = ['safe', 'harmful', 'unethical'];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const ONNX_MODEL_PATH = path.resolve(
	__dirname,
	'action_classifier_model.onnx',
);

export const ActionGuardDecision = Object.freeze({
	ALLOW: 'ALLOW',
	BLOCK: 'BLOCK',
});

export class EmbeddingModel {
	constructor(
		modelName = process.env.EMBED_MODEL_NAME,
		{ clientFactory = null } = {},
	) {
		this.modelName = modelName;
		this.client = null;
		this.clientFactory = clientFactory;
	}

	async getClient() {
		if (this.client) {
			return this.client;
		}

		const apiKey =
			process.env.EMBEDDING_API_KEY ||
			process.env.OPENAI_API_KEY ||
			'dummy';
		const baseURL = process.env.EMBEDDING_BASE_URL;

		if (this.clientFactory) {
			this.client = this.clientFactory({
				apiKey,
				...(baseURL ? { baseURL } : {}),
			});
			return this.client;
		}

		const { default: OpenAI } = await import('openai');
		this.client = new OpenAI({
			apiKey,
			...(baseURL ? { baseURL } : {}),
		});

		return this.client;
	}

	async encode(texts) {
		if (!this.modelName) {
			throw new Error('EMBED_MODEL_NAME environment variable not set.');
		}

		const client = await this.getClient();
		const response = await client.embeddings.create({
			model: this.modelName,
			input: texts,
		});

		return response.data.map((item) => item.embedding);
	}
}

export function flattenActionToText(actionData) {
	if (!actionData || typeof actionData !== 'object') {
		return `Perform action: ${String(actionData)}`;
	}

	if (actionData.type !== 'function') {
		return `Perform action: ${JSON.stringify(actionData)}`;
	}

	if (!actionData.function || typeof actionData.function !== 'object') {
		return `Call function with data: ${String(actionData.function)}`;
	}

	const funcName = actionData.function.name || 'unknown_function';
	const argsRaw = actionData.function.arguments ?? {};

	if (argsRaw && typeof argsRaw === 'object' && !Array.isArray(argsRaw)) {
		const argsFlat = Object.entries(argsRaw)
			.map(([key, value]) => `${key}=${String(value)}`)
			.join(', ');
		return `Call function ${funcName}: ${argsFlat}`;
	}

	if (typeof argsRaw === 'string') {
		try {
			const parsed = JSON.parse(argsRaw);
			if (
				parsed &&
				typeof parsed === 'object' &&
				!Array.isArray(parsed)
			) {
				const argsFlat = Object.entries(parsed)
					.map(([key, value]) => `${key}=${String(value)}`)
					.join(', ');
				return `Call function ${funcName}: ${argsFlat}`;
			}
		} catch {
			return `Call function ${funcName} with arguments: ${argsRaw}`;
		}
	}

	return `Call function ${funcName} with arguments: ${String(argsRaw)}`;
}

export const embedModel = new EmbeddingModel();
