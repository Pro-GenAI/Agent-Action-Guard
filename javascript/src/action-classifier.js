import fs from 'node:fs';

import {
	ALL_CLASSES,
	ONNX_MODEL_PATH,
	embedModel,
	flattenActionToText,
} from './runtime-utils.js';

function softmax(logits) {
	const maxLogit = Math.max(...logits);
	const exps = logits.map((value) => Math.exp(value - maxLogit));
	const sum = exps.reduce((accumulator, value) => accumulator + value, 0);
	return exps.map((value) => value / sum);
}

export class ActionClassifier {
	constructor({
		modelPath = ONNX_MODEL_PATH,
		embeddingModel = embedModel,
		sessionOptions = {},
		ortModule = null,
	} = {}) {
		this.modelPath = modelPath;
		this.embeddingModel = embeddingModel;
		this.sessionOptions = sessionOptions;
		this.ortModule = ortModule;
		this.sessionPromise = null;
	}

	async getOrt() {
		if (this.ortModule) {
			return this.ortModule;
		}

		this.ortModule = await import('onnxruntime-node');
		return this.ortModule;
	}

	async loadModel() {
		if (!fs.existsSync(this.modelPath)) {
			throw new Error(`ONNX model not found: ${this.modelPath}`);
		}

		if (!this.sessionPromise) {
			const ort = await this.getOrt();
			this.sessionPromise = ort.InferenceSession.create(this.modelPath, {
				executionProviders: ['cpu'],
				...this.sessionOptions,
			});
		}

		return this.sessionPromise;
	}

	async predict(actionDict) {
		const session = await this.loadModel();
		const text = flattenActionToText(actionDict);
		const embeddings = await this.embeddingModel.encode([text]);

		if (!Array.isArray(embeddings) || !Array.isArray(embeddings[0])) {
			throw new Error(
				'Embedding model returned an invalid embedding payload.',
			);
		}

		const vector = Float32Array.from(embeddings[0]);
		if (vector.length !== 384) {
			throw new Error(
				`Expected embedding dimension 384, received ${vector.length}.`,
			);
		}

		const ort = await this.getOrt();
		const inputTensor = new ort.Tensor('float32', vector, [
			1,
			vector.length,
		]);
		const outputs = await session.run({ input: inputTensor });
		const logitsTensor = outputs.logits;

		if (
			!logitsTensor ||
			(!Array.isArray(logitsTensor.data) &&
				!ArrayBuffer.isView(logitsTensor.data))
		) {
			throw new Error('Model did not return logits output.');
		}

		const logits = Array.from(logitsTensor.data);
		const probabilities = softmax(logits);

		let predClassIdx = 0;
		for (let index = 1; index < logits.length; index += 1) {
			if (logits[index] > logits[predClassIdx]) {
				predClassIdx = index;
			}
		}

		return {
			label: ALL_CLASSES[predClassIdx],
			confidence: probabilities[predClassIdx],
		};
	}
}

export class HarmfulActionError extends Error {
	constructor(message) {
		super(message);
		this.name = 'HarmfulActionError';
	}
}

export const classifier = new ActionClassifier();

export async function isActionHarmful(
	actionDict,
	actionClassifier = classifier,
) {
	const { label, confidence } = await actionClassifier.predict(actionDict);
	if (label === 'safe') {
		return { label: null, confidence };
	}

	return { label, confidence };
}

export async function ensureActionSafety(
	actionDict,
	{ raiseException = false, actionClassifier = classifier } = {},
) {
	const { label, confidence } = await isActionHarmful(
		actionDict,
		actionClassifier,
	);

	if (label && raiseException) {
		throw new HarmfulActionError(
			`Action classified as harmful (${label}) with confidence ${confidence.toFixed(2)}`,
		);
	}

	return !label;
}

export function actionGuarded(
	func,
	{ confThreshold = 0.5, actionClassifier = classifier } = {},
) {
	if (typeof func !== 'function') {
		throw new TypeError('actionGuarded expects a function.');
	}

	return async function guardedAction(...args) {
		const lastArg = args.at(-1);
		const kwargs =
			lastArg && typeof lastArg === 'object' && !Array.isArray(lastArg)
				? lastArg
				: {};

		const actionDict = {
			type: 'function',
			function: {
				name: func.name || 'anonymous_function',
				arguments: kwargs,
			},
		};

		const { label, confidence } = await isActionHarmful(
			actionDict,
			actionClassifier,
		);

		if (label && confidence >= confThreshold) {
			throw new HarmfulActionError(
				`Guarded action '${func.name || 'anonymous_function'}' classified as harmful (${label}) with confidence ${confidence.toFixed(2)}`,
			);
		}

		return await func.apply(this, args);
	};
}
