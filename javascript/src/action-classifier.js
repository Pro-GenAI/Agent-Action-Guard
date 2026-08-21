import fs from 'node:fs';

import {
	ALL_CLASSES,
	ONNX_MODEL_PATH,
	embedModel,
	flattenActionToText,
	normalizeOnnxRuntimeModule,
} from './runtime-utils.js';

function softmax(logits) {
	const maxLogit = Math.max(...logits);
	const exps = logits.map((value) => Math.exp(value - maxLogit));
	const sum = exps.reduce((accumulator, value) => accumulator + value, 0);
	return exps.map((value) => value / sum);
}

function getExpectedEmbeddingDimension(session) {
	const inputName = session.inputNames?.[0] ?? 'input';
	const metadata = Array.isArray(session.inputMetadata)
		? (session.inputMetadata.find((item) => item.name === inputName) ??
			session.inputMetadata[0])
		: session.inputMetadata?.[inputName];
	const dimension = metadata?.shape?.[1] ?? metadata?.dimensions?.[1];

	return Number.isInteger(dimension) ? dimension : null;
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
			this.ortModule = normalizeOnnxRuntimeModule(this.ortModule);
			return this.ortModule;
		}

		this.ortModule = normalizeOnnxRuntimeModule(
			await import('onnxruntime-node'),
		);
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
		return (await this.predictBatch([actionDict]))[0];
	}

	async predictBatch(actionDicts, { batchSize = null } = {}) {
		if (!Array.isArray(actionDicts)) {
			throw new TypeError('predictBatch expects an array of actions.');
		}
		if (
			batchSize !== null &&
			(!Number.isInteger(batchSize) || batchSize <= 0)
		) {
			throw new RangeError('batchSize must be a positive integer.');
		}
		if (actionDicts.length === 0) {
			return [];
		}

		const session = await this.loadModel();
		const ort = await this.getOrt();
		const chunkSize = batchSize ?? actionDicts.length;
		const predictions = [];

		for (let start = 0; start < actionDicts.length; start += chunkSize) {
			const chunk = actionDicts.slice(start, start + chunkSize);
			const texts = chunk.map((actionDict) =>
				flattenActionToText(actionDict),
			);
			const embeddings = await this.embeddingModel.encode(texts);
			if (
				!Array.isArray(embeddings) ||
				embeddings.length !== chunk.length
			) {
				throw new Error(
					'Embedding model returned an invalid batch payload.',
				);
			}

			const firstVector = embeddings[0];
			if (
				!Array.isArray(firstVector) &&
				!ArrayBuffer.isView(firstVector)
			) {
				throw new Error(
					'Embedding model returned an invalid batch payload.',
				);
			}
			const dimension = firstVector.length;
			const expectedDimension = getExpectedEmbeddingDimension(session);
			if (expectedDimension && dimension !== expectedDimension) {
				throw new Error(
					`Expected embedding dimension ${expectedDimension}, received ${dimension}.`,
				);
			}

			const flattened = new Float32Array(chunk.length * dimension);
			for (let index = 0; index < embeddings.length; index += 1) {
				const vector = embeddings[index];
				if (
					(!Array.isArray(vector) && !ArrayBuffer.isView(vector)) ||
					vector.length !== dimension
				) {
					throw new Error(
						'Embedding model returned inconsistent vector dimensions.',
					);
				}
				flattened.set(vector, index * dimension);
			}

			const inputTensor = new ort.Tensor('float32', flattened, [
				chunk.length,
				dimension,
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
			const classCount = ALL_CLASSES.length;
			if (logits.length !== chunk.length * classCount) {
				throw new Error(
					'Classifier returned an invalid batch logits payload.',
				);
			}

			for (let row = 0; row < chunk.length; row += 1) {
				const rowLogits = logits.slice(
					row * classCount,
					(row + 1) * classCount,
				);
				const probabilities = softmax(rowLogits);
				let predClassIdx = 0;
				for (let index = 1; index < rowLogits.length; index += 1) {
					if (rowLogits[index] > rowLogits[predClassIdx]) {
						predClassIdx = index;
					}
				}
				predictions.push({
					label: ALL_CLASSES[predClassIdx],
					confidence: probabilities[predClassIdx],
				});
			}
		}

		return predictions;
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

export async function isActionsHarmful(
	actionDicts,
	{ actionClassifier = classifier, batchSize = null } = {},
) {
	const predictions = await actionClassifier.predictBatch(actionDicts, {
		batchSize,
	});
	return predictions.map(({ label, confidence }) => ({
		label: label === 'safe' ? null : label,
		confidence,
	}));
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
