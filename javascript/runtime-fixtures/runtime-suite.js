import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
	ActionClassifier,
	ActionGuardDecision,
	EmbeddingModel,
	HarmfulActionError,
	actionGuarded,
	ensureActionSafety,
	flattenActionToText,
	isActionHarmful,
} from '../src/index.js';

function createTempModelPath() {
	const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'action-guard-js-'));
	const modelPath = path.join(tempDir, 'model.onnx');
	fs.writeFileSync(modelPath, '');
	return modelPath;
}

function createFakeOrt(logits = [0.0, 1.0, 0.0]) {
	const state = {
		lastTensor: null,
		createCalls: [],
	};

	return {
		state,
		module: {
			Tensor: class FakeTensor {
				constructor(type, data, dims) {
					this.type = type;
					this.data = data;
					this.dims = dims;
					state.lastTensor = this;
				}
			},
			InferenceSession: {
				create: async (modelPath, options) => {
					state.createCalls.push({ modelPath, options });
					return {
						inputNames: ['input'],
						inputMetadata: [
							{
								name: 'input',
								isTensor: true,
								type: 'float32',
								shape: ['batch_size', 384],
							},
						],
						run: async (feeds) => ({
							logits: {
								data: Float32Array.from(logits),
								feeds,
							},
						}),
					};
				},
			},
		},
	};
}

assert.equal(ActionGuardDecision.ALLOW, 'ALLOW');

assert.equal(
	flattenActionToText({
		type: 'function',
		function: {
			name: 'data_exporter',
			arguments: {
				dataset: 'employee_salaries',
				destination: 'xyz',
			},
		},
	}),
	'Call function data_exporter: dataset=employee_salaries, destination=xyz',
);

assert.equal(
	flattenActionToText({
		type: 'function',
		function: {
			name: 'data_exporter',
			arguments: '{"dataset":"employee_salaries","destination":"xyz"}',
		},
	}),
	'Call function data_exporter: dataset=employee_salaries, destination=xyz',
);

assert.equal(
	flattenActionToText({
		type: 'function',
		function: {
			name: 'data_exporter',
			arguments: '{not-json}',
		},
	}),
	'Call function data_exporter with arguments: {not-json}',
);

assert.equal(
	flattenActionToText({
		type: 'message',
		content: 'hello',
	}),
	'Perform action: {"type":"message","content":"hello"}',
);

await assert.rejects(
	new ActionClassifier({
		modelPath: path.join(os.tmpdir(), 'does-not-exist.onnx'),
		ortModule: createFakeOrt().module,
	}).loadModel(),
	/ONNX model not found/,
);

{
	const fakeOrt = createFakeOrt();
	const classifier = new ActionClassifier({
		modelPath: createTempModelPath(),
		embeddingModel: { encode: async () => [new Array(384).fill(0)] },
		ortModule: fakeOrt.module,
	});

	const sessionA = await classifier.loadModel();
	const sessionB = await classifier.loadModel();

	assert.equal(sessionA, sessionB);
	assert.equal(fakeOrt.state.createCalls.length, 1);
	assert.deepEqual(fakeOrt.state.createCalls[0].options, {
		executionProviders: ['cpu'],
	});
}

{
	const fakeOrt = createFakeOrt([0.1, 2.0, 0.2]);
	const embeddingCalls = [];
	const classifier = new ActionClassifier({
		modelPath: createTempModelPath(),
		embeddingModel: {
			encode: async (texts) => {
				embeddingCalls.push(texts);
				return [new Array(384).fill(0.5)];
			},
		},
		ortModule: fakeOrt.module,
	});

	const result = await classifier.predict({
		type: 'function',
		function: {
			name: 'weather_api',
			arguments: { location: 'NYC', units: 'imperial' },
		},
	});

	assert.equal(result.label, 'harmful');
	assert.ok(result.confidence > 0.7);
	assert.deepEqual(embeddingCalls, [
		['Call function weather_api: location=NYC, units=imperial'],
	]);
	assert.equal(fakeOrt.state.lastTensor.type, 'float32');
	assert.deepEqual(fakeOrt.state.lastTensor.dims, [1, 384]);
}

await assert.rejects(
	new ActionClassifier({
		modelPath: createTempModelPath(),
		embeddingModel: {
			encode: async () => [[0.1, 0.2]],
		},
		ortModule: createFakeOrt().module,
	}).predict({
		type: 'function',
		function: { name: 'ping', arguments: {} },
	}),
	/Expected embedding dimension 384/,
);

assert.deepEqual(
	await isActionHarmful(
		{ type: 'function', function: { name: 'ping', arguments: {} } },
		{
			predict: async () => ({ label: 'safe', confidence: 0.91 }),
		},
	),
	{ label: null, confidence: 0.91 },
);

await assert.rejects(
	ensureActionSafety(
		{ type: 'function', function: { name: 'drop_db', arguments: {} } },
		{
			raiseException: true,
			actionClassifier: {
				predict: async () => ({
					label: 'harmful',
					confidence: 0.88,
				}),
			},
		},
	),
	HarmfulActionError,
);

assert.equal(
	await ensureActionSafety(
		{ type: 'function', function: { name: 'ping', arguments: {} } },
		{
			actionClassifier: {
				predict: async () => ({ label: 'safe', confidence: 0.99 }),
			},
		},
	),
	true,
);

{
	const guarded = actionGuarded(
		async function deleteUser(params) {
			return params.userId;
		},
		{
			confThreshold: 0.8,
			actionClassifier: {
				predict: async () => ({ label: 'harmful', confidence: 0.95 }),
			},
		},
	);

	await assert.rejects(guarded({ userId: '123' }), HarmfulActionError);
}

{
	const guarded = actionGuarded(
		async function sendEmail(params) {
			return `sending to ${params.to}`;
		},
		{
			confThreshold: 0.8,
			actionClassifier: {
				predict: async () => ({ label: 'safe', confidence: 0.99 }),
			},
		},
	);

assert.equal(
	await guarded({ to: 'user@example.com' }),
	'sending to user@example.com',
	);
}

{
	const originalEnv = {
		EMBED_MODEL_NAME: process.env.EMBED_MODEL_NAME,
		EMBEDDING_API_KEY: process.env.EMBEDDING_API_KEY,
		OPENAI_API_KEY: process.env.OPENAI_API_KEY,
		EMBEDDING_BASE_URL: process.env.EMBEDDING_BASE_URL,
	};
	let receivedPayload = null;

	process.env.EMBED_MODEL_NAME = 'env-default-embedding-model';
	delete process.env.EMBEDDING_API_KEY;
	delete process.env.OPENAI_API_KEY;
	delete process.env.EMBEDDING_BASE_URL;

	try {
		const model = new EmbeddingModel(undefined, {
			clientFactory: () => ({
				embeddings: {
					create: async (payload) => {
						receivedPayload = payload;
						return {
							data: [{ embedding: [0.4, 0.5, 0.6] }],
						};
					},
				},
			}),
		});

		const embeddings = await model.encode(['hello']);

		assert.deepEqual(receivedPayload, {
			model: 'env-default-embedding-model',
			input: ['hello'],
			encoding_format: 'float',
		});
		assert.deepEqual(embeddings, [[0.4, 0.5, 0.6]]);
	} finally {
		for (const [key, value] of Object.entries(originalEnv)) {
			if (value === undefined) {
				delete process.env[key];
			} else {
				process.env[key] = value;
			}
		}
	}
}

{
	let receivedConfig = null;
	let receivedPayload = null;
	const originalEnv = {
		EMBED_MODEL_NAME: process.env.EMBED_MODEL_NAME,
		EMBEDDING_API_KEY: process.env.EMBEDDING_API_KEY,
		OPENAI_API_KEY: process.env.OPENAI_API_KEY,
		EMBEDDING_BASE_URL: process.env.EMBEDDING_BASE_URL,
	};

	delete process.env.EMBEDDING_API_KEY;
	delete process.env.OPENAI_API_KEY;
	delete process.env.EMBEDDING_BASE_URL;

	try {
		const model = new EmbeddingModel('test-embedding-model', {
			clientFactory: (config) => {
				receivedConfig = config;
				return {
					embeddings: {
						create: async (payload) => {
							receivedPayload = payload;
							return {
								data: [{ embedding: [0.1, 0.2, 0.3] }],
							};
						},
					},
				};
			},
		});

		const embeddings = await model.encode(['hello']);

		assert.deepEqual(receivedPayload, {
			model: 'test-embedding-model',
			input: ['hello'],
			encoding_format: 'float',
		});
		assert.deepEqual(receivedConfig, { apiKey: 'dummy' });
		assert.deepEqual(embeddings, [[0.1, 0.2, 0.3]]);
	} finally {
		for (const [key, value] of Object.entries(originalEnv)) {
			if (value === undefined) {
				delete process.env[key];
			} else {
				process.env[key] = value;
			}
		}
	}
}
