import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import {
	ActionClassifier,
	HarmfulActionError,
	actionGuarded,
	ensureActionSafety,
	isActionHarmful,
} from '../src/action-classifier.js';

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

test('loadModel raises when the model file is missing', async () => {
	const classifier = new ActionClassifier({
		modelPath: path.join(os.tmpdir(), 'does-not-exist.onnx'),
		ortModule: createFakeOrt().module,
	});

	await assert.rejects(classifier.loadModel(), /ONNX model not found/);
});

test('loadModel creates and caches an inference session', async () => {
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
});

test('predict returns the top class and confidence from logits', async () => {
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
});

test('predict rejects invalid embedding dimensions', async () => {
	const classifier = new ActionClassifier({
		modelPath: createTempModelPath(),
		embeddingModel: {
			encode: async () => [[0.1, 0.2]],
		},
		ortModule: createFakeOrt().module,
	});

	await assert.rejects(
		classifier.predict({
			type: 'function',
			function: { name: 'ping', arguments: {} },
		}),
		/Expected embedding dimension 384/,
	);
});

test('isActionHarmful returns null for safe predictions', async () => {
	const result = await isActionHarmful(
		{ type: 'function', function: { name: 'ping', arguments: {} } },
		{
			predict: async () => ({ label: 'safe', confidence: 0.91 }),
		},
	);

	assert.deepEqual(result, { label: null, confidence: 0.91 });
});

test('ensureActionSafety raises when configured and action is harmful', async () => {
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
});

test('actionGuarded blocks harmful actions above threshold', async () => {
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
});

test('actionGuarded allows safe actions', async () => {
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

	const result = await guarded({ to: 'user@example.com' });
	assert.equal(result, 'sending to user@example.com');
});
