import test from 'node:test';
import assert from 'node:assert/strict';

import { EmbeddingModel, flattenActionToText } from '../src/runtime-utils.js';

test('flattenActionToText flattens function arguments from objects', () => {
	const text = flattenActionToText({
		type: 'function',
		function: {
			name: 'data_exporter',
			arguments: {
				dataset: 'employee_salaries',
				destination: 'xyz',
			},
		},
	});

	assert.equal(
		text,
		'Call function data_exporter: dataset=employee_salaries, destination=xyz',
	);
});

test('flattenActionToText flattens function arguments from JSON strings', () => {
	const text = flattenActionToText({
		type: 'function',
		function: {
			name: 'data_exporter',
			arguments: '{"dataset":"employee_salaries","destination":"xyz"}',
		},
	});

	assert.equal(
		text,
		'Call function data_exporter: dataset=employee_salaries, destination=xyz',
	);
});

test('flattenActionToText falls back for invalid JSON argument strings', () => {
	const text = flattenActionToText({
		type: 'function',
		function: {
			name: 'data_exporter',
			arguments: '{not-json}',
		},
	});

	assert.equal(
		text,
		'Call function data_exporter with arguments: {not-json}',
	);
});

test('flattenActionToText falls back for non-function actions', () => {
	const text = flattenActionToText({
		type: 'message',
		content: 'hello',
	});

	assert.equal(text, 'Perform action: {"type":"message","content":"hello"}');
});

test('EmbeddingModel encode delegates to the configured embedding client', async () => {
	let receivedConfig = null;
	let receivedPayload = null;

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
	});
	assert.deepEqual(receivedConfig, { apiKey: 'dummy' });
	assert.deepEqual(embeddings, [[0.1, 0.2, 0.3]]);
});
