/* global fetch */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
	classifyPayload,
	createApiServer,
} from '../dist/api-server.js';
import { main } from '../dist/cli.js';

async function fakeClassify(actions, { batchSize = null } = {}) {
	assert.ok([null, 2, 4].includes(batchSize));
	return actions.map((_, index) => ({
		label: index % 2 === 0 ? null : 'harmful',
		confidence: index % 2 === 0 ? 0.9 : 0.8,
	}));
}

test('classifyPayload supports single and batch actions', async () => {
	const single = await classifyPayload(
		{ action: { id: 1 } },
		{ classifyActions: fakeClassify },
	);
	assert.deepEqual(single, {
		results: [{ label: null, confidence: 0.9, safe: true }],
		summary: { total: 1, safe: 1, unsafe: 0 },
	});

	const batch = await classifyPayload(
		{ actions: [{ id: 1 }, { id: 2 }], batch_size: 2 },
		{ classifyActions: fakeClassify },
	);
	assert.deepEqual(batch.summary, { total: 2, safe: 1, unsafe: 1 });
});

test('classifyPayload validates request shape and batch size', async () => {
	await assert.rejects(
		() => classifyPayload({}, { classifyActions: fakeClassify }),
		/exactly one/,
	);
	await assert.rejects(
		() =>
			classifyPayload(
				{ action: {}, actions: [{}] },
				{ classifyActions: fakeClassify },
			),
		/exactly one/,
	);
	await assert.rejects(
		() => classifyPayload({ actions: [] }, { classifyActions: fakeClassify }),
		/non-empty/,
	);
	await assert.rejects(
		() =>
			classifyPayload(
				{ action: {}, batch_size: 0 },
				{ classifyActions: fakeClassify },
			),
		/positive integer/,
	);
});

test('HTTP API serves health, classification, and validation errors', async (t) => {
	const server = createApiServer({ classifyActions: fakeClassify });
	await new Promise((resolve, reject) => {
		server.once('error', reject);
		server.listen(0, '127.0.0.1', () => {
			server.off('error', reject);
			resolve();
		});
	});
	t.after(() => new Promise((resolve) => server.close(resolve)));

	const address = server.address();
	assert.ok(address && typeof address === 'object');
	const baseUrl = `http://127.0.0.1:${address.port}`;

	const health = await fetch(`${baseUrl}/health`);
	assert.equal(health.status, 200);
	assert.deepEqual(await health.json(), { status: 'ok' });

	const classified = await fetch(`${baseUrl}/v1/classify`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({
			actions: [{ id: 1 }, { id: 2 }],
			batch_size: 2,
		}),
	});
	assert.equal(classified.status, 200);
	assert.deepEqual((await classified.json()).summary, {
		total: 2,
		safe: 1,
		unsafe: 1,
	});

	const invalid = await fetch(`${baseUrl}/v1/classify`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({ actions: [] }),
	});
	assert.equal(invalid.status, 400);
	assert.match((await invalid.json()).error, /non-empty/);

	const missing = await fetch(`${baseUrl}/missing`);
	assert.equal(missing.status, 404);
});

test('serve command forwards host, port, and batch size', async () => {
	const calls = [];
	const exitCode = await main(
		[
			'serve',
			'--host',
			'0.0.0.0',
			'--port',
			'9000',
			'--batch-size',
			'4',
		],
		{
			classifyActions: fakeClassify,
			serveApi: async (options) => {
				calls.push(options);
			},
		},
	);
	assert.equal(exitCode, 0);
	assert.equal(calls.length, 1);
	assert.equal(calls[0].host, '0.0.0.0');
	assert.equal(calls[0].port, 9000);
	assert.equal(calls[0].batchSize, 4);
	assert.equal(calls[0].classifyActions, fakeClassify);
});
