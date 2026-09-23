import assert from 'node:assert/strict';
import test from 'node:test';

import {
	DEFAULT_ACTIONS,
	benchmarkClassifier,
	formatReport,
	loadBenchmarkActions,
	summarizeDurations,
} from '../scripts/compare-embedding-backend-latency.js';

test('summarizeDurations reports per-action latency and throughput', () => {
	const stats = summarizeDurations([100, 200], 10);

	assert.equal(stats.meanRunMs, 150);
	assert.equal(stats.p50RunMs, 150);
	assert.equal(stats.p95RunMs, 195);
	assert.equal(stats.meanActionMs, 15);
	assert.ok(Math.abs(stats.actionsPerSecond - 1000 / 15) < 1e-9);
});

test('benchmarkClassifier runs warmups and measured iterations', async () => {
	const calls = [];
	const clock = [1000, 1100, 2000, 2200][Symbol.iterator]();
	const classifier = {
		predictBatch: async (actions, { batchSize }) => {
			calls.push({ actions, batchSize });
			return actions.map(() => ({ label: 'safe', confidence: 1 }));
		},
	};

	const stats = await benchmarkClassifier(
		classifier,
		[{ id: 1 }, { id: 2 }],
		{
			iterations: 2,
			warmup: 1,
			batchSize: 2,
			timer: () => clock.next().value,
		},
	);

	assert.equal(calls.length, 3);
	assert.ok(calls.every(({ batchSize }) => batchSize === 2));
	assert.equal(stats.meanRunMs, 150);
	assert.equal(stats.meanActionMs, 75);
});

test('loadBenchmarkActions uses built-in actions and limit', () => {
	assert.deepEqual(
		loadBenchmarkActions(null, 2),
		DEFAULT_ACTIONS.slice(0, 2),
	);
});

test('formatReport includes backend comparison metrics', () => {
	const report = formatReport(
		{
			meanRunMs: 10,
			p50RunMs: 9,
			p95RunMs: 12,
			meanActionMs: 2,
			actionsPerSecond: 500,
		},
		{
			meanRunMs: 30,
			p50RunMs: 28,
			p95RunMs: 35,
			meanActionMs: 6,
			actionsPerSecond: 166.67,
		},
		5,
	);

	assert.match(report, /ONNX/);
	assert.match(report, /API/);
	assert.match(report, /3\.00x slower/);
});
