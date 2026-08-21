import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadActions, main, summarizeResults } from '../src/cli.js';

function tempFile(name, contents) {
	const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'aag-cli-js-'));
	const filename = path.join(directory, name);
	fs.writeFileSync(filename, contents);
	return filename;
}

test('loadActions accepts direct JSON action data', () => {
	const actions = loadActions({
		actionJson: JSON.stringify({
			type: 'function',
			function: { name: 'ping', arguments: {} },
		}),
	});
	assert.equal(actions.length, 1);
	assert.equal(actions[0].function.name, 'ping');
});

test('loadActions reads JSON arrays and JSONL actions', () => {
	const jsonFile = tempFile(
		'actions.json',
		JSON.stringify([{ id: 1 }, { id: 2 }]),
	);
	const jsonlFile = tempFile('actions.jsonl', '{"id":1}\n\n{"id":2}\n');

	assert.deepEqual(loadActions({ file: jsonFile }), [{ id: 1 }, { id: 2 }]);
	assert.deepEqual(loadActions({ file: jsonlFile }), [{ id: 1 }, { id: 2 }]);
});

test('summarizeResults counts safe and unsafe labels', () => {
	assert.deepEqual(
		summarizeResults([
			{ label: null, confidence: 0.9 },
			{ label: 'harmful', confidence: 0.8 },
			{ label: 'unethical', confidence: 0.7 },
		]),
		{ safe: 1, unsafe: 2 },
	);
});

test('main classifies file actions in one batch and prints counts', async () => {
	const filename = tempFile(
		'actions.json',
		JSON.stringify([{ id: 1 }, { id: 2 }, { id: 3 }]),
	);
	const output = [];
	const calls = [];

	const exitCode = await main(['--file', filename, '--batch-size', '2'], {
		classifyActions: async (actions, options) => {
			calls.push({ actions, options });
			return [
				{ label: null, confidence: 0.9 },
				{ label: 'harmful', confidence: 0.8 },
				{ label: null, confidence: 0.7 },
			];
		},
		write: (line) => output.push(line),
	});

	assert.equal(exitCode, 0);
	assert.equal(calls[0].actions.length, 3);
	assert.deepEqual(calls[0].options, { batchSize: 2 });
	assert.deepEqual(output, [
		'Total actions: 3',
		'Safe actions: 2',
		'Unsafe actions: 1',
	]);
});
