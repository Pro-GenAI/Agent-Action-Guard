import assert from 'node:assert/strict';
import test from 'node:test';
import {
	actionGuarded,
	HarmfulActionError,
} from '../dist/action-classifier.js';
import { flattenActionToText } from '../dist/runtime-utils.js';

test('guards screen all positional and mixed arguments before execution', async () => {
	const seen = [];
	const executed = [];
	const guarded = actionGuarded(
		function tool(...args) {
			executed.push(args);
			return this;
		},
		{
			actionClassifier: {
				predict: async (action) => {
					seen.push(action.function.arguments);
					return {
						label: flattenActionToText(action).includes(
							'BLOCK_MARKER',
						)
							? 'harmful'
							: 'safe',
						confidence: 0.99,
					};
				},
			},
		},
	);
	for (const args of [
		['BLOCK_MARKER'],
		['BLOCK_MARKER', { harmless: true }],
		[{ nested: [{ command: 'BLOCK_MARKER' }] }],
		[['BLOCK_MARKER']],
	]) {
		await assert.rejects(guarded(...args), HarmfulActionError);
	}
	assert.deepEqual(executed, []);
	const receiver = {};
	assert.equal(await guarded.call(receiver, { command: 'safe' }), receiver);
	assert.deepEqual(seen.at(-1), { command: 'safe' });
});

for (const confThreshold of [NaN, Infinity, -1, 1.01, '0.5', null, true]) {
	test(`invalid threshold is rejected: ${String(confThreshold)}`, () => {
		assert.throws(
			() => actionGuarded(() => {}, { confThreshold }),
			/threshold/i,
		);
	});
}

for (const jsonArguments of [false, true]) {
	test(`nested content survives preprocessing (JSON string=${jsonArguments})`, () => {
		const argumentsObject = {
			input: { command: 'BLOCK_MARKER' },
			rows: [{ value: 'ALSO_VISIBLE' }],
			nullValue: null,
			primitive: 42,
		};
		const text = flattenActionToText({
			type: 'function',
			function: {
				name: 'tool',
				arguments: jsonArguments
					? JSON.stringify(argumentsObject)
					: argumentsObject,
			},
		});
		assert.match(text, /BLOCK_MARKER/);
		assert.match(text, /ALSO_VISIBLE/);
		assert.match(text, /primitive=42/);
		assert.match(text, /nullValue=null/);
		assert.doesNotMatch(text, /\[object Object\]/);
	});
}

test('non-serializable nested arguments fail closed', () => {
	const cyclic = {};
	cyclic.self = cyclic;
	assert.throws(
		() =>
			flattenActionToText({
				type: 'function',
				function: { name: 'tool', arguments: { cyclic } },
			}),
		/circular/i,
	);
});
