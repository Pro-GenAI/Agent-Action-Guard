import assert from 'node:assert/strict';
import test from 'node:test';

import {
	buildScenarios,
	collectDependencyVersions,
	sampleVersions,
	selectDependencyVersions,
} from '../scripts/test-dependency-matrix.js';

test('sampleVersions keeps dependency bounds and spreads intermediate samples', () => {
	assert.deepEqual(sampleVersions(['1', '2', '3', '4', '5'], 3), [
		'1',
		'3',
		'5',
	]);
	assert.deepEqual(sampleVersions(['1', '2'], 3), ['1', '2']);
});

test('selectDependencyVersions always includes the npm latest release', () => {
	assert.deepEqual(
		selectDependencyVersions(
			['4.0.0', '5.0.0', '6.0.0', '7.5.0'],
			'7.5.0',
			3,
			'openai',
		),
		['4.0.0', '6.0.0', '7.5.0'],
	);
});

test('selectDependencyVersions rejects a latest release outside the supported range', () => {
	assert.throws(
		() =>
			selectDependencyVersions(
				['4.0.0', '5.0.0', '6.0.0'],
				'7.5.0',
				3,
				'openai',
			),
		/openai@latest \(7\.5\.0\) is outside the supported range/,
	);
});

test('buildScenarios creates exact bounds and sampled one-at-a-time profiles', () => {
	const versions = {
		'@huggingface/tokenizers': ['0.1.0', '0.1.2', '0.1.3'],
		'onnxruntime-node': ['1.14.0', '1.20.0', '1.22.0'],
		openai: ['4.0.0', '6.0.0', '7.5.0'],
	};
	const scenarios = buildScenarios(versions);

	assert.equal(scenarios.length, 5);
	assert.deepEqual(scenarios[0].dependencies, {
		'@huggingface/tokenizers': '0.1.0',
		'onnxruntime-node': '1.14.0',
		openai: '4.0.0',
	});
	assert.deepEqual(scenarios[1].dependencies, {
		'@huggingface/tokenizers': '0.1.3',
		'onnxruntime-node': '1.22.0',
		openai: '7.5.0',
	});

	const onnxScenario = scenarios.find(
		(scenario) => scenario.name === 'dep-onnxruntime-node-1.20.0',
	);
	assert.deepEqual(onnxScenario.dependencies, {
		'@huggingface/tokenizers': '0.1.0',
		'onnxruntime-node': '1.20.0',
		openai: '4.0.0',
	});
	assert.deepEqual(onnxScenario.expectedVersions, onnxScenario.dependencies);
});

test('buildScenarios supports a fast bounds-only matrix', () => {
	const versions = {
		'@huggingface/tokenizers': ['0.1.0', '0.1.3'],
		'onnxruntime-node': ['1.14.0', '1.22.0'],
		openai: ['4.0.0', '7.5.0'],
	};

	assert.deepEqual(
		buildScenarios(versions, { boundsOnly: true }).map(({ name }) => name),
		['base-min', 'base-max'],
	);
});

test('collectDependencyVersions finds deduped and nested dependency releases', () => {
	const tree = {
		dependencies: {
			'onnxruntime-node': { version: '1.14.0' },
			'agent-action-guard': {
				version: '0.1.0',
				dependencies: {
					'onnxruntime-node': { version: '1.14.0' },
				},
			},
			other: {
				version: '1.0.0',
				dependencies: {
					'onnxruntime-node': { version: '1.22.0' },
				},
			},
		},
	};

	assert.deepEqual(collectDependencyVersions(tree, 'onnxruntime-node'), [
		'1.14.0',
		'1.22.0',
	]);
});
