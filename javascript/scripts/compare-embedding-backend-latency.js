import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';

import { ActionClassifier } from '../src/action-classifier.js';
import { loadActions } from '../src/cli.js';
import {
	DEFAULT_EMBED_MODEL_NAME,
	EmbeddingModel,
} from '../src/runtime-utils.js';

export const DEFAULT_ACTIONS = [
	{
		type: 'function',
		function: { name: 'get_weather', arguments: { city: 'London' } },
	},
	{
		type: 'function',
		function: {
			name: 'send_email',
			arguments: { to: 'user@example.com', subject: 'Status update' },
		},
	},
	{
		type: 'function',
		function: { name: 'delete_user', arguments: { user_id: '123' } },
	},
	{
		type: 'function',
		function: {
			name: 'export_records',
			arguments: { dataset: 'customers', destination: 'external' },
		},
	},
];

export function percentile(values, fraction) {
	if (values.length === 0) {
		throw new Error('values must not be empty');
	}
	const ordered = [...values].sort((left, right) => left - right);
	const position = (ordered.length - 1) * fraction;
	const lower = Math.floor(position);
	const upper = Math.min(lower + 1, ordered.length - 1);
	const weight = position - lower;
	return ordered[lower] * (1 - weight) + ordered[upper] * weight;
}

export function summarizeDurations(durationsMs, actionCount) {
	if (durationsMs.length === 0) {
		throw new Error('durationsMs must not be empty');
	}
	if (!Number.isInteger(actionCount) || actionCount <= 0) {
		throw new Error('actionCount must be a positive integer');
	}

	const totalMs = durationsMs.reduce((sum, value) => sum + value, 0);
	const measuredActions = actionCount * durationsMs.length;
	return {
		meanRunMs: totalMs / durationsMs.length,
		p50RunMs: percentile(durationsMs, 0.5),
		p95RunMs: percentile(durationsMs, 0.95),
		meanActionMs: totalMs / measuredActions,
		actionsPerSecond: measuredActions / (totalMs / 1000),
	};
}

export async function benchmarkClassifier(
	classifier,
	actions,
	{ iterations, warmup, batchSize, timer = () => performance.now() },
) {
	for (let index = 0; index < warmup; index += 1) {
		await classifier.predictBatch(actions, { batchSize });
	}

	const durationsMs = [];
	for (let index = 0; index < iterations; index += 1) {
		const start = timer();
		await classifier.predictBatch(actions, { batchSize });
		durationsMs.push(timer() - start);
	}
	return summarizeDurations(durationsMs, actions.length);
}

export function loadBenchmarkActions(file, limit) {
	const actions = file ? loadActions({ file }) : [...DEFAULT_ACTIONS];
	const selected = actions.slice(0, limit);
	if (selected.length === 0) {
		throw new Error('No actions available for the benchmark');
	}
	return selected;
}

export function createClassifiers(apiModel) {
	const onnxEmbeddingModel = new EmbeddingModel();
	onnxEmbeddingModel.backend = 'onnx';

	const apiEmbeddingModel = new EmbeddingModel(apiModel);
	apiEmbeddingModel.backend = 'api';

	return {
		onnxClassifier: new ActionClassifier({
			embeddingModel: onnxEmbeddingModel,
		}),
		apiClassifier: new ActionClassifier({
			embeddingModel: apiEmbeddingModel,
		}),
	};
}

function formatRow(name, stats) {
	return `${name.padEnd(8)} ${stats.meanRunMs.toFixed(2).padStart(9)} ms ${stats.p50RunMs
		.toFixed(2)
		.padStart(
			9,
		)} ms ${stats.p95RunMs.toFixed(2).padStart(9)} ms ${stats.meanActionMs
		.toFixed(2)
		.padStart(10)} ms ${stats.actionsPerSecond.toFixed(2).padStart(12)}`;
}

export function formatReport(onnxStats, apiStats, actionCount) {
	const ratio = apiStats.meanActionMs / onnxStats.meanActionMs;
	const comparison =
		ratio >= 1
			? `API / ONNX mean per-action latency: ${ratio.toFixed(2)}x slower`
			: `API / ONNX mean per-action latency: ${(1 / ratio).toFixed(2)}x faster`;
	return [
		`Measured actions per run: ${actionCount}`,
		`${'Backend'.padEnd(8)} ${'Mean/run'.padStart(12)} ${'p50/run'.padStart(12)} ${'p95/run'.padStart(12)} ${'Mean/action'.padStart(13)} ${'Actions/s'.padStart(12)}`,
		formatRow('ONNX', onnxStats),
		formatRow('API', apiStats),
		'',
		comparison,
	].join('\n');
}

function positiveInteger(value, optionName, { allowZero = false } = {}) {
	const number = Number(value);
	if (!Number.isInteger(number) || number < (allowZero ? 0 : 1)) {
		throw new Error(
			`${optionName} must be ${allowZero ? 'zero or greater' : 'a positive integer'}`,
		);
	}
	return number;
}

export async function main(argv = process.argv.slice(2)) {
	const { values } = parseArgs({
		args: argv,
		options: {
			file: { type: 'string' },
			limit: { type: 'string', default: '16' },
			iterations: { type: 'string', default: '5' },
			warmup: { type: 'string', default: '1' },
			'batch-size': { type: 'string', default: '8' },
			'api-model': {
				type: 'string',
				default:
					process.env.EMBED_MODEL_NAME ?? DEFAULT_EMBED_MODEL_NAME,
			},
			help: { type: 'boolean', short: 'h' },
		},
	});

	if (values.help) {
		console.log(
			`Usage: npm run test:latency-backends -- [options]\n\nOptions:\n  --file FILE         JSON array or JSONL actions\n  --limit N           Maximum actions (default: 16)\n  --iterations N      Measured runs/backend (default: 5)\n  --warmup N          Warmup runs; 0 includes cold start (default: 1)\n  --batch-size N      Actions per vectorized batch (default: 8)\n  --api-model MODEL   API embedding model name`,
		);
		return 0;
	}

	const limit = positiveInteger(values.limit, '--limit');
	const iterations = positiveInteger(values.iterations, '--iterations');
	const warmup = positiveInteger(values.warmup, '--warmup', {
		allowZero: true,
	});
	const batchSize = positiveInteger(values['batch-size'], '--batch-size');
	if (!process.env.EMBEDDING_API_KEY && !process.env.OPENAI_API_KEY) {
		throw new Error(
			'Set EMBEDDING_API_KEY or OPENAI_API_KEY for the API benchmark. For an unauthenticated local endpoint, EMBEDDING_API_KEY=dummy is sufficient.',
		);
	}

	const actions = loadBenchmarkActions(values.file, limit);
	const { onnxClassifier, apiClassifier } = createClassifiers(
		values['api-model'],
	);
	console.log(
		`Benchmarking ${actions.length} actions, ${iterations} measured runs/backend, ${warmup} warmup run(s), batch size ${batchSize}.`,
	);
	if (warmup) {
		console.log(
			'Warmup excludes model/client/session initialization from measured latency.',
		);
	}

	const options = { iterations, warmup, batchSize };
	const onnxStats = await benchmarkClassifier(
		onnxClassifier,
		actions,
		options,
	);
	const apiStats = await benchmarkClassifier(apiClassifier, actions, options);
	console.log();
	console.log(formatReport(onnxStats, apiStats, actions.length));
	return 0;
}

const isMain =
	!process.env.NODE_TEST_CONTEXT &&
	process.argv[1] &&
	import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;

if (isMain) {
	main().catch((error) => {
		console.error(error.message);
		process.exitCode = 1;
	});
}
