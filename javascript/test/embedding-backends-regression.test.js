import assert from 'node:assert/strict';
import {
	mkdir,
	mkdtemp,
	readFile,
	readdir,
	rm,
	writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
	DEFAULT_EMBED_MODEL_NAME,
	EmbeddingModel,
	defaultOnnxAssetUrl,
	resolveOnnxModelFiles,
} from '../src/runtime-utils.js';

const EMBEDDING_KEY_ENV = 'EMBEDDING_API_KEY';
const OPENAI_KEY_ENV = 'OPENAI_API_KEY';
const EMBEDDING_ENV_NAMES = [
	'AAG_EMBED_GGUF',
	'AAG_EMBED_ONNX',
	'EMBED_MODEL_NAME',
	'EMBEDDING_BASE_URL',
	EMBEDDING_KEY_ENV,
	OPENAI_KEY_ENV,
];

function captureEmbeddingEnv() {
	return Object.fromEntries(
		EMBEDDING_ENV_NAMES.map((name) => [name, process.env[name]]),
	);
}

function clearEmbeddingEnv() {
	for (const name of EMBEDDING_ENV_NAMES) {
		delete process.env[name];
	}
}

function restoreEmbeddingEnv(snapshot) {
	for (const [name, value] of Object.entries(snapshot)) {
		if (value === undefined) {
			delete process.env[name];
		} else {
			process.env[name] = value;
		}
	}
}

function makeDirectOnnxModel({ output, inputNames, tokenizer }) {
	class FakeTensor {
		constructor(type, data, dims) {
			this.type = type;
			this.data = data;
			this.dims = dims;
		}
	}

	const session = {
		inputNames: inputNames ?? [
			'input_ids',
			'attention_mask',
			'token_type_ids',
		],
		outputNames: ['embedding'],
		run: async () => output,
	};
	const model = new EmbeddingModel(undefined, {
		ortModule: { Tensor: FakeTensor },
	});
	model.backend = 'onnx';
	model.onnxSessionPromise = Promise.resolve(session);
	model.onnxTokenizer = tokenizer ?? {
		encode: async () => ({
			ids: [1, 2],
			attention_mask: [1, 1],
			type_ids: [0, 0],
		}),
	};
	model.onnxTokenizerMetadata = { paddingId: 0, separatorId: 102 };
	return model;
}

test('every API environment variable selects API and AAG_EMBED_ONNX overrides each one', async (t) => {
	for (const [name, value] of [
		['EMBED_MODEL_NAME', 'configured-model'],
		['EMBEDDING_BASE_URL', 'http://localhost:1234/v1'],
		[EMBEDDING_KEY_ENV, 'configured'],
		[OPENAI_KEY_ENV, 'configured'],
	]) {
		await t.test(name, () => {
			const originalEnv = captureEmbeddingEnv();
			clearEmbeddingEnv();
			try {
				process.env[name] = value;
				assert.equal(new EmbeddingModel().backend, 'api');
				process.env.AAG_EMBED_ONNX = '/tmp/custom.onnx';
				assert.equal(new EmbeddingModel().backend, 'onnx');
			} finally {
				restoreEmbeddingEnv(originalEnv);
			}
		});
	}
});

test('explicit model argument remains an API-compatible override', () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	try {
		const model = new EmbeddingModel('explicit-model');
		assert.equal(model.backend, 'api');
		assert.equal(model.modelName, 'explicit-model');
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('backend precedence is model, then API config, then default ONNX', () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	try {
		process.env.EMBED_MODEL_NAME = 'preferred-api-model';
		process.env.OPENAI_API_KEY = 'configured';
		const modelConfigured = new EmbeddingModel();
		assert.equal(modelConfigured.backend, 'api');
		assert.equal(modelConfigured.modelName, 'preferred-api-model');

		clearEmbeddingEnv();
		process.env.EMBEDDING_API_KEY = 'configured';
		const apiConfigured = new EmbeddingModel();
		assert.equal(apiConfigured.backend, 'api');
		assert.equal(apiConfigured.modelName, DEFAULT_EMBED_MODEL_NAME);

		clearEmbeddingEnv();
		const defaultModel = new EmbeddingModel();
		assert.equal(defaultModel.backend, 'onnx');
		assert.equal(defaultModel.modelName, DEFAULT_EMBED_MODEL_NAME);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('legacy model plus either API key environment calls embedding API', async (t) => {
	for (const keyEnv of [EMBEDDING_KEY_ENV, OPENAI_KEY_ENV]) {
		await t.test(keyEnv, async () => {
			const originalEnv = captureEmbeddingEnv();
			clearEmbeddingEnv();
			process.env.EMBED_MODEL_NAME = 'legacy-api-model';
			process.env[keyEnv] = 'configured';
			const configs = [];
			const payloads = [];
			try {
				const model = new EmbeddingModel(undefined, {
					clientFactory: (config) => {
						configs.push(config);
						return {
							embeddings: {
								create: async (payload) => {
									payloads.push(payload);
									return {
										data: [{ embedding: [0.25, 0.75] }],
									};
								},
							},
						};
					},
				});

				const embeddings = await model.encode(['hello']);

				assert.equal(model.backend, 'api');
				assert.equal(model.modelName, 'legacy-api-model');
				assert.deepEqual(configs, [{ apiKey: 'configured' }]);
				assert.deepEqual(payloads, [
					{
						model: 'legacy-api-model',
						input: ['hello'],
						encoding_format: 'float',
					},
				]);
				assert.deepEqual(embeddings, [[0.25, 0.75]]);
			} finally {
				restoreEmbeddingEnv(originalEnv);
			}
		});
	}
});

test('API encode preserves payload, credential precedence, and client caching', async () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	process.env.EMBED_MODEL_NAME = 'api-model';
	process.env.EMBEDDING_BASE_URL = 'http://localhost:1234/v1';
	process.env[EMBEDDING_KEY_ENV] = 'primary';
	process.env[OPENAI_KEY_ENV] = 'fallback';
	const configs = [];
	const payloads = [];
	try {
		const model = new EmbeddingModel(undefined, {
			clientFactory: (config) => {
				configs.push(config);
				return {
					embeddings: {
						create: async (payload) => {
							payloads.push(payload);
							return {
								data: [
									{ embedding: [0.1, 0.2] },
									{ embedding: [0.3, 0.4] },
								],
							};
						},
					},
				};
			},
		});

		const embeddings = await model.encode(['hello', 'world']);
		const cachedClient = await model.getClient();

		assert.equal(cachedClient, model.client);
		assert.deepEqual(configs, [
			{ apiKey: 'primary', baseURL: 'http://localhost:1234/v1' },
		]);
		assert.deepEqual(payloads, [
			{
				model: 'api-model',
				input: ['hello', 'world'],
				encoding_format: 'float',
			},
		]);
		assert.deepEqual(embeddings, [
			[0.1, 0.2],
			[0.3, 0.4],
		]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('default resolver downloads only missing cache assets', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(
		path.join(os.tmpdir(), 'aag-js-partial-cache-'),
	);
	clearEmbeddingEnv();
	try {
		const cacheDir = path.join(
			tempHome,
			'.cache',
			'agent-action-guard',
			'all-MiniLM-L6-v2',
		);
		await mkdir(cacheDir, { recursive: true });
		await writeFile(path.join(cacheDir, 'model.onnx'), 'cached-model');
		await writeFile(path.join(cacheDir, 'tokenizer.json'), '{}');
		const requested = [];

		const files = await resolveOnnxModelFiles({
			homeDir: tempHome,
			fetchImpl: async (url) => {
				requested.push(url);
				return new globalThis.Response('{}');
			},
		});

		assert.deepEqual(requested, [
			defaultOnnxAssetUrl('tokenizer_config.json'),
		]);
		assert.equal(await readFile(files.modelPath, 'utf8'), 'cached-model');
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('default resolver surfaces download failure and removes temporary files', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(
		path.join(os.tmpdir(), 'aag-js-download-error-'),
	);
	clearEmbeddingEnv();
	try {
		await assert.rejects(
			resolveOnnxModelFiles({
				homeDir: tempHome,
				fetchImpl: async () =>
					new globalThis.Response('failed', { status: 503 }),
			}),
			/Failed to download embedding asset.*HTTP 503/,
		);
		const cacheDir = path.join(
			tempHome,
			'.cache',
			'agent-action-guard',
			'all-MiniLM-L6-v2',
		);
		assert.deepEqual(await readdir(cacheDir), []);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('default resolver gives actionable error when fetch is unavailable', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(path.join(os.tmpdir(), 'aag-js-no-fetch-'));
	clearEmbeddingEnv();
	try {
		await assert.rejects(
			resolveOnnxModelFiles({ homeDir: tempHome, fetchImpl: null }),
			/a fetch implementation is required/i,
		);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('2D ONNX outputs are normalized and zero vectors remain stable', async () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	try {
		const model = makeDirectOnnxModel({
			output: {
				embedding: {
					dims: [2, 2],
					data: Float32Array.from([3, 4, 0, 0]),
				},
			},
		});

		const embeddings = await model.encode(['vector', 'zero']);

		assert.deepEqual(
			embeddings[0].map((value) => Number(value.toFixed(8))),
			[0.6, 0.8],
		);
		assert.deepEqual(embeddings[1], [0, 0]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('ONNX encode stringifies non-string inputs', async () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	const received = [];
	try {
		const model = makeDirectOnnxModel({
			output: {
				embedding: {
					dims: [2, 2],
					data: Float32Array.from([1, 0, 0, 1]),
				},
			},
			tokenizer: {
				encode: async (value) => {
					received.push(value);
					return { ids: [1], attention_mask: [1], type_ids: [0] };
				},
			},
		});

		await model.encode([123, null]);

		assert.deepEqual(received, ['123', 'null']);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('prepareEncoding pads defaults and preserves separator when truncating', () => {
	const model = new EmbeddingModel();
	const padded = model.prepareEncoding({ ids: [10, 11] }, 4, {
		paddingId: 0,
		separatorId: 102,
	});
	assert.deepEqual(padded, {
		ids: [10, 11, 0, 0],
		attentionMask: [1, 1, 0, 0],
		typeIds: [0, 0, 0, 0],
	});

	const longIds = Array.from({ length: 300 }, (_, index) => index + 1);
	longIds[longIds.length - 1] = 102;
	const truncated = model.prepareEncoding({ ids: longIds }, 256, {
		paddingId: 0,
		separatorId: 102,
	});
	assert.equal(truncated.ids.length, 256);
	assert.equal(truncated.ids[255], 102);
	assert.equal(truncated.attentionMask[255], 1);
});

test('ONNX rejects unsupported inputs, missing outputs, and invalid output ranks', async (t) => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	try {
		await t.test('unsupported input', async () => {
			const model = makeDirectOnnxModel({
				inputNames: ['input_ids', 'unsupported_input'],
				output: {
					embedding: {
						dims: [1, 2],
						data: Float32Array.from([1, 0]),
					},
				},
			});
			await assert.rejects(
				model.encode(['hello']),
				/Unsupported ONNX embedding model input/,
			);
		});

		await t.test('missing output', async () => {
			const model = makeDirectOnnxModel({ output: {} });
			model.onnxSessionPromise = Promise.resolve({
				inputNames: ['input_ids'],
				outputNames: [],
				run: async () => ({}),
			});
			await assert.rejects(
				model.encode(['hello']),
				/returned no outputs/,
			);
		});

		for (const dims of [[2], [1, 1, 1, 1]]) {
			await t.test(`invalid output rank ${dims.length}`, async () => {
				const model = makeDirectOnnxModel({
					output: {
						embedding: { dims, data: Float32Array.from([1, 2]) },
					},
				});
				await assert.rejects(
					model.encode(['hello']),
					/must return a 2D sentence embedding or 3D token embeddings/,
				);
			});
		}
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('local ONNX runtime reports each missing model/tokenizer sidecar before inference', async (t) => {
	const originalEnv = captureEmbeddingEnv();
	const tempDir = await mkdtemp(
		path.join(os.tmpdir(), 'aag-js-missing-assets-'),
	);
	clearEmbeddingEnv();
	try {
		for (const [missingName, expected] of [
			['model.onnx', /ONNX embedding model not found/],
			['tokenizer.json', /ONNX embedding tokenizer not found/],
			[
				'tokenizer_config.json',
				/ONNX embedding tokenizer config not found/,
			],
		]) {
			await t.test(missingName, async () => {
				for (const filename of [
					'model.onnx',
					'tokenizer.json',
					'tokenizer_config.json',
				]) {
					if (filename === missingName) {
						await rm(path.join(tempDir, filename), { force: true });
					} else {
						await writeFile(path.join(tempDir, filename), '{}');
					}
				}
				const model = new EmbeddingModel(undefined, {
					assetResolver: async () => ({
						modelPath: path.join(tempDir, 'model.onnx'),
						tokenizerPath: path.join(tempDir, 'tokenizer.json'),
						tokenizerConfigPath: path.join(
							tempDir,
							'tokenizer_config.json',
						),
					}),
				});
				await assert.rejects(model.getOnnxRuntime(), expected);
			});
		}
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempDir, { recursive: true, force: true });
	}
});

test('concurrent first-use cache resolution succeeds without partial assets', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(
		path.join(os.tmpdir(), 'aag-js-concurrent-cache-'),
	);
	clearEmbeddingEnv();
	try {
		const fetchImpl = async (url) => {
			await new Promise((resolve) => globalThis.setTimeout(resolve, 2));
			return new globalThis.Response(
				`asset:${path.basename(new globalThis.URL(url).pathname)}`,
			);
		};

		const [first, second] = await Promise.all([
			resolveOnnxModelFiles({ homeDir: tempHome, fetchImpl }),
			resolveOnnxModelFiles({ homeDir: tempHome, fetchImpl }),
		]);

		assert.deepEqual(first, second);
		assert.equal(
			await readFile(first.modelPath, 'utf8'),
			'asset:model.onnx',
		);
		assert.equal(
			await readFile(first.tokenizerPath, 'utf8'),
			'asset:tokenizer.json',
		);
		assert.equal(
			await readFile(first.tokenizerConfigPath, 'utf8'),
			'asset:tokenizer_config.json',
		);
		assert.deepEqual(
			(await readdir(path.dirname(first.modelPath))).sort(),
			['model.onnx', 'tokenizer.json', 'tokenizer_config.json'],
		);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});
