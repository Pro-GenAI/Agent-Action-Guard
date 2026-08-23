import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
	DEFAULT_EMBED_MODEL_NAME,
	EmbeddingModel,
	defaultOnnxAssetUrl,
	resolveGgufModelFile,
	resolveOnnxModelFiles,
} from '../src/runtime-utils.js';

const EMBEDDING_KEY_ENV = 'EMBEDDING_API_KEY'
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

test('EmbeddingModel defaults to ONNX when no embedding configuration exists', () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	try {
		const model = new EmbeddingModel();
		assert.equal(model.backend, 'onnx');
		assert.equal(model.modelName, DEFAULT_EMBED_MODEL_NAME);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('EmbeddingModel uses API when API model configuration exists', () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	process.env.EMBED_MODEL_NAME = 'configured-api-model';
	try {
		assert.equal(new EmbeddingModel().backend, 'api');
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('AAG_EMBED_ONNX takes precedence over API configuration', () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	process.env.AAG_EMBED_ONNX = '/tmp/custom-embedding.onnx';
	process.env.EMBED_MODEL_NAME = 'configured-api-model';
	try {
		assert.equal(new EmbeddingModel().backend, 'onnx');
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('AAG_EMBED_GGUF takes precedence over ONNX and API configuration', () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	process.env.AAG_EMBED_GGUF = '/tmp/custom-embedding.gguf';
	process.env.AAG_EMBED_ONNX = '/tmp/custom-embedding.onnx';
	process.env.EMBED_MODEL_NAME = 'configured-api-model';
	try {
		assert.equal(new EmbeddingModel().backend, 'gguf');
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('resolveGgufModelFile accepts a GGUF filepath or directory', async () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	try {
		process.env.AAG_EMBED_GGUF = '/tmp/custom.gguf';
		assert.equal(await resolveGgufModelFile(), path.resolve('/tmp/custom.gguf'));
		process.env.AAG_EMBED_GGUF = '/tmp/models';
		assert.equal(
			await resolveGgufModelFile(),
			path.resolve('/tmp/models/model.gguf'),
		);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('ONNX URL downloads assets into normalized cache names', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(path.join(os.tmpdir(), 'aag-js-url-cache-'));
	const requested = [];
	clearEmbeddingEnv();
	process.env.AAG_EMBED_ONNX =
		'https://Example.com/models/My%20Model%20(v1).onnx?download=true';
	try {
		const files = await resolveOnnxModelFiles({
			homeDir: tempHome,
			fetchImpl: async (url) => {
				requested.push(url);
				return new globalThis.Response('asset');
			},
		});
		assert.equal(path.basename(files.modelPath), 'my-model-v1.onnx');
		assert.equal(path.basename(files.tokenizerPath), 'tokenizer.json');
		assert.equal(path.basename(path.dirname(files.modelPath)), path.basename(path.dirname(files.tokenizerPath)));
		assert.match(path.basename(path.dirname(files.modelPath)), /^[a-z0-9._-]+$/);
		assert.deepEqual(requested, [
			'https://example.com/models/My%20Model%20(v1).onnx?download=true',
			'https://example.com/models/tokenizer.json?download=true',
			'https://example.com/models/tokenizer_config.json?download=true',
		]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('Hugging Face repo IDs resolve standard ONNX and GGUF assets', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(path.join(os.tmpdir(), 'aag-js-hf-cache-'));
	const requested = [];
	const fetchImpl = async (url) => {
		requested.push(url);
		return new globalThis.Response('asset');
	};
	clearEmbeddingEnv();
	try {
		process.env.AAG_EMBED_ONNX = 'acme/example-onnx';
		await resolveOnnxModelFiles({ homeDir: tempHome, fetchImpl });
		assert.deepEqual(requested, [
			'https://huggingface.co/acme/example-onnx/resolve/main/model.onnx',
			'https://huggingface.co/acme/example-onnx/resolve/main/tokenizer.json',
			'https://huggingface.co/acme/example-onnx/resolve/main/tokenizer_config.json',
		]);

		requested.length = 0;
		delete process.env.AAG_EMBED_ONNX;
		process.env.AAG_EMBED_GGUF = 'acme/example-gguf';
		const modelPath = await resolveGgufModelFile({ homeDir: tempHome, fetchImpl });
		assert.equal(path.basename(modelPath), 'model.gguf');
		assert.deepEqual(requested, [
			'https://huggingface.co/acme/example-gguf/resolve/main/model.gguf',
		]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('GGUF URL uses a normalized downloaded filename', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(path.join(os.tmpdir(), 'aag-js-gguf-url-'));
	const requested = [];
	clearEmbeddingEnv();
	process.env.AAG_EMBED_GGUF =
		'https://models.example/Embed%20Model%20(Q8_0).GGUF?version=1';
	try {
		const modelPath = await resolveGgufModelFile({
			homeDir: tempHome,
			fetchImpl: async (url) => {
				requested.push(url);
				return new globalThis.Response('asset');
			},
		});
		assert.equal(path.basename(modelPath), 'embed-model-q8_0.gguf');
		assert.match(path.basename(path.dirname(modelPath)), /^[a-z0-9._-]+$/);
		assert.deepEqual(requested, [
			'https://models.example/Embed%20Model%20(Q8_0).GGUF?version=1',
		]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('resolveOnnxModelFiles accepts an ONNX filepath or directory', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempDir = await mkdtemp(path.join(os.tmpdir(), 'aag-js-paths-'));
	clearEmbeddingEnv();
	try {
		process.env.AAG_EMBED_ONNX = path.join(tempDir, 'custom.onnx');
		let files = await resolveOnnxModelFiles();
		assert.equal(files.modelPath, path.join(tempDir, 'custom.onnx'));
		assert.equal(files.tokenizerPath, path.join(tempDir, 'tokenizer.json'));
		assert.equal(
			files.tokenizerConfigPath,
			path.join(tempDir, 'tokenizer_config.json'),
		);

		process.env.AAG_EMBED_ONNX = tempDir;
		files = await resolveOnnxModelFiles();
		assert.equal(files.modelPath, path.join(tempDir, 'model.onnx'));
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempDir, { recursive: true, force: true });
	}
});

test('resolveOnnxModelFiles auto-downloads and caches default ONNX assets', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempHome = await mkdtemp(path.join(os.tmpdir(), 'aag-js-cache-'));
	const requested = [];
	clearEmbeddingEnv();
	try {
		const files = await resolveOnnxModelFiles({
			homeDir: tempHome,
			fetchImpl: async (url) => {
				requested.push(url);
				return new globalThis.Response(
					`asset:${path.basename(new globalThis.URL(url).pathname)}`,
				);
			},
		});

		assert.deepEqual(requested, [
			defaultOnnxAssetUrl('model.onnx'),
			defaultOnnxAssetUrl('tokenizer.json'),
			defaultOnnxAssetUrl('tokenizer_config.json'),
		]);
		assert.equal(
			await readFile(files.modelPath, 'utf8'),
			'asset:model.onnx',
		);
		assert.equal(
			await readFile(files.tokenizerPath, 'utf8'),
			'asset:tokenizer.json',
		);
		assert.equal(
			await readFile(files.tokenizerConfigPath, 'utf8'),
			'asset:tokenizer_config.json',
		);

		requested.length = 0;
		await resolveOnnxModelFiles({
			homeDir: tempHome,
			fetchImpl: async () => null,
		});
		assert.deepEqual(requested, []);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempHome, { recursive: true, force: true });
	}
});

test('EmbeddingModel generates normalized embeddings with local GGUF inference', async () => {
	const originalEnv = captureEmbeddingEnv();
	clearEmbeddingEnv();
	process.env.AAG_EMBED_GGUF = '/tmp/custom.gguf';
	try {
		const model = new EmbeddingModel();
		model.ggufRuntimePromise = Promise.resolve({
			context: {
				getEmbeddingFor: async (text) => ({
					vector: text === 'hello' ? [3, 4] : [0, 2],
				}),
			},
		});
		const embeddings = await model.encode(['hello', 'world']);
		assert.equal(model.backend, 'gguf');
		assert.deepEqual(embeddings, [[0.6, 0.8], [0, 1]]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
	}
});

test('EmbeddingModel generates normalized embeddings with local ONNX inference', async () => {
	const originalEnv = captureEmbeddingEnv();
	const tempDir = await mkdtemp(path.join(os.tmpdir(), 'aag-js-onnx-'));
	clearEmbeddingEnv();
	process.env.AAG_EMBED_ONNX = path.join(tempDir, 'model.onnx');
	try {
		await writeFile(path.join(tempDir, 'model.onnx'), 'fake model');
		await writeFile(
			path.join(tempDir, 'tokenizer.json'),
			JSON.stringify({ model: { vocab: { '[PAD]': 0, '[SEP]': 102 } } }),
		);
		await writeFile(
			path.join(tempDir, 'tokenizer_config.json'),
			JSON.stringify({ pad_token: '[PAD]', sep_token: '[SEP]' }),
		);

		class FakeTensor {
			constructor(type, data, dims) {
				this.type = type;
				this.data = data;
				this.dims = dims;
			}
		}

		const session = {
			inputNames: ['input_ids', 'attention_mask', 'token_type_ids'],
			outputNames: ['last_hidden_state'],
			run: async (feed) => {
				assert.deepEqual(feed.input_ids.dims, [2, 3]);
				assert.equal(feed.input_ids.type, 'int64');
				assert.deepEqual(Array.from(feed.attention_mask.data), [
					1n,
					1n,
					1n,
					1n,
					1n,
					0n,
				]);
				return {
					last_hidden_state: {
						dims: [2, 3, 2],
						data: Float32Array.from([
							2, 0, 2, 0, 2, 0, 0, 3, 0, 3, 99, 99,
						]),
					},
				};
			},
		};
		const ortModule = {
			Tensor: FakeTensor,
			InferenceSession: {
				create: async (filename, options) => {
					assert.equal(filename, path.join(tempDir, 'model.onnx'));
					assert.deepEqual(options, { executionProviders: ['cpu'] });
					return session;
				},
			},
		};
		const tokenizerFactory = async () => ({
			encode: async (text) =>
				text === 'hello'
					? {
							ids: [101, 10, 102],
							attention_mask: [1, 1, 1],
							type_ids: [0, 0, 0],
						}
					: {
							ids: [101, 102],
							attention_mask: [1, 1],
							type_ids: [0, 0],
						},
		});

		const model = new EmbeddingModel('configured-api-model', {
			ortModule,
			tokenizerFactory,
		});
		const embeddings = await model.encode(['hello', 'world']);

		assert.equal(model.backend, 'onnx');
		assert.deepEqual(embeddings, [
			[1, 0],
			[0, 1],
		]);
	} finally {
		restoreEmbeddingEnv(originalEnv);
		await rm(tempDir, { recursive: true, force: true });
	}
});
