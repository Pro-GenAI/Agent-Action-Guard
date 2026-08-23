import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';

import {
	resolveGgufModelFile,
	resolveOnnxModelFiles,
} from '../src/runtime-utils.js';

const REAL_ONNX_BASE_URL =
	'https://huggingface.co/llmware/bling-tiny-llama-onnx/resolve/main/';
const REAL_GGUF_URL =
	'https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf';

function restoreEnv(name, value) {
	if (value === undefined) {
		delete process.env[name];
	} else {
		process.env[name] = value;
	}
}


test('real ONNX URL downloads once and then uses cache', async (t) => {
	const original = process.env.AAG_EMBED_ONNX;
	t.after(() => restoreEnv('AAG_EMBED_ONNX', original));
	process.env.AAG_EMBED_ONNX = REAL_ONNX_BASE_URL;

	const files = await resolveOnnxModelFiles();
	const firstStat = await stat(files.modelPath);
	assert.ok(firstStat.size > 100_000);
	assert.ok((await stat(files.tokenizerPath)).size > 100_000);
	assert.ok((await stat(files.tokenizerConfigPath)).size > 100);

	const cached = await resolveOnnxModelFiles({
		fetchImpl: async () => {
			throw new Error('cached ONNX assets unexpectedly used the network');
		},
	});
	const secondStat = await stat(cached.modelPath);
	assert.equal(cached.modelPath, files.modelPath);
	assert.equal(secondStat.ino, firstStat.ino);
	assert.equal(secondStat.mtimeMs, firstStat.mtimeMs);
});


test('real GGUF URL downloads once and then uses cache', async (t) => {
	const original = process.env.AAG_EMBED_GGUF;
	t.after(() => restoreEnv('AAG_EMBED_GGUF', original));
	process.env.AAG_EMBED_GGUF = REAL_GGUF_URL;

	const modelPath = await resolveGgufModelFile();
	const firstStat = await stat(modelPath);
	assert.equal(path.basename(modelPath), 'mmproj-tinygemma3.gguf');
	assert.equal(firstStat.size, 1_039_072);
	assert.equal((await readFile(modelPath)).subarray(0, 4).toString('ascii'), 'GGUF');

	const cachedPath = await resolveGgufModelFile({
		fetchImpl: async () => {
			throw new Error('cached GGUF asset unexpectedly used the network');
		},
	});
	const secondStat = await stat(cachedPath);
	assert.equal(cachedPath, modelPath);
	assert.equal(secondStat.ino, firstStat.ino);
	assert.equal(secondStat.mtimeMs, firstStat.mtimeMs);
});
