import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import { mkdir, readFile, rename, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { URL, fileURLToPath } from 'node:url';

export const ALL_CLASSES = ['safe', 'harmful', 'unethical'];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const ONNX_MODEL_PATH = path.resolve(
	__dirname,
	'action_classifier_model.onnx',
);

export const DEFAULT_EMBED_MODEL_NAME =
	'sentence-transformers/all-MiniLM-L6-v2';
export const AAG_EMBED_GGUF_ENV = 'AAG_EMBED_GGUF';
export const AAG_EMBED_ONNX_ENV = 'AAG_EMBED_ONNX';

export function normalizeOnnxRuntimeModule(module) {
	return module?.InferenceSession ? module : (module?.default ?? module);
}

const DEFAULT_ONNX_REPO = 'onnx-models/all-MiniLM-L6-v2-onnx';
const HF_BASE_URL = 'https://huggingface.co';
const DEFAULT_MAX_SEQUENCE_LENGTH = 256;

export function defaultOnnxAssetUrl(filename) {
	return `${HF_BASE_URL}/${DEFAULT_ONNX_REPO}/resolve/main/${filename}`;
}

function isHttpUrl(value) {
	try {
		const parsed = new URL(value);
		return ['http:', 'https:'].includes(parsed.protocol) && Boolean(parsed.host);
	} catch {
		return false;
	}
}

function normalizeSourceComponent(value, fallback = 'source') {
	let normalized;
	try {
		normalized = decodeURIComponent(value);
	} catch {
		normalized = value;
	}
	normalized = normalized
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9._-]+/g, '-')
		.replace(/-+/g, '-')
		.replace(/-\./g, '.')
		.replace(/^[ ._-]+|[ ._-]+$/g, '');
	return normalized || fallback;
}

function normalizedUrlCacheName(url) {
	const parsed = new URL(url);
	const readable = normalizeSourceComponent(
		`${parsed.protocol.replace(':', '')}-${parsed.host}-${parsed.pathname}`,
		'remote',
	).slice(0, 96);
	parsed.hash = '';
	const fingerprint = createHash('sha256')
		.update(parsed.toString())
		.digest('hex')
		.slice(0, 12);
	return `${readable}-${fingerprint}`;
}

function looksLikeHfRepoId(value) {
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
		return false;
	}
	const candidate = path.resolve(value);
	if (fs.existsSync(candidate) || fs.existsSync(path.dirname(candidate))) {
		return false;
	}
	return !value.startsWith('/') && !value.startsWith('./') && !value.startsWith('../') && !value.startsWith('~') && !value.includes('\\');
}

function hfRepoBaseUrl(repoId) {
	return `${HF_BASE_URL}/${repoId}/resolve/main/`;
}

function normalizeHfSourceUrl(url) {
	const parsed = new URL(url);
	if (!['huggingface.co', 'www.huggingface.co'].includes(parsed.hostname.toLowerCase())) {
		return { url: parsed.toString(), isBase: parsed.pathname.endsWith('/') };
	}

	const parts = parsed.pathname.split('/').filter(Boolean);
	if (parts.length === 2) {
		parsed.pathname = `/${parts[0]}/${parts[1]}/resolve/main/`;
		return { url: parsed.toString(), isBase: true };
	}
	if (parts.length >= 4 && parts[2] === 'blob') {
		parsed.pathname = `/${[parts[0], parts[1], 'resolve', ...parts.slice(3)].join('/')}`;
		return { url: parsed.toString(), isBase: false };
	}
	if (parts.length >= 4 && parts[2] === 'tree') {
		parsed.pathname = `/${[parts[0], parts[1], 'resolve', ...parts.slice(3)].join('/')}/`;
		return { url: parsed.toString(), isBase: true };
	}
	if (parts.length === 4 && parts[2] === 'resolve') {
		parsed.pathname = `/${parts.join('/')}/`;
		return { url: parsed.toString(), isBase: true };
	}
	return { url: parsed.toString(), isBase: parsed.pathname.endsWith('/') };
}

function remoteModelUrl(source, defaultFilename) {
	const sourceUrl = looksLikeHfRepoId(source) ? hfRepoBaseUrl(source) : source;
	const normalized = normalizeHfSourceUrl(sourceUrl);
	if (!normalized.isBase) {
		return normalized.url;
	}
	const parsed = new URL(normalized.url);
	parsed.pathname = `${parsed.pathname.endsWith('/') ? parsed.pathname : `${parsed.pathname}/`}${defaultFilename}`;
	return parsed.toString();
}

function siblingUrl(url, filename) {
	const parsed = new URL(url);
	const parts = parsed.pathname.split('/');
	parts[parts.length - 1] = filename;
	parsed.pathname = parts.join('/');
	return parsed.toString();
}

function remoteCachePath(sourceUrl, kind, defaultFilename, homeDir = os.homedir()) {
	const parsed = new URL(sourceUrl);
	let basename = path.posix.basename(parsed.pathname);
	try {
		basename = decodeURIComponent(basename);
	} catch {
		// Keep the encoded basename when it is not valid percent-encoding.
	}
	let normalizedFilename = normalizeSourceComponent(basename, defaultFilename);
	const expectedExtension = path.extname(defaultFilename).toLowerCase();
	if (expectedExtension && !normalizedFilename.endsWith(expectedExtension)) {
		normalizedFilename += expectedExtension;
	}
	return path.join(
		homeDir,
		'.cache',
		'agent-action-guard',
		kind,
		normalizedUrlCacheName(sourceUrl),
		normalizedFilename,
	);
}

async function downloadFile(url, destination, fetchImpl = globalThis.fetch) {
	if (fs.existsSync(destination)) {
		return;
	}
	if (typeof fetchImpl !== 'function') {
		throw new Error(
			'A fetch implementation is required to download embedding assets.',
		);
	}

	await mkdir(path.dirname(destination), { recursive: true });
	const temporaryPath = `${destination}.${process.pid}.${Date.now()}.${randomUUID()}.tmp`;
	try {
		const response = await fetchImpl(url);
		if (!response.ok || !response.body) {
			throw new Error(
				`Failed to download embedding asset ${url}: HTTP ${response.status}`,
			);
		}

		await pipeline(
			Readable.fromWeb(response.body),
			fs.createWriteStream(temporaryPath, { flags: 'wx' }),
		);
		await rename(temporaryPath, destination);
	} finally {
		await rm(temporaryPath, { force: true });
	}
}

async function resolveRemoteOnnxModelFiles(
	source,
	{ fetchImpl = globalThis.fetch, homeDir = os.homedir() } = {},
) {
	const modelUrl = remoteModelUrl(source, 'model.onnx');
	const modelPath = remoteCachePath(modelUrl, 'onnx', 'model.onnx', homeDir);
	const directory = path.dirname(modelPath);
	const files = {
		modelPath,
		tokenizerPath: path.join(directory, 'tokenizer.json'),
		tokenizerConfigPath: path.join(directory, 'tokenizer_config.json'),
	};
	await downloadFile(modelUrl, files.modelPath, fetchImpl);
	await downloadFile(
		siblingUrl(modelUrl, 'tokenizer.json'),
		files.tokenizerPath,
		fetchImpl,
	);
	await downloadFile(
		siblingUrl(modelUrl, 'tokenizer_config.json'),
		files.tokenizerConfigPath,
		fetchImpl,
	);
	return files;
}

export async function resolveOnnxModelFiles({
	fetchImpl = globalThis.fetch,
	homeDir = os.homedir(),
} = {}) {
	const configured = process.env[AAG_EMBED_ONNX_ENV];
	if (configured) {
		if (isHttpUrl(configured) || looksLikeHfRepoId(configured)) {
			return resolveRemoteOnnxModelFiles(configured, { fetchImpl, homeDir });
		}
		const configuredPath = path.resolve(configured);
		const modelPath = configuredPath.toLowerCase().endsWith('.onnx')
			? configuredPath
			: path.join(configuredPath, 'model.onnx');
		const directory = path.dirname(modelPath);
		return {
			modelPath,
			tokenizerPath: path.join(directory, 'tokenizer.json'),
			tokenizerConfigPath: path.join(directory, 'tokenizer_config.json'),
		};
	}

	const cacheDir = path.join(
		homeDir,
		'.cache',
		'agent-action-guard',
		'all-MiniLM-L6-v2',
	);
	const files = {
		modelPath: path.join(cacheDir, 'model.onnx'),
		tokenizerPath: path.join(cacheDir, 'tokenizer.json'),
		tokenizerConfigPath: path.join(cacheDir, 'tokenizer_config.json'),
	};

	await downloadFile(
		defaultOnnxAssetUrl('model.onnx'),
		files.modelPath,
		fetchImpl,
	);
	await downloadFile(
		defaultOnnxAssetUrl('tokenizer.json'),
		files.tokenizerPath,
		fetchImpl,
	);
	await downloadFile(
		defaultOnnxAssetUrl('tokenizer_config.json'),
		files.tokenizerConfigPath,
		fetchImpl,
	);
	return files;
}

export async function resolveGgufModelFile({
	fetchImpl = globalThis.fetch,
	homeDir = os.homedir(),
} = {}) {
	const configured = process.env[AAG_EMBED_GGUF_ENV];
	if (!configured) {
		throw new Error(`${AAG_EMBED_GGUF_ENV} is not configured`);
	}

	if (isHttpUrl(configured) || looksLikeHfRepoId(configured)) {
		const modelUrl = remoteModelUrl(configured, 'model.gguf');
		const modelPath = remoteCachePath(modelUrl, 'gguf', 'model.gguf', homeDir);
		await downloadFile(modelUrl, modelPath, fetchImpl);
		return modelPath;
	}

	const configuredPath = path.resolve(configured);
	return configuredPath.toLowerCase().endsWith('.gguf')
		? configuredPath
		: path.join(configuredPath, 'model.gguf');
}

export const ActionGuardDecision = Object.freeze({
	ALLOW: 'ALLOW',
	BLOCK: 'BLOCK',
});

export class EmbeddingModel {
	constructor(
		modelName = undefined,
		{
			clientFactory = null,
			ggufModule = null,
			ortModule = null,
			tokenizerFactory = null,
			assetResolver = resolveOnnxModelFiles,
			fetchImpl = globalThis.fetch,
		} = {},
	) {
		const envModelName = process.env.EMBED_MODEL_NAME;
		this.modelName = modelName ?? envModelName ?? DEFAULT_EMBED_MODEL_NAME;
		this.client = null;
		this.clientFactory = clientFactory;
		this.ggufModule = ggufModule;
		this.ggufRuntimePromise = null;
		this.ortModule = ortModule;
		this.tokenizerFactory = tokenizerFactory;
		this.assetResolver = assetResolver;
		this.fetchImpl = fetchImpl;
		this.onnxSessionPromise = null;
		this.onnxTokenizer = null;
		this.onnxTokenizerMetadata = null;

		const localGgufConfigured = Boolean(process.env[AAG_EMBED_GGUF_ENV]);
		const localOnnxConfigured = Boolean(process.env[AAG_EMBED_ONNX_ENV]);
		const modelConfigured = Boolean(
			modelName !== undefined || envModelName,
		);
		const apiConfigured = Boolean(
			process.env.EMBEDDING_BASE_URL ||
			process.env.EMBEDDING_API_KEY ||
			process.env.OPENAI_API_KEY,
		);

		if (localGgufConfigured) {
			this.backend = 'gguf';
		} else if (localOnnxConfigured) {
			this.backend = 'onnx';
		} else if (modelConfigured) {
			this.backend = 'api';
		} else if (apiConfigured) {
			this.backend = 'api';
		} else {
			this.backend = 'onnx';
		}
	}

	async getGgufRuntime() {
		if (this.ggufRuntimePromise) {
			return this.ggufRuntimePromise;
		}

		const modelPath = await resolveGgufModelFile({ fetchImpl: this.fetchImpl });
		if (!fs.existsSync(modelPath)) {
			throw new Error(`GGUF embedding model not found: ${modelPath}`);
		}

		this.ggufRuntimePromise = (async () => {
			let module = this.ggufModule;
			if (!module) {
				try {
					module = await import('node-llama-cpp');
				} catch (error) {
					const wrapped = new Error(
						'GGUF embeddings require node-llama-cpp. Install it with `npm install node-llama-cpp`.',
					);
					wrapped.cause = error;
					throw wrapped;
				}
			}

			const llama = await module.getLlama();
			const model = await llama.loadModel({ modelPath });
			const context = await model.createEmbeddingContext();
			return { llama, model, context };
		})();

		try {
			return await this.ggufRuntimePromise;
		} catch (error) {
			this.ggufRuntimePromise = null;
			throw error;
		}
	}

	async encodeGguf(texts) {
		const { context } = await this.getGgufRuntime();
		return Promise.all(
			texts.map(async (text) => {
				const result = await context.getEmbeddingFor(String(text));
				const vector = Array.from(result?.vector ?? []);
				if (
					vector.length === 0 ||
					vector.some((value) => !Number.isFinite(value))
				) {
					throw new Error(
						'GGUF embedding model returned an invalid embedding vector.',
					);
				}
				const norm = Math.sqrt(
					vector.reduce((sum, value) => sum + value * value, 0),
				);
				return vector.map((value) => value / Math.max(norm, 1e-12));
			}),
		);
	}

	async getClient() {
		if (this.client) {
			return this.client;
		}

		const apiKey =
			process.env.EMBEDDING_API_KEY ||
			process.env.OPENAI_API_KEY ||
			'dummy';
		const baseURL = process.env.EMBEDDING_BASE_URL;

		if (this.clientFactory) {
			this.client = this.clientFactory({
				apiKey,
				...(baseURL ? { baseURL } : {}),
			});
			return this.client;
		}

		const { default: OpenAI } = await import('openai');
		this.client = new OpenAI({
			apiKey,
			...(baseURL ? { baseURL } : {}),
		});

		return this.client;
	}

	async getOrt() {
		if (!this.ortModule) {
			this.ortModule = normalizeOnnxRuntimeModule(
				await import('onnxruntime-node'),
			);
		} else {
			this.ortModule = normalizeOnnxRuntimeModule(this.ortModule);
		}
		return this.ortModule;
	}

	async getOnnxRuntime() {
		if (this.onnxSessionPromise && this.onnxTokenizer) {
			return {
				session: await this.onnxSessionPromise,
				tokenizer: this.onnxTokenizer,
				metadata: this.onnxTokenizerMetadata,
			};
		}

		const files = await this.assetResolver({ fetchImpl: this.fetchImpl });
		for (const [label, filename] of [
			['ONNX embedding model', files.modelPath],
			['ONNX embedding tokenizer', files.tokenizerPath],
			['ONNX embedding tokenizer config', files.tokenizerConfigPath],
		]) {
			if (!fs.existsSync(filename)) {
				throw new Error(`${label} not found: ${filename}`);
			}
		}

		const tokenizerJson = JSON.parse(
			await readFile(files.tokenizerPath, 'utf8'),
		);
		const tokenizerConfig = JSON.parse(
			await readFile(files.tokenizerConfigPath, 'utf8'),
		);
		if (this.tokenizerFactory) {
			this.onnxTokenizer = await this.tokenizerFactory(
				tokenizerJson,
				tokenizerConfig,
			);
		} else {
			const { Tokenizer } = await import('@huggingface/tokenizers');
			this.onnxTokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
		}

		const vocab = tokenizerJson?.model?.vocab ?? {};
		const paddingSymbol = tokenizerConfig.pad_token ?? '[PAD]';
		const separatorSymbol = tokenizerConfig.sep_token ?? '[SEP]';
		this.onnxTokenizerMetadata = {
			paddingId: vocab[paddingSymbol] ?? 0,
			separatorId: vocab[separatorSymbol] ?? null,
		};

		const ort = await this.getOrt();
		this.onnxSessionPromise = ort.InferenceSession.create(files.modelPath, {
			executionProviders: ['cpu'],
		});

		return {
			session: await this.onnxSessionPromise,
			tokenizer: this.onnxTokenizer,
			metadata: this.onnxTokenizerMetadata,
		};
	}

	prepareEncoding(encoded, maxLength, { paddingId, separatorId }) {
		let ids = Array.from(encoded.ids ?? []);
		let attentionMask = Array.from(
			encoded.attention_mask ?? encoded.attentionMask ?? ids.map(() => 1),
		);
		let typeIds = Array.from(
			encoded.type_ids ?? encoded.typeIds ?? ids.map(() => 0),
		);

		if (ids.length > maxLength) {
			const preserveSeparator =
				separatorId !== null && ids[ids.length - 1] === separatorId;
			ids = ids.slice(0, maxLength);
			attentionMask = attentionMask.slice(0, maxLength);
			typeIds = typeIds.slice(0, maxLength);
			if (preserveSeparator) {
				ids[maxLength - 1] = separatorId;
				attentionMask[maxLength - 1] = 1;
			}
		}

		while (ids.length < maxLength) {
			ids.push(paddingId);
			attentionMask.push(0);
			typeIds.push(0);
		}

		return { ids, attentionMask, typeIds };
	}

	async encodeOnnx(texts) {
		const { session, tokenizer, metadata } = await this.getOnnxRuntime();
		const encodings = await Promise.all(
			texts.map((text) => tokenizer.encode(String(text))),
		);
		const maxLength = Math.min(
			DEFAULT_MAX_SEQUENCE_LENGTH,
			Math.max(
				1,
				...encodings.map((encoding) => encoding.ids?.length ?? 0),
			),
		);
		const prepared = encodings.map((encoding) =>
			this.prepareEncoding(encoding, maxLength, metadata),
		);

		const batchSize = prepared.length;
		const flatten = (key) => prepared.flatMap((item) => item[key]);
		const ort = await this.getOrt();
		const inputs = {
			input_ids: new ort.Tensor(
				'int64',
				BigInt64Array.from(flatten('ids'), (value) => BigInt(value)),
				[batchSize, maxLength],
			),
			attention_mask: new ort.Tensor(
				'int64',
				BigInt64Array.from(flatten('attentionMask'), (value) =>
					BigInt(value),
				),
				[batchSize, maxLength],
			),
			token_type_ids: new ort.Tensor(
				'int64',
				BigInt64Array.from(flatten('typeIds'), (value) =>
					BigInt(value),
				),
				[batchSize, maxLength],
			),
		};
		const feed = {};
		for (const inputName of session.inputNames ?? Object.keys(inputs)) {
			if (!inputs[inputName]) {
				throw new Error(
					`Unsupported ONNX embedding model input: ${inputName}`,
				);
			}
			feed[inputName] = inputs[inputName];
		}

		const outputs = await session.run(feed);
		const output = session.outputNames?.length
			? outputs[session.outputNames[0]]
			: Object.values(outputs)[0];
		if (!output?.data) {
			throw new Error('ONNX embedding model returned no outputs.');
		}

		const dimensions = output.dims ?? output.dimensions ?? [];
		let embeddings;
		if (dimensions.length === 3) {
			const [, sequenceLength, hiddenSize] = dimensions;
			embeddings = prepared.map((item, batchIndex) => {
				const vector = Array(hiddenSize).fill(0);
				let activePositions = 0;
				for (
					let position = 0;
					position < sequenceLength;
					position += 1
				) {
					if (!item.attentionMask[position]) {
						continue;
					}
					activePositions += 1;
					const offset =
						(batchIndex * sequenceLength + position) * hiddenSize;
					for (let index = 0; index < hiddenSize; index += 1) {
						vector[index] += output.data[offset + index];
					}
				}
				return vector.map(
					(value) => value / Math.max(activePositions, 1),
				);
			});
		} else if (dimensions.length === 2) {
			const [, hiddenSize] = dimensions;
			embeddings = Array.from({ length: batchSize }, (_, batchIndex) =>
				Array.from(
					output.data.slice(
						batchIndex * hiddenSize,
						(batchIndex + 1) * hiddenSize,
					),
				),
			);
		} else {
			throw new Error(
				'ONNX embedding model must return a 2D sentence embedding or 3D token embeddings.',
			);
		}

		return embeddings.map((vector) => {
			const norm = Math.sqrt(
				vector.reduce((sum, value) => sum + value * value, 0),
			);
			return vector.map((value) => value / Math.max(norm, 1e-12));
		});
	}

	async encode(texts) {
		if (this.backend === 'gguf') {
			return this.encodeGguf(texts);
		}
		if (this.backend === 'onnx') {
			return this.encodeOnnx(texts);
		}

		const client = await this.getClient();
		const response = await client.embeddings.create({
			model: this.modelName,
			input: texts,
			encoding_format: 'float',
		});

		return response.data.map((item) => item.embedding);
	}
}

export function flattenActionToText(actionData) {
	if (!actionData || typeof actionData !== 'object') {
		return `Perform action: ${String(actionData)}`;
	}

	if (actionData.type !== 'function') {
		return `Perform action: ${JSON.stringify(actionData)}`;
	}

	if (!actionData.function || typeof actionData.function !== 'object') {
		return `Call function with data: ${String(actionData.function)}`;
	}

	const funcName = actionData.function.name || 'unknown_function';
	const argsRaw = actionData.function.arguments ?? {};

	if (argsRaw && typeof argsRaw === 'object' && !Array.isArray(argsRaw)) {
		const argsFlat = Object.entries(argsRaw)
			.map(([key, value]) => `${key}=${String(value)}`)
			.join(', ');
		return `Call function ${funcName}: ${argsFlat}`;
	}

	if (typeof argsRaw === 'string') {
		try {
			const parsed = JSON.parse(argsRaw);
			if (
				parsed &&
				typeof parsed === 'object' &&
				!Array.isArray(parsed)
			) {
				const argsFlat = Object.entries(parsed)
					.map(([key, value]) => `${key}=${String(value)}`)
					.join(', ');
				return `Call function ${funcName}: ${argsFlat}`;
			}
		} catch {
			return `Call function ${funcName} with arguments: ${argsRaw}`;
		}
	}

	return `Call function ${funcName} with arguments: ${String(argsRaw)}`;
}

export const embedModel = new EmbeddingModel();
