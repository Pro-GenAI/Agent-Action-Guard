import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import { mkdir, readFile, rename, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

export const ALL_CLASSES = ['safe', 'harmful', 'unethical'];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const ONNX_MODEL_PATH = path.resolve(
	__dirname,
	'action_classifier_model.onnx',
);

export const DEFAULT_EMBED_MODEL_NAME =
	'sentence-transformers/all-MiniLM-L6-v2';
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

async function downloadFile(url, destination, fetchImpl = globalThis.fetch) {
	if (fs.existsSync(destination)) {
		return;
	}
	if (typeof fetchImpl !== 'function') {
		throw new Error(
			'A fetch implementation is required to download ONNX assets.',
		);
	}

	await mkdir(path.dirname(destination), { recursive: true });
	const temporaryPath = `${destination}.${process.pid}.${Date.now()}.${randomUUID()}.tmp`;
	try {
		const response = await fetchImpl(url);
		if (!response.ok || !response.body) {
			throw new Error(
				`Failed to download ONNX embedding asset ${url}: HTTP ${response.status}`,
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

export async function resolveOnnxModelFiles({
	fetchImpl = globalThis.fetch,
	homeDir = os.homedir(),
} = {}) {
	const configured = process.env[AAG_EMBED_ONNX_ENV];
	if (configured) {
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

export const ActionGuardDecision = Object.freeze({
	ALLOW: 'ALLOW',
	BLOCK: 'BLOCK',
});

export class EmbeddingModel {
	constructor(
		modelName = undefined,
		{
			clientFactory = null,
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
		this.ortModule = ortModule;
		this.tokenizerFactory = tokenizerFactory;
		this.assetResolver = assetResolver;
		this.fetchImpl = fetchImpl;
		this.onnxSessionPromise = null;
		this.onnxTokenizer = null;
		this.onnxTokenizerMetadata = null;

		const localOnnxConfigured = Boolean(process.env[AAG_EMBED_ONNX_ENV]);
		const modelConfigured = Boolean(
			modelName !== undefined || envModelName,
		);
		const apiConfigured = Boolean(
			process.env.EMBEDDING_BASE_URL ||
			process.env.EMBEDDING_API_KEY ||
			process.env.OPENAI_API_KEY,
		);

		if (localOnnxConfigured) {
			this.backend = 'onnx';
		} else if (modelConfigured) {
			this.backend = 'api';
		} else if (apiConfigured) {
			this.backend = 'api';
		} else {
			this.backend = 'onnx';
		}
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
