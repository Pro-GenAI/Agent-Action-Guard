import { Buffer } from 'node:buffer';
import { createServer } from 'node:http';
import { URL } from 'node:url';

import { isActionsHarmful } from './action-classifier.js';

export const DEFAULT_HOST = '127.0.0.1';
export const DEFAULT_PORT = 8000;
export const DEFAULT_MAX_BODY_BYTES = 1_048_576;

export async function classifyPayload(
	payload,
	{ classifyActions = isActionsHarmful, batchSize = null } = {},
) {
	if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
		throw new Error('Request body must be a JSON object');
	}

	const hasAction = Object.hasOwn(payload, 'action');
	const hasActions = Object.hasOwn(payload, 'actions');
	if (hasAction === hasActions) {
		throw new Error("Provide exactly one of 'action' or 'actions'");
	}

	const actions = hasActions ? payload.actions : [payload.action];
	if (!Array.isArray(actions) || actions.length === 0) {
		throw new Error("'actions' must be a non-empty JSON array");
	}
	for (let index = 0; index < actions.length; index += 1) {
		const action = actions[index];
		if (!action || typeof action !== 'object' || Array.isArray(action)) {
			throw new Error(`action ${index + 1} must be a JSON object`);
		}
	}

	const requestBatchSize = payload.batch_size ?? batchSize;
	if (
		requestBatchSize !== null &&
		(!Number.isInteger(requestBatchSize) || requestBatchSize <= 0)
	) {
		throw new Error("'batch_size' must be a positive integer");
	}

	const results = await classifyActions(actions, { batchSize: requestBatchSize });
	const items = results.map(({ label, confidence }) => ({
		label,
		confidence,
		safe: label === null,
	}));
	const safe = items.filter((item) => item.safe).length;
	return {
		results: items,
		summary: {
			total: items.length,
			safe,
			unsafe: items.length - safe,
		},
	};
}

function sendJson(response, status, payload) {
	const data = JSON.stringify(payload);
	response.writeHead(status, {
		'content-type': 'application/json',
		'content-length': Buffer.byteLength(data),
	});
	response.end(data);
}

export function createApiServer({
	classifyActions = isActionsHarmful,
	batchSize = null,
	maxBodyBytes = DEFAULT_MAX_BODY_BYTES,
} = {}) {
	return createServer((request, response) => {
		const url = new URL(request.url ?? '/', 'http://localhost');

		if (request.method === 'GET' && url.pathname === '/health') {
			sendJson(response, 200, { status: 'ok' });
			return;
		}
		if (url.pathname !== '/v1/classify') {
			sendJson(response, 404, { error: 'Not found' });
			return;
		}
		if (request.method !== 'POST') {
			sendJson(response, 405, { error: 'Method not allowed' });
			return;
		}

		let size = 0;
		const chunks = [];
		let rejected = false;
		request.on('data', (chunk) => {
			if (rejected) return;
			size += chunk.length;
			if (size > maxBodyBytes) {
				rejected = true;
				sendJson(response, 413, { error: 'Request body too large' });
				request.resume();
				return;
			}
			chunks.push(chunk);
		});
		request.on('end', async () => {
			if (rejected) return;
			if (size === 0) {
				sendJson(response, 400, { error: 'Request body is required' });
				return;
			}
			let payload;
			try {
				payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
			} catch {
				sendJson(response, 400, { error: 'Request body must be valid JSON' });
				return;
			}
			try {
				const result = await classifyPayload(payload, {
					classifyActions,
					batchSize,
				});
				sendJson(response, 200, result);
			} catch (error) {
				if (error instanceof Error) {
					sendJson(response, 400, { error: error.message });
					return;
				}
				sendJson(response, 500, { error: 'Classification failed' });
			}
		});
	});
}

export async function startApiServer({
	host = DEFAULT_HOST,
	port = DEFAULT_PORT,
	batchSize = null,
	classifyActions = isActionsHarmful,
} = {}) {
	const server = createApiServer({ classifyActions, batchSize });
	await new Promise((resolve, reject) => {
		server.once('error', reject);
		server.listen(port, host, () => {
			server.off('error', reject);
			resolve(undefined);
		});
	});
	const address = server.address();
	const actualPort =
		typeof address === 'object' && address !== null ? address.port : port;
	console.log(`Agent Action Guard API listening on http://${host}:${actualPort}`);
	return server;
}
