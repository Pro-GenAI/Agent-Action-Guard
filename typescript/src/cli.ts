import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';

import { isActionsHarmful } from './action-classifier.js';
import { DEFAULT_HOST, DEFAULT_PORT, startApiServer } from './api-server.js';

function normalizeActions(value, source) {
	const actions = Array.isArray(value) ? value : [value];
	for (let index = 0; index < actions.length; index += 1) {
		const action = actions[index];
		if (!action || typeof action !== 'object' || Array.isArray(action)) {
			throw new Error(
				`${source}: action ${index + 1} must be a JSON object`,
			);
		}
	}
	return actions;
}

export function loadActions({ actionJson = null, file = null } = {}) {
	if (Boolean(actionJson) === Boolean(file)) {
		throw new Error('Provide exactly one of ACTION_JSON or --file');
	}

	if (actionJson !== null) {
		let value;
		try {
			value = JSON.parse(actionJson);
		} catch (error) {
			throw new Error(`Invalid ACTION_JSON: ${error.message}`);
		}
		return normalizeActions(value, 'ACTION_JSON');
	}

	let text;
	try {
		text = fs.readFileSync(file, 'utf8');
	} catch (error) {
		throw new Error(`Unable to read ${file}: ${error.message}`);
	}

	if (path.extname(file).toLowerCase() === '.jsonl') {
		const actions = [];
		for (const [index, line] of text.split(/\r?\n/).entries()) {
			if (!line.trim()) {
				continue;
			}
			let value;
			try {
				value = JSON.parse(line);
			} catch (error) {
				throw new Error(
					`${file}:${index + 1}: invalid JSON: ${error.message}`,
				);
			}
			actions.push(...normalizeActions(value, `${file}:${index + 1}`));
		}
		return actions;
	}

	let value;
	try {
		value = JSON.parse(text);
	} catch (error) {
		throw new Error(`${file}: invalid JSON: ${error.message}`);
	}
	return normalizeActions(value, file);
}

export function summarizeResults(results) {
	const safe = results.filter(({ label }) => label === null).length;
	return { safe, unsafe: results.length - safe };
}

export function usage() {
	return `Usage:
  agent-action-guard 'ACTION_JSON'
  agent-action-guard --file actions.json [--batch-size N]
  agent-action-guard --file actions.jsonl [--batch-size N]
  agent-action-guard serve [--host HOST] [--port PORT] [--batch-size N]`;
}

export function serveUsage() {
	return `Usage: agent-action-guard serve [--host HOST] [--port PORT] [--batch-size N]`;
}

export async function main(
	argv = process.argv.slice(2),
	{
		classifyActions = isActionsHarmful,
		serveApi = startApiServer,
		write = console.log,
	} = {},
) {
	if (argv[0] === 'serve') {
		const { values, positionals } = parseArgs({
			args: argv.slice(1),
			allowPositionals: true,
			options: {
				host: { type: 'string' },
				port: { type: 'string' },
				'batch-size': { type: 'string' },
				help: { type: 'boolean', short: 'h' },
			},
		});
		if (values.help) {
			write(serveUsage());
			return 0;
		}
		if (positionals.length > 0) {
			throw new Error(`Unexpected positional arguments.\n${serveUsage()}`);
		}
		const port = values.port === undefined ? DEFAULT_PORT : Number(values.port);
		if (!Number.isInteger(port) || port < 1 || port > 65535) {
			throw new Error('--port must be an integer between 1 and 65535');
		}
		let batchSize = null;
		if (values['batch-size'] !== undefined) {
			batchSize = Number(values['batch-size']);
			if (!Number.isInteger(batchSize) || batchSize <= 0) {
				throw new Error('--batch-size must be a positive integer');
			}
		}
		await serveApi({
			host: values.host ?? DEFAULT_HOST,
			port,
			batchSize,
			classifyActions,
		});
		return 0;
	}

	const { values, positionals } = parseArgs({
		args: argv,
		allowPositionals: true,
		options: {
			file: { type: 'string' },
			'batch-size': { type: 'string' },
			help: { type: 'boolean', short: 'h' },
		},
	});

	if (values.help) {
		write(usage());
		return 0;
	}
	if (positionals.length > 1) {
		throw new Error(
			`Expected at most one ACTION_JSON positional argument.\n${usage()}`,
		);
	}

	let batchSize = null;
	if (values['batch-size'] !== undefined) {
		batchSize = Number(values['batch-size']);
		if (!Number.isInteger(batchSize) || batchSize <= 0) {
			throw new Error('--batch-size must be a positive integer');
		}
	}

	const actions = loadActions({
		actionJson: positionals[0] ?? null,
		file: values.file ?? null,
	});
	const results = await classifyActions(actions, { batchSize });
	const { safe, unsafe } = summarizeResults(results);
	write(`Total actions: ${results.length}`);
	write(`Safe actions: ${safe}`);
	write(`Unsafe actions: ${unsafe}`);
	return 0;
}
