import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import assert from 'node:assert/strict';

const fixturesDir = path.resolve('runtime-fixtures');
const activeNodeCommand = process.execPath;

function supportsNodeTypeStripping() {
	const [major] = process.versions.node.split('.').map(Number);

	return major >= 22;
}

const runtimeMatrix = [
	{
		name: 'node-js',
		command: activeNodeCommand,
		displayCommand: 'node',
	args: ['runtime-fixtures/runtime-suite.js'],
	},
	{
		name: 'node-ts',
		command: activeNodeCommand,
		displayCommand: 'node',
		args: [
			'--experimental-strip-types',
			'runtime-fixtures/runtime-suite.ts',
		],
		optional: true,
		requiresNodeTypeStripping: true,
	},
	{
		name: 'deno-js',
		command: 'deno',
		args: [
			'run',
			'--allow-env',
			'--allow-read',
			'--allow-write',
			'runtime-fixtures/runtime-suite.js',
		],
		optional: true,
	},
	{
		name: 'deno-ts',
		command: 'deno',
		args: [
			'run',
			'--allow-env',
			'--allow-read',
			'--allow-write',
			'runtime-fixtures/runtime-suite.ts',
		],
		optional: true,
	},
	{
		name: 'bun-js',
		command: 'bun',
	args: ['run', 'runtime-fixtures/runtime-suite.js'],
		optional: true,
	},
	{
		name: 'bun-ts',
		command: 'bun',
	args: ['run', 'runtime-fixtures/runtime-suite.ts'],
		optional: true,
	},
];

function commandExists(command) {
	if (path.isAbsolute(command)) {
		return fs.existsSync(command);
	}

	const pathEntries = (process.env.PATH || '').split(path.delimiter);
	const executableNames =
		process.platform === 'win32'
			? [command, `${command}.exe`, `${command}.cmd`, `${command}.bat`]
			: [command];

	return pathEntries.some((entry) =>
		executableNames.some((candidate) =>
			fs.existsSync(path.join(entry, candidate)),
		),
	);
}

for (const runtime of runtimeMatrix) {
	const available = commandExists(runtime.command);
	const missingOptional = !available && runtime.optional;
	const unsupportedNodeTypeStripping =
		runtime.requiresNodeTypeStripping && !supportsNodeTypeStripping();
	const skip = missingOptional || unsupportedNodeTypeStripping;
	const commandName = runtime.displayCommand || runtime.command;

	test(
		`${runtime.name} executes the runtime assertion suite`,
		{
			skip: skip
				? unsupportedNodeTypeStripping
					? `${commandName} does not support type stripping`
					: `${commandName} is not installed`
				: false,
		},
		() => {
			assert.equal(available, true, `${commandName} is not installed`);

			const result = spawnSync(runtime.command, runtime.args, {
				cwd: path.resolve('.'),
				encoding: 'utf8',
				env: process.env,
			});

			assert.equal(
				result.status,
				0,
				[
					`Runtime command failed: ${runtime.command} ${runtime.args.join(' ')}`,
					result.stdout,
					result.stderr,
				]
					.filter(Boolean)
					.join('\n'),
			);
		},
	);
}

test('runtime smoke fixtures exist', () => {
	assert.ok(fs.existsSync(path.join(fixturesDir, 'runtime-suite.js')));
	assert.ok(fs.existsSync(path.join(fixturesDir, 'runtime-suite.ts')));
});
