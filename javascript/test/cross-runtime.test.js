import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import assert from 'node:assert/strict';

const fixturesDir = path.resolve('runtime-fixtures');

const runtimeMatrix = [
	{
		name: 'node-js',
		command: 'node',
		args: ['runtime-fixtures/runtime-smoke.js'],
	},
	{
		name: 'node-ts',
		command: 'node',
		args: [
			'--experimental-strip-types',
			'runtime-fixtures/runtime-smoke.ts',
		],
	},
	{
		name: 'deno-js',
		command: 'deno',
		args: ['run', '--allow-env', 'runtime-fixtures/runtime-smoke.js'],
		optional: true,
	},
	{
		name: 'deno-ts',
		command: 'deno',
		args: ['run', '--allow-env', 'runtime-fixtures/runtime-smoke.ts'],
		optional: true,
	},
	{
		name: 'bun-js',
		command: 'bun',
		args: ['run', 'runtime-fixtures/runtime-smoke.js'],
		optional: true,
	},
	{
		name: 'bun-ts',
		command: 'bun',
		args: ['run', 'runtime-fixtures/runtime-smoke.ts'],
		optional: true,
	},
];

function commandExists(command) {
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
	const skip = !available && runtime.optional;

	test(
		`${runtime.name} executes the ESM smoke fixture`,
		{ skip: skip ? `${runtime.command} is not installed` : false },
		() => {
			assert.equal(
				available,
				true,
				`${runtime.command} is not installed`,
			);

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
	assert.ok(fs.existsSync(path.join(fixturesDir, 'runtime-smoke.js')));
	assert.ok(fs.existsSync(path.join(fixturesDir, 'runtime-smoke.ts')));
});
