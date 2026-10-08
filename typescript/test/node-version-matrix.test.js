import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
	DEFAULT_NODE_VERSIONS,
	assertBashAvailable,
	buildNvmTestCommand,
	normalizeNodeVersions,
	resolveNodeVersions,
	resolveNvmScript,
	runNodeVersionMatrix,
	shellQuote,
} from '../scripts/test-all-versions.js';

test('Bash prerequisite fails early for a missing or unusable Bash executable', () => {
	assert.doesNotThrow(() => assertBashAvailable(() => ({ status: 0 })));
	assert.throws(() => assertBashAvailable(() => ({ error: new Error('ENOENT'), status: null })), /requires Bash and nvm/);
	assert.throws(() => assertBashAvailable(() => ({ status: 1 })), /Bash was not found or could not start/);
});

test('default Node matrix covers supported LTS/current majors', () => {
	assert.deepEqual(DEFAULT_NODE_VERSIONS, ['20', '22', '24', '26']);
});

test('normalizeNodeVersions supports arguments, comma lists, and deduplication', () => {
	assert.deepEqual(normalizeNodeVersions(['20,22', '22', '20', '24.1.0']), [
		'20',
		'22',
		'24.1.0',
	]);
});

test('normalizeNodeVersions rejects unsupported version syntax', () => {
	assert.throws(
		() => normalizeNodeVersions(['lts/*']),
		/Invalid Node version/,
	);
	assert.throws(() => normalizeNodeVersions(['v22']), /Invalid Node version/);
});

test('matrix rejects Node 19 and earlier, including explicitly selected versions', () => {
	for (const version of ['18', '19', '19.9.0', '0']) {
		assert.throws(() => normalizeNodeVersions([version]), /Node 20 or later is required/);
	}
	assert.throws(() => resolveNodeVersions([], { NODE_TEST_VERSIONS: '19' }), /Node 20 or later is required/);
});

test('resolveNodeVersions prefers CLI arguments, then environment, then defaults', () => {
	assert.deepEqual(
		resolveNodeVersions(['22', '24'], { NODE_TEST_VERSIONS: '20' }),
		['22', '24'],
	);
	assert.deepEqual(resolveNodeVersions([], { NODE_TEST_VERSIONS: '20,22' }), [
		'20',
		'22',
	]);
	assert.deepEqual(resolveNodeVersions([], {}), DEFAULT_NODE_VERSIONS);
});

test('resolveNvmScript supports NVM_SH override and reports a missing installation', () => {
	const tempDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'aag-nvm-'));
	const nvmScript = path.join(tempDirectory, 'nvm.sh');
	fs.writeFileSync(nvmScript, '# test fixture\n');

	assert.equal(
		resolveNvmScript({ NVM_SH: nvmScript }, tempDirectory),
		nvmScript,
	);
	assert.throws(
		() =>
			resolveNvmScript({
				NVM_SH: path.join(tempDirectory, 'missing.sh'),
			}),
		/nvm\.sh not found/,
	);
});

test('shellQuote safely quotes apostrophes for bash', () => {
	assert.equal(shellQuote("a'b"), `'a'"'"'b'`);
});

test('buildNvmTestCommand installs and selects Node with nvm before npm ci and tests', () => {
	const command = buildNvmTestCommand({
		nvmScript: '/home/test/.nvm/nvm.sh',
		version: '22',
		testFiles: ['test/a.test.js', 'test/file with space.test.js'],
	});

	assert.match(command, /\. '\/home\/test\/\.nvm\/nvm\.sh'/);
	assert.match(command, /nvm install '22'/);
	assert.match(command, /nvm use '22'/);
	assert.match(command, /ONNXRUNTIME_NODE_INSTALL=skip npm ci --no-audit --no-fund/);
	assert.match(
		command,
		/node --test 'test\/a\.test\.js' 'test\/file with space\.test\.js'/,
	);
	assert.ok(command.indexOf('nvm use') < command.indexOf('npm ci'));
	assert.ok(command.indexOf('npm ci') < command.indexOf('npm run build'));
	assert.ok(command.indexOf('npm run build') < command.indexOf('node --test'));
});

test('runNodeVersionMatrix continues after failures by default and returns non-zero', () => {
	const calls = [];
	const output = [];
	const statuses = [0, 3, 0];
	const status = runNodeVersionMatrix({
		versions: ['18', '20', '22'],
		nvmScript: '/tmp/nvm.sh',
		testFiles: ['test/example.test.js'],
		env: {},
		spawn: (command, args) => {
			calls.push({ command, args });
			return { status: statuses[calls.length - 1] };
		},
		write: (message) => output.push(message),
	});

	assert.equal(status, 1);
	assert.equal(calls.length, 3);
	assert.ok(calls.every(({ command }) => command === 'bash'));
	assert.ok(output.includes('Succeeded versions: 18, 22'));
	assert.ok(output.includes('Failed versions: 20'));
	assert.ok(output.includes('Result: Node version matrix failed.'));
});

test('runNodeVersionMatrix can stop on the first failure and still prints a summary', () => {
	let calls = 0;
	const output = [];
	const status = runNodeVersionMatrix({
		versions: ['18', '20', '22'],
		nvmScript: '/tmp/nvm.sh',
		testFiles: ['test/example.test.js'],
		env: { CONTINUE_ON_FAILURE: '0' },
		spawn: () => {
			calls += 1;
			return { status: calls === 2 ? 7 : 0 };
		},
		write: (message) => output.push(message),
	});

	assert.equal(status, 7);
	assert.equal(calls, 2);
	assert.ok(output.includes('Succeeded versions: 18'));
	assert.ok(output.includes('Failed versions: 20'));
	assert.ok(output.includes('Result: Node version matrix failed.'));
});
