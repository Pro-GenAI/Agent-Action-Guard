import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MAKEFILE = fs.readFileSync(path.join(ROOT, 'Makefile'), 'utf8');
const PACKAGE_JSON = JSON.parse(
	fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'),
);

test('security audit aborts before scanning when Bash is unavailable', () => {
	const result = spawnSync(process.execPath, ['scripts/security-audit.js'], {
		cwd: ROOT,
		env: { ...process.env, PATH: '' },
		encoding: 'utf8',
	});
	assert.equal(result.status, 1);
	assert.match(result.stderr, /Bash was not found/);
	assert.doesNotMatch(result.stdout, /dangerous-patterns/);
});

test('install-dev provisions all TypeScript security audit tools', () => {
	assert.match(MAKEFILE, /npm install --include=dev/);
	assert.match(MAKEFILE, /rm -rf \$\(SECURITY_VENV\)/);
	assert.match(MAKEFILE, /\$\(UV\) venv --python 3\.12 \$\(SECURITY_VENV\)/);
	assert.match(
		MAKEFILE,
		/\$\(UV\) pip install --python \$\(SECURITY_VENV\)\/bin\/python semgrep detect-secrets "cryptography<50" --only-binary cryptography/,
	);
	assert.equal(typeof PACKAGE_JSON.devDependencies.pnpm, 'string');
});

test('security uses local npm and Python security tool paths', () => {
	assert.match(
		MAKEFILE,
		/PATH="\$\(CURDIR\)\/\$\(SECURITY_VENV\)\/bin:\$\(CURDIR\)\/node_modules\/\.bin:\$\$PATH"/,
	);
});

test('security audit summarizes failed scanners in red on terminals', () => {
	const source = fs.readFileSync(
		path.join(ROOT, 'scripts', 'security-audit.js'),
		'utf8',
	);
	assert.match(source, /const failures = \[\]/);
	assert.match(
		source,
		/record\('dangerous-patterns', builtinPatternScan\(\)\)/,
	);
	assert.match(source, /process\.stdout\.isTTY === true/);
	assert.match(source, /\\x1b\[31m/);
	assert.match(source, /TypeScript security audit summary/);
	assert.match(source, /Result: SECURITY AUDIT FAILED/);
});

test('test-matrix exposes selectable Node versions through the Makefile', () => {
	assert.match(MAKEFILE, /NODE_TEST_VERSIONS \?=/);
	assert.match(
		MAKEFILE,
		/test-matrix:\n\tNODE_TEST_VERSIONS="\$\(NODE_TEST_VERSIONS\)" npm run test:all-versions/,
	);
});

test('OpenAI remains optional so Node 20 installs do not force a Node-22-only SDK', () => {
	assert.equal(PACKAGE_JSON.dependencies.openai, undefined);
	assert.equal(PACKAGE_JSON.devDependencies.openai, '^6.49.0');
	assert.equal(
		PACKAGE_JSON.peerDependencies.openai,
		'^4.0.0 || ^5.0.0 || ^6.0.0 || ^7.0.0',
	);
	assert.equal(PACKAGE_JSON.peerDependenciesMeta.openai.optional, true);
});
