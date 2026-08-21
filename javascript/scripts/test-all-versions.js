import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const DEFAULT_NODE_VERSIONS = ['18', '20', '22', '24'];

export function normalizeNodeVersions(values) {
	const versions = [];
	for (const value of values) {
		for (const version of String(value).split(/[,\s]+/)) {
			if (!version) {
				continue;
			}
			if (!/^\d+(?:\.\d+){0,2}$/.test(version)) {
				throw new Error(
					`Invalid Node version ${JSON.stringify(version)}; expected MAJOR, MAJOR.MINOR, or MAJOR.MINOR.PATCH`,
				);
			}
			if (!versions.includes(version)) {
				versions.push(version);
			}
		}
	}
	return versions;
}

export function resolveNodeVersions(args = [], env = process.env) {
	const argumentVersions = normalizeNodeVersions(args);
	if (argumentVersions.length > 0) {
		return argumentVersions;
	}

	const environmentVersions = normalizeNodeVersions([
		env.NODE_TEST_VERSIONS ?? '',
	]);
	return environmentVersions.length > 0
		? environmentVersions
		: [...DEFAULT_NODE_VERSIONS];
}

export function resolveNvmScript(
	env = process.env,
	homeDirectory = os.homedir(),
) {
	const nvmDirectory = env.NVM_DIR || path.join(homeDirectory, '.nvm');
	const nvmScript = env.NVM_SH || path.join(nvmDirectory, 'nvm.sh');
	if (!fs.existsSync(nvmScript)) {
		throw new Error(
			`nvm.sh not found at ${nvmScript}. Install nvm or set NVM_DIR/NVM_SH.`,
		);
	}
	return nvmScript;
}

export function discoverTestFiles(rootDirectory = process.cwd()) {
	const testDirectory = path.join(rootDirectory, 'test');
	return fs
		.readdirSync(testDirectory)
		.filter((fileName) => fileName.endsWith('.test.js'))
		.sort()
		.map((fileName) => path.join('test', fileName));
}

export function shellQuote(value) {
	return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

export function buildNvmTestCommand({ nvmScript, version, testFiles }) {
	const quotedTests = testFiles.map(shellQuote).join(' ');
	return [
		'set -euo pipefail',
		`. ${shellQuote(nvmScript)}`,
		`nvm install ${shellQuote(version)}`,
		`nvm use ${shellQuote(version)}`,
		`requested=${shellQuote(version)}`,
		`actual="$(node -p 'process.versions.node')"`,
		'case "$actual" in "$requested"|"$requested".*) ;; *) echo "Expected Node $requested, got $actual" >&2; exit 2 ;; esac',
		'echo "Using Node $actual ($(npm --version | sed \'s/^/npm /\'))"',
		'npm ci --no-audit --no-fund',
		`node --test ${quotedTests}`,
	].join('\n');
}

export function runNodeVersionMatrix({
	versions,
	nvmScript,
	testFiles,
	env = process.env,
	spawn = spawnSync,
	write = console.log,
} = {}) {
	const selectedVersions = versions ?? resolveNodeVersions([], env);
	const selectedNvmScript = nvmScript ?? resolveNvmScript(env);
	const selectedTestFiles = testFiles ?? discoverTestFiles();
	const continueOnFailure = env.CONTINUE_ON_FAILURE !== '0';
	const succeeded = [];
	const failed = [];

	for (const version of selectedVersions) {
		write(`\n=== Node ${version} ===`);
		const result = spawn(
			'bash',
			[
				'-lc',
				buildNvmTestCommand({
					nvmScript: selectedNvmScript,
					version,
					testFiles: selectedTestFiles,
				}),
			],
			{
				cwd: process.cwd(),
				env,
				stdio: 'inherit',
			},
		);

		if (result.error) {
			throw result.error;
		}
		if (result.status === 0) {
			succeeded.push(version);
			continue;
		}

		failed.push(version);
		if (!continueOnFailure) {
			return result.status ?? 1;
		}
	}

	write('\n=== Node version matrix summary ===');
	write(`Succeeded: ${succeeded.length}`);
	write(`Failed: ${failed.length}`);
	if (failed.length > 0) {
		write(`Failed versions: ${failed.join(', ')}`);
		return 1;
	}
	write('Result: all Node version runs succeeded.');
	return 0;
}

export function main(argv = process.argv.slice(2), env = process.env) {
	const versions = resolveNodeVersions(argv, env);
	return runNodeVersionMatrix({ versions, env });
}

const isMainModule =
	process.argv[1] &&
	path.resolve(process.argv[1]) ===
		path.resolve(fileURLToPath(import.meta.url));

if (isMainModule) {
	try {
		process.exitCode = main();
	} catch (error) {
		console.error(error.message);
		process.exitCode = 2;
	}
}
