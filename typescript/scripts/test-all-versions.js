import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const DEFAULT_NODE_VERSIONS = ['20', '22', '24', '26'];

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
			if (Number(version.split('.')[0]) < 20) {
				throw new Error(`Unsupported Node version ${version}; Node 20 or later is required`);
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
		'ONNXRUNTIME_NODE_INSTALL=skip npm ci --no-audit --no-fund',
		'npm run build',
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

	const writeSummary = () => {
		write('\n=== Node version matrix summary ===');
		write(`Succeeded: ${succeeded.length}`);
		write(`Failed: ${failed.length}`);
		write(
			`Succeeded versions: ${succeeded.length > 0 ? succeeded.join(', ') : 'none'}`,
		);
		write(
			`Failed versions: ${failed.length > 0 ? failed.join(', ') : 'none'}`,
		);
	};

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
			writeSummary();
			write('Result: Node version matrix failed.');
			return result.status ?? 1;
		}
	}

	writeSummary();
	if (failed.length > 0) {
		write('Result: Node version matrix failed.');
		return 1;
	}
	write('Result: all Node version runs succeeded.');
	return 0;
}

export function assertBashAvailable(spawn = spawnSync) {
	const probe = spawn('bash', ['--version'], { stdio: 'ignore', shell: false });
	if (probe.error || probe.status !== 0) {
		throw new Error('Node test matrix requires Bash and nvm. Bash was not found or could not start; install Bash (for example via WSL on Windows) before running make test-matrix.');
	}
}

export function main(argv = process.argv.slice(2), env = process.env) {
	assertBashAvailable();
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
