import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const DEPENDENCY_RANGES = Object.freeze({
	'@huggingface/tokenizers': '^0.1.0',
	'onnxruntime-node': '^1.14.0',
	openai: '^4.0.0 || ^5.0.0 || ^6.0.0 || ^7.0.0',
});

const DEFAULT_SAMPLES = 3;
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDirectory, '..');

function run(
	command,
	args,
	{ cwd = packageRoot, env = process.env, inherit = false } = {},
) {
	const result = spawnSync(command, args, {
		cwd,
		env,
		encoding: 'utf8',
		stdio: inherit ? 'inherit' : 'pipe',
	});

	if (result.error) {
		throw result.error;
	}

	if (result.status !== 0) {
		const output = [result.stdout, result.stderr]
			.filter(Boolean)
			.join('\n')
			.trim();
		throw new Error(
			`${command} ${args.join(' ')} failed with exit code ${result.status ?? 1}${
				output ? `\n${output}` : ''
			}`,
		);
	}

	return (result.stdout ?? '').trim();
}

function runNpm(args, options = {}) {
	const command = process.platform === 'win32' ? 'npm.cmd' : 'npm';
	return run(command, args, options);
}

export function sampleVersions(versions, count) {
	if (versions.length === 0) {
		return [];
	}
	if (versions.length <= count) {
		return [...versions];
	}

	const selected = [];
	for (let index = 0; index < count; index += 1) {
		const versionIndex = Math.round(
			(index * (versions.length - 1)) / (count - 1),
		);
		const version = versions[versionIndex];
		if (!selected.includes(version)) {
			selected.push(version);
		}
	}
	return selected;
}

function parsePublishedVersions(output, dependencyName) {
	let parsed;
	try {
		parsed = JSON.parse(output);
	} catch (error) {
		throw new Error(
			`Unable to parse published versions for ${dependencyName}: ${error.message}`,
		);
	}

	const versions = Array.isArray(parsed) ? parsed : [parsed];
	const normalized = versions.filter(
		(version) => typeof version === 'string' && version.length > 0,
	);
	if (normalized.length === 0) {
		throw new Error(`No published versions found for ${dependencyName}`);
	}
	return normalized;
}

export function selectDependencyVersions(
	versions,
	latestVersion,
	samples,
	dependencyName = 'dependency',
) {
	if (!versions.includes(latestVersion)) {
		throw new Error(
			`${dependencyName}@latest (${latestVersion}) is outside the supported range`,
		);
	}

	const selected = sampleVersions(versions, samples);
	if (!selected.includes(latestVersion)) {
		selected[selected.length - 1] = latestVersion;
	}
	return [...new Set(selected)];
}

function resolveDependencyVersions(dependencyName, range, samples) {
	const output = runNpm([
		'view',
		`${dependencyName}@${range}`,
		'version',
		'--json',
	]);
	const latestOutput = runNpm(['view', dependencyName, 'version', '--json']);
	const latestVersion = parsePublishedVersions(
		latestOutput,
		dependencyName,
	)[0];
	return selectDependencyVersions(
		parsePublishedVersions(output, dependencyName),
		latestVersion,
		samples,
		dependencyName,
	);
}

export function buildScenarios(
	dependencyVersions,
	{ boundsOnly = false } = {},
) {
	const entries = Object.entries(dependencyVersions);
	const minimums = Object.fromEntries(
		entries.map(([name, versions]) => [name, versions[0]]),
	);
	const maximums = Object.fromEntries(
		entries.map(([name, versions]) => [name, versions.at(-1)]),
	);

	const scenarios = [
		{
			name: 'base-min',
			dependencies: minimums,
			expectedVersions: minimums,
		},
		{
			name: 'base-max',
			dependencies: maximums,
			expectedVersions: maximums,
		},
	];

	if (boundsOnly) {
		return scenarios;
	}

	for (const [targetName, versions] of entries) {
		for (const version of versions.slice(1, -1)) {
			const dependencies = {
				...minimums,
				[targetName]: version,
			};
			const slug = targetName.replace(/^@/, '').replaceAll('/', '-');
			scenarios.push({
				name: `dep-${slug}-${version}`,
				dependencies,
				expectedVersions: dependencies,
			});
		}
	}

	return scenarios;
}

export function collectDependencyVersions(tree, dependencyName) {
	const versions = [];
	const visit = (node) => {
		if (!node || typeof node !== 'object') {
			return;
		}
		const dependencies = node.dependencies;
		if (!dependencies || typeof dependencies !== 'object') {
			return;
		}

		const dependency = dependencies[dependencyName];
		if (dependency && typeof dependency.version === 'string') {
			versions.push(dependency.version);
		}
		for (const child of Object.values(dependencies)) {
			visit(child);
		}
	};

	visit(tree);
	return [...new Set(versions)];
}

function createPackageTarball(workRoot) {
	const output = runNpm(['pack', '--json', '--pack-destination', workRoot], {
		cwd: packageRoot,
	});
	const packed = JSON.parse(output);
	const filename = packed?.[0]?.filename;
	if (!filename) {
		throw new Error('npm pack did not return a package filename');
	}
	return path.resolve(workRoot, filename);
}

function createScenarioProject(scenarioDirectory, tarballPath, scenario) {
	fs.rmSync(scenarioDirectory, { recursive: true, force: true });
	fs.mkdirSync(scenarioDirectory, { recursive: true });

	const packageJson = {
		name: `aag-${scenario.name}`,
		private: true,
		type: 'module',
		dependencies: {
			'agent-action-guard': pathToFileURL(tarballPath).href,
			...scenario.dependencies,
		},
	};

	fs.writeFileSync(
		path.join(scenarioDirectory, 'package.json'),
		`${JSON.stringify(packageJson, null, 2)}\n`,
	);
}

function verifyInstalledVersions(scenarioDirectory, expectedVersions) {
	const names = Object.keys(DEPENDENCY_RANGES);
	const output = runNpm(['ls', ...names, '--all', '--json'], {
		cwd: scenarioDirectory,
	});
	const tree = JSON.parse(output);

	for (const [dependencyName, expectedVersion] of Object.entries(
		expectedVersions,
	)) {
		const installedVersions = collectDependencyVersions(
			tree,
			dependencyName,
		);
		if (installedVersions.length === 0) {
			throw new Error(`${dependencyName} was not installed`);
		}
		if (
			installedVersions.length !== 1 ||
			installedVersions[0] !== expectedVersion
		) {
			throw new Error(
				`${dependencyName}: expected only ${expectedVersion}, found ${installedVersions.join(', ')}`,
			);
		}
	}
}

function runSmokeTest(scenarioDirectory) {
	const snippet = `
		const guard = await import('agent-action-guard');
		const tokenizers = await import('@huggingface/tokenizers');
		const ortModule = await import('onnxruntime-node');
		const ort = ortModule.InferenceSession ? ortModule : ortModule.default;
		const openaiModule = await import('openai');

		if (typeof guard.isActionHarmful !== 'function') throw new Error('Action Guard import failed');
		if (typeof tokenizers.Tokenizer !== 'function') throw new Error('Tokenizer API missing');
		if (typeof ort?.InferenceSession?.create !== 'function') throw new Error('ONNX Runtime API missing');
		if (typeof openaiModule.default !== 'function') throw new Error('OpenAI default export missing');

		await guard.classifier.loadModel();
		const embeddingModel = new guard.EmbeddingModel({ modelName: 'matrix-smoke' });
		const client = await embeddingModel.getClient();
		if (typeof client.embeddings?.create !== 'function') throw new Error('OpenAI embeddings API missing');
	`;

	run(process.execPath, ['--input-type=module', '--eval', snippet], {
		cwd: scenarioDirectory,
	});
}

function parseArgs(argv) {
	const args = {
		boundsOnly: false,
		samples: DEFAULT_SAMPLES,
		workRoot: path.join(packageRoot, '.dependency-matrix'),
	};

	for (let index = 0; index < argv.length; index += 1) {
		const value = argv[index];
		if (value === '--bounds-only') {
			args.boundsOnly = true;
			continue;
		}
		if (value === '--samples') {
			args.samples = Number.parseInt(argv[index + 1], 10);
			index += 1;
			continue;
		}
		if (value === '--work-root') {
			args.workRoot = path.resolve(argv[index + 1]);
			index += 1;
			continue;
		}
		throw new Error(`Unknown argument: ${value}`);
	}

	if (!Number.isInteger(args.samples) || args.samples < 2) {
		throw new Error('--samples must be an integer of at least 2');
	}
	return args;
}

export function main(argv = process.argv.slice(2)) {
	let args;
	try {
		args = parseArgs(argv);
	} catch (error) {
		console.error(error.message);
		return 2;
	}

	fs.mkdirSync(args.workRoot, { recursive: true });
	console.log(`Node runtime: ${process.version}`);
	console.log(`Dependency samples per package: ${args.samples}`);

	let dependencyVersions;
	let tarballPath;
	try {
		dependencyVersions = Object.fromEntries(
			Object.entries(DEPENDENCY_RANGES).map(([name, range]) => {
				const versions = resolveDependencyVersions(
					name,
					range,
					args.samples,
				);
				console.log(`${name}: ${versions.join(', ')}`);
				return [name, versions];
			}),
		);
		tarballPath = createPackageTarball(args.workRoot);
	} catch (error) {
		console.error(`Matrix setup failed: ${error.message}`);
		return 1;
	}

	const scenarios = buildScenarios(dependencyVersions, {
		boundsOnly: args.boundsOnly,
	});
	const failed = [];

	for (const scenario of scenarios) {
		console.log(`\n========= Scenario: ${scenario.name} =========`);
		const scenarioDirectory = path.join(
			args.workRoot,
			'scenarios',
			scenario.name,
		);

		try {
			createScenarioProject(scenarioDirectory, tarballPath, scenario);
			console.log('Installing scenario dependencies');
			runNpm(
				['install', '--no-package-lock', '--no-audit', '--no-fund'],
				{
					cwd: scenarioDirectory,
					inherit: true,
				},
			);
			verifyInstalledVersions(
				scenarioDirectory,
				scenario.expectedVersions,
			);
			console.log('Running package smoke test');
			runSmokeTest(scenarioDirectory);
			console.log(`Scenario succeeded: ${scenario.name}`);
		} catch (error) {
			failed.push(scenario.name);
			console.error(`Scenario failed: ${scenario.name}`);
			console.error(error.message);
		}
	}

	console.log('\n========= Summary =========');
	console.log(`Succeeded: ${scenarios.length - failed.length}`);
	console.log(`Failed: ${failed.length}`);
	if (failed.length > 0) {
		for (const scenarioName of failed) {
			console.log(`  - ${scenarioName}`);
		}
		return 1;
	}

	console.log('Result: all dependency-version scenarios succeeded.');
	return 0;
}

const isMain =
	!process.env.NODE_TEST_CONTEXT &&
	process.argv[1] &&
	import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
	process.exitCode = main();
}
