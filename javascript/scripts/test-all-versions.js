import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const defaultNodeVersions = ['18', '20', '22', '24'];
const nodeVersions = (process.env.NODE_TEST_VERSIONS || '')
	.split(/[,\s]+/)
	.map((version) => version.trim())
	.filter(Boolean);
const versionsToTest =
	nodeVersions.length > 0 ? nodeVersions : defaultNodeVersions;
const testFiles = fs
	.readdirSync('test')
	.filter((fileName) => fileName.endsWith('.test.js'))
	.sort()
	.map((fileName) => path.join('test', fileName));

for (const version of versionsToTest) {
	console.log(`\n==> Running tests with Node ${version}`);

	const result = spawnSync(
		'npx',
		[
			'--yes',
			'--package',
			`node@${version}`,
			'node',
			'--test',
			...testFiles,
		],
		{
			cwd: process.cwd(),
			env: process.env,
			stdio: 'inherit',
		},
	);

	if (result.error) {
		throw result.error;
	}

	if (result.status !== 0) {
		process.exit(result.status ?? 1);
	}
}
