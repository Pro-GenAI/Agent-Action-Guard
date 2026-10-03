#!/usr/bin/env node
/**
 * Defense-in-depth security audit for the TypeScript/JavaScript package.
 *
 * No single scanner detects every vulnerability class. This orchestrates npm's
 * advisory audit, Semgrep when available, secret scanning when available, and
 * repository-local dangerous-API heuristics. It never installs tools.
 */

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REPO_ROOT = path.resolve(ROOT, '..');
const REPORT_DIR = path.join(ROOT, 'security-reports');
const STRICT_TOOLS = process.argv.includes('--strict-tools');
const NO_SEMGREP = process.argv.includes('--no-semgrep');

const SOURCE_DIRS = [
	'src',
	'bin',
	'examples',
	'scripts',
	'test',
	'runtime-fixtures',
];
const EXCLUDED = new Set([
	'node_modules',
	'dist',
	'build',
	'coverage',
	'security-reports',
]);
const RISK_PATTERNS = [
	['dynamic-code', /(?<![.\w])(?:eval|Function)\s*\(/],
	[
		'child-process',
		/(?:node:)?child_process|\b(?:exec|execSync|spawn|spawnSync)\s*\(/,
	],
	['shell-true', /\bshell\s*:\s*true\b/],
	['weak-hash', /createHash\s*\(\s*['"](?:md5|sha1)['"]\s*\)/i],
	[
		'insecure-tls',
		/NODE_TLS_REJECT_UNAUTHORIZED|rejectUnauthorized\s*:\s*false/,
	],
	['unsafe-prototype-key', /(?:__proto__|constructor|prototype)\s*\]/],
	[
		'hardcoded-secret',
		/\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*['"][^'"]{8,}['"]/i,
	],
];

const BLOCKING_PATTERN_CATEGORIES = new Set([
	'dynamic-code',
	'shell-true',
	'weak-hash',
	'insecure-tls',
	'unsafe-prototype-key',
]);

function ensureReportDir() {
	fs.mkdirSync(REPORT_DIR, { recursive: true });
}

function commandExists(command) {
	const probe = spawnSync(command, ['--version'], {
		cwd: ROOT,
		stdio: 'ignore',
		shell: false,
	});
	return !probe.error && probe.status === 0;
}

function runCheck(name, command, args, reportName, cwd = ROOT) {
	ensureReportDir();
	const result = spawnSync(command, args, {
		cwd,
		encoding: 'utf8',
		shell: false,
		maxBuffer: 64 * 1024 * 1024,
	});
	const output = `${result.stdout ?? ''}${result.stderr ?? ''}`;
	fs.writeFileSync(path.join(REPORT_DIR, reportName), output, 'utf8');
	const status = result.error ? 127 : (result.status ?? 1);
	console.log(
		`[${name}] exit=${status} report=security-reports/${reportName}`,
	);
	return status;
}

function runDetectSecrets(commandArgs) {
	const status = runCheck(
		'detect-secrets',
		'detect-secrets',
		commandArgs,
		'detect-secrets.json',
		REPO_ROOT,
	);
	if (status !== 0) return status;

	let payload;
	try {
		payload = JSON.parse(
			fs.readFileSync(
				path.join(REPORT_DIR, 'detect-secrets.json'),
				'utf8',
			),
		);
	} catch {
		console.log('[detect-secrets] invalid JSON report');
		return 1;
	}

	let findingCount = 0;
	let ignoredTestKeywords = 0;
	for (const [filename, entries] of Object.entries(payload.results ?? {})) {
		const isTestFixture =
			filename.includes('/test/') ||
			filename.includes('/tests/') ||
			filename.includes('/runtime-fixtures/');
		for (const entry of entries) {
			if (isTestFixture && entry.type === 'Secret Keyword') {
				ignoredTestKeywords += 1;
				continue;
			}
			findingCount += 1;
		}
	}
	console.log(
		`[detect-secrets] findings=${findingCount} ignored-test-keywords=${ignoredTestKeywords}`,
	);
	return findingCount > 0 ? 1 : 0;
}

function walkFiles(directory, files = []) {
	if (!fs.existsSync(directory)) return files;
	for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
		if (EXCLUDED.has(entry.name)) continue;
		const fullPath = path.join(directory, entry.name);
		if (entry.isDirectory()) {
			walkFiles(fullPath, files);
		} else if (/\.(?:c?js|mjs|ts|json)$/i.test(entry.name)) {
			files.push(fullPath);
		}
	}
	return files;
}

function builtinPatternScan() {
	const findings = [];
	for (const sourceDir of SOURCE_DIRS) {
		for (const file of walkFiles(path.join(ROOT, sourceDir))) {
			if (
				path.resolve(file) ===
				path.resolve(fileURLToPath(import.meta.url))
			)
				continue;
			const content = fs.readFileSync(file, 'utf8');
			const lines = content.split(/\r?\n/);
			lines.forEach((line, index) => {
				for (const [category, pattern] of RISK_PATTERNS) {
					if (pattern.test(line)) {
						findings.push({
							category,
							file: path.relative(ROOT, file),
							line: index + 1,
							text: line.trim().slice(0, 300),
						});
					}
				}
			});
		}
	}
	ensureReportDir();
	fs.writeFileSync(
		path.join(REPORT_DIR, 'typescript-dangerous-patterns.json'),
		`${JSON.stringify(findings, null, 2)}\n`,
		'utf8',
	);
	const blocking = findings.filter((finding) =>
		BLOCKING_PATTERN_CATEGORIES.has(finding.category),
	);
	console.log(
		`[dangerous-patterns] findings=${findings.length} blocking=${blocking.length} report=security-reports/typescript-dangerous-patterns.json`,
	);
	return blocking.length > 0 ? 1 : 0;
}

function main() {
	let failures = 0;
	const missing = [];

	failures += builtinPatternScan() !== 0 ? 1 : 0;

	failures +=
		runCheck(
			'npm-audit',
			'npm',
			['audit', '--json', '--audit-level=low'],
			'npm-audit.json',
		) !== 0
			? 1
			: 0;

	if (fs.existsSync(path.join(ROOT, 'pnpm-lock.yaml'))) {
		if (commandExists('pnpm')) {
			failures +=
				runCheck(
					'pnpm-audit',
					'pnpm',
					['audit', '--json'],
					'pnpm-audit.json',
				) !== 0
					? 1
					: 0;
		} else {
			console.log(
				'[pnpm-audit] skipped: pnpm-lock.yaml exists but pnpm is not installed',
			);
			missing.push('pnpm');
		}
	}

	if (!NO_SEMGREP) {
		if (commandExists('semgrep')) {
			failures +=
				runCheck(
					'semgrep',
					'semgrep',
					[
						'scan',
						'--config',
						'auto',
						'--severity',
						'ERROR',
						'--error',
						'--json',
						'--exclude',
						'node_modules',
						'--exclude',
						'dist',
						'--exclude',
						'build',
						'--exclude',
						'coverage',
						'--exclude',
						'security-reports',
						'--exclude',
						'scripts/security-audit.js',
						'src',
						'bin',
						'examples',
						'scripts',
						'test',
						'runtime-fixtures',
					],
					'semgrep.json',
				) !== 0
					? 1
					: 0;
		} else {
			console.log('[semgrep] skipped: executable not installed');
			missing.push('semgrep');
		}
	}

	if (commandExists('detect-secrets')) {
		failures +=
			runDetectSecrets([
				'scan',
				'--all-files',
				'--exclude-files',
				'(^|/)(node_modules|dist|build|coverage|security-reports|unused)(/|$)',
				'typescript/src',
				'typescript/bin',
				'typescript/examples',
				'typescript/scripts',
				'typescript/test',
				'typescript/runtime-fixtures',
				'typescript/package.json',
				'typescript/package-lock.json',
			]) !== 0
				? 1
				: 0;
	} else {
		console.log('[detect-secrets] skipped: executable not installed');
		missing.push('detect-secrets');
	}

	if (missing.length > 0) {
		console.log(
			`Missing scanners/tools: ${[...new Set(missing)].sort().join(', ')}`,
		);
		if (STRICT_TOOLS) failures += 1;
	}

	process.exitCode = failures > 0 ? 1 : 0;
}

main();
