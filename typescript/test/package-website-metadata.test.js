import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const CANONICAL_WEBSITE = 'https://action-guard.pro-genai.com/';
const packageJson = JSON.parse(
	readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
);
const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');

test('npm package metadata points to the canonical project website', () => {
	assert.equal(packageJson.homepage, CANONICAL_WEBSITE);
});

test('npm package README links to the canonical website and docs', () => {
	assert.ok(readme.includes(`[Website](${CANONICAL_WEBSITE})`));
	assert.ok(readme.includes(`[Documentation](${CANONICAL_WEBSITE}docs/)`));
	assert.doesNotMatch(readme, /https?:\\/\\/action-guard\\.github\\.io(?:[/?#]|$)/i);
});
