import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const MAKEFILE = fs.readFileSync(new URL('../Makefile', import.meta.url), 'utf8');

test('make test and Node version matrix use configurable parallel workers', () => {
	assert.match(MAKEFILE, /TEST_WORKERS \?=/);
	assert.match(MAKEFILE, /npm test -- --test-concurrency=\$\(TEST_WORKERS\)/);
	assert.match(MAKEFILE, /TEST_WORKERS="\$\(TEST_WORKERS\)" npm run test:all-versions/);
});
