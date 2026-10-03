import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MAKEFILE = fs.readFileSync(path.join(ROOT, 'Makefile'), 'utf8');
const PACKAGE_JSON = JSON.parse(
  fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'),
);

test('install-dev provisions all TypeScript security audit tools', () => {
  assert.match(MAKEFILE, /npm install --include=dev/);
  assert.match(MAKEFILE, /rm -rf \$\(SECURITY_VENV\)/);
  assert.match(
    MAKEFILE,
    /\$\(UV\) venv --python 3\.12 \$\(SECURITY_VENV\)/,
  );
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
