const path = require('node:path');
const { defineConfig } = require('@playwright/test');
const root = path.resolve(__dirname, '../..');

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: '**/*.browser.spec.cjs',
  timeout: 30000,
  retries: process.env.CI ? 1 : 0,
  workers: 2,
  reporter: [['list'], ['html', { open: 'never', outputFolder: path.join(root, 'docs/browser-tests/playwright-report') }]],
  outputDir: path.join(root, 'docs/browser-tests/test-results'),
  use: { trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: ['chromium', 'webkit'].flatMap((browserName) => [
    { name: `${browserName}-desktop`, use: { browserName, viewport: { width: 1280, height: 900 } } },
    { name: `${browserName}-tablet`, use: { browserName, viewport: { width: 760, height: 900 } } },
    { name: `${browserName}-mobile`, use: { browserName, viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' } },
  ]),
  webServer: [
    { command: 'python3 -m http.server 8765 --bind 127.0.0.1 --directory docs/website', cwd: root, url: 'http://127.0.0.1:8765/', reuseExistingServer: !process.env.CI },
    { command: 'python3 -m http.server 8766 --bind 127.0.0.1 --directory docs/agent-leaderboard-site', cwd: root, url: 'http://127.0.0.1:8766/', reuseExistingServer: !process.env.CI },
  ],
});
