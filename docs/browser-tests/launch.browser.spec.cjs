const { test, expect } = require('@playwright/test');

for (const [port, paths] of [[8765, ['/', '/docs/', '/privacy/', '/terms/']], [8766, ['/', '/privacy/', '/terms/']]]) {
  for (const route of paths) {
    test(`${port}${route} has a usable mobile/desktop page and sharing card`, async ({ page }) => {
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      const response = await page.goto(`http://127.0.0.1:${port}${route}`);
      expect(response.status()).toBe(200);
      await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
      await page.keyboard.press('Tab');
      await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused();
      await page.keyboard.press('Enter');
      await expect(page.locator('#main-content')).toBeFocused();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await expect(page.locator('meta[property="og:image"]')).toHaveAttribute('content', /https:\/\/[^/]+\/assets\/social-card.jpg$/);
      await expect(page.getByRole('navigation', { name: 'Footer', exact: true }).getByRole('link', { name: 'Data requests' })).toBeVisible();
      expect(errors).toEqual([]);
    });
  }
}

test('homepage tabs retain keyboard focus and panel visibility', async ({ page }) => {
  await page.goto('http://127.0.0.1:8765/');
  for (const [first, second, panel] of [['pip', 'npm', '#install-npm'], ['Live Demo Execution', 'System Workflow', '#panel-flow'], ['Python', 'JavaScript', '#code-js']]) {
    const tab = page.getByRole('tab', { name: first, exact: true });
    await tab.click();
    await tab.press('ArrowRight');
    await expect(page.getByRole('tab', { name: second, exact: true })).toBeFocused();
    await expect(page.locator(panel)).toBeVisible();
    await page.keyboard.press('Home');
    await expect(tab).toBeFocused();
  }
});

test('benchmark content remains visible with scripting disabled', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:8766/');
  await expect(page.getByRole('heading', { name: 'Performance Rankings' })).toBeVisible();
  await expect(page.locator('tbody tr')).toHaveCount(19);
  await context.close();
});

test('copy controls announce failure and success without losing keyboard focus', async ({ page }) => {
  await page.addInitScript(() => {
    let calls = 0;
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async () => { if (++calls === 1) throw new Error('Clipboard denied'); },
    } });
  });
  await page.goto('http://127.0.0.1:8765/');
  const button = page.locator('#hero-copy-install');
  await button.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#install-copy-status')).toContainText('Copy unavailable');
  await expect(button).toBeFocused();
  await expect(button).not.toHaveClass(/is-copied/);
  await page.keyboard.press('Enter');
  await expect(page.locator('#install-copy-status')).toHaveText('Install command copied to clipboard.');
  await expect(button).toBeFocused();
  await page.locator('#code-copy').click();
  await expect(page.locator('#code-copy-status')).toHaveText('Code copied to clipboard.');
});

test('unavailable fallback copy never reports success and restores focus', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    document.execCommand = () => false;
  });
  await page.goto('http://127.0.0.1:8765/');
  const button = page.locator('#hero-copy-install');
  await button.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#install-copy-status')).toContainText('copy it manually');
  await expect(button).toBeFocused();
  await expect(button).not.toHaveClass(/is-copied/);
  await expect(page.locator('textarea')).toHaveCount(0);
});

test('content security policy blocks injected scripts while bundled code remains usable', async ({ page }) => {
  await page.goto('http://127.0.0.1:8765/');
  expect(await page.evaluate(() => {
    const script = document.createElement('script');
    script.textContent = 'window.__aagUnexpectedScriptExecuted = true';
    document.head.appendChild(script);
    return window.__aagUnexpectedScriptExecuted;
  })).toBeUndefined();
  const javascriptTab = page.getByRole('tab', { name: 'JavaScript', exact: true });
  await javascriptTab.click();
  await expect(javascriptTab).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#code-js')).toBeVisible();
});
