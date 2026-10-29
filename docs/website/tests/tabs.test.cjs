const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Execute the shipped script with a small DOM fixture, without browser/test dependencies.
const source = fs.readFileSync(path.join(__dirname, '../assets/home.js'), 'utf8');
function fixture({ count = 2, selected = 0, missingPanels = false, clipboard = 'success',
  secure = true, execResult = true, execError = false, selectError = false,
  writeText, missingCode = false, missingFeedback = false } = {}) {
  const ids = new Map();
  const copied = [];
  const timers = new Map();
  let timerId = 0;
  const temporary = new Set();
  const document = {
    documentElement: { scrollTop: 0, scrollHeight: 1000 },
    getElementById: (id) => ids.get(id) || null,
    querySelector: () => null,
  };
  class Element {
    constructor(attrs = {}) {
      this.attrs = { ...attrs };
      this.listeners = new Map();
      this.dataset = {};
      const classes = new Set();
      this.classList = { add: (name) => classes.add(name), remove: (name) => classes.delete(name),
        toggle: (name, force) => force ? classes.add(name) : classes.delete(name), contains: (name) => classes.has(name) };
      this.textContent = '';
      this.hidden = false;
      this.tabIndex = 0;
    }
    getAttribute(name) { return this.attrs[name] ?? null; }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    addEventListener(name, callback) {
      const callbacks = this.listeners.get(name) || [];
      this.listeners.set(name, [...callbacks, callback]);
    }
    querySelector() { return null; }
    focus() { document.activeElement = this; }
    async dispatch(name, details = {}) {
      const event = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...details };
      await Promise.all((this.listeners.get(name) || []).map((callback) => callback(event)));
      return event;
    }
    click() { return this.dispatch('click'); }
    key(key) { return this.dispatch('keydown', { key }); }
  }
  const groups = ['demo', 'code', 'install'].map((name) => {
    const tabs = Array.from({ length: count }, (_, index) => {
      const panelId = `${name}-panel-${index}`;
      const tab = new Element({ 'aria-selected': String(index === selected), 'aria-controls': panelId });
      tab.dataset.cmd = `${name} command ${index}`;
      if (!missingPanels) ids.set(panelId, new Element());
      return tab;
    });
    return tabs;
  });
  const selectors = new Map([
    ['#demo .tab', groups[0]], ['.code-head .tab', groups[1]], ['.install-tab', groups[2]],
  ]);
  document.querySelectorAll = (selector) => selectors.get(selector) || [];
  document.body = { appendChild: (element) => temporary.add(element) };
  document.createElement = () => {
    const element = new Element();
    element.style = {};
    element.select = () => {
      element.focus();
      if (selectError) throw new Error('selection failed');
    };
    element.remove = () => temporary.delete(element);
    return element;
  };
  document.execCommand = () => {
    if (execError) throw new Error('copy failed');
    if (execResult) copied.push([...temporary][0].value);
    return execResult;
  };
  const copyButton = new Element();
  ids.set('hero-copy-install', copyButton);
  const codeButton = new Element();
  const codeLabel = new Element();
  const codeIcon = new Element();
  codeButton.querySelector = (selector) => selector === 'span' ? codeLabel : codeIcon;
  ids.set('code-copy', codeButton);
  const codePanels = groups[1].map((tab, index) => {
    const panel = ids.get(tab.getAttribute('aria-controls')) || new Element();
    panel.querySelector = () => missingCode ? null : { innerText: `code example ${index}` };
    return panel;
  });
  selectors.set('.code-panel', codePanels);
  if (!missingFeedback) {
    ids.set('install-copy-status', new Element());
    ids.set('code-copy-status', new Element());
  }
  vm.runInNewContext(source, {
    document,
    window: {
      matchMedia: (query) => ({ matches: query.includes('reduced-motion') }),
      innerHeight: 800, scrollY: 0, isSecureContext: secure, addEventListener() {},
    },
    navigator: { clipboard: clipboard === 'missing' ? undefined : { writeText: writeText || (async (value) => {
      if (clipboard === 'denied') throw new Error('permission denied');
      copied.push(value);
    }) } },
    setTimeout(callback) { const id = ++timerId; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); }, console,
  });
  function check(group, active) {
    group.forEach((tab, index) => {
      assert.equal(tab.getAttribute('aria-selected'), String(index === active));
      assert.equal(tab.tabIndex, index === active ? 0 : -1);
      const panel = ids.get(tab.getAttribute('aria-controls'));
      if (panel) assert.equal(panel.hidden, index !== active);
    });
  }
  return { groups, document, copied, copyButton, codeButton, codeLabel, codeIcon,
    ids, timers, temporary, check };
}

for (const [index, name] of ['demo', 'code', 'install'].entries()) {
  test(`${name}: keyboard navigation wraps, supports Home/End, and updates focus/panels`, async () => {
    const { groups, document, check } = fixture({ count: 3 });
    const tabs = groups[index];
    check(tabs, 0);
    assert.equal((await tabs[0].key('ArrowLeft')).defaultPrevented, true);
    check(tabs, 2);
    assert.equal(document.activeElement, tabs[2]);
    await tabs[2].key('ArrowRight');
    check(tabs, 0);
    await tabs[0].key('End');
    check(tabs, 2);
    await tabs[2].key('Home');
    check(tabs, 0);
    for (const other of groups.filter((group) => group !== tabs)) check(other, 0);
  });
  test(`${name}: mouse selection retains one tab stop; Tab and unrelated keys are not trapped`, async () => {
    const { groups, check } = fixture();
    const tabs = groups[index];
    await tabs[1].click();
    check(tabs, 1);
    for (const key of ['Tab', 'Escape', 'ArrowUp', 'a']) {
      assert.equal((await tabs[1].key(key)).defaultPrevented, false);
      check(tabs, 1);
    }
  });
}

test('a non-first preselected tab is honored; missing selection defaults to the first', () => {
  for (const selected of [1, -1]) {
    const { groups, check } = fixture({ selected });
    groups.forEach((group) => check(group, Math.max(0, selected)));
  }
});
test('empty groups and missing optional panels do not stop initialization', async () => {
  fixture({ count: 0 });
  const { groups, check } = fixture({ missingPanels: true });
  await groups[0][1].click();
  check(groups[0], 1);
});
test('single-tab groups keep keyboard focus and selection stable', async () => {
  const { groups, check } = fixture({ count: 1 });
  for (const tabs of groups) {
    for (const key of ['ArrowRight', 'ArrowLeft', 'Home', 'End']) await tabs[0].key(key);
    check(tabs, 0);
  }
});
test('copy install command follows keyboard and mouse selection across repeated switches', async () => {
  const { groups, copied, copyButton } = fixture();
  const tabs = groups[2];
  await copyButton.click();
  await tabs[0].key('ArrowRight');
  await copyButton.click();
  await tabs[0].click();
  await copyButton.click();
  assert.deepEqual(copied, ['install command 0', 'install command 1', 'install command 0']);
});

test('successful copying announces the selected code and retains a named retry control', async () => {
  const { groups, copied, codeButton, codeLabel, codeIcon, ids, timers } = fixture();
  await groups[1][1].click();
  await codeButton.click();
  assert.deepEqual(copied, ['code example 1']);
  assert.equal(ids.get('code-copy-status').textContent, 'Code copied to clipboard.');
  assert.equal(codeButton.getAttribute('aria-busy'), 'false');
  assert.equal(codeButton.getAttribute('aria-label'), 'Copy code: copied');
  assert.equal(codeLabel.textContent, 'Copied');
  assert.equal(codeIcon.getAttribute('href'), '#i-check');
  [...timers.values()][0]();
  assert.equal(codeButton.getAttribute('aria-label'), 'Copy code');
  assert.equal(codeLabel.textContent, 'Copy');
  assert.equal(codeIcon.getAttribute('href'), '#i-copy');
});

for (const options of [{ clipboard: 'missing' }, { secure: false }]) {
  test(`fallback copy restores focus and cleans up temporary text: ${JSON.stringify(options)}`, async () => {
    const { document, copyButton, copied, temporary, ids } = fixture(options);
    copyButton.focus();
    await copyButton.click();
    assert.equal(document.activeElement, copyButton);
    assert.equal(temporary.size, 0);
    assert.deepEqual(copied, ['install command 0']);
    assert.equal(ids.get('install-copy-status').textContent, 'Install command copied to clipboard.');
  });
}

for (const options of [{ clipboard: 'denied' },
  { clipboard: 'missing', execResult: false },
  { clipboard: 'missing', execError: true },
  { clipboard: 'missing', selectError: true }]) {
  test(`failed copy reports manual selection without claiming success: ${JSON.stringify(options)}`, async () => {
    const { document, copyButton, copied, temporary, ids } = fixture(options);
    copyButton.focus();
    await copyButton.click();
    assert.equal(copyButton.classList.contains('is-copied'), false);
    assert.equal(copyButton.getAttribute('aria-label'), 'Copy install command: retry');
    assert.equal(copyButton.getAttribute('aria-busy'), 'false');
    assert.match(ids.get('install-copy-status').textContent, /Copy unavailable.*manually/);
    assert.equal(temporary.size, 0);
    assert.equal(document.activeElement, copyButton);
    assert.deepEqual(copied, []);
  });
}

test('repeated copying replaces only that button’s timer; independent controls keep their feedback', async () => {
  const { copyButton, codeButton, timers, ids } = fixture();
  await copyButton.click();
  const oldTimer = [...timers.keys()][0];
  await codeButton.click();
  await copyButton.click();
  assert.equal(timers.has(oldTimer), false);
  assert.equal(timers.size, 2);
  assert.equal(ids.get('code-copy-status').textContent, 'Code copied to clipboard.');
});

test('pending copies ignore duplicate activation while other controls remain usable', async () => {
  let resolve;
  let calls = 0;
  const { copyButton, codeButton, ids } = fixture({ writeText: async () => {
    if (++calls === 1) await new Promise((done) => { resolve = done; });
  } });
  const pending = copyButton.click();
  assert.equal(copyButton.getAttribute('aria-busy'), 'true');
  assert.equal(ids.get('install-copy-status').textContent, 'Copying…');
  await copyButton.click();
  assert.equal(calls, 1);
  await codeButton.click();
  assert.equal(calls, 2);
  resolve();
  await pending;
  assert.equal(copyButton.getAttribute('aria-busy'), 'false');
});

test('missing selections, optional feedback, and absent code do not break copying', async () => {
  const empty = fixture({ count: 0 });
  await empty.copyButton.click();
  await empty.codeButton.click();
  assert.deepEqual(empty.copied, []);
  const missing = fixture({ missingCode: true, missingFeedback: true });
  await missing.codeButton.click();
  await missing.copyButton.click();
  assert.deepEqual(missing.copied, ['install command 0']);
});
