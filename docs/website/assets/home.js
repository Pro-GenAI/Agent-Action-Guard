(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia('(pointer: fine)').matches;
  const root = document.documentElement;
  const nav = document.querySelector('.site-nav');
  const progress = document.getElementById('scroll-progress');
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Ambient cursor light.
  if (!reducedMotion && finePointer) {
    let frame = null;
    window.addEventListener('pointermove', (event) => {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        root.style.setProperty('--mouse-x', `${event.clientX}px`);
        root.style.setProperty('--mouse-y', `${event.clientY}px`);
      });
    }, { passive: true });
  }

  // Scroll progress and glass navigation.
  const onScroll = () => {
    const top = window.scrollY || root.scrollTop;
    const max = root.scrollHeight - window.innerHeight;
    if (progress) progress.style.transform = `scaleX(${max > 0 ? Math.min(top / max, 1) : 0})`;
    nav?.classList.toggle('is-scrolled', top > 16);
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });

  // Constellation background in the hero.
  const canvas = document.getElementById('constellation');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    let width = 0, height = 0, nodes = [], running = true, mouse = { x: -9999, y: -9999 };
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.min(90, Math.round((width * height) / 16000));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: reducedMotion ? 0 : (Math.random() - 0.5) * 0.28,
        vy: reducedMotion ? 0 : (Math.random() - 0.5) * 0.28,
        r: Math.random() * 1.4 + 0.6,
        hot: Math.random() < 0.12,
      }));
    };
    const draw = () => {
      if (!running) return;
      ctx.clearRect(0, 0, width, height);
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > width) n.vx *= -1;
        if (n.y < 0 || n.y > height) n.vy *= -1;
      }
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 19600) {
            const alpha = (1 - d2 / 19600) * 0.22;
            ctx.strokeStyle = `rgba(121, 255, 91, ${alpha})`;
            ctx.lineWidth = 0.7;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        const mdx = a.x - mouse.x, mdy = a.y - mouse.y;
        const md = Math.sqrt(mdx * mdx + mdy * mdy);
        const near = md < 180 ? 1 - md / 180 : 0;
        if (near > 0) {
          ctx.strokeStyle = `rgba(121, 255, 91, ${near * 0.45})`;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
        }
        ctx.fillStyle = a.hot ? `rgba(121, 255, 91, ${0.7 + near * 0.3})` : `rgba(187, 198, 226, ${0.45 + near * 0.5})`;
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + near * 1.6, 0, Math.PI * 2); ctx.fill();
      }
      if (!reducedMotion) requestAnimationFrame(draw);
    };
    resize();
    window.addEventListener('resize', () => { resize(); if (reducedMotion) draw(); }, { passive: true });
    canvas.parentElement.parentElement.addEventListener('pointermove', (event) => {
      const rect = canvas.getBoundingClientRect();
      mouse = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      if (reducedMotion) requestAnimationFrame(draw);
    }, { passive: true });
    canvas.parentElement.parentElement.addEventListener('pointerleave', () => { mouse = { x: -9999, y: -9999 }; });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([entry]) => {
        const wasRunning = running;
        running = entry.isIntersecting;
        if (running && !wasRunning && !reducedMotion) requestAnimationFrame(draw);
      }).observe(canvas);
    }
    requestAnimationFrame(draw);
  }

  // Illustrative replay console in the hero.
  const consoleEl = document.getElementById('console');
  if (consoleEl) {
    const feed = [
      { agent: 'support-agent', name: 'send_email', args: { to: 'user@example.com', subject: 'Status update', body: 'Hello from Action Guard' }, harmful: false },
      { agent: 'ops-agent', name: 'file_delete', args: { target: '/important/data.txt' }, harmful: true },
      { agent: 'calendar-agent', name: 'create_event', args: { title: 'Design review', when: 'Thu 10:00' }, harmful: false },
      { agent: 'data-agent', name: 'data_exporter', args: { dataset: 'employee_salaries', destination: 'xyz' }, harmful: true },
      { agent: 'research-agent', name: 'web_search', args: { query: 'onnx runtime cpu benchmarks' }, harmful: false },
      { agent: 'devops-agent', name: 'run_shell', args: { command: 'rm -rf / --no-preserve-root' }, harmful: true },
    ];
    const $ = (id) => document.getElementById(id);
    const text = $('c-text'), code = $('c-code'), stamp = $('c-stamp'), meter = $('c-meter');
    const score = $('c-score'), lat = $('c-lat'), log = $('c-log'), agent = $('c-agent');
    const counts = { total: 0, allowed: 0, blocked: 0 };
    let visible = true;
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(consoleEl);
    }
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const tokens = (item) => {
      const out = [['{\n', 'p'], ['  "name"', 'k'], [': ', 'p'], [`"${item.name}"`, 'fn'], [',\n', 'p'], ['  "arguments"', 'k'], [': {\n', 'p']];
      const entries = Object.entries(item.args);
      entries.forEach(([key, value], index) => {
        out.push([`    "${key}"`, 'k'], [': ', 'p'], [`"${value}"`, 's'], [index < entries.length - 1 ? ',\n' : '\n', 'p']);
      });
      out.push(['  }\n}', 'p']);
      return out;
    };
    const setStamp = (kind, label, icon) => {
      stamp.className = `stamp ${kind}`;
      stamp.innerHTML = kind === 'wait'
        ? `<span class="spinner"></span><span>${label}</span>`
        : `${icon ? `<svg class="ico"><use href="#${icon}"/></svg>` : ''}<span>${label}</span>`;
    };
    const pushLog = (item, ms) => {
      const li = document.createElement('li');
      li.innerHTML = item.harmful
        ? `<svg class="ico no"><use href="#i-x"/></svg><span class="nm">${esc(item.name)}</span><span class="tag no">blocked</span><span class="ms">${ms}</span>`
        : `<svg class="ico ok"><use href="#i-check"/></svg><span class="nm">${esc(item.name)}</span><span class="tag ok">allowed</span><span class="ms">${ms}</span>`;
      log.prepend(li);
      while (log.children.length > 4) log.lastElementChild.remove();
    };
    const updateCounts = () => {
      $('c-total').textContent = counts.total;
      $('c-allowed').textContent = counts.allowed;
      $('c-blocked').textContent = counts.blocked;
    };
    const run = async () => {
      let index = 0;
      while (true) {
        if (document.hidden || !visible) { await sleep(400); continue; }
        const item = feed[index % feed.length];
        index += 1;
        consoleEl.classList.remove('is-blocked');
        agent.textContent = item.agent;
        setStamp('', 'Incoming');
        meter.style.width = '0';
        score.textContent = '—';
        lat.textContent = '—';
        text.innerHTML = '';
        for (const [chunk, cls] of tokens(item)) {
          const span = document.createElement('span');
          span.className = cls;
          text.appendChild(span);
          for (let i = 0; i < chunk.length; i += 2) {
            span.textContent = chunk.slice(0, i + 2);
            await sleep(14);
          }
        }
        setStamp('wait', 'Screening');
        code.classList.add('is-scanning');
        await sleep(950);
        code.classList.remove('is-scanning');
        const ms = (15 + Math.random() * 9).toFixed(1);
        const harm = item.harmful ? 0.9 + Math.random() * 0.09 : 0.01 + Math.random() * 0.06;
        counts.total += 1;
        if (item.harmful) {
          counts.blocked += 1;
          consoleEl.classList.add('is-blocked');
          setStamp('harm', 'Blocked', 'i-shield-x');
        } else {
          counts.allowed += 1;
          setStamp('safe', 'Allowed', 'i-shield-check');
        }
        meter.style.width = `${(harm * 100).toFixed(0)}%`;
        score.textContent = harm.toFixed(2);
        lat.textContent = `${ms} ms`;
        updateCounts();
        await sleep(2100);
        pushLog(item, `${ms}ms`);
      }
    };
    run();
  }

  // Reveal on scroll.
  const reveals = [...document.querySelectorAll('.rv')];
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('in');
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add('in'));
  }

  // Count-up statistics.
  const counters = [...document.querySelectorAll('[data-count]')];
  const countUp = (el) => {
    const target = Number(el.dataset.count);
    const decimals = Number(el.dataset.decimals || 0);
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min((now - start) / 1600, 1);
      const eased = 1 - Math.pow(1 - t, 4);
      el.textContent = (target * eased).toFixed(decimals);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if ('IntersectionObserver' in window) {
    const co = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        countUp(entry.target);
        co.unobserve(entry.target);
      });
    }, { threshold: 0.6 });
    counters.forEach((el) => co.observe(el));
  }

  // Spotlight + tilt on cards.
  if (!reducedMotion && finePointer) {
    document.querySelectorAll('.card').forEach((card) => {
      card.addEventListener('pointermove', (event) => {
        const rect = card.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        card.style.setProperty('--cx', `${x}px`);
        card.style.setProperty('--cy', `${y}px`);
        const rx = (0.5 - y / rect.height) * 3.5;
        const ry = (x / rect.width - 0.5) * 3.5;
        card.style.transform = `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-3px)`;
      });
      card.addEventListener('pointerleave', () => { card.style.transform = ''; });
    });
  }

  // Generic tab groups.
  const bindTabs = (tabs, onChange) => {
    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        tabs.forEach((other) => {
          const selected = other === tab;
          other.setAttribute('aria-selected', String(selected));
          const panelId = other.getAttribute('aria-controls');
          const panel = panelId && document.getElementById(panelId);
          if (panel) {
            panel.hidden = !selected;
            if (selected) { panel.classList.remove('panel-anim'); void panel.offsetWidth; panel.classList.add('panel-anim'); }
          }
        });
        onChange?.(tab);
      });
      tab.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
        const index = tabs.indexOf(tab);
        const next = tabs[(index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
        next.focus();
        next.click();
      });
    });
  };
  bindTabs([...document.querySelectorAll('#demo .tab')]);
  bindTabs([...document.querySelectorAll('.code-head .tab')]);

  // Copy helpers.
  const copyText = async (value) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const textarea = document.createElement('textarea');
    textarea.value = value;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  };
  const flashCopied = (button, label) => {
    const use = button.querySelector('use');
    const span = button.querySelector('span');
    button.classList.add('is-copied');
    use?.setAttribute('href', '#i-check');
    if (span) span.textContent = 'Copied';
    button.setAttribute('aria-label', 'Copied');
    setTimeout(() => {
      button.classList.remove('is-copied');
      use?.setAttribute('href', '#i-copy');
      if (span) span.textContent = 'Copy';
      button.setAttribute('aria-label', label);
    }, 1400);
  };

  const installCode = document.getElementById('install-code');
  const installTabs = [...document.querySelectorAll('.install-tab')];
  installTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      installTabs.forEach((other) => other.setAttribute('aria-selected', String(other === tab)));
      installCode.textContent = tab.dataset.cmd;
    });
  });
  const heroCopy = document.getElementById('hero-copy-install');
  heroCopy?.addEventListener('click', async () => {
    try {
      await copyText(installCode.textContent);
      flashCopied(heroCopy, 'Copy install command');
    } catch (error) {
      console.warn('Copy failed', error);
    }
  });

  const codeCopy = document.getElementById('code-copy');
  codeCopy?.addEventListener('click', async () => {
    const panel = [...document.querySelectorAll('.code-panel')].find((p) => !p.hidden);
    const code = panel?.querySelector('pre code');
    if (!code) return;
    try {
      await copyText(code.innerText);
      flashCopied(codeCopy, 'Copy code');
    } catch (error) {
      console.warn('Copy failed', error);
    }
  });

  window.addEventListener("DOMContentLoaded", () => {
    if (!window.hljs) return;
    document.querySelectorAll("#docs pre code").forEach((code) => window.hljs.highlightElement(code));
  });

  // Keep navigation state synced with the section in view.
  const navLinks = [...document.querySelectorAll('.nav-link[href^="#"]')];
  const targets = navLinks
    .map((link) => ({ link, target: document.querySelector(link.getAttribute('href')) }))
    .filter((item) => item.target);
  if ('IntersectionObserver' in window) {
    const navObserver = new IntersectionObserver((entries) => {
      const visibleEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visibleEntry) return;
      const match = targets.find((item) => item.target === visibleEntry.target);
      if (match) navLinks.forEach((link) => link.classList.toggle('is-active', link === match.link));
    }, { rootMargin: '-35% 0px -55% 0px', threshold: 0 });
    targets.forEach((item) => navObserver.observe(item.target));
  }
})();
