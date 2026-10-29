# action-guard.github.io
Web page code of Agent Action Guard project.

Link: https://action-guard.github.io/

Main repository link:
https://github.com/Pro-GenAI/Agent-Action-Guard

`./tests` folder contains the `pytest` test cases for the project.

Run the website regression suite without additional dependencies:

```bash
python3 -m unittest discover -s docs/website/tests -v
node --test docs/website/tests/tabs.test.cjs
```

Run that command from the repository root. All browser assets are served locally,
including licensed fonts and Highlight.js 11.11.1; see `assets/THIRD_PARTY.md`.
The `/privacy/` page documents website and runtime data handling, cookie use,
and access/correction/deletion requests. `/terms/` explains licensing, current
free-access/payment/refund scope, benchmark limitations, and project contact.
Keep policy navigation, skip links, tab semantics, and sitemap entries synchronized.
See the twenty-area inventory in `../PRIVACY_REVIEW.md` before adding
accounts, tracking, email, billing, uploads, or hosted action processing.

Changed browser assets use a version query in HTML. Bump it when a markup/script
or stylesheet contract changes so cached files do not conflict with new HTML.

The homepage and documentation must keep the supported pre-tool-use hook targets synchronized with the Python package: Codex, Claude Code, Cursor, Kiro, OpenCode, Antigravity CLI (`agy`), GitHub Copilot CLI, OpenClaw, and Hermes Agent.

See `../LAUNCH_REVIEW.md` for the launch checklist, external-link checking,
HTTPS/search handoff, image optimization, and verified source snapshots. Browser
automation lives in `../browser-tests/` and adds no runtime dependencies to the
static site. `.github/workflows/website-checks.yml` runs the local checks and
desktop/mobile Chromium/WebKit tests after push. Native Safari remains a separate
device check. Sharing-card SVG/JPEG assets and the public privacy inventory are
served from this site; keep their source and exported copies synchronized.

The public HTML includes a hash-based content security policy. After editing
inline scripts or event handlers, run `python3 docs/scripts/site_security.py`
from the repository root; CI checks it with `--check`. Script execution and
same-origin assets are exercised by the browser tests. See
[security maintenance](../security-maintenance.md) for deployment requirements
and the response-header controls that GitHub Pages cannot configure here.
