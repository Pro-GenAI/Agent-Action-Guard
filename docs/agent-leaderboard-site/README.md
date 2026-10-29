# agent-leaderboard.github.io
Leaderboard for AI agent action safety

Fonts, provider marks, and styles are served from this site's own origin.
See `assets/THIRD_PARTY.md` for bundled licenses. Small UI icons are inline SVGs.
Keep `/privacy/`, `robots.txt`, and `sitemap.xml` synchronized with public routes.

After changing utility classes or the theme, rebuild the checked-in CSS from the
repository root (Node/npm is needed only when rebuilding):

```bash
npx --yes --package tailwindcss@3.4.19 tailwindcss -c docs/agent-leaderboard-site/tailwind.config.cjs -i docs/agent-leaderboard-site/tailwind.input.css -o docs/agent-leaderboard-site/assets/tailwind.css --minify
python3 -m unittest discover -s docs/website/tests -v
node --test docs/website/tests/tabs.test.cjs
```

The static site needs no package install or runtime Tailwind CDN script to serve.
See `../PRIVACY_REVIEW.md` for the six-area feature/privacy review.

That review also records the follow-up twenty-area checklist. Keep `/terms/`,
cookie information and data-request links in the footer, preserve the skip link
and focus styles, and leave leaderboard content visible when JavaScript is
unavailable. The methodology link points to the published evaluation documentation.

See `../LAUNCH_REVIEW.md` for the launch checks and backup/restore procedure.
Sharing metadata uses this origin's `assets/social-card.jpg` (1200×630), with an
editable SVG beside it. The public policy inventory is `assets/privacy-review.md`.
Browser checks live in `../browser-tests/`; Chromium/WebKit coverage is distinct
from a native Safari/device pass. The website workflow retains source snapshots
on successful non-PR runs after it is pushed.

The public HTML includes a hash-based content security policy. After editing
inline scripts or event handlers, run `python3 docs/scripts/site_security.py`
from the repository root; CI checks it with `--check`. Script execution and
same-origin assets are exercised by the browser tests. See
[security maintenance](../security-maintenance.md) for deployment requirements
and the response-header controls that GitHub Pages cannot configure here.
