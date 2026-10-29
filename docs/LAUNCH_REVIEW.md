# Website launch review

Reviewed 29 October 2026 against the [launch-checklist reel](https://www.instagram.com/reel/DecGaf1A4Fe/).
Scope: the project documentation and both static sites. Changes in this checkout
are local until committed and published; this review does not claim that the
production sites already contain them.

The operator explicitly chose to retain tracking-free sites and the current
public contact email. Google Analytics and a domain mailbox were not introduced.

| Area | Applied checks and current status |
| --- | --- |
| Small screens | Responsive layouts retain dense benchmark results. Fixed documentation grid minimums and contained wide tables in keyboard-focusable scroll regions so long examples do not widen the entire page. Local browser QA checks 390px views; the browser suite covers 390px, 760px, and 1280px on Chromium/WebKit. |
| Navigation | Offline tests resolve every local and cross-site page link and fragment. Repository README links now point to real files, including the Python README's sibling TypeScript path. Empty badge destinations are removed. The optional external checker distinguishes real HTTP errors from blocked/time-out checks. |
| Contact | Existing email links and the data-request subject use the published maintainer address. No contact form or form backend exists, so no test message is sent. Actual mailbox delivery is an operator check. |
| Loading and media | The demo has a lossless WebP alternative: 73,586 bytes versus the 92,348-byte GIF (18,762 bytes / 20.3% saved). All 42 decoded RGBA frames and frame durations were compared. The original GIF remains the fallback. Below-fold images load lazily, decode asynchronously where appropriate, and reserve dimensions. Other trial conversions increased size and were not adopted. Asset budgets are regression-tested. |
| Transport | Read-only requests to both public homepages succeeded over HTTPS; HTTP requests redirected to HTTPS. No fake browser-padlock UI or certificate bypass is added. Custom-domain DNS/certificate changes, if introduced, require the site operator. |
| Browser coverage | Dependency-free local tests cover tab/copy behavior. The new CI suite defines 66 Chromium/WebKit desktop/tablet/mobile scenarios with retained failure diagnostics. Test discovery is checked locally. The CI workflow has not run from this unpublished checkout; WebKit is not a claim of native Safari verification. Run native Safari/iOS checks before release. |
| Copy and runtime facts | Corrected the documentation and structured metadata to Python 3.10–3.14 and Node.js 20+. Removed stale absolute safety claims from homepage metadata. Existing benchmark values remain unchanged. |
| Messaging links | There is no WhatsApp contact button or known project phone number. No phone number, chat service, or unsupported integration is invented. Open Graph sharing metadata is updated independently of a WhatsApp button. |
| Mail identity | Retained the current public email per the operator's choice. No domain ownership, DNS, or mailbox provisioning is claimed. |
| Search discovery | Existing robots files, full public-route sitemaps, canonical URLs, and ownership-verification files are preserved. Analytics is omitted by request. After publication, the owner can submit each sitemap in Search Console; submission/indexing is not claimed here. |
| Sharing previews | All seven public HTML pages have same-origin 1200×630 JPEG cards, Open Graph image dimensions/type/alt text, and matching Twitter image metadata. The main site and leaderboard have distinct editable SVG artwork. No WhatsApp message is sent, and third-party preview-cache refresh still depends on publication and the platform. |
| Recoverability | `backup_sites.py` creates a new allowlisted ZIP including uncommitted site sources, tooling, policy inventories, selected project documentation, and a SHA-256 manifest. It excludes local tools, reports, caches, `.env*`, and symlinks. Verification and restore-content checks run locally. Successful non-PR CI runs retain a source artifact for 30 days once the workflow is pushed. A local ZIP is not an off-device backup by itself. |
| Checkout | There is no payment page, billing provider, recurring subscription, or project charge. The published terms explain that scope; payment testing becomes relevant if a paid offering is introduced. |

## Repeatable checks

From the repository root:

```bash
python3 -m unittest discover -s docs/website/tests -v
node --test docs/website/tests/tabs.test.cjs
python3 docs/scripts/check_external_links.py
python3 docs/scripts/backup_sites.py
python3 docs/scripts/backup_sites.py --verify .cache/site-backups/<snapshot>.zip
```

External checks use bounded parallel requests and default TLS verification.
They fail for 404/410 responses. Authentication walls, rate limits, challenges,
and timeouts remain unverified and need a manual check; a 200 challenge page is
not counted as proof that the requested content loaded. The JSON report is under
`.cache/site-launch/` and is not published. Own-site links resolve against the
checkout so new, unpublished policy pages do not cause false live-site failures.

The article reference now uses the working `www.mind-verse.de` origin. The
earlier pass replaced unavailable stargazer-directory links with public
profiles. The subsequent eleven-point review removed those three cards and the
like card because profile/post links did not establish the claimed interaction.
Retained source-linked mentions omit employer titles; social-source access may
still require manual verification before release. See the follow-up in
`docs/PRIVACY_REVIEW.md`.

Browser automation is isolated from shipped website assets:

```bash
cd docs/browser-tests
npm ci --ignore-scripts
npx playwright install --with-deps chromium webkit
npm run test:browser
```

For native Safari/iOS, open both sites at desktop/mobile sizes, tab through skip
links and all three homepage tab groups, switch PIP/NPM and Python/JavaScript,
check the demo animation/fallback, check horizontal table scrolling, and inspect
the privacy/terms links and sharing image. Use the published HTTPS origins for
the final device pass. Chromium/WebKit automation does not substitute for every
device or assistive-technology check.

## Source assets and restore procedure

The sharing cards are code-native SVG artwork with generated JPEG exports. To
regenerate them, or re-check a changed demo animation:

```bash
uv run --no-project --with cairosvg --with pillow python docs/scripts/render_social_cards.py
uv run --no-project --with pillow python docs/scripts/optimize_demo.py
```

Keep the GIF and its attribution, and regenerate `demo-optimization.json` after
changing it. The optimizer verifies decoded pixels/timing and a smaller output
before replacing the WebP. These optional authoring dependencies are not needed
to serve the static websites or run the standard-library checks.

Verify a snapshot before restoration. Extract it into a new empty directory,
inspect `manifest.json` and the selected files, then copy reviewed files into a
recovery branch. The backup command never overwrites existing snapshots. Git
history plus the CI source artifacts provide versioned recovery once changes
are committed/pushed; retain an independent off-device copy for disaster recovery.

## Before publishing

- Run all local checks and the full browser matrix; inspect any unverified
  external URLs manually instead of describing them as passed.
- Check native Safari/iOS and real email delivery. No form/payment/WhatsApp
  delivery checks apply to the current feature set.
- Commit the reviewed source and publish both site directories through their
  existing GitHub Pages process. This task does not push or deploy automatically.
- Recheck live HTTPS, social-card URLs, policy URLs, robots, and sitemaps; then
  submit/update the sitemaps in the owner's Search Console account.
- Confirm the source artifact is retained on a successful workflow run and keep
  an independent backup. Retention is finite and local `.cache/` can be deleted.

## Primary references

- [GitHub Pages HTTPS](https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https)
- [Google sitemap publication and submission](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap)
- [Open Graph metadata](https://ogp.me/)
- [Playwright browser support and WebKit](https://playwright.dev/docs/browsers)
