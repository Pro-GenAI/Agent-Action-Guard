# Security maintenance and October 2026 repairs

The callable guards now include positional and keyword inputs. Python also screens inspectable defaults and arguments bound by `functools.partial`, caches signatures at wrapping time, and preserves sync/async behavior. JavaScript retains its single-object argument shape and represents other calls with an `args` array; nested JSON values retain their contents during classification. Invalid thresholds fail immediately. Existing callers do not need to change their tool signatures.

Reinstall native hooks after upgrading: `agent-action-guard hooks install --target <target>`. Existing Claude guard entries are upgraded in place, preserving unrelated hooks. Use **Claude Code 2.1.295 or later** for `onFailure: "block"`, which blocks startup errors and timeouts as well as script failures. Older versions do not provide that native guarantee. All nine adapters render classifier/bridge failures into their blocking protocols, without returning provider errors or action contents.

The Node HTTP API returns JSON 400 for malformed request targets and remains available for subsequent requests. Both HTTP APIs still assume trusted clients; they provide no authentication or TLS termination. Follow the proxy/network/concurrency guidance in `USAGE.md` before exposing a service remotely.

## Dependencies and platform limits

The runtime dependency configuration requires a patched certifi trust bundle (`>=2026.10.19`) and filelock (`>=4.2.1`) on Python 3.11+. Compatible resolved development tools were updated, as were both local scanner environments. The JavaScript tokenizer retains 0.1.x compatibility and additionally supports 0.2.x; both npm and pnpm lockfiles are synchronized. The npm manifest now reflects the already documented ONNX Runtime Node 1.14+ support, so the oldest dependency profile cannot silently install a second newer runtime.

Some upstream constraints still prevent a completely clean audit of every supported development environment:

| Environment/dependency | Remaining constraint | Operational boundary |
| --- | --- | --- |
| Python 3.10 / filelock | Patched 4.2.x releases require Python 3.11+. The compatible lockfile branch remains advisory-bearing. | Use Python 3.11+ for a clean runtime dependency audit. Keep user/model caches private to the operator; never process untrusted lock databases. Action Guard's local tokenizer uses `from_file`, not Hugging Face Hub's lock-backed download APIs. Python 3.10 runtime support is retained. |
| Intel macOS scanner cryptography | Patched 50.x wheels omit Intel macOS. The Intel-only `>=48.0.1,<49` constraint preserves binary installation and retains the reported PKCS#7 advisory. | Run security scanning on Linux or Apple Silicon with patched wheels. Scanning does not require an S/MIME/PKCS#7 decryption service. Do not introduce such a service using this scanner environment. Do not silently replace the fallback with a source build or remove Intel support. |
| Semgrep / MCP and OpenTelemetry | Semgrep 1.182.1 requires `mcp==1.29.0` and `opentelemetry-api~=1.37.0`; reported fixes require newer incompatible constraints. | Use Semgrep only as a local scanner, not a public MCP server or telemetry receiver. Track upstream compatible releases and audit the scanner environment separately. |
| FastEmbed / Pillow | FastEmbed 0.8.0 requires Pillow `<13`; reported fixes require Pillow 13. | The core classifier does not use Pillow. Keep this optional example/development environment away from untrusted images; update when FastEmbed supports the patched version. |

These are remaining advisories, not a claim that every reported vulnerable API is reachable through the classifier. They are also not suppressed in the evidence. Do not override upstream dependency metadata, downgrade to older vulnerable libraries, or claim a full development-environment audit passed merely because the runtime export passed.

## Validation and release controls

Run `make test` and `make security SECURITY_ARGS=--strict-tools` in both package directories. The strict argument makes missing scanners fail CI. The reusable security workflow must succeed before the PyPI job can publish; external actions are pinned to verified commit SHAs. A GitHub Release still requires an appropriate new package version.

Python formatting/linting uses `make format` and `make lint`. Mypy runs in an isolated minimum-supported Python environment so it reads compatible NumPy stubs. This does not replace testing the other Python runtimes. TypeScript linting excludes its generated scanner environment rather than linting third-party Python-package JavaScript.

The Docker image runs as UID 10001 with a writable `/home/aag` cache. It copies only package metadata and runtime files; `.env*`, keys, scanner environments, dependency trees, and reports are excluded from the context. Operator-mounted files must remain readable by that unprivileged user.

## Website publishing and browser controls

Both static sites already contain `/privacy/` and `/terms/` in this checkout. Those directories and their same-origin assets must be published to the existing repositories for the public 404s to disappear:

- `docs/website/` → `action-guard/action-guard.github.io`
- `docs/agent-leaderboard-site/` → `agent-leaderboard/agent-leaderboard.github.io`

Validated deployment ZIPs are available locally in `.cache/security-fixes/deploy/`, with file checksums in `manifest.json`. They contain public site files and policies, without tests, tool environments, or credentials.

No website was published without an authenticated publishing connection. Check deployment status and all four live policy URLs after publishing; local checks alone cannot establish the live state.

Each public page now has a restrictive content security policy. Script execution is limited to same-origin scripts and hashes of the authored inline scripts/handlers; browser connections, forms, external frames, and object embedding are disabled. Inline styles remain allowed for the existing design and motion effects. Regenerate hashes after editing inline scripts:

```bash
python3 docs/scripts/site_security.py
python3 docs/scripts/site_security.py --check
python3 -m unittest discover -s docs/website/tests -v
node --test docs/website/tests/tabs.test.cjs
cd docs/browser-tests
npm run test:browser
```

GitHub Pages controls HTTP response headers. The meta CSP cannot set `frame-ancestors`, and meta tags cannot implement `X-Content-Type-Options` or `X-Frame-Options`. Adding those controls requires a hosting/proxy configuration that can emit real headers; there is no claim that the current source change supplies them. HSTS is already supplied by the observed hosting platform.

The code example window remains stationary during reveals so copy buttons do not move between pointer positioning and activation. Browser tests cover both clipboard failure/retry and actual CSP rejection of an injected inline script while the bundled tab code continues to work.

## Classifier accuracy remains a separate limitation

A local CPU/ONNX smoke test returned valid public result shapes in both
runtimes, but a synthetic nested deletion command was still classified as safe
(Python confidence approximately 0.413; JavaScript approximately 0.389). The
command was only text passed to the classifier and was never executed. The
preprocessing repair ensures nested text reaches the model; it cannot guarantee
the trained model's label. Keep independent tool permissions and deterministic
restrictions for destructive operations. Changing the model or default decision
policy requires a separately evaluated compatibility and accuracy change.

## Verification scope

Python lint and strict package security checks pass; supported-runtime tests pass
on Python 3.10 and 3.14. The Python 3.14 run has 285 passing tests. Node 20, 22,
24, and 26 pass the TypeScript tests, and both oldest/newest dependency-profile
checks pass. Website source checks (50), tab/copy unit cases (20), Chromium
desktop/tablet/mobile cases (36), and additional browser security probes (7) pass.

The local WebKit runtime initially lacked native libraries. After providing
those in its user-owned browser cache, basic route checks worked, but its
non-reduced-motion renderer crashed during tab navigation on this unsupported
Linux host. The reduced-motion WebKit mobile project passes all 12 cases, including CSP and clipboard behavior. Desktop/tablet WebKit coverage is therefore not claimed to pass;
retained logs are in `.cache/security-fixes/webkit-browser.log`. Native Safari,
Apple Silicon CI, and Docker image execution remain separate environment checks.
No system library installation or production deployment was performed.
