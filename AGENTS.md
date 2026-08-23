# Project: Agent Action Guard

Agent Action Guard classifies proposed AI agent actions as safe or harmful and blocks or flags harmful actions. This repository provides the model, dataset, integration helpers, and example MCP-compatible tooling to enable runtime action screening in agent loops.
- Repository URL: https://github.com/Pro-GenAI/Agent-Action-Guard

The repository also ships a JavaScript runtime package under `javascript/` that exposes `isActionHarmful()`, `ensureActionSafety()`, and `actionGuarded()` for Node.js tool screening.

## Why it matters

- Helps prevent autonomous agents from executing harmful, unethical, or risky operations.
- Provides a reproducible benchmark (HarmActionsEval) and dataset (HarmActions) for evaluating agent safety.
- Lightweight model for easy integration into MCP or custom agent frameworks.

## Quick Usage (for agents)

1. Install the package (recommended in a venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install agent-action-guard
```

2. Start or configure an embedding server if using vector features (see `USAGE.md`).

3. In your agent runtime, call the convenience API to check actions before execution:

```python
from agent_action_guard import is_action_harmful, action_guarded

# Manual Check
is_harmful, confidence = is_action_harmful(action_dict)
if is_harmful:
    raise Exception("Harmful action blocked")

# Decorator (Automatic safety check based on function name and kwargs)
@action_guarded(conf_threshold=0.8)
def send_email(to, subject, body):
    # This tool will be blocked if the model classifies the 'send_email' action as harmful
    print(f"Sending email to {to}")
```

### JavaScript runtime package

For Node.js projects, use the npm package in `javascript/`.

```bash
cd javascript
npm install
npm test
```

Embedding configuration is optional: set `AAG_EMBED_GGUF` or `AAG_EMBED_ONNX` to a local model path/directory, a direct HTTP(S) model URL, a base URL ending in `/`, or a Hugging Face `owner/repo` ID; use the existing API environment variables for an OpenAI-compatible endpoint; or leave embedding variables unset to auto-download and cache the default MiniLM ONNX assets. Remote model assets are cached under normalized filesystem-safe names derived from their URLs. If both GGUF and ONNX variables are set, GGUF takes precedence.

```js
import { actionGuarded, ensureActionSafety, isActionHarmful } from "agent-action-guard";

const action = {
    type: "function",
    function: {
        name: "send_email",
        arguments: {
            to: "user@example.com",
            subject: "Status update",
            body: "Hello",
        },
    },
};

const decision = await isActionHarmful(action);
if (decision.label) {
    throw new Error(`Blocked: ${decision.label}`);
}

await ensureActionSafety(action, { raiseException: true });
```

## Key Files & Structure

- `python` — Python package for the classifier, dataset loading, and runtime helpers.
    - `python/agent_action_guard/` — implementation package (classifier, runtime helpers, dataset loaders).
    - `python/training/` — training scripts and dataset artifacts used to produce the classifier.
    - `python/examples/` — sample integrations and MCP server examples.
    - `python/tests/` — unit tests validating core behavior.
- `javascript/` — npm runtime package for Node.js action screening.
- `USAGE.md` — detailed usage examples and environment setup.
- `README.md` — project overview, demos, and citations.

## Architecture Overview

- Input: proposed agent action (structured dict describing tool call, intent, parameters).
- Preprocessing: optional embedding + metadata normalization.
- Classifier: lightweight NN (PyTorch / ONNX) outputs harmful/safe logits and confidence.
- Policy: decision layer in the agent runtime that blocks, allows, or requests human approval.

## Development & CI

- Formatting and linting: run `make format` and `make lint` from `python/`.
- Tests: run `pytest` from `python/` (configured by `python/pytest.ini`) to run test cases in `python/tests/`.
- For every code change, update or add the corresponding tests so the changed behavior is covered and regressions are caught. Tests must be very comprehensive: cover happy paths, edge cases, invalid inputs, error handling, async/concurrent behavior, configuration precedence, platform/runtime differences, dependency/version differences, performance-sensitive paths, and backward-compatibility scenarios where applicable. Prefer regression tests that reproduce any bug before fixing it.
- Treat backward compatibility and user impact as release-blocking concerns. Code changes must not unexpectedly break existing users, public APIs, documented behavior, environment-variable semantics, supported runtimes, installation flows, or established defaults. Add compatibility/regression tests for affected behavior, preserve existing behavior unless a breaking change is explicitly intended, and require a clear migration path and documentation for any deliberate breaking change.
- For runtime, embedding, cache, download, or dependency changes, include regression coverage for repeated calls, concurrent first use, cache reuse, partial/failing downloads, missing or malformed assets, API fallback/precedence, supported runtime versions, and existing public return shapes. A change is not complete until the relevant Python and JavaScript suites, lint/format checks, and supported-runtime smoke tests pass.
- For every code change, update `AGENTS.md` to reflect any affected behavior, workflows, commands, files, architecture, or development constraints; keep these instructions synchronized with the implementation.
- Add all new user-facing usage/runtime features to both the Python and JavaScript packages so their public capabilities stay in sync. Training-only functionality belongs in Python only (for example under `python/training/`) and does not need a JavaScript counterpart.
- Keep `aag-classify` available from both packages. It accepts one direct JSON action or `--file` with a JSON array/JSONL actions, supports bounded batch processing, and reports total/safe/unsafe counts. Programmatic batch classification must stay vectorized (`is_actions_harmful` in Python and `isActionsHarmful` in JavaScript) rather than looping through the single-action API.
- Keep the manual ONNX-vs-API latency benchmarks in sync across Python and JavaScript (`python/examples/scripts/compare_embedding_backend_latency.py` and `javascript/scripts/compare-embedding-backend-latency.js`). They must benchmark end-to-end Action Guard classification with separate embedding backends, support warmup/iterations/batch-size/custom JSON or JSONL inputs, and report mean/p50/p95 run latency, per-action latency, throughput, and the API/ONNX ratio. These benchmarks require a live compatible embedding API and are not CI tests; `make latency-backends` runs them in either package.
- Keep both the Python and JavaScript packages lightweight and ultra-fast. Treat startup time, inference latency, memory usage, install size, dependency count, and unnecessary network or filesystem work as core engineering constraints. Prefer simple, lazy, cache-aware implementations; avoid heavyweight dependencies, unnecessary abstractions, eager initialization, duplicate work, and performance regressions unless there is a clear measured benefit.
- The shipped Python runtime uses NumPy directly for ONNX tensor preparation, embedding pooling/normalization, and classifier softmax/argmax. Keep `numpy>=1.21.6` as an explicit runtime dependency, matching ONNX Runtime's minimum requirement, and preserve NumPy ndarray return/feed behavior in compatibility-sensitive paths.
- The Python package supports Python 3.8 through 3.14. Keep PyPI classifiers, runtime-version tests, dependency-matrix defaults, and user-facing documentation synchronized when adding or removing a supported Python minor.
- Keep runtime dependency ranges intentionally broad where the APIs used by Action Guard are compatible. Python supports OpenAI SDK 1.x-3.x where the SDK supports the selected Python runtime; OpenAI 3.3.1 is supported on Python 3.10+ while pip resolves an older compatible OpenAI release on Python 3.8-3.9. Python supports tokenizers 0.13.3+ on Python <3.13 (0.21+ on Python 3.13+). JavaScript supports `@huggingface/tokenizers` 0.1.x, ONNX Runtime Node 1.14+, and OpenAI JS 4.x-7.x. JavaScript ONNX imports must normalize both the default-only ESM wrapper exposed by older ONNX Runtime Node releases and the named-export shape exposed by newer releases. When dependency bounds change, verify both an older supported release and a recent supported release instead of testing only lockfile-resolved versions.
- Use `python/scripts/test_dependency_matrix.py` for Python dependency compatibility testing. It must exercise multiple Python minors and multiple dependency releases per minor, use `uv` for interpreter provisioning and dependency installation, and return a non-zero exit status when any matrix scenario fails. Per-dependency scenarios pin only the dependency under test and let `uv` resolve compatible companion dependencies within supported ranges using binary wheels; NumPy 2.x scenarios require ONNX Runtime 1.19+ because older released wheels use the NumPy 1.x ABI. `make dependency-matrix-bounds` is the fast oldest/newest core-dependency check; `make dependency-matrix` runs the broader sampled matrix.
- Use `javascript/scripts/test-all-versions.js` for Node.js runtime compatibility testing. It must use nvm (sourcing `NVM_SH` or `$NVM_DIR/nvm.sh`) to install/select each requested runtime, run `npm ci` under that runtime, and then execute the JavaScript tests. The default matrix is Node.js 18, 20, 22, and 24; CLI version arguments override `NODE_TEST_VERSIONS`. Continue through failures by default and return non-zero after the matrix, with `CONTINUE_ON_FAILURE=0` available for fail-fast runs.
- Use `javascript/scripts/test-dependency-matrix.js` for JavaScript dependency compatibility testing. It must test `@huggingface/tokenizers` 0.1.x, ONNX Runtime Node 1.14+, and OpenAI JS 4.x-7.x across old, sampled, and recent releases; pack the local package and install it into isolated scenario projects so the selected versions are actually used; and return a non-zero exit status when any scenario fails. The base scenarios test the exact oldest/newest dependency profiles, and the matrix must query npm's current `latest` tag for every dependency and fail if that release falls outside the declared supported range; sampled scenarios vary one intermediate dependency release at a time against the oldest supported companion-dependency baseline to avoid redundant heavyweight installs. `make dependency-matrix-bounds` is the fast oldest/newest dependency check; `make dependency-matrix` runs the broader sampled matrix. Keep this separate from `test:all-versions`, which exercises supported Node.js runtime majors through nvm.
- Python and JavaScript embedding selection must follow this explicit order: an explicit `AAG_EMBED_GGUF` source overrides everything; otherwise an explicit `AAG_EMBED_ONNX` source overrides API configuration; otherwise, if `EMBED_MODEL_NAME` (or an explicit model constructor argument) exists, use the OpenAI-compatible embedding API with that model; otherwise, if API configuration exists (`EMBEDDING_API_KEY`, `OPENAI_API_KEY`, or `EMBEDDING_BASE_URL`), use the embedding API; otherwise automatically download/cache and use the default `sentence-transformers/all-MiniLM-L6-v2` ONNX model locally. Preserve `EMBEDDING_API_KEY` precedence over `OPENAI_API_KEY`. Each local source variable accepts a local model file, a local directory containing the standard filename, a direct HTTP(S) model URL, a base URL ending in `/`, or a Hugging Face `owner/repo` ID. Hugging Face repo IDs resolve `main/model.gguf` or `main/model.onnx`; use a direct URL when the remote filename is nonstandard. Direct ONNX URLs derive `tokenizer.json` (and JavaScript `tokenizer_config.json`) as sibling URLs. Remote assets are cached under URL-derived names after normalization to lowercase filesystem-safe `[a-z0-9._-]` components plus a short source fingerprint. Existing or obvious filesystem paths take priority over the `owner/repo` repo-ID heuristic. GGUF inference loads optional llama.cpp bindings lazily (`llama-cpp-python` for Python and `node-llama-cpp` for JavaScript).
- Keep real URL download/cache integration coverage in `python/tests/test_embedding_real_url_cache.py` and `javascript/test/embedding-real-url-cache.test.js`. These tests intentionally use lightweight public ONNX/GGUF fixtures and the normal persistent runtime cache: a cold cache performs the real download, while later runs must reuse the cached inode/mtime and avoid repeated model downloads. Keep deterministic mocked URL-resolution tests separate from these network/cache integration tests.

## Guidance for AI agents reading this repo

- Use `USAGE.md` and `python/examples/` for integration patterns rather than reproducing code.
- Prefer runtime API `is_action_harmful()` for decision making.
- Respect model limitations: the classifier is trained on a limited dataset; combine with rule-based checks for high-risk systems.

## Where to look next (quick links)

- Full details and demo: [README.md](README.md)
- Integration and examples: [USAGE.md](USAGE.md) and `python/examples/`
- Implementation: `python/agent_action_guard/`
- Training scripts & dataset: `python/training/`
