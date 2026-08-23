<p align="center">
  <img src="https://raw.githubusercontent.com/Pro-GenAI/Agent-Action-Guard/main/assets/cover.jpg" alt="Agent Action Guard" height="220" />
</p>

<h1 align="center">Agent Action Guard for JavaScript</h1>

<p align="center">
  <strong>Block harmful AI agent tool calls before they execute.</strong><br />
  A lightweight runtime safety layer for Node.js agents, MCP tools, and tool-calling applications.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/agent-action-guard"><img src="https://img.shields.io/npm/v/agent-action-guard?style=for-the-badge&logo=npm&logoColor=white&color=CB3837" alt="npm version" /></a>
  <a href="https://www.npmjs.com/package/agent-action-guard"><img src="https://img.shields.io/npm/dm/agent-action-guard?style=for-the-badge&logo=npm&logoColor=white" alt="npm downloads" /></a>
  <img src="https://img.shields.io/badge/Node.js-%3E%3D18-339933?style=for-the-badge&logo=node.js&logoColor=white" alt="Node.js 18+" />
  <a href="https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/License-CC_BY_4.0-darkgreen.svg?style=for-the-badge" alt="CC BY 4.0 license" /></a>
</p>

---

## Install

```bash
npm install agent-action-guard
```

Or with pnpm:

```bash
pnpm add agent-action-guard
```

**Node.js 18+ · ESM · usable with zero embedding configuration.**

On first use, Action Guard can automatically download and cache the default MiniLM ONNX embedding assets. You can also point it at your own local ONNX model or an OpenAI-compatible embedding API.

## Why add Action Guard?

An AI agent can decide to call a tool even when the resulting action is unsafe. Action Guard gives you a dedicated checkpoint between **the model deciding what to do** and **your application actually doing it**.

- **Block before execution** — screen tool calls before they reach email, filesystem, API, database, or other side-effecting code.
- **Minimal integration** — classify an action directly or wrap a function with `actionGuarded()`.
- **Local-first option** — use cached ONNX embeddings without sending action text to an embedding API.
- **Bring your own embeddings** — use a custom ONNX model or OpenAI-compatible endpoint when needed.
- **Batch + CLI support** — classify multiple actions programmatically or from JSON/JSONL files.
- **Built for agent loops** — the package contains the runtime guard; training and benchmark tooling stay outside the JavaScript runtime package.

## 60-second quick start

### 1. Check a tool call before it runs

```js
import { isActionHarmful } from 'agent-action-guard';

const action = {
  type: 'function',
  function: {
    name: 'send_email',
    arguments: {
      to: 'user@example.com',
      subject: 'Status update',
      body: 'Hello',
    },
  },
};

const { label, confidence } = await isActionHarmful(action);

if (label) {
  throw new Error(`Blocked: ${label} (${confidence.toFixed(2)})`);
}

// Safe path: execute the real tool here.
```

A safe action returns `label: null`. A classified harmful action returns its label plus a confidence score.

### 2. Guard a function automatically

```js
import { actionGuarded } from 'agent-action-guard';

const guardedSendEmail = actionGuarded(
  async function sendEmail(params) {
    // Call your real email provider here.
    return `sending to ${params.to}`;
  },
  { confThreshold: 0.8 },
);

await guardedSendEmail({
  to: 'user@example.com',
  subject: 'Status update',
  body: 'Hello',
});
```

`actionGuarded()` derives the action from the wrapped function name and its object argument. When a harmful classification meets the configured confidence threshold, it throws `HarmfulActionError` instead of executing the wrapped function.

### 3. Use a boolean safety gate

```js
import { ensureActionSafety } from 'agent-action-guard';

const safe = await ensureActionSafety(action);

if (!safe) {
  // Reject, escalate, log, or request approval.
}
```

Or make harmful actions throw automatically:

```js
await ensureActionSafety(action, { raiseException: true });
```

## Batch classification

Use the vectorized batch API when you already have multiple proposed actions:

```js
import { isActionsHarmful } from 'agent-action-guard';

const decisions = await isActionsHarmful(actions, { batchSize: 32 });

for (const decision of decisions) {
  console.log(decision.label, decision.confidence);
}
```

## CLI included

Installing the package also installs `aag-classify`.

```bash
aag-classify '{"type":"function","function":{"name":"send_email","arguments":{}}}'
```

Classify JSON arrays or JSONL files in batches:

```bash
aag-classify --file actions.json --batch-size 32
aag-classify --file actions.jsonl --batch-size 32
```

The CLI reports total, safe, and unsafe action counts.

## Embedding backends

You do not need to configure an embedding backend to try the package.

| Mode | When to use it |
| --- | --- |
| **Zero-config local ONNX** | Fastest path to getting started. Default MiniLM ONNX assets are downloaded and cached automatically. |
| **Configured GGUF** | Use a local file/directory, direct model URL, base URL ending in `/`, or Hugging Face `owner/repo` ID. Install `node-llama-cpp`. |
| **Configured ONNX** | Use a local file/directory, direct model URL, base URL ending in `/`, or Hugging Face `owner/repo` ID. |
| **OpenAI-compatible API** | Use an existing embedding service or centrally managed endpoint. |

Backend selection follows this precedence:

1. `AAG_EMBED_GGUF` configured GGUF source
2. `AAG_EMBED_ONNX` configured ONNX source
3. Explicit `EMBED_MODEL_NAME`
4. API configuration via `EMBEDDING_API_KEY`, `OPENAI_API_KEY`, or `EMBEDDING_BASE_URL`
5. Automatically downloaded/cached default `sentence-transformers/all-MiniLM-L6-v2` ONNX assets

If both model variables are set, GGUF takes precedence. Remote assets are cached under normalized filesystem-safe names derived from their source URLs. Hugging Face repo IDs resolve standard `main/model.gguf` or `main/model.onnx` filenames; use a direct URL for nonstandard GGUF filenames/quantizations. Direct ONNX URLs derive tokenizer sidecars from sibling URLs.

For environment variables, tokenizer files, custom endpoints, and configuration examples, see the full [USAGE.md](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md).

## Public runtime API

```js
import {
  ActionClassifier,
  HarmfulActionError,
  actionGuarded,
  ensureActionSafety,
  isActionHarmful,
  isActionsHarmful,
} from 'agent-action-guard';
```

| API | Purpose |
| --- | --- |
| `isActionHarmful(action)` | Classify one proposed action and return `{ label, confidence }`. |
| `isActionsHarmful(actions, options)` | Vectorized classification for multiple actions. |
| `ensureActionSafety(action, options)` | Return a boolean safety decision or throw on harmful actions. |
| `actionGuarded(fn, options)` | Wrap a function so harmful calls can be blocked before execution. |
| `ActionClassifier` | Create/configure a classifier instance directly. |
| `HarmfulActionError` | Error type thrown by blocking helpers. |

See [`examples/basic-usage.js`](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/javascript/examples/basic-usage.js) for a minimal runnable example.

## Where it fits

```text
User request
    ↓
LLM / Agent
    ↓
Proposed tool call
    ↓
┌────────────────────┐
│ Agent Action Guard │  ← classify before execution
└────────────────────┘
    ↓ safe                     ↓ harmful
Execute tool                 Block / escalate
```

Use Action Guard as a runtime safety layer alongside your existing authentication, authorization, validation, rate limits, allowlists, and human-approval policies.

## Compatibility

The package supports **Node.js 18+** and is tested across Node.js 18, 20, 22, and 24. Runtime dependency compatibility is tested separately across supported versions of `@huggingface/tokenizers`, `onnxruntime-node`, and the OpenAI JavaScript SDK.

For contributors:

```bash
npm test
npm run lint
npm run test:all-versions
npm run test:dependency-matrix-bounds
```

The broader dependency matrix is available with:

```bash
npm run test:dependency-matrix
```

## Important limitation

Action Guard is a learned classifier, not a complete security boundary. For high-risk systems, combine its decision with deterministic policy checks, strict tool permissions, input validation, least-privilege credentials, and human approval where appropriate.

## Learn more

- [Main repository](https://github.com/Pro-GenAI/Agent-Action-Guard)
- [Full usage and configuration](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md)
- [Runnable JavaScript example](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/javascript/examples/basic-usage.js)
- [npm package](https://www.npmjs.com/package/agent-action-guard)

---

<p align="center">
  <strong>Put a safety check between your agent and its tools.</strong><br />
  <code>npm install agent-action-guard</code>
</p>
