# Agent Action Guard for JavaScript

JavaScript runtime for screening proposed AI agent actions as `safe`, `harmful`, or `unethical` using the packaged ONNX classifier in this folder.

## Install

```bash
cd javascript
npm install
```

If you prefer pnpm:

```bash
cd javascript
pnpm install
```

## Test

```bash
npm test
```

```bash
pnpm test
```

`npm test` runs the Node unit tests plus a runtime smoke matrix. `node` is required. If `deno` or `bun` are installed on `PATH`, the same smoke fixture is also executed there for both `.js` and `.ts` entrypoints; otherwise those checks are skipped.

The `npm_publish.sh` helper also supports pnpm. Set `PACKAGE_MANAGER=pnpm` to use `pnpm install --frozen-lockfile`, `pnpm test`, and `pnpm publish` during the publish flow.

## Required environment

The classifier expects embeddings from the same embedding model used during training.

```bash
export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY="your-embedding-key"
```

`EMBEDDING_BASE_URL` should point to any OpenAI-compatible embeddings endpoint. `EMBEDDING_API_KEY` falls back to `OPENAI_API_KEY`.

## Usage

```js
import {
	actionGuarded,
	ensureActionSafety,
	isActionHarmful,
} from 'agent-action-guard';

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

await ensureActionSafety(action, { raiseException: true });

const guardedSendEmail = actionGuarded(
	async function sendEmail(params) {
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

See [examples/basic-usage.js](examples/basic-usage.js) for a minimal runnable example.

## Notes

- This folder implements only the Action Guard runtime.
- Benchmark, dataset, and training code remain in the Python side of the repository.
- The ONNX model lives in [src/action_classifier_model.onnx](src/action_classifier_model.onnx) and expects a 384-dimensional embedding vector.
