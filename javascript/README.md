# Agent Action Guard

Framework to block harmful AI agent actions before they cause harm — lightweight, real-time, easy-to-use

## Install

```bash
npm i agent-action-guard
```

If you prefer pnpm:

```bash
pnpm install agent-action-guard
```

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

// --------- Create a guarded version of the function ---------

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

See [examples/basic-usage.js](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/javascript/examples/basic-usage.js) for a minimal runnable example.

## Required environment

The classifier expects embeddings from the same embedding model used during training.

```bash
export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY="your-embedding-key"
```

`EMBEDDING_BASE_URL` should point to any OpenAI-compatible embeddings endpoint. `EMBEDDING_API_KEY` falls back to `OPENAI_API_KEY`.

## Notes

- This folder implements only the Action Guard runtime.
- Benchmark, dataset, and training code remain in the Python side of the repository.
- The ONNX model expects a 384-dimensional embedding vector.
- The model training script is at [../python/training/](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/python/training/train_nn.py).
