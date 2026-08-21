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

For embedding backend options and configuration details—including zero-config ONNX embeddings, custom local ONNX models, OpenAI-compatible embedding APIs, environment variables, and backend precedence—read [USAGE.md](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md). The package also installs `aag-classify` for direct JSON, JSON arrays, and JSONL batch classification.

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

## Compatibility testing

The JavaScript package keeps runtime-version and dependency-version coverage separate:

```bash
npm run test:all-versions
npm run test:all-versions -- 18 24
NODE_TEST_VERSIONS="20,22" npm run test:all-versions
npm run test:dependency-matrix-bounds
npm run test:dependency-matrix
```

`test:all-versions` uses [nvm](https://github.com/nvm-sh/nvm) to install/select each requested Node.js version, runs `npm ci` under that runtime, and then runs the JavaScript test suite. By default it tests Node.js 18, 20, 22, and 24. Pass versions after `--` or set `NODE_TEST_VERSIONS`; set `NVM_DIR` or `NVM_SH` when nvm is installed somewhere other than `~/.nvm`. The runner continues through failures by default and returns non-zero after the matrix; set `CONTINUE_ON_FAILURE=0` to stop at the first failure.

The dependency matrix remains separate: it packs the local package and installs it into isolated scenario projects so the smoke test uses the selected dependency releases rather than the repository lockfile. It explicitly checks npm's current `latest` release for every runtime dependency and fails if `latest` falls outside the declared support range. The bounds command checks the oldest and newest supported dependency profiles; the full matrix additionally samples intermediate releases and varies one dependency at a time against the oldest supported companion-dependency baseline.

## Notes

- This folder implements only the Action Guard runtime.
- Benchmark, dataset, and training code remain in the Python side of the repository.
- Local embedding inference uses `onnxruntime-node` and `@huggingface/tokenizers`.
- The Action Guard classifier expects a 384-dimensional embedding vector.
- The model training script is at [../python/training/](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/python/training/train_nn.py).
