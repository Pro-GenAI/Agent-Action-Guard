### Usage:
1. Install the runtime package.

```bash
pip install agent-action-guard
```

2. Choose the embedding backend. ONNX is the default and requires no embedding server.

#### Default ONNX usage

With no embedding-related environment variables, the first classification call automatically downloads and caches the ONNX export of `sentence-transformers/all-MiniLM-L6-v2` plus its tokenizer. Later calls reuse the cached files.

```python
from agent_action_guard import is_action_harmful

is_harmful, confidence = is_action_harmful({
    "type": "function",
    "function": {
        "name": "send_email",
        "arguments": {"to": "user@example.com"},
    },
})
```

#### Use an existing local ONNX model

Set `AAG_EMBED_ONNX` to either the `.onnx` filepath or a directory containing `model.onnx`. Keep the matching `tokenizer.json` beside the model.

```bash
export AAG_EMBED_ONNX="/models/all-MiniLM-L6-v2/model.onnx"
python app.py
```

Python embedding backend precedence is:
1. `AAG_EMBED_ONNX` local ONNX configuration.
2. OpenAI-compatible API configuration (`EMBED_MODEL_NAME`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, or `OPENAI_API_KEY`).
3. Automatic download/cache of the default `sentence-transformers/all-MiniLM-L6-v2` ONNX model and tokenizer.

#### Use an OpenAI-compatible embedding API

To use an embedding server instead, set its model, base URL, and credentials in the environment before starting it:

```bash
pip install "agent-action-guard[all]"

export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY

python examples/scripts/host_embed_fastembed.py
```
If you are running from this repository checkout, use:

```bash
cd python
python examples/scripts/host_embed_fastembed.py
```

**Alternatives:**
Any OpenAI-compatible embedding server such as [LMStudio](https://lmstudio.ai/download) can be used.

The classifier is trained against embeddings from `sentence-transformers/all-MiniLM-L6-v2`; use a compatible embedding model when overriding the default.

3. Use Action Guard in your own project:

```python
from agent_action_guard import is_action_harmful, action_guarded

# Method 1: Manual Check
is_harmful, confidence = is_action_harmful(action_dict)
if is_harmful:
    raise Exception("Harmful action blocked")

# Method 2: Decorator (Recommended for tool functions)
@action_guarded(conf_threshold=0.8)
def delete_user(user_id: str):
    # This tool will be blocked if the 'delete_user' action is classified as harmful
    print(f"Deleting user {user_id}...")
```

PyPI package scope:
- `pip install agent-action-guard` installs the runtime classifier and dependencies for local ONNX embedding inference. The default embedding model and tokenizer are downloaded and cached on first use rather than bundled in the wheel.
- Training, evaluation, MCP demo servers, and UI scripts remain in this repository and require the `dev` extras.

Repository-local development commands run from `python/`:

```bash
cd python
make format
make lint
pytest
```

### HarmActionsEval CLI (standalone)

After installing `agent-action-guard[harmactionseval]`, run:

```bash
python -m agent_action_guard.harmactionseval --k 3
```

Common arguments:
- `--k`: Number of attempts per prompt (HarmActions@k).
- `--offset`: Start index within harmful/unethical rows.
- `--limit`: Maximum number of harmful/unethical rows to evaluate.
- `--cache-path`: Path to cache JSON file.
- `--output`: Path to output JSON file.
- `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL`.

Environment variables:
- Required: `OPENAI_MODEL` and provider credentials (`OPENAI_API_KEY` or Azure equivalents).

### Usage of Action Guard with OpenAI Agents

```python
import json
from agents import (
    Agent,
    Runner,
    ToolGuardrailFunctionOutput,
    function_tool,
    tool_input_guardrail,
)
from agent_action_guard import is_action_harmful

@tool_input_guardrail
def block_harmful(data):
    args = json.loads(data.context.tool_arguments or "{}")
    is_harmful, confidence = is_action_harmful(args)
    if is_harmful:
        return ToolGuardrailFunctionOutput.reject_content(
            message="Harmful action detected",
            output_info={"confidence": confidence}
        )
    return ToolGuardrailFunctionOutput.allow()


@function_tool(tool_input_guardrails=[block_harmful])
def classify_text(text: str) -> str:
    """Classify text for internal routing."""
    return f"length:{len(text)}"


agent = Agent(name="Classifier", tools=[classify_text])
result = Runner.run_sync(agent, "hello world")
print(result.final_output)
```

### JavaScript Runtime Package

The repository also includes an npm package in [javascript/package.json](javascript/package.json) that exposes the same action-screening runtime for Node.js.

Install and test it from the package directory:

```bash
cd javascript
npm install
npm test
```

Use it in a Node.js agent loop:

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

const { label, confidence } = await isActionHarmful(action);
if (label) {
    throw new Error(`Blocked: ${label} (${confidence.toFixed(2)})`);
}

await ensureActionSafety(action, { raiseException: true });

const guardedSendEmail = actionGuarded(async function sendEmail(params) {
    return `sending to ${params.to}`;
});
```

#### JavaScript default ONNX usage

No embedding configuration is required. The first Action Guard call downloads and caches the default MiniLM ONNX model and tokenizer assets; later calls reuse the cache.

```js
import { isActionHarmful } from "agent-action-guard";

const result = await isActionHarmful({
    type: "function",
    function: {
        name: "send_email",
        arguments: { to: "user@example.com" },
    },
});

console.log(result);
```

To use an existing local ONNX model:

```bash
export AAG_EMBED_ONNX="/models/all-MiniLM-L6-v2/model.onnx"
node app.js
```

JavaScript embedding backend precedence matches Python:
1. `AAG_EMBED_ONNX` uses a local ONNX embedding model first.
2. Otherwise, `EMBED_MODEL_NAME`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, or `OPENAI_API_KEY` select the OpenAI-compatible embeddings API.
3. If none are set, the runtime automatically downloads and caches the default `sentence-transformers/all-MiniLM-L6-v2` ONNX model, `tokenizer.json`, and `tokenizer_config.json`, then generates embeddings locally.

`AAG_EMBED_ONNX` may point to a `.onnx` file or a directory containing `model.onnx`; keep `tokenizer.json` and `tokenizer_config.json` beside the JavaScript embedding model.

#### JavaScript OpenAI-compatible embedding API

To use an OpenAI-compatible embedding endpoint instead of local ONNX embeddings, configure the model and endpoint before starting Node:

```bash
export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY
node app.js
```

`EMBEDDING_API_KEY` takes precedence over `OPENAI_API_KEY`. The classifier is trained against `sentence-transformers/all-MiniLM-L6-v2` embeddings, so custom API or ONNX embedding models must produce compatible 384-dimensional vectors.

The JavaScript Action Guard classifier itself continues to use the packaged ONNX model in [javascript/src/action_classifier_model.onnx](javascript/src/action_classifier_model.onnx). Local embedding inference depends on `onnxruntime-node` and `@huggingface/tokenizers`; API inference depends on `openai`.

### Batch classification and `aag-classify` CLI

Both the Python and npm packages install an `aag-classify` command. Pass one action directly as JSON:

```bash
aag-classify '{"type":"function","function":{"name":"send_email","arguments":{"to":"user@example.com"}}}'
```

For multiple actions, pass a JSON file containing an array:

```bash
aag-classify --file actions.json
```

Or pass a JSONL file containing one JSON action per non-empty line:

```bash
aag-classify --file actions.jsonl
```

Large inputs can be processed in bounded vectorized chunks:

```bash
aag-classify --file actions.jsonl --batch-size 64
```

The command prints totals such as:

```text
Total actions: 100
Safe actions: 82
Unsafe actions: 18
```

`Unsafe actions` includes every non-`safe` label (`harmful` and `unethical`). Batch classification uses one embedding request/inference and one Action Guard ONNX classifier inference per batch chunk rather than classifying actions one by one.

Python exposes the same batch path programmatically:

```python
from agent_action_guard import is_actions_harmful

results = is_actions_harmful(actions, batch_size=64)
```

JavaScript exposes the camelCase equivalent:

```js
import { isActionsHarmful } from 'agent-action-guard';

const results = await isActionsHarmful(actions, { batchSize: 64 });
```

### ONNX vs API latency benchmark

The repository includes manual latency benchmarks that run the same actions through separate Action Guard classifier instances using local ONNX embeddings and an OpenAI-compatible embedding API. They report mean, p50, and p95 latency per full run, mean per-action latency, throughput, and the API/ONNX latency ratio.

Configure an API endpoint that serves embeddings compatible with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). For an unauthenticated local OpenAI-compatible server, a dummy API key is sufficient:

```bash
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY="dummy"
export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

Run the Python benchmark from `python/`:

```bash
make latency-backends
# or
python examples/scripts/compare_embedding_backend_latency.py --iterations 10 --batch-size 8
```

Run the JavaScript benchmark from `javascript/`:

```bash
make latency-backends
# or
npm run test:latency-backends -- --iterations 10 --batch-size 8
```

Both scripts use built-in representative actions by default. Pass `--file actions.json` or `--file actions.jsonl` to benchmark your own actions, and `--limit N` to bound API usage. The default warmup is one run per backend, which excludes model download/session/client initialization from measured steady-state latency. Use `--warmup 0` when you explicitly want cold-start cost included.
