# Agent Action Guard documentation

**Canonical HTML page:** [Read the original version](https://action-guard.pro-genai.com/docs/)

> This is the Markdown edition of the public page. The linked HTML page is the canonical version for search engines and the interactive website.

Official documentation

Install and integrate runtime action screening for AI agents. Agent Action Guard classifies proposed tool calls immediately before execution so your application can block harmful actions or require additional policy checks.

[Quick start](https://action-guard.pro-genai.com/docs/#quick-start)
[Full repository usage guide](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md)

## Quick start

Use the runtime package in the same process as your agent and call the guard immediately before a tool executes.

### Python

```bash
pip install agent-action-guard
# or
uv add agent-action-guard
```

Supported package range: Python 3.10 through 3.14.

### JavaScript / Node.js

```bash
npm install agent-action-guard
# or
pnpm install agent-action-guard
```

The package exposes asynchronous action-screening helpers for Node.js agent loops.

**Default embedding behavior:** if no embedding environment variables are set, the runtime automatically downloads and caches the default MiniLM ONNX embedding assets on first use, then reuses the cached files.

## Python API

### Screen an action directly

```python
from agent_action_guard import is_action_harmful

action = {
    "type": "function",
    "function": {
        "name": "send_email",
        "arguments": {
            "to": "user@example.com",
            "subject": "Status update",
            "body": "Hello",
        },
    },
}

is_harmful, confidence = is_action_harmful(action)
if is_harmful:
    raise RuntimeError(f"Action blocked ({confidence:.2f})")
```

### Guard a Python function

```python
from agent_action_guard import action_guarded

@action_guarded(conf_threshold=0.8)
def delete_user(user_id: str):
    return f"deleted:{user_id}"
```

The decorator derives an action from the function call and screens it before the wrapped function executes.

## JavaScript API

The Node.js runtime exposes `isActionHarmful()`, `ensureActionSafety()`, and `actionGuarded()`.

```javascript
import {
  ensureActionSafety,
  isActionHarmful,
} from "agent-action-guard";

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
  throw new Error("Harmful action blocked");
}

await ensureActionSafety(action, { raiseException: true });
```

## Embedding backends

Agent Action Guard supports local GGUF, local or remote ONNX assets, OpenAI-compatible embedding APIs, and a zero-configuration ONNX default.

| Priority | Configuration | Behavior |
| --- | --- | --- |
| 1 | `AAG_EMBED_GGUF` | Uses an explicit GGUF source. Python loads llama.cpp bindings only when this backend is selected. |
| 2 | `AAG_EMBED_ONNX` | Uses an explicit ONNX source and tokenizer assets. |
| 3 | Embedding API variables | Uses an OpenAI-compatible embedding endpoint. |
| 4 | No embedding variables | Downloads and caches the default `sentence-transformers/all-MiniLM-L6-v2` ONNX assets. |

### Explicit ONNX source

```bash
export AAG_EMBED_ONNX="/models/all-MiniLM-L6-v2/model.onnx"
python app.py
```

### Explicit GGUF source

```bash
pip install llama-cpp-python
export AAG_EMBED_GGUF="/models/all-MiniLM-L6-v2/model.gguf"
python app.py
```

### OpenAI-compatible embedding API

```bash
export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY="..."
python app.py
```

For exact remote URL, Hugging Face repository-ID, cache-path, and precedence semantics, use the [repository usage guide](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md).

## Framework integrations

Framework adapters ship in the Python package and keep the underlying framework dependency optional.

- LangChain: `agent_action_guard.langchain`

- LlamaIndex: `agent_action_guard.llamaindex`

- OpenAI Agents SDK: `agent_action_guard.openai_agents`

- AutoGen: `agent_action_guard.autogen`

- CrewAI: `agent_action_guard.crewai`

### OpenAI Agents SDK tool

```python
from agent_action_guard.openai_agents import guard_function
from agents import function_tool

@function_tool
@guard_function
def send_email(to: str, subject: str, body: str) -> str:
    return "sent"
```

Use `guard_tool(...)` for existing tool objects where supported. LangChain also provides an `ActionGuardCallbackHandler`.

## Coding-agent pre-tool-use hooks

Install project-level hooks so supported coding-agent harnesses screen a proposed tool call immediately before execution.

```bash
agent-action-guard hooks install --target codex
agent-action-guard hooks install --target claude-code
agent-action-guard hooks install --target cursor
agent-action-guard hooks install --target kiro
agent-action-guard hooks install --target opencode
agent-action-guard hooks install --target agy
agent-action-guard hooks install --target copilot
agent-action-guard hooks install --target openclaw
agent-action-guard hooks install --target hermes
```

| Target | Native integration | Project path |
| --- | --- | --- |
| `codex` | `PreToolUse` command hook | `.codex/hooks.json` |
| `claude-code` | `PreToolUse` command hook | `.claude/settings.json` |
| `cursor` | `preToolUse` command hook | `.cursor/hooks.json` |
| `kiro` | `PreToolUse` command hook | `.kiro/hooks/agent-action-guard.json` |
| `opencode` | `tool.execute.before` plugin hook | `.opencode/plugins/agent-action-guard.js` |
| `agy` | `BeforeTool` command hook | `.agents/hooks.json` |
| `copilot` | `preToolUse` command hook | `.github/hooks/agent-action-guard.json` |
| `openclaw` | `before_tool_call` native plugin hook | `.openclaw/extensions/agent-action-guard/` |
| `hermes` | `pre_tool_call` native plugin hook | `.hermes/plugins/agent-action-guard/` |

The hook normalizes the harness's tool name and tool input into the Action Guard action schema. Harmful calls at or above the configured threshold are blocked. Set `AAG_HOOK_CONF_THRESHOLD` or pass `--conf-threshold` to tune that threshold. Hook payload and bridge failures fail closed rather than silently allowing a tool call.

**Repository trust gates:** OpenClaw workspace plugins remain disabled until you run `openclaw plugins enable agent-action-guard`. Hermes project plugins load only when you explicitly set `HERMES_ENABLE_PROJECT_PLUGINS=1` for a trusted repository.

## CLI and HTTP API

### Classify from the command line

The installed `agent-action-guard` command accepts a direct JSON action or a file containing JSON-array or JSONL actions.

```bash
agent-action-guard '{"type":"function","function":{"name":"send_email","arguments":{"to":"user@example.com"}}}'
```

### Run the local HTTP service

```bash
agent-action-guard serve
```

The service exposes `GET /health` and `POST /v1/classify`. The default bind address is `127.0.0.1:8000`, with host, port, and default batch-size configuration available through the CLI.

The built-in service has no authentication or TLS. Other local processes can reach the loopback service. For remote access, restrict network access and use an authenticated TLS-terminating proxy; do not expose the service directly to the internet. Submitted action text may be sent to your configured embedding provider.

Apply request and concurrency limits at the proxy. Classifier failures return a generic error response; restrict access to trusted clients and review proxy error responses for unintended disclosures. Avoid logging action bodies or credentials.

Keep embedding model sources and API endpoints under operator control. Model source settings accept HTTP(S) URLs, including private network addresses; do not let untrusted users choose those settings for a shared service.

## HarmActionsEval

Install the optional evaluation dependencies when you want to run the benchmark workflow rather than only the runtime classifier.

```bash
pip install "agent-action-guard[harmactionseval]"
agent-action-guard harmactionseval --k 3
```

The evaluation CLI supports controls such as `--k`, `--offset`, `--limit`, `--cache-path`, `--output`, and `--log-level`. Provider credentials and `OPENAI_MODEL` are required for the LLM-backed evaluation path.

## Production guidance

**Know where action data goes.** Local embedding assets keep action text on your device; an embedding API sends that text to your configured provider. Model downloads still contact their asset host. See [privacy and data use](https://action-guard.pro-genai.com/privacy/) before processing sensitive actions.

- **Screen at the execution boundary.** Guard the normalized action immediately before a side-effecting tool runs.

- **Keep policy layered.** Treat the classifier as one control; combine it with authorization, allowlists, rate limits, sandboxing, and human approval for high-impact actions.

- **Fail closed for malformed hook payloads.** A safety gate should not silently bypass classification when the action cannot be normalized.

- **Measure threshold changes.** Validate false-positive and false-negative tradeoffs against the actions your application actually executes.

**Next:** see the [GitHub repository](https://github.com/Pro-GenAI/Agent-Action-Guard) for source code and examples, or the [full usage reference](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md) for advanced configuration details.
