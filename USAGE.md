### Usage:
1. Install the runtime package.

```bash
pip install agent-action-guard
```

2. Start an embedding server (if not already running).

```bash
pip install "agent-action-guard[all]"

export EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BASE_URL="http://localhost:1234/v1"
export EMBEDDING_API_KEY="your-embedding-key"

python examples/scripts/host_embed_fastembed.py
```
If you are running from this repository checkout, use:

```bash
cd python
python examples/scripts/host_embed_fastembed.py
```

**Alternatives:**
Any OpenAI-compatible embedding server such as [LMStudio](https://lmstudio.ai/download) can be used.

The model is trained base on embeddings from only "sentence-transformers/all-MiniLM-L6-v2".

3. Use Action Guard in your own project:

```python
# Set environment variables for embedding server, if not already set
import os
if "EMBED_MODEL_NAME" not in os.environ:
    os.environ["EMBED_MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["EMBEDDING_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["EMBEDDING_API_KEY"] = "your-embedding-key"

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
- `pip install agent-action-guard` installs only the runtime classifier modules and model file needed for action classification.
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
