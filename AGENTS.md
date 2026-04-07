# Project: Agent Action Guard

Agent Action Guard classifies proposed AI agent actions as safe or harmful and blocks or flags harmful actions. This repository provides the model, dataset, integration helpers, and example MCP-compatible tooling to enable runtime action screening in agent loops.
- Repository URL: https://github.com/Pro-GenAI/Agent-Action-Guard

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

# Decorator
@action_guarded(conf_threshold=0.5)
def execute_action(action_dict, *args, **kwargs):
    pass
```

## Key Files & Structure

- `agent_action_guard/` — implementation package (classifier, runtime helpers, dataset loaders).
- `training/` — training scripts and dataset artifacts used to produce the classifier.
- `examples/` — sample integrations and MCP server examples.
- `tests/` — unit tests validating core behavior.
- `USAGE.md` — detailed usage examples and environment setup.
- `README.md` — project overview, demos, and citations.

## Architecture Overview

- Input: proposed agent action (structured dict describing tool call, intent, parameters).
- Preprocessing: optional embedding + metadata normalization.
- Classifier: lightweight NN (PyTorch / ONNX) outputs harmful/safe logits and confidence.
- Policy: decision layer in the agent runtime that blocks, allows, or requests human approval.

## Development & CI

- Formatting and linting: `make format` and `make lint` (created through `Makefile`).
- Tests: run `pytest` (configured by `pytest.ini`) to run test cases in [`tests/`](tests/) directory.

## Guidance for AI agents reading this repo

- Use `USAGE.md` and `examples/` for integration patterns rather than reproducing code.
- Prefer runtime API `is_action_harmful()` for decision making.
- Respect model limitations: the classifier is trained on a limited dataset; combine with rule-based checks for high-risk systems.

## Where to look next (quick links)

- Full details and demo: [README.md](README.md)
- Integration and examples: [USAGE.md](USAGE.md) and `examples/`
- Implementation: `agent_action_guard/`
- Training scripts & dataset: `training/`
