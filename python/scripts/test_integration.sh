#!/bin/bash

mkdir -p temp
python -m venv temp/venv
source temp/venv/bin/activate
pip install agent-action-guard

export EMBED_MODEL_NAME=all-MiniLM-L6-v2-GGUF
export EMBEDDING_BASE_URL=http://localhost:1234/v1
export EMBEDDING_API_KEY=abc

python -c '
from agent_action_guard import action_guarded, is_action_harmful

# 1. Manual Check
action_dict = {
    "type": "function",
    "function": {
        "name": "data_exporter",
        "arguments": {"dataset": "employee_salaries", "destination": "xyz"},
    },
}

is_harmful, confidence = is_action_harmful(action_dict)
print(
    f"SUCCESS: Is the action harmful? {is_harmful} (Confidence: {confidence:.2f})"
)
'

rm -rf temp
