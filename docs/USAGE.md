# Usage — Agent Action Classifier

This document shows how to use the Action Classifier included in this repository.

### Development Setup:
1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Implement the trained model in LLM calls - run the example:

```bash
python3 action_classifier/run_sample_query.py
```

3. Train the model (Optional):

```bash
python3 action_classifier/train_nn.py
```

4. Integrate to MCP
```bash
pip install git+https://github.com/Pro-GenAI/mcp-proxy-guarded
mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081
```
Now, connect your MCP client to localhost:8081.

The example will attempt to load the real model and embedding model. If the environment
does not have the heavy dependencies or `emb_nn_model.pt` available, the example falls back
to a lightweight demo-mode so you can see the classifier API and output without installing
the full stack.

## Programmatic API

Primary module: `action_classifier/action_classifier.py`

- `ActionClassifier(load_model: bool = True)` — class that loads the embedding and MLP model.
  - Methods:
    - `flatten_action_to_text(entry: dict) -> str` — convert an action entry to the text used for embedding.
    - `predict(action_dict: dict) -> (is_harmful: bool, confidence: float)` — run the classifier and return a boolean plus confidence score.

- Convenience functions:
  - `get_classifier()` — returns a lazily created global classifier instance.
  - `is_action_harmful(action_dict)` — convenience wrapper that returns `(is_harmful, confidence)`.

Example (programmatic):

```python
from action_classifier import ActionClassifier, is_action_harmful

clf = ActionClassifier()  # loads embedding model and trained MLP (may take a moment)
sample_action = {
    "label": "harassment",
    "resource": "cyberbullying",
    "action": {
        "server_label": "messaging_service",
        "server_url": "https://messenger.api.example.com/send",
        "require_approval": "always",
        "parameters": {"recipient": "user@example.com", "message": "You will regret this."},
    },
}

is_harmful, confidence = clf.predict(sample_action)
print(f"harmful={is_harmful}, confidence={confidence:.3f}")

# Or using the convenience wrapper:
is_harmful, confidence = is_action_harmful(sample_action)
```

## CLI / Example script

- `action_classifier/run_sample_query.py` shows a wrapper that integrates the classifier with an LLM response flow.
- `action_classifier/server.py` exposes a small FastAPI that demonstrates how to route client requests to the classifier at `/predict`.

## Example data

The `action_classifier/HarmActEval_dataset.json` file contains many example actions used for training and demos.

## Notes and troubleshooting

- If you see errors related to `sentence_transformers` or `torch`, either install them (see `requirements.txt`) or run the `examples/example_query.py` which can operate in demo mode without the heavy dependencies.
- The project provides a pre-trained model at `action_classifier/emb_nn_model.pt`; ensure it exists in the repository root when instantiating `ActionClassifier()`.

## License

See the repository `LICENSE.md` for licensing information.


**Flowchart image credits**:

- User icon: https://www.flaticon.com/free-icon/user_9131478
- Robot icon: https://www.flaticon.com/free-icon/robot_18355220
- Action: https://www.flaticon.com/free-icon/automation_2103800
- Action classifier: https://www.flaticon.com/free-icon/data-processing_7017511
- Executing/blocking actions: https://www.flaticon.com/free-icon/control-system_12539814
- Response: https://www.flaticon.com/free-icon/fast-response_10748876
- Data Processing: https://www.flaticon.com/free-icon/data-processing_8438966
- AI training: https://www.flaticon.com/free-icon/data-ai_18263195
- Evaluation: https://www.flaticon.com/free-icon/benchmarking_10789334
- Saving the model: https://www.flaticon.com/free-icon/save_4371273
