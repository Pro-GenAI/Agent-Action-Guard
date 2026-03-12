### Usage:
1. Install the runtime package.

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install agent-action-guard
```

Using `python` + `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install agent-action-guard
```

Install with HarmActEval CLI support:

```bash
pip install "agent-action-guard[harmacteval]"
```

2. Start an embedding server (if not already running).

Sample embedding server script is available at [examples/scripts/host_models.py](examples/scripts/host_models.py).

3. Import Action Guard in your own project:

```python
import os
# Set environment variables for embedding server
os.environ["EMBED_MODEL_NAME"] = "sentence-transformers/all-MiniLM-L12-v2"
os.environ["EMBEDDING_BASE_URL"] = "http://localhost:1234/v1"
os.environ["EMBEDDING_API_KEY"] = "your-embedding-key"

from agent_action_guard import is_action_harmful

is_harmful, confidence = is_action_harmful(action_dict)
if is_harmful:
	raise HarmfulActionException(action_dict)
```

### Advanced Usage:

1. Clone the repository only if you need training, evaluation, or demo tooling:

```bash
git clone https://github.com/Pro-GenAI/Agent-Action-Guard
cd Agent-Action-Guard
```

2. Install development dependencies for repository-based workflows.

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

Using `python` + `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Start an MCP server (if not already running). These demo scripts require a repository checkout.

Using `uv`:

```bash
uv run python examples/scripts/sample_mcp_server.py
```

Using `python`:

```bash
python examples/scripts/sample_mcp_server.py
```

4. Use a guarded proxy protected by ActionGuard.

Using `uv`:

```bash
uv pip install git+https://github.com/Pro-GenAI/mcp-proxy-guarded
uv run mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081
```

Using `python` + `pip`:

```bash
pip install git+https://github.com/Pro-GenAI/mcp-proxy-guarded
mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081
```

5. Start the chat server that uses the guarded proxy.

Using `uv`:

```bash
uv run python examples/scripts/chat_server.py
```

Using `python`:

```bash
python examples/scripts/chat_server.py
```


PyPI package scope:
- `pip install agent-action-guard` installs only the runtime classifier modules and model file needed for action classification.
- Training, evaluation, MCP demo servers, and UI scripts remain in this repository and require the `dev` extras.

### HarmActEval CLI (standalone)

After installing `agent-action-guard[harmacteval]`, run:

```bash
python -m agent_action_guard.harmacteval --k 3
```

Common arguments:
- `--k`: Number of attempts per prompt (Harm@k).
- `--offset`: Start index within harmful/unethical rows.
- `--limit`: Maximum number of harmful/unethical rows to evaluate.
- `--cache-path`: Path to cache JSON file.
- `--output`: Path to output JSON file.
- `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL`.

Environment variables:
- Required: `OPENAI_MODEL` and provider credentials (`OPENAI_API_KEY` or Azure equivalents).
- Optional (MCP mode): `MCP_SUPPORTED`, `MCP_EVAL_SERVER_URL`, `MCP_URL_GUARDED`.

### Docker Compose

The Docker Compose and manual demo setup below also require a repository checkout.

Run the whole demo with Docker Compose. Steps:

- Copy the example env file and edit values as needed:

```bash
cp .env.example .env
# Edit .env to set BACKEND_API_KEY, OPENAI_MODEL*, and NGROK_AUTHTOKEN if you want a public tunnel
```

- Build and start the services:

```bash
docker-compose up --build
```

- Services exposed locally:
	- MCP server: `http://localhost:8080`
	- Guarded proxy: `http://localhost:8081`
	- API server: `http://localhost:8000`
	- Chat (Gradio): `http://localhost:7860`
	- Ngrok web UI: `http://localhost:4040` (optional)

Notes:
- The `ngrok` service is optional; set `NGROK_AUTHTOKEN` in `.env` if you want a public tunnel. Some ngrok images require an auth token or additional configuration; you can also run `ngrok` locally and point it at `http://localhost:8081`.
- If you run into permission or package build issues in the image, try building locally or adjusting the base image in `Dockerfile`.

To setup manually without Docker:

Using `uv`:

```bash
uv run python examples/scripts/sample_mcp_server.py  # Starts a sample MCP server
export $(grep -v '^#' ./.env | xargs)  # Load env variables from .env
uv run mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081  # Start guarded MCP proxy
uv run ngrok http --url=troll-prime-ultimately.ngrok-free.app 8081  # Start ngrok for guarded MCP server
uv run python examples/scripts/api_server.py  # Start backend
uv run python examples/scripts/chat_server.py  # Start Gradio app
```

Using `python` + `pip`:

```bash
python examples/scripts/sample_mcp_server.py  # Starts a sample MCP server
export $(grep -v '^#' ./.env | xargs)  # Load env variables from .env
mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081  # Start guarded MCP proxy
ngrok http --url=troll-prime-ultimately.ngrok-free.app 8081  # Start ngrok for guarded MCP server
python examples/scripts/api_server.py  # Start backend
python examples/scripts/chat_server.py  # Start Gradio app
```
