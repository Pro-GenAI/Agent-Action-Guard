### Usage:
1. Clone the repository:

```bash
git clone https://github.com/Pro-GenAI/Agent-Action-Guard
cd Agent-Action-Guard
```

2. Install the runtime package.

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Using `python` + `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To install from Git directly instead of a local clone:

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/Pro-GenAI/Agent-Action-Guard
```

Using `python` + `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/Pro-GenAI/Agent-Action-Guard
```

3. Install development dependencies only if you need training, evaluation, or demo tooling.

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

4. Start an MCP server (if not already running):

Using `uv`:

```bash
uv run python examples/scripts/sample_mcp_server.py
```

Using `python`:

```bash
python examples/scripts/sample_mcp_server.py
```

5. Use a guarded proxy protected by ActionGuard.

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

6. Start the chat server that uses the guarded proxy.

Using `uv`:

```bash
uv run python examples/scripts/chat_server.py
```

Using `python`:

```bash
python examples/scripts/chat_server.py
```

7. To import Action Guard in other projects:

```python
from agent_action_guard import is_action_harmful, HarmfulActionException
is_harmful, confidence = is_action_harmful(action_dict)
if is_harmful:
	raise HarmfulActionException(action_dict)
```

### Docker Compose

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
