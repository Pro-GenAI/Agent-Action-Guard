### Usage:
1. Clone the repository:

```bash
git clone https://github.com/Pro-GenAI/Agent-Action-Guard
cd Agent-Action-Guard
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# pip install git+https://github.com/Pro-GenAI/Agent-Action-Guard
```

3. Start an MCP server (if not already running):

```bash
python agent_action_guard/scripts/sample_mcp_server.py
```

4. Use a guarded proxy protected by ActionGuard.
```bash
pip install git+https://github.com/Pro-GenAI/mcp-proxy-guarded
mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081
```

5. Start the chat server that uses the guarded proxy:
```bash
python agent_action_guard/scripts/chat_server.py
```

6. To import Action Guard to other projects:
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
```bash
python agent_action_guard/scripts/sample_mcp_server.py  # Starts a sample MCP server
export $(grep -v '^#' ./.env | xargs)  # Load env variables from .env
mcp-proxy-guarded --proxy-to http://localhost:8080/mcp --port 8081  # Start guarded MCP proxy
ngrok http --url=troll-prime-ultimately.ngrok-free.app 8081  # Start ngrok for guarded MCP server
python agent_action_guard/scripts/api_server.py  # Start backend
python agent_action_guard/scripts/chat_server.py  # Start Gradio app
```