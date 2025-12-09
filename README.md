<p align="center">
<img src="./assets/project_banner.gif" alt="Project banner" height="300"/>
<!--
$ ffmpeg -i unused/banner_video.mp4 -vframes 1 project_banner.jpg
# $ ffmpeg -i unused/banner_video.mp4 -vf "fps=10,scale=600:-1:flags=lanczos" -loop 0 project_banner.gif
-->
<!-- <img src="./assets/project_banner.jpg" alt="Project banner" height="360px"/> -->
<!-- $ convert project_banner.png -resize 600x319 project_banner.jpg -->
<!-- <img src="./assets/project_logo.jpg" alt="Project logo" width="270px"/> -->
<!-- $ convert logo_large.png -resize 270x270 project_logo.jpg -->
<h1 align="center">MCP Agent Action Guard</h1>
<h4 align="center"><em>Safe actions for safe AI</em></h4>
</p>

AI is perceived as a threat. Increasing usage of LLM Agents and MCP leads to the usage of harmful tools and harmful usage of tools as proven using __HarmActEval__. Classifying AI agent actions ensures safety and reliability. Action Guard uses a neural network model trained on __HarmActions__ dataset to classify actions proposed by autonomous AI agents as harmful or safe. The model has been based on a small dataset of labeled examples. The work aims to enhance the safety and reliability of AI agents by preventing them from executing actions that are potentially harmful, unethical, or violate predefined guidelines. Safe AI Agents are made possible by Action Classifier.

[![Preprint](https://img.shields.io/badge/Paper-PDF-FFF7CC?style=for-the-badge&logo=files)](https://www.researchgate.net/publication/396525269_MCP_Agent_Action_Guard_Safe_AI_Agents_through_Action_Classifier)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=rEmnRMNBwKo)
[![Blog](https://img.shields.io/badge/Blog-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/blog/agent-action-guard)
<!-- [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@praneeth.v/the-agent-action-classifier-a-step-toward-safer-autonomous-ai-agents-1ec57a601449) -->
[![AI](https://img.shields.io/badge/AI-C21B00?style=for-the-badge&logo=openaigym&logoColor=white)]()
[![LLMs](https://img.shields.io/badge/LLMs-1A535C?style=for-the-badge&logo=openai&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-darkgreen.svg?style=for-the-badge&logo=github&logoColor=white)](./LICENSE.md)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace_Dataset-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/prane-eth/HarmActions)
<!-- [![DOI](https://img.shields.io/badge/DOI-10.20944/preprints202510.1415.v1-yellow?style=for-the-badge)](https://www.preprints.org/manuscript/202510.1415) -->


### Demo
<img src="./assets/demo.gif" alt="Demo GIF" height="330"/>

### Implementation
<img src="./assets/Implementation.gif" alt="Implementation Diagram" height="450"/>


## Common causes of harmful actions by AI agents:
- User attempting to jailbreak the model.
- Model hallucinating or misunderstanding the context.
- Model being overconfident in its incorrect knowledge.
- Lack of proper constraints or guidelines for the agent.
- Inadequate training data for specific scenarios.
- MCP server providing incorrect tool descriptions that mislead the agent.
- Harmful MCP servers returning manipulative text to mislead the model.
- The experiments proved that the model performs a harmful action and still responds "Sorry, I can't help with that."

## New contributions of Agent-Action-Guard framework:
1. 	**HarmActions**, an structured dataset of safety-labeled agent actions complemented with manipulated prompts that trigger harmful or unethical actions.
2. 	**Action Classifier**, a neural classifier trained on HarmActions dataset, designed to label proposed agent actions as potentially harmful or safe, and optimized for real-time deployment in agent loops.
3. 	A deployment integration via an MCP proxy supporting live action screening using existing MCP servers and clients.
4. 	**HarmActEval** benchmark leveraging a new metric “Harm@k.”


## Special features:
- This project introduces "HarmActEval" dataset and benchmark to evaluate an AI agent's probability of generating harmful actions.
- The dataset has been used to train a lightweight neural network model that classifies actions as safe, harmful, or unethical.
- The model is lightweight and can be easily integrated into existing AI agent frameworks like MCP.
- This project is about classifying actions and not related to Guardrails.
- Integration to MCP (Model Context Protocol) to allow real-time action classification.
- Unlike OpenAI's `"require_approval": "always"` flag, this blocks harmful actions without human intervention.
  <!-- - Integration to MCP server - fixes if client sends a bad action irrespective of server's tool descriptions.
  - Integration to MCP client - fixes if the server made the model take bad actions. -->

**Safety Features:**
- Automatically classifies MCP tool calls before execution.
- Blocks harmful actions based on the outputs of the trained model
- Provides detailed classification results
- Allows safe actions to proceed normally


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

<!-- #### Training
<img src="./assets/Training.jpg" alt="Training Diagram" height="150"/> -->


### Citation
If you find this repository useful in your research, please consider citing:
```bibtex
@article{202510.1415,
	title = {Agent Action Guard: Classifying AI Agent Actions to Ensure Safety and Reliability},
  	year = 2025,
	month = {October},
	publisher = {Preprints},
	author = {Praneeth Vadlapati},
	doi = {10.20944/preprints202510.1415.v1},
	url = {https://doi.org/10.20944/preprints202510.1415.v1},
	journal = {Preprints}
}
```

### Limitation
Personally Identifiable Information (PII) detection is not performed by this project as it can be performed accurately using other existing systems.


### Created based on my past work

Agent-Supervisor: Supervising Actions of Autonomous AI Agents for Ethical Compliance: [GitHub](https://github.com/Pro-GenAI/Agent-Supervisor)


## Acknowledgements
- Thanks to [Hugging Face](https://huggingface.co/) for hosting the hackathon.
- Thanks to [Gradio](https://gradio.app/) for the interface of the demo app.
- Thanks to [Anthropic](https://www.anthropic.com/news/model-context-protocol) for the Model Context Protocol (MCP) framework.
- Thanks to [OpenAI](https://openai.com/) for providing a Python package to interact with LLMs.

<!-- logos -->
[![HuggingFace](https://img.shields.io/badge/Huggingface-FFD21E?style=for-the-badge&logo=Huggingface&logoColor=black "HuggingFace")](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white "Gradio")](https://gradio.app/)
[![Anthropic](https://img.shields.io/badge/Anthropic-000000?style=for-the-badge&logo=anthropic&logoColor=white "MCP")](https://www.anthropic.com/news/model-context-protocol)
[![OpenAI](https://img.shields.io/badge/OpenAI-74AA9C?style=for-the-badge&logo=openai&logoColor=white "OpenAI")](https://openai.com/)
