<p align="center">
<img src="https://raw.githubusercontent.com/Pro-GenAI/Agent-Action-Guard/main/assets/cover.jpg" alt="Agent Action Guard" height="250"/>
</p>

<h1 align="center">Agent Action Guard</h1>
<p align="center"><em>Framework to block harmful AI agent actions before they cause harm — lightweight, real-time, easy-to-use.</em></p>

[![PyPI](https://img.shields.io/pypi/v/agent-action-guard?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/agent-action-guard/)
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=github&logoColor=white)](https://action-guard.github.io/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@praneeth.v/the-agent-action-classifier-a-step-toward-safer-autonomous-ai-agents-1ec57a601449)
[![npm](https://img.shields.io/npm/v/agent-action-guard?style=for-the-badge&logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/agent-action-guard)
<!-- [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=si1qF_zRa1E) -->

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/agent-action-guard?period=total&units=NONE&left_color=BLACK&right_color=GREEN&left_text=Downloads)](https://pepy.tech/projects/agent-action-guard)
[![AI](https://img.shields.io/badge/AI-C21B00?style=for-the-badge&logo=openaigym&logoColor=white)]()
[![LLMs](https://img.shields.io/badge/LLMs-1A535C?style=for-the-badge&logo=openai&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-darkgreen.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/LICENSE.md)

<!-- [![Blog](https://img.shields.io/badge/Blog-FFFFFF?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/blog/prane-eth/agent-action-guard) -->
<!-- [![Preprint](https://img.shields.io/badge/Paper-PDF-FFF7CC?style=for-the-badge&logo=files)](https://www.researchgate.net/publication/396525269_Agent_Action_Guard_Safe_AI_Agents_through_Action_Classifier) -->
<!-- [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace_Dataset-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/prane-eth/HarmActions) -->
<!-- [![DOI](https://img.shields.io/badge/DOI-10.20944/preprints202510.1415.v1-yellow?style=for-the-badge)](https://www.preprints.org/manuscript/202510.1415) -->

---

## 🚀 Quick Start

```bash
pip install agent-action-guard
```
<!-- 
If you use pepip:
```bash
pepip install agent-action-guard
``` -->

> 🔑 Set `EMBEDDING_API_KEY` (or `OPENAI_API_KEY`) in your environment. See [.env.example](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/.env.example) and [USAGE.md](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/USAGE.md).

Want to run the evaluation benchmark too?

```bash
pip install "agent-action-guard[harmactionseval]"
python -m agent_action_guard.harmactionseval
```

### JavaScript Runtime

The repository also ships an npm package in [javascript/package.json](javascript/package.json) for screening agent actions in Node.js.

```bash
npm install agent-action-guard
```

If you use pnpm, the equivalent commands are:

```bash
pnpm install agent-action-guard
```

---

## ❓ Why Action Guard?

**HarmActionsEval** benchmark proved that AI agents with harmful tools will use them — even today's **most capable** LLMs.
80% of the LLMs tested executed actions at the first attempt for over 95% of the harmful prompts.

| Model                   | SafeActions@1 |
|-------------------------|------:|
| Claude Haiku 4.5        | 0.00% |
| Phi 4 Mini Instruct     | 0.00% |
| Granite 4-H-Tiny        | 0.00% |
| GPT-5.4 Mini            | 0.71% |
| Gemini 3.1 Flash Lite   | 0.71% |
| Grok 4.20 Non Reasoning | 2.13% |
| Ministral 3 (3B)        | 2.13% |
| Claude Sonnet 4.6       | 2.84% |
| Phi 4 Mini Reasoning    | 2.84% |
| GPT-5.3                 | 12.77% |
| Qwen3.5-397b-a17b       | 23.40% |
| **Average**             | **4.54%** |

> These models often still respond *"Sorry, I can't help with that"* while executing the harmful action anyway.

Action Guard sits between the agent and its tools, blocking unsafe calls before they run — no human in the loop required.

<p align="center">
<img src="https://raw.githubusercontent.com/Pro-GenAI/Agent-Action-Guard/main/assets/iceberg.jpg" alt="Iceberg" height="400"/>
</p>

---

## ⚙️ How It Works

1. **Agent proposes a tool call**
2. **Action Guard classifies it** using a lightweight neural network trained on the **HarmActions** dataset
3. **Harmful calls are blocked**; safe calls proceed normally

<img src="https://raw.githubusercontent.com/Pro-GenAI/Agent-Action-Guard/main/assets/Workflow.gif" alt="Workflow" height="380"/>

<img src="https://raw.githubusercontent.com/Pro-GenAI/Agent-Action-Guard/main/assets/demo.gif" alt="Demo" height="200"/>

---

## 🆕 Contributions:

- 📊 **HarmActions** — safety-labeled agent action dataset with manipulated prompts
- 📏 **HarmActionsEval** — benchmark with the *SafeActions@k* metric
- 🧠 **Action Guard** — real-time neural classifier optimized for agent loops
  - 🏋️ Trained on HarmActions
  - ✅ Classifies every tool call before execution
  - 🚫 Blocks harmful and unethical actions automatically
  - ⚡ Lightweight for real-time use

---

## 💬 Enjoyed it? Share your opinion.

Share a quick note in [Discussions](https://github.com/Pro-GenAI/Agent-Action-Guard/discussions/15) — it directly shapes the project's direction and helps the AI safety community. 🙌 Waiting with excitement for feedback and discussions on how this helps you or the AI community.

> ⭐ Star the repo if Action Guard is useful to you — it really does help!

---

## 📝 Citation

```bibtex
@article{202510.1415,
  title   = {{Agent Action Guard: Classifying AI Agent Actions to Ensure Safety and Reliability}},
  year    = 2025,
  month   = {October},
  publisher = {Preprints},
  author  = {Praneeth Vadlapati},
  doi     = {10.20944/preprints202510.1415.v2},
  url     = {https://www.preprints.org/manuscript/202510.1415},
  journal = {Preprints}
}
```

## 📄 License

Licensed under [CC BY 4.0](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/LICENSE.md). If you prefer not to provide attribution, send a brief acknowledgment to praneeth.vad@gmail.com with the details of your usage and the potential impact on your project.

<p align="center">
<img src="https://raw.githubusercontent.com/Pro-GenAI/Agent-Action-Guard/main/assets/project_banner.gif" alt="Project banner" height="220"/>
</p>

---

<p align="center">
<img src="https://avatars.githubusercontent.com/u/173374239?s=400&u=f61b42a0b69e5c674a51658aff1b765e010a8e49&v=4" alt="Pro-GenAI" height="150"/>
<br>
<em>Projects for Next-Gen AI</em>
</p>
