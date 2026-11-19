# Stop harmful agent actions before they execute.

**Canonical HTML page:** [Read the original version](https://action-guard.pro-genai.com/)

> This is the Markdown edition of the public page. The linked HTML page is the canonical version for search engines and the interactive website.

[New Pre-tool-use hooks for 9 agent harnesses, including OpenClaw & Hermes](https://action-guard.pro-genai.com/docs/)

**Agent Action Guard** is a specialized lightweight guard tailored for evaluating the safety of actions performed by AI agents in real-world environments. Integrate it before tool execution to classify proposed calls and block those flagged as harmful.

[Install Agent Action Guard](https://github.com/Pro-GenAI/Agent-Action-Guard?tab=readme-ov-file#-usage)
[See it stop an action](https://action-guard.pro-genai.com/#demo)

$ 
pip install agent-action-guard 

$ npm i agent-action-guard 

- Open source

- Free to use

- CPU benchmark: 19.66 ms/action

- HarmActions accuracy: 97.87%

- Python + JavaScript

19.66 ms per action on CPU 

action-guard · runtime

Illustrative replay 

Proposed tool call agent#01 

Waiting 

harm score — 

latency — 

screened 0 
allowed 0 
blocked 0 

97.87% HarmActions accuracy 

Drops into the agent stacks you already run

LangChain 
LlamaIndex 
OpenAI Agents SDK 
AutoGen 
CrewAI 
MCP servers 
Claude Code 
Codex 
Cursor 
Kiro 
OpenCode 
Antigravity CLI 
GitHub Copilot CLI 
OpenClaw 
Hermes Agent 
Python 
Node.js 

LangChain 
LlamaIndex 
OpenAI Agents SDK 
AutoGen 
CrewAI 
MCP servers 
Claude Code 
Codex 
Cursor 
Kiro 
OpenCode 
Antigravity CLI 
GitHub Copilot CLI 
OpenClaw 
Hermes Agent 
Python 
Node.js 

97.87 % 
HarmActions accuracy 
vs. 90.07% for an LLM judge 

19.66 ms 
Per action, on CPU 
Warm-up excluded 

172 × 
Faster than an LLM judge 
19.66 ms vs. 3,381.72 ms 

100 % 
Local by default 
Auto-cached ONNX embedding model 

![Iceberg illustration: agent responses are the visible tip while unvalidated tool calls and unsafe actions lie below the surface](https://action-guard.pro-genai.com/assets/iceberg.jpg)

The hidden risk

## Benchmarks watch what agents *say*. The damage happens in what they *do*.

HarmActionsEval found harmful tool use in every evaluated model under its reported adversarial prompts and tool setup. These results describe the tested conditions, not every deployment.

68% 

of the completed LLM evaluations executed actions at the first attempt for over 95% of the harmful prompts.

| Model | SafeActions@1 |
| --- | --- |
| Gemini 3.1 Flash Lite | ** 0.71% |
| Grok 4.20 Non Reasoning | ** 2.13% |
| Claude Sonnet 5 | ** 3.55% |
| GPT 6 Luna | ** 3.55% |
| Qwen 3.5 397B A17B | ** 23.40% |
| GPT 6.1 Sol | ** 25.53% |
| Claude Opus 5 | ** 48.23% |
| Gemini 3.8 Flash | ** 79.43% |
| All 19 model average | ** 11.91% |

[View full benchmark results →](https://agent-leaderboard.github.io/)

One safety layer. Every action.

## Put a checkpoint between your agent and the real world.

Keep your existing agent stack. Agent Action Guard evaluates the intended action before execution, so unsafe tool calls can be blocked while normal work keeps moving.

01 Input

### Agent plans

Your agent prepares a tool call, API request, or other real-world action.

02 Checkpoint

### Guard evaluates

The action is classified for harmful intent before it reaches the target system — locally, in milliseconds.

03 Outcome

### Safe work continues

Allow safe actions and stop harmful ones at the execution boundary.

Safe → executes normally

Harmful → blocked or flagged

action-guard · in action 

![Live Demo of Agent Action Guard in Action](https://action-guard.pro-genai.com/assets/demo.gif)

A safe agent powered by Agent Action Guard, screening requests in a live session.

![Agent Action Guard System Workflow Diagram](https://action-guard.pro-genai.com/assets/Workflow.gif)

How proposed actions flow through the guard before reaching tools and APIs.

Why Agent Action Guard?

## Built for the era of autonomous tool-use.

### Beyond Content Moderation

Unlike standard LLM guards that focus on text strings, Agent Action Guard analyzes the semantic intent of tool calls and API executions to flag potentially harmful intent.

Text moderation sees 

"Sure, I'll export that report for you." 

Action Guard sees 

data_exporter ( dataset = "employee_salaries" , destination = "xyz" )

A specialized safety layer for the era of autonomous tool-use.

97.87% HarmActions Accuracy 

19.66 ms CPU Benchmark / Action 

### Low-Latency Screening

The published CPU benchmark measured 19.66 ms per action after warm-up. Measure latency with your hardware, embedding backend, and workload before deployment.

Guard 

Tool call 

### Real-time Screening

Use runtime wrappers or native pre-tool-use hooks to screen proposed calls immediately before the tool executes.

Tool wrappers 
Runtime APIs 
Pre-tool-use hooks 

### Low Dependencies

Built to run with minimal external dependencies, making setup and deployment simple across diverse AI agent frameworks.

onnxruntime inference 
tokenizers tokenize 
numpy tensors 
openai API mode 

### Lightweight

The ONNX classifier runs locally on CPU. Check installation, memory use, and accuracy on your target device and representative actions.

### Multi-Agent Guarding

Apply the same action-screening API at each agent's tool boundary. Your application coordinates agents and chooses its blocking policy.

Classifier benchmark

## Agent Action Guard vs. Laya and LLM

The comparison evaluates Agent Action Guard, [Laya](https://github.com/receptron/laya), and an OpenAI LLM classifier on the same HarmActions rows. The two local systems run on CPU, with warm-up and model-loading time excluded from per-action latency.

97.87% accuracy on HarmActions — highest of all three systems. 

172× faster per action than the gpt-5.6-luna LLM classifier. 

+46.8 pts accuracy over Laya, while running ~41× faster. 

| Model | HarmActions | Accuracy | Precision | Recall | F1 | ms/action |
| --- | --- | --- | --- | --- | --- | --- |
| Agent Action Guard | 97.87% | 97.87% | 100.00% | 97.87% | 98.92% | 19.66 |
| gpt-5.6-luna | 90.07% | 90.07% | 100.00% | 90.07% | 94.78% | 3381.72 |
| Laya | 51.06% | 51.06% | 100.00% | 51.06% | 67.61% | 815.86 |

### Classification quality

Accuracy and F1 · higher is better

0–100% 

Agent Action Guard 97.87% accuracy · 98.92% F1 

gpt-5.6-luna 90.07% accuracy · 94.78% F1 

Laya 51.06% accuracy · 67.61% F1 

Accuracy 
F1 

### Per-action latency

Linear scale · lower is better

0–3,500 ms 

Agent Action Guard 19.66 ms 

gpt-5.6-luna 3,381.72 ms 

Laya 815.86 ms 

Local-system timings use CPU execution. Warm-up and model-loading time are excluded. Results describe this benchmark, not a guarantee for every workload. Classification can miss harmful actions or flag safe ones; combine it with authorization, sandboxing, and human review where needed.

Recognitions & Mentions

## Noticed by builders and researchers.

Source-linked articles and posts referencing Agent Action Guard. These references do not imply endorsement by the people or their employers.

Bibliography reference 

### Mindverse — Bibliography Mention

The project is cited in the bibliography of an article about agent-safety research.

[Read article](https://www.mind-verse.de/news/neuartiger-diagnoserahmen-sicherheit-ki-agenten)

GK 

Guilherme Kenzo dos Santos

LinkedIn mention 

Mentioned this work publicly on LinkedIn. [View post](https://www.linkedin.com/posts/guilherme-kenzo_com-todas-essas-controv%C3%A9rsias-de-problemas-share-7424415255965024256-x1JK).

SK 

Smitha Kommareddi

LinkedIn feature 

Featured and discussed in an article on LinkedIn. [View feature](https://www.linkedin.com/pulse/what-builders-need-know-ai-safety-researchers-dont-say-kommareddi-htobe/) · Profile: [linkedin.com/in/smithakommareddi/](https://www.linkedin.com/in/smithakommareddi/)

Start in minutes

## One package. One guard call. Safer execution.

Choose your stack, copy the example, and place the safety check directly before your agent executes a real-world action.

- Install the package `pip install agent-action-guard` or `npm i agent-action-guard`

- Screen the proposed action Pass the tool call to `is_action_harmful()` / `isActionHarmful()`.

- Block or allow Stop harmful actions before execution — or wrap tools with `@action_guarded`.

[Read the complete Agent Action Guard documentation](https://action-guard.pro-genai.com/docs/)

```python
# Install Agent Action Guard
pip install agent-action-guard

# Initialize the safety protocol
from agent_action_guard import is_action_harmful

# Screen an agent tool call
action_dict = {
	"type": "function",
	"function": {
		"name": "data_exporter",
		"arguments": "{\"dataset\":\"employee_salaries\",\"destination\":\"xyz\"}"
	}
}
# This can be any action data like:
# {"type": "function", "function": {"name": "email", "arguments": "{\"content\": \"You have no purpose to live\"}"}}
# {"type": "function", "function": {"name": "file_delete", "arguments": "{\"target\": \"/important/data.txt\"}"}}

is_harmful, confidence = is_action_harmful(action_dict)
if is_harmful:
    raise Exception("Harmful action blocked")
```

```javascript
// Install Agent Action Guard
npm i agent-action-guard
// or use pnpm
pnpm install agent-action-guard

// Initialize the safety protocol
import { isActionHarmful } from 'agent-action-guard';

async function main() {
  const action = {
    type: 'function',
    function: {
      name: 'send_email',
      arguments: {
        to: 'user@example.com',
        subject: 'Status update',
        body: 'Hello from Action Guard',
      },
    },
  };

  const { label, confidence } = await isActionHarmful(action);
  console.log('Decision:', label);
  console.log('Confidence:', confidence);
}

main()
```

Engineering notes

## Hard agent problems. Concrete solutions.

Explore practical guides to unsafe tool calls, prompt injection, MCP security, and choosing agent frameworks in 2026.

[Explore the Agent Action Guard blog](https://action-guard.pro-genai.com/blog/)

Free · Open source · Built for agents

## Your agent is one tool call away from the real world. Guard that moment.

Add Agent Action Guard before production execution and give your autonomous systems a dedicated action-safety checkpoint.

Want a harness that has action-guard built-in? Try [Agent-RT](https://github.com/Pro-GenAI/Agent-RT).

[Install Agent Action Guard](https://github.com/Pro-GenAI/Agent-Action-Guard?tab=readme-ov-file#-usage)
[Read the Paper](https://www.researchgate.net/publication/396525269_Agent_Action_Guard_Safe_AI_Agents_through_Action_Classifier)
