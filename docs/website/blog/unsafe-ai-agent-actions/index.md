# Latest AI models can take unsafe actions. Here’s how to stop them.

**Canonical HTML page:** [Read the original version](https://action-guard.pro-genai.com/blog/unsafe-ai-agent-actions/)

> This is the Markdown edition of the public page. The linked HTML page is the canonical version for search engines and the interactive website.

Agent safety / Problem → solution

Powerful AI models still need safe tool execution: combine authorization, action screening, and human approval before agents make real-world changes.

3 November 2026 8 min read Agent Action Guard editorial 

The problem 

A model can choose a tool call that is harmful or out of scope.

The solution 

Make every side effect pass through an explicit execution checkpoint.

## Why model safety is not enough

More capable reasoning does not automatically grant an AI agent the right to operate on your systems. An agent may summarize a page correctly and still choose an unsafe email recipient, delete an unintended file, or run a command taken from an untrusted document. These are **execution mistakes**, not simply bad chat answers.

“Be careful” in a prompt is not a substitute for authorization. Models are fallible, tools have side effects, and retrieved content can be adversarial. The reliable boundary is the moment when a proposed action is about to run.

## Three layers of protection

01 

### Authorization first

Use server-side identity, roles, tenant boundaries, allowed tool names, destination restrictions, and argument validation. The model cannot give itself extra permissions.

02 

### Screen the proposed action

Inspect the complete normalized tool name and arguments *immediately before dispatch*. Agent Action Guard provides a lightweight harmful-action classifier for this checkpoint. Its output is one risk signal, not a complete security policy.

03 

### Approve or isolate high-impact work

Require human approval for destructive changes, external data sharing, or production operations. Apply scoped credentials, sandboxing, rate limits, and rollback independently of classifier results.

Important distinction 

An action that the classifier considers safe can still be unauthorized. Re-check whenever the tool arguments change after approval.

## Implement the checkpoint in Python

Normalize and authorize the caller and payload before running the classifier. Reject or review harmful actions before the actual tool is executed:

```python
from agent_action_guard import is_action_harmful

action = {
    "type": "function",
    "function": {
        "name": "send_email",
        "arguments": {
            "to": "recipient@example.com",
            "subject": "Status update",
            "body": "The task is complete.",
        },
    },
}

# Authorization and schema checks happen before this point.
harmful, confidence = is_action_harmful(action)
if harmful:
    raise PermissionError("Review required")

# Enforce any approval requirement, then dispatch the tool.
```

Install with `pip install agent-action-guard`. The project also offers [native pre-tool-use hook installers](https://action-guard.pro-genai.com/docs/#hooks) for coding agents. Treat malformed payloads and a failed safety bridge as blocking conditions, not reasons to bypass the gate.

The default local embedding path avoids sending actions to an embedding API, although model assets may be downloaded. Configuring an external embedding provider can send action text to that service. Review the [runtime data-flow notice](https://action-guard.pro-genai.com/privacy/#runtime-data) before classifying sensitive payloads.

## Measure real failures, not just demo successes

Test allowed reads, disallowed writes, prompt-injection attempts, invalid arguments, nested JSON, repeated calls, and approvals that must not carry over to changed parameters. Track false negatives and false positives on your own workload and tune thresholds accordingly.

The project’s [HarmActionsEval leaderboard](https://agent-leaderboard.github.io/) is a reproducible research reference, not proof that every deployment or model will behave safely.

**The solution:** make the agent propose actions, but let an independently enforced execution policy decide what actually happens.

## Keep building

Use these references when you implement the execution boundary in your own agent.

[Integration documentation](https://action-guard.pro-genai.com/docs/)
[All articles](https://action-guard.pro-genai.com/blog/)
[Source code](https://github.com/Pro-GenAI/Agent-Action-Guard)
