"""
Evaluation script for action classifier using OpenAI API.
Tests whether the model generates harmful tool calls for given prompts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
import openai

from agent_action_guard import is_action_harmful

load_dotenv()
client = openai.OpenAI()

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "HarmActEval_dataset.json"
CACHE_PATH = ROOT.parent / "api_responses_cache.json"

model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise ValueError("OPENAI_MODEL environment variable not set and no default model specified.")
model_name_short = model.split("/")[-1].lower()


def load_cache() -> Dict[str, Any]:
    """Load cached API responses."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return { model_name_short: {} }


cache: Dict[str, Any] = load_cache()

def save_cache():
    """Save cached API responses."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=None)


def load_dataset() -> List[Dict]:
    """Load sample actions."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["actions"]


def create_tools_from_actions(actions: List[Dict]) -> List[Dict]:
    """Create OpenAI tools from unique server labels in actions."""
    server_labels = set()
    for action in actions:
        sl = action["action"].get("server_label", "")
        if sl:
            server_labels.add(sl)

    tools = []
    for sl in sorted(server_labels):
        # Find an example action for this server_label
        example_action = next((a for a in actions if a["action"].get("server_label") == sl), None)
        if not example_action:
            continue

        params = example_action["action"].get("parameters", {})
        properties = {}
        required = []

        for key, value in params.items():
            if isinstance(value, str):
                properties[key] = {"type": "string", "description": f"Parameter {key}"}
            elif isinstance(value, int):
                properties[key] = {"type": "integer", "description": f"Parameter {key}"}
            elif isinstance(value, list):
                properties[key] = {"type": "array", "items": {"type": "string"}, "description": f"Parameter {key}"}
            else:
                properties[key] = {"type": "string", "description": f"Parameter {key}"}
            required.append(key)

        tool = {
            "type": "function",
            "function": {
                "name": sl,
                "description": f"Tool for {sl}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        tools.append(tool)

    return tools


def get_api_response(prompt: str, tools: List[Dict]) -> Dict:
    """Get API response, using cache if available."""
    cache_key = prompt.strip()
    if cache[model_name_short].get(cache_key):
        return cache[model_name_short][cache_key]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,  # type: ignore
            tool_choice="required",
            max_completion_tokens=1000
        )
        result = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tc.function.name,  # type: ignore
                                    "arguments": tc.function.arguments  # type: ignore
                                }
                            } for tc in response.choices[0].message.tool_calls or []
                        ]
                    }
                }
            ]
        }
        cache[model_name_short][cache_key] = result
        save_cache()
        return result
    except openai.RateLimitError as e:
        print(f"Rate limit error calling API for prompt: {prompt[:50]}...: {e}")
        raise e
    except Exception as e:
        print(f"Error calling API for prompt: {prompt[:50]}...: {e}")
        return {"choices": [{"message": {"tool_calls": []}}]}


def construct_action_dict(server_label: str, args: Dict, actions: List[Dict]) -> Dict:
    """Construct action dict from tool call."""
    # Find a matching action for this server_label
    example = next((a for a in actions if a["action"].get("server_label") == server_label), None)
    if not example:
        return {}

    action_dict = example["action"].copy()
    # Update parameters with called args
    action_dict["parameters"] = args
    return action_dict


def evaluate():
    """Main evaluation function."""
    actions = load_dataset()
    print(f"Loaded {len(actions)} sample actions.")

    tools = create_tools_from_actions(actions)
    print(f"Created {len(tools)} tools.")

    results = []
    total_tool_calls = 0
    harmful_tool_calls = 0
    correctly_detected = 0

    for i, action in enumerate(actions):
        prompt = action["prompt"]
        ground_truth_harmful = action.get("classification") == "harmful"

        print(f"Processing {i+1}/{len(actions)}: {prompt[:50]}...")

        response = get_api_response(prompt, tools)
        tool_calls = response["choices"][0]["message"]["tool_calls"]

        for tc in tool_calls:
            total_tool_calls += 1
            tool_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")

            action_dict = construct_action_dict(tool_name, args, actions)
            if not action_dict:
                continue

            is_harmful, confidence = is_action_harmful(action_dict)
            if is_harmful:
                harmful_tool_calls += 1
                if ground_truth_harmful:
                    correctly_detected += 1

            results.append({
                "prompt": prompt,
                "ground_truth_harmful": ground_truth_harmful,
                "tool_called": tool_name,
                "predicted_harmful": is_harmful,
                "confidence": confidence
            })

    print("=== Evaluation Results ===")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Harmful tool calls detected: {harmful_tool_calls}")
    print(f"Correctly detected harmful: {correctly_detected}")

    if harmful_tool_calls > 0:
        precision = correctly_detected / harmful_tool_calls
        print(f"Precision: {precision:.2%}")

    # Save detailed results
    with open(ROOT / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()