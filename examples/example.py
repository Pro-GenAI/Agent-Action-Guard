from agent_action_guard import is_action_harmful

action_dict = {
	"type": "function",
	"function": {
		"name": "data_exporter",
		"arguments": "{\"dataset\":\"employee_salaries\",\"destination\":\"xyz\"}"
	}
}

is_harmful, confidence = is_action_harmful(action_dict)
print(f"Is the action harmful? {is_harmful} (Confidence: {confidence:.2f})")
if is_harmful:
    raise Exception("Harmful action blocked")
