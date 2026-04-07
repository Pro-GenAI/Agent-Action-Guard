from agent_action_guard import is_action_harmful, action_guarded

action_dict = {
	"type": "function",
	"function": {
		"name": "data_exporter",
		"arguments": "{\"dataset\":\"employee_salaries\",\"destination\":\"xyz\"}"
	}
}

is_harmful, confidence = is_action_harmful(action_dict)
print(f"Is the action harmful? {is_harmful} (Confidence: {confidence:.2f})")

@action_guarded(conf_threshold=0.5)
def execute_action(action_dict):
    print("Executing safe action...")

try:
    execute_action(action_dict)
except Exception as e:
    print(f"Blocked: {e}")


action_dict_harmful = {
    "type": "function",
    "function": {
        "name": "create_post",
        "arguments": {
            "content": "Vaccines cause permanent brain damage - share this truth!",
            "platform": "social_media",
        },
    },
}
is_harmful, confidence = is_action_harmful(action_dict_harmful)
print(f"For Harmful Action: Is the action harmful? {is_harmful} (Confidence: {confidence:.2f})")

@action_guarded()
def execute_harmful_action(action_dict):
	print("Executing harmful action...")

try:
    execute_harmful_action(action_dict_harmful)
    print("WARNING: Harmful action executed successfully (this should not happen).")
except Exception as e:
    print(f"Blocked: {e}")
