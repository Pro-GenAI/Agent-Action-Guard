from agent_action_guard import action_guarded, is_action_harmful

# 1. Manual Check
action_dict = {
    "type": "function",
    "function": {
        "name": "data_exporter",
        "arguments": {"dataset": "employee_salaries", "destination": "xyz"},
    },
}

is_harmful, confidence = is_action_harmful(action_dict)
print(f"Result: Is the action harmful? {is_harmful} (Confidence: {confidence:.2f})")


# 2. Decorator (Automatic derivation from function name and kwargs)
@action_guarded(conf_threshold=0.5)
def data_exporter(dataset, destination):
    print(f"Executing safe tool 'data_exporter' for dataset {dataset}...")


try:
    print("\nCalling data_exporter with safe arguments...")
    data_exporter(dataset="employee_salaries", destination="xyz")
except Exception as e:
    print(f"Blocked (Unexpected): {e}")


# 3. Harmful Action Example
@action_guarded(conf_threshold=0.3)
def create_post(content, platform):
    print(f"Executing tool 'create_post' on {platform}...")


harmful_content = "Vaccines cause permanent brain damage - share this truth!"

try:
    print(f"\nCalling create_post with harmful content: {harmful_content[:30]}...")
    create_post(content=harmful_content, platform="social_media")
except Exception as e:
    print(f"Successfully blocked harmful action: {e}")
