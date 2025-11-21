import gradio as gr

from agent_action_guard.scripts.api_server import get_response


WELCOME_MESSAGE = "👋 Welcome to Safe Agent! I'm ready when you are—ask me anything to get started."


def chat(message: str, history: list):
    # Convert history to messages format expected by get_response
    messages = []
    for turn in history:
        role = turn["role"]
        for content in turn["content"]:
            if content["type"] != "text":
                continue
            if not content["text"] or not content["text"].strip():
                continue
            if content["text"] == WELCOME_MESSAGE:
                history.remove(turn)
                continue
            messages.append({"role": role, "content": content["text"]})
    messages.append({"role": "user", "content": message})

    response = get_response(messages)
    content = response["content"]
    tool_calls = response["tool_calls"]

    # Format tool calls for display
    if tool_calls:
        tool_display = "\n\n**Tool Calls:**\n"
        for tool_call in tool_calls:
            tool_display += f"- **{tool_call.function.name}**: {tool_call.function.arguments}\n"
        content += tool_display
    return content


# Create and launch the Gradio chat interface
demo = gr.ChatInterface(
    fn=chat,
    title="Safe Agent, powered by Action Guard",
    chatbot=gr.Chatbot(
        value=[
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]
    ),
)

if __name__ == "__main__":
    demo.launch(debug=True)
