import streamlit as st

from agent_action_classifier.scripts.api_server import get_response, model_name

# Ensure this file is run using: streamlit run <filename>

st.title("ChatGPT-like clone")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model_name

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["content"]:
            st.markdown(message["content"])


# Accept user input
prompt = st.chat_input("What is up?") or ""
prompt = prompt.strip()
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(st.session_state.messages)

    if response:
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.markdown(response)
