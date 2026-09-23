import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Mode Chatbot", page_icon="🤖")

# ---- Mode definitions (same as original script) ----
MODES = {
    "1": ("😡 Angry mode", "You are a angry AI. Always angrry and frustated, you are also impateint, and roast the user for his questions asked"),
    "2": ("😂 Funny mode", "You are a funny AI, give replies with humour and funny way, you always try to make laugh the user and also crack jokes"),
    "3": ("📝 Poetic mode", "You are a poet AI, you always give answers like a poem"),
    "4": ("👩‍🏫 Teacher mode", "You are a tutor AI, you explain eveything in detail with examples and in simple terms"),
}

# ---- Session state init ----
if "mode_selected" not in st.session_state:
    st.session_state.mode_selected = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("🤖 Mode Chatbot")

# ---- Mode selection screen (mirrors the CLI choice prompt) ----
if not st.session_state.mode_selected:
    st.write("😡 Press 1 for Angry mode")
    st.write("😂 Press 2 for Funny mode")
    st.write("📝 Press 3 for Poetic mode")
    st.write("👩‍🏫 Press 4 for Teacher mode")

    choice = st.text_input("🔢 Enter your choice :", key="choice_input")

    if st.button("🚀 Start Chat"):
        if choice in MODES:
            mode_name, mode_prompt = MODES[choice]
        else:
            mode_name, mode_prompt = "🙂 Helpful mode", "You are a helpful AI assistant"
            st.warning("⚠️ Invalid choice, defaulting to helpful mode")

        st.session_state.messages = [SystemMessage(content=mode_prompt)]
        st.session_state.mode_name = mode_name
        st.session_state.mode_selected = True
        st.rerun()

# ---- Chat screen ----
else:
    st.caption(f"Mode: **{st.session_state.mode_name}**")

    # Replay chat history (skip the SystemMessage)
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg.content)

    prompt = st.chat_input("💬 You :")
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="🧑"):
            st.write(prompt)

        response = st.session_state.model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response))
        with st.chat_message("assistant", avatar="🤖"):
            st.write(response)

    if st.button("🔄 Restart / Change mode"):
        st.session_state.mode_selected = False
        st.session_state.messages = []
        st.rerun()