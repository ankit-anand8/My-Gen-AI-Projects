from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import streamlit as st

from dataBase import build_retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI


# ---------------------------- Page Setup ----------------------------
st.set_page_config(page_title="AskDoc AI", page_icon="📚")
st.title("📚 AskDoc AI")
st.caption("🤖 Ask questions about your document")


# ---------------------------- Prompt Template ----------------------------
template = ChatPromptTemplate([
    (
        "system",
        """
        You are an AI tutor who explains things in detail and in simple terms.
        Use ONLY the provided context to answer the question.
        If the answer is not present in the context, say:
        "I Could Not Find The Answer In The Document."
        """
    ),
    (
        "user",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])


# ---------------------------- LLM ----------------------------
model = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# ---------------------------- Upload File ----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your PDF / TXT / DOCX file",
    type=["pdf", "txt", "docx"]
)


if uploaded_file is None:
    # No file -> clear old state and show a hint
    st.session_state.pop("retriever", None)
    st.session_state.pop("file_name", None)
    st.session_state.pop("messages", None)
    st.info("👆 Please upload a document to start chatting.")

else:
    # Build the retriever only once per uploaded file
    if st.session_state.get("file_name") != uploaded_file.name:

        extension = os.path.splitext(uploaded_file.name)[1]

        # build_retriever() needs a file path, so save the upload temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        with st.spinner("⏳ Reading your document and creating the RAG system..."):
            st.session_state.retriever = build_retriever(tmp_path)

        os.remove(tmp_path)

        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []

    st.success("✅ RAG System Created! Ask questions below.")

    # ---------------------------- Show Chat History ----------------------------
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # ---------------------------- Chat Input ----------------------------
    query = st.chat_input("💬 Ask a question about your document...")

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query)

        # Retrieve relevant chunks
        docs = st.session_state.retriever.invoke(query)

        # Combine retrieved chunks into context
        context = ""
        for doc in docs:
            context += doc.page_content + "\n\n"

        # Create final prompt
        final_prompt = template.invoke({
            "context": context,
            "question": query
        })

        # Ask LLM
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                response = model.invoke(final_prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})