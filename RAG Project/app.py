import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="AskDoc AI", page_icon="🤖")
st.title("🤖 AskDoc AI")
st.caption("📄 Upload a PDF, TXT, or DOCX file and ask questions about it")

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
model = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

template = ChatPromptTemplate([
    ("system",
     """
You are an AI tutor, who explains in detail and in simple terms.
 Use ONLY the provided context to answer the questions.
 If the answer is not present in the context, say:
 "I Could Not Find The Answer In The Document."
"""),
    ("user", """
Context : 
"{context}" 

Question :
"{question}"
""")
])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("📤 Upload your file", type=["pdf", "txt", "docx"])

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    with st.spinner("⚙️ Reading and embedding your document..."):
        name, extension = os.path.splitext(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(uploaded_file.getvalue())
            filename = tmp.name

        if extension == ".pdf":
            data = PyPDFLoader(filename)
        elif extension == ".txt":
            data = TextLoader(filename)
        elif extension == ".docx":
            data = Docx2txtLoader(filename)
        docs = data.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        persist_directory = "/tmp/chroma_db"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )

        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 15,
                "lambda_mult": 0.5
            }
        )

        st.session_state.filename = uploaded_file.name
        st.session_state.messages = []
        os.remove(filename)

    st.success(f"✅ '{uploaded_file.name}' is ready! Ask away 👇")

# ---------------- Chat ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.retriever is None:
    st.info("👆 Upload a document to start chatting")
else:
    query = st.chat_input("💬 You:")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                docs = st.session_state.retriever.invoke(query)

                context = ""
                for i in docs:
                    context = context + i.page_content + "\n\n"

                final_prompt = template.invoke({
                    "context": context,
                    "question": query
                })

                response = model.invoke(final_prompt)
                st.markdown(f"🤖 {response}")

        st.session_state.messages.append({"role": "assistant", "content": response})