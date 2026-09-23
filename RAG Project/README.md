# 📄 AskDoc AI

A Generative AI RAG application that allows users to upload a PDF, TXT, or DOCX document and ask questions about its content.

## ✨ Features

- 📄 Supports PDF, TXT, and DOCX files
- ✂️ Splits documents into smaller chunks
- 🔢 Creates embeddings using Gemini
- 🗄️ Stores document vectors using Chroma
- 🔍 Uses MMR-based retrieval
- 🤖 Generates answers using Google Gemini
- 💬 Interactive chat interface using Streamlit
- 📚 Answers questions based only on the uploaded document

## 🛠️ Technologies Used

- Python
- LangChain
- Google Gemini
- Chroma
- Streamlit
- RecursiveCharacterTextSplitter

## 🔄 How It Works

The uploaded document is loaded using the appropriate document loader and split into smaller chunks.

The chunks are converted into embeddings and stored in Chroma.

When the user asks a question, the question is converted into an embedding and relevant document chunks are retrieved using MMR.

The retrieved chunks are provided as context to Gemini, which generates the final answer based on the uploaded document.

## 🌐 Live Demo

[Try AskDoc AI](https://my-gen-ai-projects-urbis5fisarfj4beaynewf.streamlit.app/)

## 📁 Project Files

- `app.py` — Streamlit application and complete RAG workflow
- `dataBase.py` — document loading, splitting, embedding, and Chroma database creation
- `main.py` — RAG retrieval and question-answering logic

## 🎯 Purpose

This project was built to practice and strengthen Generative AI and RAG concepts, including document loading, chunking, embeddings, vector stores, retrieval, context-based prompting, and LLM generation.