from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


def build_retriever(filename: str):

    # Load Document
    name, extension = os.path.splitext(filename)

    extension = extension.lower()

    if extension==".pdf":
        data=PyPDFLoader(filename)
    elif extension==".txt":
        data=TextLoader(filename)
    elif extension==".docx":
        data=Docx2txtLoader(filename)
    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF, TXT, or DOCX file."
        )

    documents = data.load()

    #Split into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    #Create Vector Store 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma-db"
    )

    # Create Retriever 
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )

    return retriever
