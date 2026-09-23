#load
from langchain_community.document_loaders import TextLoader,PyPDFLoader,Docx2txtLoader

filename= input("Upload your file ")
import os
name,extension=os.path.splitext(filename)

if extension==".pdf":
    data=PyPDFLoader(filename)
elif extension==".txt":
    data=TextLoader(filename)
elif extension==".docx":
    data=Docx2txtLoader(filename)
docs=data.load()

#split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.split_documents(docs)

#embedding
'''from langchain_huggingface import HuggingFaceEmbeddings
embedding_model=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)'''

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


#Vectorstore
from langchain_chroma import Chroma
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
