from dotenv import load_dotenv
load_dotenv()

#embedding model for embedding the query
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Embedding model used to convert the user's query into a vector during retrieval
from langchain_chroma import Chroma
vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":15,
        "lambda_mult":0.5
    }
)

from langchain_core.prompts import ChatPromptTemplate
template=ChatPromptTemplate([
    ("system",
     """
You are an AI tutor, who explains in detail and in simple terms.
 Use ONLY the provided context to answer the questions.
 If the answer is not present in the context, say:
 "I Could Not Find The Answer In The Document."
"""),
("user","""
Context : 
"{context}" 

Question :
"{question}"
""")
])

from langchain_google_genai import GoogleGenerativeAI
model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

print("-----------------------------RAG System Created--------------------------")
print("Press 0 to Exit")

while True:
    query=input("You : ")
    if query=="0":
        break
    docs=retriever.invoke(query)

    context=""
    for i in docs:
        context=context+ i.page_content+"\n\n"

    final_prompt=template.invoke({
        "context":context,
        "question" :query
        })
    #We could have used template.format_messages() as well, 
    # but invoke() follows the common Runnable interface in LangChain.
    
    response=model.invoke(final_prompt)
    print("Bot : ", response)
