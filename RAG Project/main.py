from dotenv import load_dotenv
load_dotenv()

from dataBase import build_retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI


# Get File 
file_path = input("Enter the path of your PDF/TXT/DOCX file: ")


# Create Retriever
retriever = build_retriever(file_path) #Build the searcher.

# build_retriever() will:
# 1. Detect the file type
# 2. Load the document
# 3. Split it into chunks
# 4. Create embeddings
# 5. Store the embeddings in Chroma
# 6. Return a Retriever


# Prompt Template
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


#LLM
model = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


print("\n----------------------------- RAG System Created --------------------------")
print("Ask questions about your document.")
print("Press 0 to Exit.\n")


while True:

    query = input("You : ")

    if query == "0":
        break
    # Retrieve relevant chunks
    docs = retriever.invoke(query) #Use the searcher.

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
    response = model.invoke(final_prompt)
    print("Bot :", response,"\n")
