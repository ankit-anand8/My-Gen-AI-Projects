from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

print("Press 1 for Angry mode")
print("Press 2 for Funny mode")
print("Press 3 for Poetic mode")
print("Press 4 for Teacher mode")

ch=input("Enter your choice : ")
if ch=="1":
    mode="You are a angry AI. Always angrry and frustated, you are also impateint, and roast the user for his questions asked"
elif ch=="2":
    mode="You are a funny AI, give replies with humour and funny way, you always try to make laugh the user and also crack jokes"
elif ch=="3":
    mode="You are a poet AI, you always give answers like a poem"
elif ch=="4":
    mode="You are a tutor AI, you explain eveything in detail with examples and in simple terms"

else: 
    mode="You are a helpful AI assistant"
    print("Invalid choice, defaulting to helpful mode")
#If the user presses any other numbers or letters, then the default mode of the chatbot will be "helpful mode"

from langchain_google_genai import GoogleGenerativeAI
model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

messages=[SystemMessage(content=mode)]
print("--------------WELCOME TYPE 0 to exit the application-----------------")

while True:
    prompt=input("You : ")
    if prompt=="0":
        break
    messages.append(HumanMessage(content=prompt))
    response=model.invoke(messages)
    messages.append(AIMessage(content=response))
    print("Bot : ",response)

print(messages)