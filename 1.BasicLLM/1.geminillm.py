from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
response = llm.invoke("Hello, langchain! Explain yourself in a single sentence")
print(response.content[0]['text'])
