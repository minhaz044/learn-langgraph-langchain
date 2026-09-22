from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


# Schema
class Review(TypedDict):
    summary: str
    sentiment: str


prompt = ("This is the Era of Ai Revolution. "
          "Yes Ai has its own pros and cons yet we need to adopt it in shortest possible time")

structured_model=chat_model.with_structured_output(Review)
response=structured_model.invoke(prompt)
print(response)