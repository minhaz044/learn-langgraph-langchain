from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic'])

prompt2 = PromptTemplate(
    template="Generate a 3 point summary of the following {text}",
    input_variables=['text'])

parser = StrOutputParser()

chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

response = chain.invoke({"topic": "Bangladesh"})
print(response)
