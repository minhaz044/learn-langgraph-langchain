from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=['topic'])

template2 = PromptTemplate(template="Write a 5 line summary on the following {text}", input_variables=['text'])

parser = StrOutputParser()

chain = template1 | chat_model | parser | template2 | chat_model | parser

response = chain.invoke({'topic':'Black Hole'})
print(response)
