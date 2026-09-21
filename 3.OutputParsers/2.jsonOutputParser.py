from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name age and city of a fictional person , and the name and city has to be Bangladeshi. {format_instruction}",
    input_variables=[], partial_variables={"format_instruction": parser.get_format_instructions()})

template2 = PromptTemplate(template="Write a 5 line summary on the following {text}", input_variables=['text'])

chain = template | chat_model | parser

response = chain.invoke({})
print(response)
