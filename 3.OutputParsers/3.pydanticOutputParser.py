from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class Person(BaseModel):
    name: str = Field(description="The Persons full name")
    age: int = Field(gt=18, lt=60, description="The persons age,must not less then 18 and grater then 60")
    city: str = Field(description="The city where the person live in")


chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name age and city of a fictional person"
             "make sure the age is grater tgen 18 and less then 60"
             "The Person must be from {place}"
             "Return the response in the following format."
             " {format_instruction}",
    input_variables=['place'], partial_variables={"format_instruction": parser.get_format_instructions()})

chain = template | chat_model | parser

response = chain.invoke({"place":"Bangladesh"})
print(response)
print(type(response))
