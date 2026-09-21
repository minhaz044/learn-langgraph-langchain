from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = PromptTemplate(template="Generate 3 facts about a topic {topic}", input_variables=['topic'])

parser = StrOutputParser()

chain = prompt | chat_model | parser

response = chain.invoke({'topic': 'Bangladesh'})
print(response)
