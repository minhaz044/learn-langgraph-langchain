from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = PromptTemplate(
    template="""
    Important rules:
        - Do not say "Based on the provided text" unless the user explicitly asks you to summarize or answer only from the provided text.
        - Answer the question directly and naturally.
        - If the provided context contains outdated information, do not blindly repeat it.
        - For questions about current people, political offices, laws, prices, events, or other time-sensitive information, prefer up-to-date information when available.
        - If you only have old information in the context and cannot verify the current information, clearly say that the information may be outdated.
        - Do not invent facts that are not supported by the context or reliable knowledge.

    Answer the following question directly and naturally {question} from the following text {text}
    """,
    input_variables=["question", "text"]
)

url = "https://www.bbc.com/news/world-south-asia-12650940"

loader = WebBaseLoader(url)

info = loader.load()

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({
    "question": "Who is the current PM of Bangladesh",
    "text": info[0].page_content
})

print(response)
