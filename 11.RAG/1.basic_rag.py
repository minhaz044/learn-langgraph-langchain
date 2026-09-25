from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load the Document
loader = PyPDFLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/audit-report.pdf")

documents = loader.load()

# Split the Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

# Create embedding

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# vector stores

vectorDb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="rag_collection"
)

# Retriever
retriever = vectorDb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "labda_mult": 0.5}
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = PromptTemplate(
    template="""
    Your are an AI assistant , use the following context to answer the question.
    if the answer is not present in the context , say you dont know
    Do not mention "Based on the provided context"
    
    Context: {context}
    
    question: {question}
    """,
    input_variables=["context", "question"]
)

parser = StrOutputParser()


def format_docs(similar_docs):
    return "\n\n".join(doc.page_content for doc in similar_docs)


rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        } | prompt | llm | parser
)
query = "What is this document mainly about"

while query != "exit":
    response = rag_chain.invoke(query)
    print(response)
    query = input("\nQuestion or type 'exit': ")
    print("Loading...............\n")
