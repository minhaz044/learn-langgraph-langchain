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
query = "What is this document mainly about"

retrieved_docs = vectorDb.similarity_search(
    query=query, k=3
)
context_text = ""
sources = []
for i, doc in enumerate(retrieved_docs):
    page = doc.metadata.get("page", "N/A")
    source = doc.metadata.get("source", "N/A")

    context_text += f"\n chunk {i + 1} : \n {doc.page_content} \n"
    sources.append({
        "chunk": i + 1,
        "page": page,
        "source": source,
        "content": doc.page_content
    })

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = PromptTemplate(
    template="""
    Your are an AI assistant , use the following context to answer the question.
    
    Rules: 
    1. Ans form the context
    2. Mention which page and chunk you get the information
    3. Do not mention "Based on the provided context"
    4. if the answer is not present in the context , say you dont know
    
    
    Context: {context}
    
    question: {question}
    """,
    input_variables=["context", "question"]
)

parser = StrOutputParser()

rag_chain = prompt | llm | parser

response = rag_chain.invoke({
    "context": context_text,
    "question": query
})

print(response)

for src in sources:
    print("-----------------------------------------------------------")
    print(f" chunk : {src['chunk']} | Page : {src['page']}")
    print(src['content'])
    print(f"chunk : {src['chunk']} | Page : {src['page']}")
