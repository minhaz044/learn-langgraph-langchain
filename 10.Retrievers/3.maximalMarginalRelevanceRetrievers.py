from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="langchain makes it easy to work with LLMs"),
    Document(page_content="langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="OpenAI provides powerful embedding models")
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorStore = FAISS.from_documents(documents=documents, embedding=embedding)

retriever = vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.25},
)

query = "what is langchain"
result = retriever.invoke(query)

for match in result:
    print(match.page_content)
    print("\n==========\n")

print("======================Without diversity============\n")

retriever = vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 1.00},
)

result = retriever.invoke(query)

for match in result:
    print(match.page_content)
    print("\n==========\n")
