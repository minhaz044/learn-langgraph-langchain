from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Large language model are trained on massive datasets"),
    Document(page_content="large language model(llms) are particularly trained using transformer"),
    Document(page_content="Chroma is a lightweight vector structured in langchain"),
    Document(page_content="Embedding convert text into numerical representation")
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorStore = Chroma.from_documents(documents=docs, embedding=embedding)

retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

query = "tell me more about llms"
result = retriever.invoke(query)

for match in result:
    print(match.page_content)
    print("\n==========\n")
