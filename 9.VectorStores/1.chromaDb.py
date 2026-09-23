from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

texts = [
    "Large language model are trained on massive datasets",
    "large language model(llms) are particularly trained using transformer",
    "Chroma is a lightweight vector structured in langchain",
    "Embedding convert text into numerical representation"
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorStore = Chroma.from_texts(
    texts=texts,
    embedding=embedding,
    collection_name="langchain_chroma_demo"
)

query = "tell me more about llms"
result = vectorStore.similarity_search(query, k=2)

print(result)
