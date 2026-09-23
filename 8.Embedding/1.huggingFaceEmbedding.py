from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "My name is Minhaz Uddin"
result = embedding.aembed_query(text)
print("=======================================================")
print(result)
