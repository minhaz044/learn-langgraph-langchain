from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

bangladesh_texts = [
    "Bangladesh is a vibrant South Asian country known for its rich cultural heritage, lush green landscapes, and extensive river networks.",
    "Dhaka, the capital city of Bangladesh, is a bustling metropolis famous for its historical Mughal architecture and vibrant street life.",
    "The Sundarbans, located in southwestern Bangladesh, is the world's largest mangrove forest and home to the iconic Royal Bengal Tiger.",
    "Cricket is the most popular sport in Bangladesh, with a passionate national fanbase and a competitive international team.",
    "The national economy is significantly driven by its thriving ready-made garment (RMG) industry, which exports textiles worldwide.",
    "Cox's Bazar, situated along the Bay of Bengal, boasts one of the longest unbroken natural sandy sea beaches in the world."
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorStore = FAISS.from_texts(
    texts=bangladesh_texts,
    embedding=embedding
)

query = "city of bangladesh"
result = vectorStore.similarity_search(query, k=2)

print(result)
