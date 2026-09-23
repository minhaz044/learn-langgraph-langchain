from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/audit-report.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator=""
)

data = splitter.split_documents(docs)

print(data[0].page_content)
print("\n")
print(data[0].metadata)

print("=======================================================")

print(data[1].page_content)
print("\n")
print(data[1].metadata)
