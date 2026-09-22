from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Hot Loader
pdfs = loader.load()

print(len(pdfs))
for pdf in pdfs:
    print(pdf.metadata)

# Lazy Loader
pdfs = loader.lazy_load()

for pdf in pdfs:
    print(pdf.metadata)
