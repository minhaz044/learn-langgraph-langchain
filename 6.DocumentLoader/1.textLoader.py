from langchain_community.document_loaders import TextLoader

loader = TextLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/cricket.txt", encoding="utf-8")

docs = loader.load()

print(type(docs))
print('\n')
print(docs[0])
print('\n')
print(docs)
