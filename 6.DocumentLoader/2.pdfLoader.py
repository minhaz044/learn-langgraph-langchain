from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/audit-report.pdf")

pdf = loader.load()

print(type(pdf))
print('\n')
print(pdf[0].metadata)
print('\n')
print(pdf)
