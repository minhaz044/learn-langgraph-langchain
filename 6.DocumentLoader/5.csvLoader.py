from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/countries of the world.csv")

csvRows = loader.load()

for row in csvRows:
    print(row.page_content)
