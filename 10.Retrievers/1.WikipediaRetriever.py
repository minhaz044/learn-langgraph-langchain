from dotenv import load_dotenv

load_dotenv()

from langchain_community.retrievers import WikipediaRetriever

retrievers = WikipediaRetriever(top_k_results=2, lang="en")

query = "What is the Current PM of Bangladesh"

docs = retrievers.invoke(query)
for doc in docs:
    print(doc.page_content)
    print("\n==============\n")
