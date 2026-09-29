from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os

DOC_FOLDER_PATH = "/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/multidocsample/"

all_documents: List[Document] = []

for file in os.listdir(DOC_FOLDER_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOC_FOLDER_PATH, file))
        docs = loader.load()

        # Attach document level metadata

        for d in docs:
            d.metadata['source_document'] = file

        all_documents.extend(docs)

print(f"loaded {len(all_documents)} pages from multiple documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)

print(f"Total chunk created : {len(chunks)}")

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
                You are a helpful AI assistant.
                Answer the user's question using the provided context and chat history.

    
                Rules:
                1. Use the PDF context as the primary source for questions about the PDF.
                2. Use conversation history for information the user previously provided,
                    such as their name, preferences, or previous statements or any useful information.
                3. Use conversation history to resolve references such as "he", "they",
                    "it", "this", or "that".
                4. check the sources and if multiple documents are involved , clearly mentioned them    
                4. Do not make up information.
                5. If the answer is explicitly available in the conversation history,
                    use it even if it is not present in the PDF context.
                6. If the information cannot be found in either the conversation history
                    or the PDF context, say "I don't know."
                7. Do not mention "Based on the provided context".
             """
        ),
        (
            "human",
            "context : \n {context} \n \n question: \n {question}"
        )

    ]
)


# RAG function

def multi_document_rag(query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join(
        f"""
        [Document : {d.metadata.get("source_document")}]
        \n
        {d.page_content} 
        """ for d in docs
    )
    message = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(message)
    return response.content[0]['text'], docs


print("Multi document RAG ,Type 'exit' to quit")

while True:
    query = input("You :")
    if query == 'exit':
        break
    res, sources = multi_document_rag(query)

    print("AI answer : ", res)

    for i, d in enumerate(sources):
        print(f"""{i + 1}. {d.metadata.get('source_document')} """)
