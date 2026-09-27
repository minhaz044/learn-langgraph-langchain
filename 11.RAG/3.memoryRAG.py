from typing import List

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the Document
loader = PyPDFLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/audit-report.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorDb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="rag_collection"
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

retriever = vectorDb.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
                You are a helpful AI assistant.
                Answer the user's question using the provided context.

    
                Rules:
                1. Use the context as the primary source of information.
                2. Do not make up information that is not supported by the context.
                3. You may use the conversation history to understand references
                such as "he", "they", "it", or "that".
                4. Do not mention "Based on the provided context".
                5. If the information needed to answer the question is not in the
                    context, say "I don't know."
             """
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "context : \n {context} \n \n question: \n {input}"
        )

    ]
)


def conversational_rag(user_input: str, chat_history: List[BaseMessage]):
    docs = retriever.invoke(user_input)
    context = "\n\n".join(
        f"[Page {d.metadata.get('page', 'N/A')}] \n {d.page_content}" for d in docs
    )
    messages = prompt.invoke({
        "input": user_input,
        "context": context,
        "chat_history": chat_history
    })
    response = llm.invoke(messages)
    return response, docs


chat_history: List[BaseMessage] = []

print("Conversational RAG")

user_input = ""
while True:
    user_input = input("You:")
    if user_input == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    response, sources = conversational_rag(user_input, chat_history)
    chat_history.append(AIMessage(content=response.content[0]['text']))
    print("\n AI : ", response.content[0]['text'])


print(chat_history)