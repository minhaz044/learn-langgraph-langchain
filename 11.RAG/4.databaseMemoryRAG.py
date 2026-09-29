from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import sqlite3
import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

connection = sqlite3.connect("chat_memory1.db", check_same_thread=False)
cursor = connection.cursor()

cursor.execute(
    """
    create table if not exists chat_history
    (
        session_id TEXT,
        role TEXT,
        content TEXT
    )
    """
)
connection.commit()


def save_message(session_id: str, role: str, content: str):
    cursor.execute(
        "INSERT INTO chat_history values(?,?,?)",
        (session_id, role, content)
    )
    connection.commit()


def load_chat_history(session_id: str):
    cursor.execute(
        "SELECT role,content from chat_history where session_id= ? ", (session_id,)
    )
    rows = cursor.fetchall()

    history: List[BaseMessage] = []

    for role, content in rows:
        if role == 'human':
            history.append(HumanMessage(content=content))
        elif role == 'ai':
            history.append(AIMessage(content=content))
    return history


def get_all_session():
    cursor.execute(
        "SELECT DISTINCT session_id from chat_history order by rowid desc"
    )
    return [row[0] for row in cursor.fetchall()]


# Streamlit configurations
st.set_page_config(page_title="Conversational RAG", layout="wide")
st.title("Conversational Rag with memory")

st.sidebar.title("Chats")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

if st.sidebar.button("Now chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

st.sidebar.markdown("previous conversation")

for sid in get_all_session():
    if st.sidebar.button(sid[:8]):
        st.session_state.session_id = sid
        st.session_state.chat_history = load_chat_history(sid)

session_id = st.session_state.session_id


@st.cache_resource
def load_vector_storage():
    loader = PyPDFLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/audit-report.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vector_db


vector_store = load_vector_storage()

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

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
                4. Do not make up information.
                5. If the answer is explicitly available in the conversation history,
                    use it even if it is not present in the PDF context.
                6. If the information cannot be found in either the conversation history
                    or the PDF context, say "I don't know."
                7. Do not mention "Based on the provided context".
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


# load session history
if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history(session_id)

# chat window
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)

user_input = st.chat_input("Ask a question for the PDF: ")

if user_input:
    st.chat_message("user").write(user_input)
    save_message(session_id, "human", user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    response, source = conversational_rag(
        user_input,
        st.session_state.chat_history
    )

    st.chat_message("AI").write(response.content[0]['text'])
    save_message(session_id, "ai", response.content[0]['text'])
    st.session_state.chat_history.append(AIMessage(content=response.content[0]['text']))
