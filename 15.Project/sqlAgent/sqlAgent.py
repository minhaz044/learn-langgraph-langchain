from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///bank.db")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type='zero-shot-react-description',
    verbose=True
)

response = agent_executor.invoke({
    "input": "can you find out how many personal loan has been assigned"
})

print(response["output"])
print("Application exiting...........")
