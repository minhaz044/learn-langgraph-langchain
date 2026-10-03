from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


@tool()
def multiply(a: int, b: int):
    """Multiply two number"""
    return a * b


@tool()
def add(a: int, b: int):
    """Add two number"""
    return a + b


tools = [add, multiply]

system_prompt = """
You are a helpful ai agents , use tools when necessary , if no tools required , answered directly  
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

print("langchain basic agent")

while True:
    user_input = input("You : ")
    if user_input == 'exit':
        break

    response = agent.invoke(
        {"messages" : [{"role": "user", "content": user_input}]}
    )

    print("\n AI:", response["messages"][-1].content[-1]['text'])
    print("\n ===================AI agent debug Mode============\n", response)
    print("\n\n")