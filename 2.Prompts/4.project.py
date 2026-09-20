from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

print("Blog post generator")
print("Provide ideas r topic for blog post")

topic = input("Enter your Post topic:")

chat_prompts_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a professional blog post writer. Help generate informative, "
            "engaging and well structured blog post about {topic}."),
        HumanMessagePromptTemplate.from_template("Write a details blog post about {topic}.")
    ]
)

chat_history = []

while True:
    user_input = input("Ideas or interaction or type exit : ")
    if user_input.lower() == "exit":
        break

    message = chat_prompts_template.format_messages(topic=topic)
    for prev in chat_history:
        message.append(prev)

    human_message = HumanMessagePromptTemplate.from_template(user_input).format_messages(user_input=user_input)[0]
    message.append(human_message)
    chat_history.append(human_message)

    response = chat_model.invoke(message)
    chat_history.append(AIMessage(content=response.content[0]['text']))
    print("Blog post content: \n", response.content[0]['text'])

print(chat_history)