# SYSTEM : AI behavior/Tone or constrain
# User : User Query or input or question
# AI : Contain the previous Response , Used to maintain the Chat

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that provides information about {subject}."),
        HumanMessagePromptTemplate.from_template("can you tell me something interesting about {subject}.")
    ]
)

prompt_text = chat_prompts.format_messages(subject="AI")
print(prompt_text)
