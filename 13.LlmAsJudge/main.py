from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import QUESTION_PROMPT, GEMINI_VERSION
from judge import evaluate_answer

llm = ChatGoogleGenerativeAI(model=GEMINI_VERSION)


def generate_answer(user_input: str):
    prompt = QUESTION_PROMPT.format(question=user_input)
    response = llm.invoke(prompt)
    return response


questions = [
    "What is the capital of Bangladesh?",
    "what is the largest city of bangladesh? ",
    "what is the current ODI cricket rank position of bangladesh?",
    "What is the current PM of bangladesh?"
]

for question in questions:
    answer = generate_answer(question)
    evaluation = evaluate_answer(question, answer.text)
    print("\nQuestion:", question)
    print("\nAnswer:", answer.text)
    print("\nEvaluation:", evaluation)
    print("\n===================================\n")
