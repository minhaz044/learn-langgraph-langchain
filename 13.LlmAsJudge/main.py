from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import QUESTION_PROMPT, GEMINI_VERSION
from judge import evaluate_answer, Evaluation

llm = ChatGoogleGenerativeAI(model=GEMINI_VERSION)


def generate_answer(user_input: str):
    prompt = QUESTION_PROMPT.format(question=user_input)
    response = llm.invoke(prompt)
    return response


def ask_until_good_answer(ques: str, threshold: int = 8, max_attempt: int = 2):
    attempt = 0

    while attempt < max_attempt:
        ans = generate_answer(ques)
        print("Answer : \n\n ", ans.text)
        _eval: Evaluation = evaluate_answer(ques, ans.text)
        print("Evaluation : \n\n ", _eval)
        if _eval.accuracy >= threshold and _eval.hallucination != True:
            print("Answer is accurate , No correction needed")
            break
        else:
            print("Answer is below threshold ,regenerating..........")
            attempt += 1
    print("===========================")


questions = [
    "What is the capital of Bangladesh?",
    "what is the largest city of bangladesh? ",
    "what is the current ODI cricket rank position of bangladesh?",
    "What is the current PM of bangladesh?"
]

for question in questions:
    ask_until_good_answer(question)
