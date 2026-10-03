from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import JUDGE_PROMPT, GEMINI_VERSION
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Evaluation(BaseModel):
    accuracy: int = Field(description="score from 0 to 10")
    hallucination: bool = Field(description="true if answer contain hallucination except false")
    feedback: str = Field(description="Brief comments on the quality of the answer")


def evaluate_answer(question: str, answer: str):
    judge_llm = ChatGoogleGenerativeAI(model=GEMINI_VERSION)
    structured_judge_llm = judge_llm.with_structured_output(Evaluation)
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    response = structured_judge_llm.invoke(prompt)
    return response


# question = "what is the capital city of Bangladesh"
#
# answer = "It is Cumilla"
#
# result = evaluate_answer(question, answer)
#
# print(result)
