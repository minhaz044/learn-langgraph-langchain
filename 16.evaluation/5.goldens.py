from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.dataset import Golden, EvaluationDataset
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from LLM import llm as evalutionLLM

load_dotenv()
chat_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

correctness_metric = GEval(
    name="Correctness metrics",
    model=evalutionLLM,
    threshold=0.5,
    criteria="Determine the 'actual output' is correct based on the 'expected output'"
             "It has to be correct in its essence. Ignore format issue",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]
)

goldens = [
    Golden(input="What is the capital of Bangladesh?", expected_output="Dhaka"),
    Golden(input="What is 12 * 6 ?", expected_output="72"),
    Golden(input="What is 12 * 3 ?", expected_output="Thirty six"),

]
evaluation_dataset = EvaluationDataset(goldens=goldens)

for golden in goldens:
    evaluation_dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=chat_llm.invoke(golden.input).text
        )
    )

evaluate(test_cases=evaluation_dataset.test_cases, metrics=[correctness_metric])
