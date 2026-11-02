from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from LLM import llm

answer_relevancy_metric = AnswerRelevancyMetric(
    model=llm,
    threshold=0.5,
    include_reason=True
)

test_case_1 = LLMTestCase(
    input="What is the capital of Bangladesh?",
    actual_output="Dhaka"
)

test_case_2 = LLMTestCase(
    input="What is the capital of Bangladesh?",
    actual_output="This is a good question. The capital of Bangladesh is Dhaka."
                  " It is one of the beautiful cities in the Asia You should check it out"
)

evaluate(test_cases=[test_case_1, test_case_2], metrics=[answer_relevancy_metric])

