from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from LLM import llm

faithfulness_metric = FaithfulnessMetric(
    model=llm,
    threshold=0.5,
    include_reason=True
)

test_case_1 = LLMTestCase(
    input="What is the capital of Bangladesh?",
    actual_output="Chittagong",
    retrieval_context=['The capital of Bangladesh is Chittagong']
)

test_case_2 = LLMTestCase(
    input="What is the capital of Bangladesh?",
    actual_output="Dhaka",
    retrieval_context=['The capital of Bangladesh is Chittagong']
)

evaluate(test_cases=[test_case_1, test_case_2], metrics=[faithfulness_metric])
