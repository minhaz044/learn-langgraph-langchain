
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

llm = GeminiModel(model="gemini-3.5-flash-lite")

strict_correctness_matrics = GEval(
    model=llm,
    name='Correctness',
    criteria="check if the output is exact match. if exact math return 1 else 0",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5
)
correctness_matrics = GEval(
    model=llm,
    name='Correctness',
    criteria="check if the output is correct match."
             "Result dont need to be identical but need to be correct and same in meanning."
             " if correct return 1 else 0",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5
)


def test_correctness_identical():
    test_case1 = LLMTestCase(
        input="What is 5 divided by 2 ?",
        expected_output="2.5",
        actual_output="The result is 2.5"
    )

    assert_test(test_case1, [strict_correctness_matrics])


def test_correctness():
    test_case2 = LLMTestCase(
        input="What is 5 divided by 2 ?",
        expected_output="2.5",
        actual_output="The result is 2.5"
    )
    assert_test(test_case2, [correctness_matrics])
