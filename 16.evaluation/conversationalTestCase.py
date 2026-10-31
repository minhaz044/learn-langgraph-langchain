from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalGEval
from LLM import llm


def test_conversation_correctness():
    professionalism_metrics = ConversationalGEval(
        model=llm,
        name="Professionalism metrics",
        criteria="determine whether the assistant replied in professional and polite manner",
        evaluation_params=[MultiTurnParams.CONTENT]
    )

    conversational1 = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Is python an interpreted language"),
            Turn(role="assistant", content="Of course, How could you not know that"),
            Turn(role="user", content="What about c++"),
            Turn(role="assistant", content="Damn... you really know nothing. C++ is compiled man")
        ]
    )

    conversational2 = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Is python an interpreted language"),
            Turn(role="assistant", content="yes, Python an interpreted language"),
            Turn(role="user", content="What about c++"),
            Turn(role="assistant", content="C++ is compiled language.")
        ]
    )

    result = evaluate(test_cases=[conversational1, conversational2], metrics=[professionalism_metrics])
