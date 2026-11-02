import os

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval import evaluate
from langchain_community.document_loaders import TextLoader
from meetingSummarizer import MeetingSummarizer
from deepeval.metrics import GEval
from llm import llm

CONVERSATION_DOC_FOLDER_PATH = "/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/transcripts/"

transcripts = []
goldens: list[Golden] = []
for file in os.listdir(CONVERSATION_DOC_FOLDER_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(CONVERSATION_DOC_FOLDER_PATH, file))
        docs = loader.load()
        page_content = docs[0].page_content
        transcripts.append(page_content)
        goldens.append(Golden(input=page_content))

summarizer = MeetingSummarizer()

summary_test_cases: list[LLMTestCase] = []
action_item_test_cases: list[LLMTestCase] = []

for golden in goldens:
    summary, action_item = summarizer.summarize(golden.input)
    summary_test_case = LLMTestCase(
        input=golden.input,
        actual_output=summary
    )
    action_item_test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(action_item)
    )
    summary_test_cases.append(summary_test_case)
    summary_test_cases.append(action_item_test_case)

summary_matrics = GEval(
    model=llm,
    name="Summary Concision",
    criteria="Assess whether the summary is concise and focused on the point so of the meetings?"
             " It should avoid repetition, irrelevant details, and unnecessary elaboration",
    threshold=0.5,
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
)

action_item_matrics = GEval(
    model=llm,
    name="Action Item Accuracy",
    criteria="Are the action items accurate , complete , and clearly reflect the key tasks or follow-up mentions in Meeting?",
    threshold=0.9,
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
)

evaluate(test_cases=summary_test_cases, metrics=[summary_matrics])
evaluate(test_cases=action_item_test_cases, metrics=[action_item_matrics])

print("Application exiting........")
