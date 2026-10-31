import os

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval import evaluate
from langchain_community.document_loaders import TextLoader
from meetingSummarizer import MeetingSummarizer
from deepeval.metrics import GEval

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
        actual_output=action_item
    )
    summary_test_cases.append(summary_test_case)
    summary_test_cases.append(action_item_test_case)

summary_matrics = GEval(
    name="Summary Concision",
    criteria="Assess whether the summary is concise and focused on the point so of the meetings?"
             " It should avoid repetition, irrelevant details, and unnecessary elaboration",
    threshold=0.5,
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
)

action_item_matrics = GEval(
    name="Action Item Accuracy",
    criteria="Are the action items accurate , complete , and clearly reflect the key tasks or follow-up mentions in Meeting?",
    threshold=0.5,
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
)

result = evaluate(test_cases=summary_test_cases, matrics=[summary_matrics])
print(result)
result = evaluate(test_cases=action_item_test_cases, matrics=[action_item_matrics])
print(result)

print("Application exiting........")
