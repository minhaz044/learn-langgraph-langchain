from dotenv import load_dotenv

load_dotenv()

import os
import logging
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

LOG_DIR = "logs"

try:
    os.mkdir(LOG_DIR)
except FileExistsError:
    pass

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "execution.log"),
    level=logging.INFO,
    format="%(asctime)s- %(levelname)s - %(message)s",
    filemode="w"
)

logger = logging.getLogger(__name__)


def write_text_file(filename: str, content: str):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def planner_agent(state):
    logger.info("planner agent started..")
    user_query = state['user_query']
    logger.info(f"Users query : {user_query}")
    prompt = f"""
    you are a planner agent in a multiagent system
    
    your job is to read the user query and create a short plan for how the worker agent should answer

    user query : {user_query}
    
    Return a short actionable plan
    """
    response = llm.invoke(prompt)
    plan = response.text

    state['plan'] = plan
    write_text_file("planner_output.txt", plan)
    logger.info(f"planner output: {plan}")
    return state


def worker_agent(state):
    state['worker_call'] += 1
    feedback = state.get('review_reason', "")
    logger.info(f"Worker agent call #{state['worker_call']}")
    prompt = f"""
    You are a worker agent
    
    user query : {state['user_query']}
    
    plan : {state['plan']}
    
    previous reviewer : {feedback}
    
    write the best possible improved response. if reviewer feedback existing, explicitly fix those issues
    """
    response = llm.invoke(prompt)
    draft_response = response.text
    state['draft_response'] = draft_response

    write_text_file(f"worker_output_{state['worker_call']}.txt", draft_response)
    logger.info(f"worker output {state['worker_call']} : {draft_response}")

    return state


def reviewer_agent(state):
    state['reviewer_call'] += 1
    logger.info(f"Reviewer agent call #{state['reviewer_call']}")

    prompt = f"""
    You are a strict reviewer agent
    
    User query : {state['user_query']}
    
    Draft response : {state['draft_response']}
    
    check for : 
    - concrete example 
    - implementation details
    - tradeoffs
    - clarity
    - actionable recommendation
    
    if anything is missing revise.
    Return EXACTLY in this format:
    
    Decision: approve OR revise 
    Reason : brief reason
    """

    response = llm.invoke(prompt)
    review_res = response.text

    decision = 'approve' if "decision: approve" in review_res.lower() else "review"

    reason = next(line for line in review_res.splitlines() if line.lower().startswith("reason"))
    reason = reason.replace("Reason: ", "").strip() if reason else "No reason provided"

    state['review_decision'] = decision
    state['review_reason'] = reason
    write_text_file(f"reviewer_output_{state['reviewer_call']}.txt", review_res)
    logger.info(f"worker output {state['reviewer_call']} : {review_res}")
    return state


def review_router(state):
    logger.info("checking Review decision for farther review")

    if state['review_decision'] == 'approve' or state.get('revision_count', 0) >= 2:
        return END
    state['revision_count'] = state.get('revision_count', 0) + 1
    return 'worker_agent'


PLANNER_AGENT_NODE = "planner_agent"
WORKER_AGENT_NODE = "worker_agent"
REVIEWER_AGENT_NODE = "reviewer_agent"

workflow = StateGraph(dict)

workflow.add_node(PLANNER_AGENT_NODE, planner_agent)
workflow.add_node(WORKER_AGENT_NODE, worker_agent)
workflow.add_node(REVIEWER_AGENT_NODE, reviewer_agent)

workflow.set_entry_point(PLANNER_AGENT_NODE)
workflow.add_edge(PLANNER_AGENT_NODE, WORKER_AGENT_NODE)
workflow.add_edge(WORKER_AGENT_NODE, REVIEWER_AGENT_NODE)

workflow.add_conditional_edges(
    REVIEWER_AGENT_NODE, review_router,
    {
        WORKER_AGENT_NODE: WORKER_AGENT_NODE,
        END: END
    })

app = workflow.compile()

try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(png_data)
    print("Graph saved to workflow.png")
except Exception:
    logger.exception("Failed to render the Mermaid graph")

user_query = input("Enter your query?")
logger.info(f"User input {user_query}")

initial_state = {
    'user_query': user_query,
    "plan": "",
    "draft_response": "",
    "review_decision": "",
    "review_reason": "",
    "worker_call": 0,
    "reviewer_call": 0,
    "revision_count": 0
}

result = app.invoke(initial_state)

print(result['draft_response'])
