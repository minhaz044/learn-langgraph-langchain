from langchain_core.prompts import PromptTemplate

QUESTION_PROMPT = PromptTemplate(
    template="Answer the following Question as accurately as possible : \n {question}",
    input_variables=['question']
)

GEMINI_VERSION = "gemini-3.5-flash-lite"

JUDGE_PROMPT = PromptTemplate(
    template="""
    You are an expert AI judge , evaluate the following answer to the question \n\n 
    Question : {question} \n
    
    Answer : {answer} \n 
    
    Assess the answer based on the following criteria : \n
    1. accuracy (0-10) \n
    2. hallucination (true/false) \n
    3. feedback (brief comments) \n
    return your evaluation as json only
    """, input_variables=['question', 'answer']
)