from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

chat_model1 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
chat_model2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template="Generate a short and simple note from the following topic, {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 short Question Answer from the following text, {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(template="Merge the provided notes and question answer into single document.{ {notes}, {qa}}",
                         input_variables=["notes", "qa"])

parser = StrOutputParser()

runnable_chain = RunnableParallel({
    "notes": prompt1 | chat_model1 | parser,
    "qa": prompt2 | chat_model2 | parser
})

merge_chain = prompt3 | chat_model1 | parser

final_chain = runnable_chain | merge_chain

response = final_chain.invoke({
    "topic": "Bangladesh. ",
    "text": "Germany is one of the largest Country in EU , "
            "Its GDP is the largest in the EU and 5th in the World. Its capital is Berlin,"
            "Which is also the business city. Germany is the home of some of the Major Automobile Company"
})
print(response)
