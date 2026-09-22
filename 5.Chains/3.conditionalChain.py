from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback ,must be positive or the negative")


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify the sentiment of the following text into positive or negative {feedback},"
             "{format_instruction}",
    input_variables=['feedback'], partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | chat_model | parser2

prompt2 = PromptTemplate(
    template="Write exactly ONE appropriate response to this positive feedback  {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write exactly ONE appropriate response to this negative feedback  {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x['classifier'].sentiment == "positive", prompt2 | chat_model | parser),
    (lambda x: x['classifier'].sentiment == "negative", prompt3 | chat_model | parser),
    RunnableLambda(lambda x: "No valida sentiment found")
)

chain = (
        RunnablePassthrough.assign(classifier=classifier_chain)
        | branch_chain
)

response = chain.invoke({"feedback": "Bangladesh is a beautiful country. "
                                     "It has a lots of place to visit and Its food is also amazing"})
print(response)
