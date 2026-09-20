from langchain_core.prompts import PromptTemplate

static_prompts = PromptTemplate(
    input_variables=[],
    template="Write a short fun fact about AI"
)

prompt_text = static_prompts.format()

print(prompt_text)
