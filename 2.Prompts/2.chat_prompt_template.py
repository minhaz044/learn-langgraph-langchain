from langchain_core.prompts import PromptTemplate
from langgraph.channels import topic

dynamic_prompts = PromptTemplate(
    input_variables=["topic", "style"],
    template="Write a short paragraph about {topic} in a {style} style"
)

prompt_text = dynamic_prompts.format(topic="AI", style="humarious")
print(prompt_text)

prompt_text = dynamic_prompts.format(topic="Block chain", style="professional")
print(prompt_text)
