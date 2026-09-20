# Learn-LangGraph

My personal notes and practice code while learning **LangChain / LangGraph** with **Google Gemini**.

## Setup

```bash
# install dependencies
uv sync          # or: pip install -r requirements.txt

# create a .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# run any file
python 1.BasicLLM/1.geminillm.py
```

## What's inside

| Folder | Topic |
| --- | --- |
| `1.BasicLLM/` | Calling a Gemini model with LangChain |
| `2.Prompts/` | Prompt templates and a small chat app |
| `2.StructuredResponse/` | Getting structured output from the model |

### 1.BasicLLM
- `1.geminillm.py` — load `.env`, create a `ChatGoogleGenerativeAI` model and invoke it.

### 2.Prompts
- `1.static_prompt.py` — `PromptTemplate` with no input variables.
- `2.chat_prompt_template.py` — `PromptTemplate` with `{topic}` and `{style}` variables.
- `3.dynamic_prompt.py` — `ChatPromptTemplate` using system + human messages.
- `4.project.py` — a small interactive blog post generator that keeps chat history.

### 2.StructuredResponse
Same task (reviewing a text) done four ways:
- `1.typedDict.py` — `TypedDict` schema with `with_structured_output`.
- `2.annnotatedTypedDict.py` — `TypedDict` + `Annotated` field descriptions.
- `3.pydanticResponse.py` — Pydantic `BaseModel` with `Field` descriptions and `Literal`.
- `4.jsonSchema.py` — raw JSON Schema passed to the model.

## Key concepts learned

- **Prompt templates** — static vs dynamic, `PromptTemplate` vs `ChatPromptTemplate`.
- **Message roles** — System (behavior/tone), Human (user input), AI (past responses).
- **Structured output** — `model.with_structured_output(schema)` forces the LLM to return a fixed shape (TypedDict, Pydantic, or JSON Schema).
- **Strict mode** — passing `strict=True` enforces the schema more tightly.

## Notes

- Model used: `gemini-3.6-flash`.
- Python `>=3.13`, managed with `uv`.
- `.env` is git-ignored — never commit API keys.
