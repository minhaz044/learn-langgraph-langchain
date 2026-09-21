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
| `3.OutputParsers/` | Parsing model output with LCEL chains |
| `4.Chains/` | Composing prompts, models and parsers into chains |

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

### 3.OutputParsers
Chains built with the `|` pipe operator (LCEL), each ending in an output parser:
- `1.strOutputParser.py` — `StrOutputParser` in a two-step chain (long report → 5-line summary).
- `2.jsonOutputParser.py` — `JsonOutputParser`, injecting `get_format_instructions()` as a partial variable.
- `3.pydanticOutputParser.py` — `PydanticOutputParser` with a `Person` model (`Field` constraints, e.g. `gt`/`lt` on age).

### 4.Chains
- `1.simpleChain.py` — basic `prompt | model | parser` chain generating facts about a topic.
- `2.sequentialChain.py` — two prompts chained together (detailed report → 3-point summary).
- `3.conditionalChain.py` — `RunnableBranch` routing feedback by sentiment: a classifier chain tags it positive/negative, then the matching reply prompt runs.
- `4.parallelChain.py` — `RunnableParallel` runs a notes chain and a Q&A chain at the same time, then a merge prompt combines both outputs.

## Key concepts learned

- **Prompt templates** — static vs dynamic, `PromptTemplate` vs `ChatPromptTemplate`.
- **Message roles** — System (behavior/tone), Human (user input), AI (past responses).
- **Structured output** — `model.with_structured_output(schema)` forces the LLM to return a fixed shape (TypedDict, Pydantic, or JSON Schema).
- **Strict mode** — passing `strict=True` enforces the schema more tightly.
- **Output parsers** — `StrOutputParser`, `JsonOutputParser`, and `PydanticOutputParser` turn raw messages into plain strings, dicts, or validated models.
- **LCEL chains** — the `|` operator composes prompt → model → parser into a pipeline (and chains can be sequenced, e.g. report then summary).
- **Simple vs sequential chains** — a simple chain is one pass through prompt/model/parser; a sequential chain feeds one step's output into the next step's prompt.
- **Conditional chains** — `RunnableBranch` picks a branch based on a condition function, with a fallback runnable for the no-match case.
- **RunnablePassthrough.assign** — adds a new key (e.g. `classifier`) to the input dict while passing the rest through unchanged.
- **Parallel chains** — `RunnableParallel` fans the same input out to multiple chains at once and returns their results as a dict of named keys.

## Notes

- Model used: `gemini-3.6-flash`.
- Python `>=3.13`, managed with `uv`.
- `.env` is git-ignored — never commit API keys.
