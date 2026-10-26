# LangChain & LangGraph — Class / Tool Reference

A detailed catalogue of the **classes, tools and features** used in this repo. Every entry:

- **What it is** · **Import** · **Key params / methods** · **Used in** · **Note**.

Companion to `HANDBOOK.md` (which explains the flow); this file is the lookup table.

---

# PART A — LangChain core

## A1. Models

### `ChatGoogleGenerativeAI`
- **What it is:** the chat-model Runnable — the "brain". Takes a list of messages, returns an `AIMessage`.
- **Import:** `from langchain_google_genai import ChatGoogleGenerativeAI`
- **Key params:** `model=` (e.g. `"gemini-3.6-flash"`, `"gemini-3.5-flash-lite"`).
- **Methods:** `.invoke(input)` → one call · `.stream(input)` → tokens · `.batch(list)` → many inputs · `.ainvoke` → async · `.bind_tools(tools)` → returns a model that may emit `tool_calls` · `.with_structured_output(schema, strict=False)` → returns a model bound to a schema.
- **Used in:** every module (`1.BasicLLM`, all RAG, agents, LangGraph nodes).
- **Note:** Gemini text lives at `response.content[0]['text']` or `response.text`.

### Messages (`BaseMessage` subclasses)
- **What they are:** the typed turns you send/receive.
- **Import:** `from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage`
- **Roles:** `SystemMessage` = behavior/tone/constraints · `HumanMessage` = user turn · `AIMessage` = model reply (carries `.tool_calls`) · `ToolMessage` = a tool's result (carries `tool_call_id`).
- **Used in:** `2.Prompts/4.project.py`, `11.RAG/3.memoryRAG.py`, all LangGraph notebooks.

---

## A2. Prompts

### `PromptTemplate`
- **What it is:** a **string** template with `{variables}`.
- **Import:** `from langchain_core.prompts import PromptTemplate`
- **Key params:** `template=`, `input_variables=[...]`, `partial_variables={...}` (pre-filled at build time).
- **Methods:** `.format(**kwargs)` → final string.
- **Used in:** `2.Prompts/1.static_prompt.py`, `2.Prompts/2.chat_prompt_template.py`, `13.LlmAsJudge/prompt.py`.

### `ChatPromptTemplate`
- **What it is:** a **list of role messages** (the chat analogue of `PromptTemplate`).
- **Import:** `from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate`
- **Methods:** `.from_messages([...])` (build), `.format_messages(**kwargs)` → `list[BaseMessage]`.
- **Used in:** `2.Prompts/3.dynamic_prompt.py`, `2.Prompts/4.project.py`, `11.RAG/3.memoryRAG.py`, `11.RAG/4.databaseMemoryRAG.py`, `11.RAG/5.multiDocumentRAG.py`.

### `MessagesPlaceholder`
- **What it is:** a slot in a chat prompt where a **list of prior messages** gets injected.
- **Import:** `from langchain_core.prompts import MessagesPlaceholder`
- **Usage:** `MessagesPlaceholder(variable_name="chat_history")` inside `.from_messages([...])`.
- **Used in:** `11.RAG/3.memoryRAG.py`, `11.RAG/4.databaseMemoryRAG.py`.
- **Note:** this is *the* mechanism for conversation memory in a prompt.

### `partial_variables` (feature, not a class)
- **What it is:** values bound into a template at **build time** (so they aren't needed on every `.invoke`).
- **Used in:** `4.OutputParsers/2.jsonOutputParser.py`, `4.OutputParsers/3.pydanticOutputParser.py`, `5.Chains/3.conditionalChain.py` (inject `parser.get_format_instructions()`).

---

## A3. Structured output

### `with_structured_output(schema, strict=False)` (method)
- **What it is:** forces the model to answer in a fixed shape using the provider's native tool-calling.
- **Import:** method on any chat model.
- **Schemas accepted:** `TypedDict` · `Annotated` `TypedDict` · Pydantic `BaseModel` · raw JSON Schema dict.
- **Key param:** `strict=True` enforces the schema more tightly.
- **Returns:** a dict (TypedDict/JSON) or a validated object (Pydantic).
- **Used in:** `3.StructuredResponse/*`, `13.LlmAsJudge/judge.py`.

### Pydantic `BaseModel` + `Field` / `Literal` / `Optional`
- **What it is:** schema with validation + constraints.
- **Import:** `from pydantic import BaseModel, Field` · `from typing import Literal, Optional`
- **Constraints seen:** `Literal["Pos","Neg"]` (enum), `Field(gt=18, lt=60)` (range), `Optional[...]`, `default=None`.
- **Used in:** `3.StructuredResponse/3.pydanticResponse.py`, `4.OutputParsers/3.pydanticOutputParser.py`, `5.Chains/3.conditionalChain.py`, `13.LlmAsJudge/judge.py`.

---

## A4. Output parsers

| Class | Import | Input → Output | Used in |
| --- | --- | --- | --- |
| `StrOutputParser` | `langchain_core.output_parsers` | `AIMessage` → `str` | `4.*`, `5.*`, RAG |
| `JsonOutputParser` | `langchain_core.output_parsers` | text → `dict` | `4.OutputParsers/2.jsonOutputParser.py` |
| `PydanticOutputParser` | `langchain_core.output_parsers` | text → validated object | `4.OutputParsers/3.pydanticOutputParser.py`, `5.Chains/3.conditionalChain.py` |

- **Key method:** `parser.get_format_instructions()` → a string that tells the model the exact format (injected via `partial_variables`).
- **Note:** `PydanticOutputParser(pydantic_object=Person)` validates `gt`/`lt` on parse.

---

## A5. Runnables & LCEL (the `|` operator)

### `Runnable` / LCEL
- **What it is:** any composable unit; `a | b | c` chains them into one `Runnable`.
- **Import:** base classes in `langchain_core.runnables`.
- **Call:** `.invoke(x)` · `.stream(x)` · `.batch([x])` · async variants.

### `RunnableParallel`
- **What it is:** fan the **same input** to several chains; returns a dict of named results.
- **Usage:** `RunnableParallel({"notes": chainA, "qa": chainB})`.
- **Used in:** `5.Chains/4.parallelChain.py`.

### `RunnableBranch`
- **What it is:** runtime if/elif/else for runnables.
- **Usage:** `RunnableBranch((cond_fn, runnable), ..., fallback_runnable)`.
- **Used in:** `5.Chains/3.conditionalChain.py`.
- **Note:** always give a fallback.

### `RunnablePassthrough.assign(key=chain)`
- **What it is:** run a chain and **add its result under a key** while passing the original input through unchanged.
- **Used in:** `5.Chains/3.conditionalChain.py` (adds `classifier`).

### `RunnableLambda`
- **What it is:** wrap a plain Python function as a runnable.
- **Used in:** `5.Chains/3.conditionalChain.py` (fallback message), RAG (query pass-through).

---

## A6. `Document` (the data unit)

- **What it is:** the universal container for loaded data.
- **Import:** `from langchain_core.documents import Document`
- **Shape:** `page_content: str` + `metadata: dict` (`source`, `page`, custom keys).
- **Used in:** all loaders/splitters/retrievers; multi-doc RAG adds `metadata['source_document']`.

---

# PART B — Data pipeline (load → split → embed → store → retrieve)

## B1. Document loaders — `langchain_community.document_loaders`

| Loader | Turns into Documents | Params used | Used in |
| --- | --- | --- | --- |
| `TextLoader` | one text file | `encoding="utf-8"` | `6.DocumentLoader/1.textLoader.py` |
| `PyPDFLoader` | **one per page** | path | `2.pdfLoader.py`, RAG, `9.ragAgent` |
| `DirectoryLoader` | all files in a folder | `path`, `glob="*.pdf"`, `loader_cls=PyPDFLoader` | `3.directoryLoader.py` |
| `WebBaseLoader` | a URL's text | `url` | `4.webBaseLoader.py` |
| `CSVLoader` | **one per row** | `file_path` | `5.csvLoader.py` |

- **Methods:** `.load()` → `list[Document]` (eager) · `.lazy_load()` → generator (lazy).

## B2. Text splitters — `langchain_text_splitters`

| Splitter | Splits on | Used in |
| --- | --- | --- |
| `CharacterTextSplitter` | one `separator` | `7.TextSplitter/1.characterTextSplitter.py` |
| `RecursiveCharacterTextSplitter` | `\n\n → \n → sentence → word` until size fits | `2.recursiveTextSplitter.py`, all RAG |
| `PythonCodeTextSplitter` | code syntax (class/def/statement) | `3.pythonCodeSplitter.py` |

- **Key params:** `chunk_size` (target), `chunk_overlap` (carry-over so meaning isn't cut).
- **Methods:** `.split_documents(docs)` (keeps metadata) · `.split_text(str)` → `list[str]`.

## B3. Embeddings

| Class | Import | Params | Used in |
| --- | --- | --- | --- |
| `HuggingFaceEmbeddings` | `langchain_huggingface` | `model="sentence-transformers/all-MiniLM-L6-v2"` | `8.Embedding`, all RAG |
| `GoogleGenerativeAIEmbeddings` | `langchain_google_genai` | `model="gemini-embedding-2"` | `14.LangGraph/9.ragAgent.ipynb` |

- **Methods:** `.embed_query(text)` / `.aembed_query` (async) · `.embed_documents([...])`.

## B4. Vector stores

| Class | Import | Constructors | Search | Used in |
| --- | --- | --- | --- | --- |
| `Chroma` | `langchain_chroma` | `from_texts(...)` / `from_documents(..., collection_name=...)` | `.similarity_search(q, k)` · `.as_retriever(...)` | `9.VectorStores/1.chromaDb.py`, all RAG |
| `FAISS` | `langchain_community.vectorstores` | `from_texts(...)` / `from_documents(...)` | `.similarity_search(q, k)` | `9.VectorStores/2.faissDb.py`, `10.Retrievers/3.*` |

- **`Chroma`** = persistent, named collections; **`FAISS`** = fast, in-memory.

## B5. Retrievers

| Retriever | Import | Key params | Used in |
| --- | --- | --- | --- |
| `WikipediaRetriever` | `langchain_community.retrievers` | `top_k_results=2`, `lang="en"` | `10.Retrievers/1.*` |
| store-as-retriever | `.as_retriever(search_kwargs={"k": n})` | `search_type="mmr"`, `lambda_mult` | `10.Retrievers/2.*`, `3.*`, RAG |

- **Method:** `.invoke(query)` → `list[Document]`.
- **MMR feature:** `search_type="mmr"` + `lambda_mult` (`~0` diverse, `~1` similar).

---

# PART C — Tools, agents & SQL

### `@tool` decorator
- **What it is:** exposes a Python function to a model.
- **Import:** `from langchain_core.tools import tool`
- **Rule:** the function's **type hints = arg schema**, its **docstring = the description** the model reads.
- **Used in:** `12.Agents/1.toolCallingAgents.py`, `14.LangGraph/7.toolsCalls.ipynb`, `8.emailDrafter.ipynb`, `9.ragAgent.ipynb`, `15.Project/claw.ipynb`.

### `model.bind_tools(tools)` (method)
- **What it is:** returns a model that **may emit `tool_calls`** instead of answering directly.
- **Used in:** all agent notebooks + `15.Project/claw.ipynb`.

### `create_agent` (LangChain agent)
- **What it is:** a ready-made agent loop: model → (tool calls) → observe → repeat → answer.
- **Import:** `from langchain.agents import create_agent`  (older: `create_tool_calling_agent` + `AgentExecutor`)
- **Params:** `model`, `tools`, `system_prompt`.
- **Returns:** runnable; `invoke({"messages": [...]})` gives the **full message trace**.
- **Used in:** `12.Agents/1.toolCallingAgents.py`.

### `SQLDatabase`
- **What it is:** wraps a SQL database and exposes its **tables, schema and query runner** as tools.
- **Import:** `from langchain_community.utilities import SQLDatabase`
- **Usage:** `SQLDatabase.from_uri("sqlite:///bank.db")`.
- **Used in:** `15.Project/sqlAgent/sqlAgent.py`.

### `create_sql_agent`
- **What it is:** an agent specialised for NL → SQL, built on `SQLDatabase`.
- **Import:** `from langchain_community.agent_toolkits import create_sql_agent`
- **Params:** `llm`, `db`, `agent_type='zero-shot-react-description'` (ReAct), `verbose=True` (prints thought/action/observation).
- **Used in:** `15.Project/sqlAgent/sqlAgent.py`.

---

# PART D — LangGraph (building block by building block)

## D1. State & graph

### `StateGraph`
- **What it is:** the graph object parameterised by your **state schema**.
- **Import:** `from langgraph.graph import StateGraph`
- **Usage:** `StateGraph(State)` where `State` is a `TypedDict` (or `dict`).
- **Methods:** `.add_node`, `.add_edge`, `.add_conditional_edges`, `.set_entry_point`, `.compile()`.

### State = `TypedDict` (shared memory)
- **What it is:** the dict every node reads from and returns.
- **Import:** `from typing import TypedDict`
- **Used in:** every LangGraph notebook (`MsgState`, `AgentState`, …).

### `START` / `END`
- **What they are:** the virtual entry/exit markers.
- **Import:** `from langgraph.graph import START, END`
- **Usage:** `add_edge(START, "first")`, `add_edge("last", END)`.

### `add_node(name, fn)`
- **What it is:** registers a step. `fn(state) -> state` (or a dict of updates).
- **Used in:** all notebooks.

### `add_edge(a, b)`
- **What it is:** fixed transition `a → b`.

### `add_conditional_edges(source, router, path_map)`
- **What it is:** the **branching** feature. `router(state)` returns a **key**; `path_map` maps key → node name.
- **Used in:** `3.conditionalEdges.ipynb`, `4.complexConditionalEdges.ipynb`, `5.loopNumberGuess.ipynb`, `7.toolsCalls.ipynb`, `9.ragAgent.ipynb`, `15.Project/multiAgent/multiAgent.py`.
- **Note:** the "router" is often a **pass-through node** `lambda s: s` so the branch has a stable source.

### Loops (feature)
- **What it is:** a conditional edge whose target is an **earlier node** (even its own source).
- **Guard:** a counter (`MAX_COUNTER`, `revision_count`) so it terminates.
- **Used in:** `5.loopNumberGuess.ipynb`, `7/8/9` agent loops, `multiAgent`.

### `compile()` / `invoke()` / `stream()`
- **What it is:** turn the graph into a runnable, then run it.
- **Used in:** all notebooks.
- **`stream(input, stream_mode="values")`** = yield the state after each step (see each message/tool call live). Used in `7.toolsCalls.ipynb`.
- **Recursion cap:** `config={"recursion_limit": 50}` bounds long loops (`15.Project/claw.ipynb`).
- **Graph export:** `graph.get_graph().draw_mermaid_png()` (`15.Project/multiAgent` → `workflow.png`).

## D2. Messages state & reducers

### `add_messages` reducer
- **What it is:** tells the graph **how to merge** new messages — append, not overwrite.
- **Import:** `from langgraph.graph.message import add_messages`
- **Usage:** `messages: Annotated[Sequence[BaseMessage], add_messages]`
- **Used in:** `7.toolsCalls.ipynb`, `8.emailDrafter.ipynb`, `9.ragAgent.ipynb`, `15.Project/claw.ipynb`.
- **Note:** `Annotated[Type, reducer]` is the general reducer mechanism.

## D3. Tool execution inside a graph

### `ToolNode`
- **What it is:** a ready-made node that runs every `tool_call` in the last `AIMessage` and returns `ToolMessage`s.
- **Import:** `from langgraph.prebuilt import ToolNode`
- **Usage:** `ToolNode(tools=tools)`.
- **Used in:** `7.toolsCalls.ipynb`, `8.emailDrafter.ipynb`, `15.Project/claw.ipynb`.

### The agent loop (pattern)
- **What it is:** `agent_node → conditional(should_continue) → ToolNode → agent_node`, ending when the last message has **no** `tool_calls`.
- **Router:** check `last_message.tool_calls`.
- **Used in:** `7`, `8`, `9`, `15.Project/claw.ipynb`.
- **Note:** this hand-built loop is exactly what `create_agent` gives you for free.

### Agentic RAG (pattern)
- **What it is:** wrap `retriever.invoke(q)` in a `@tool` so the model decides **when** to retrieve (and may search more than once).
- **Used in:** `14.LangGraph/9.ragAgent.ipynb`.

---

# PART E — Feature → class → where it lives

| Feature you want to do | Class / tool | File |
| --- | --- | --- |
| Call an LLM | `ChatGoogleGenerativeAI.invoke` | `1.BasicLLM` |
| Reusable string prompt | `PromptTemplate` | `2.Prompts/1,2` |
| Role-based chat prompt | `ChatPromptTemplate` | `2.Prompts/3,4` |
| Inject conversation history | `MessagesPlaceholder` | `11.RAG/3,4` |
| Fixed output shape | `with_structured_output` | `3.StructuredResponse` |
| Parse model output in a chain | `*OutputParser` | `4.OutputParsers` |
| Compose a pipeline | LCEL `\|` | `5.Chains` |
| Branch by a condition | `RunnableBranch` | `5.Chains/3` |
| Run chains in parallel | `RunnableParallel` | `5.Chains/4` |
| Load any data | `*Loader` | `6.DocumentLoader` |
| Cut text into chunks | `*TextSplitter` | `7.TextSplitter` |
| Text → vectors | `HuggingFaceEmbeddings` | `8.Embedding` |
| Store vectors | `Chroma` / `FAISS` | `9.VectorStores` |
| Fetch relevant docs | `*Retriever` / `as_retriever` | `10.Retrievers` |
| Answer from your docs | RAG chain | `11.RAG` |
| Let the LLM use tools | `@tool` + `create_agent` | `12.Agents` |
| Grade an answer | judge + `with_structured_output` | `13.LlmAsJudge` |
| Build a state machine | `StateGraph` | `14.LangGraph` |
| Branch inside a graph | `add_conditional_edges` | `14.LangGraph/3,4` |
| Loop inside a graph | conditional edge → source node | `14.LangGraph/5` |
| Merge messages in state | `add_messages` reducer | `14.LangGraph/7` |
| Run tools in a graph | `ToolNode` + `bind_tools` | `14.LangGraph/7,8` |
| Query a SQL DB with NL | `SQLDatabase` + `create_sql_agent` | `15.Project/sqlAgent` |
| Drive a real browser | `@tool` over Playwright + graph loop | `15.Project/claw.ipynb` |
| Plan → write → review agents | `StateGraph` + reviewer router | `15.Project/multiAgent` |

---

## Import cheat-sheet

```python
# models / embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# messages
from langchain_core.messages import (SystemMessage, HumanMessage, AIMessage,
                                     ToolMessage, BaseMessage)
# prompts
from langchain_core.prompts import (PromptTemplate, ChatPromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate, MessagesPlaceholder)
# parsers
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
# runnables (LCEL)
from langchain_core.runnables import (RunnableParallel, RunnableBranch,
                                      RunnableLambda, RunnablePassthrough)
# data
from langchain_core.documents import Document
from langchain_community.document_loaders import (TextLoader, PyPDFLoader,
                                                  DirectoryLoader, WebBaseLoader, CSVLoader)
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter, PythonCodeTextSplitter)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import WikipediaRetriever
# tools / agents / sql
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
# langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
```
