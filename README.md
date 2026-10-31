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
| `3.StructuredResponse/` | Getting structured output from the model |
| `4.OutputParsers/` | Parsing model output with LCEL chains |
| `5.Chains/` | Composing prompts, models and parsers into chains |
| `6.DocumentLoader/` | Loading text, PDF, directory, web and CSV data as `Document`s |
| `7.TextSplitter/` | Splitting text and code into chunks with `langchain_text_splitters` |
| `8.Embedding/` | Turning text into vectors with Hugging Face embeddings |
| `9.VectorStores/` | Storing embeddings in Chroma and FAISS for similarity search |
| `10.Retrievers/` | Fetching relevant documents from a source (Wikipedia) |
| `11.RAG/` | Putting it all together: retrieval-augmented generation |
| `12.Agents/` | Building a tool-calling agent with `create_agent` |
| `13.LlmAsJudge/` | Using an LLM as a judge to evaluate another model's answers |
| `14.LangGraph/` | State graphs: multi-node flows, conditional routing, loops and tool-calling agents |
| `15.Project/` | Hands-on agents: a browser-use agent (Playwright), a SQL agent over a generated `bank.db`, and a planner/worker/reviewer multi-agent system |
| `16.evaluation/` | Testing and scoring LLM apps with DeepEval: test cases, `GEval`, conversational and reference-free metrics, and a meeting-summarizer eval harness |

### 1.BasicLLM
- `1.geminillm.py` — load `.env`, create a `ChatGoogleGenerativeAI` model and invoke it.

### 2.Prompts
- `1.static_prompt.py` — `PromptTemplate` with no input variables.
- `2.chat_prompt_template.py` — `PromptTemplate` with `{topic}` and `{style}` variables.
- `3.dynamic_prompt.py` — `ChatPromptTemplate` using system + human messages.
- `4.project.py` — a small interactive blog post generator that keeps chat history.

### 3.StructuredResponse
Same task (reviewing a text) done four ways:
- `1.typedDict.py` — `TypedDict` schema with `with_structured_output`.
- `2.annnotatedTypedDict.py` — `TypedDict` + `Annotated` field descriptions.
- `3.pydanticResponse.py` — Pydantic `BaseModel` with `Field` descriptions and `Literal`.
- `4.jsonSchema.py` — raw JSON Schema passed to the model.

### 4.OutputParsers
Chains built with the `|` pipe operator (LCEL), each ending in an output parser:
- `1.strOutputParser.py` — `StrOutputParser` in a two-step chain (long report → 5-line summary).
- `2.jsonOutputParser.py` — `JsonOutputParser`, injecting `get_format_instructions()` as a partial variable.
- `3.pydanticOutputParser.py` — `PydanticOutputParser` with a `Person` model (`Field` constraints, e.g. `gt`/`lt` on age).

### 5.Chains
- `1.simpleChain.py` — basic `prompt | model | parser` chain generating facts about a topic.
- `2.sequentialChain.py` — two prompts chained together (detailed report → 3-point summary).
- `3.conditionalChain.py` — `RunnableBranch` routing feedback by sentiment: a classifier chain tags it positive/negative, then the matching reply prompt runs.
- `4.parallelChain.py` — `RunnableParallel` runs a notes chain and a Q&A chain at the same time, then a merge prompt combines both outputs.

### 6.DocumentLoader
Loaders from `langchain_community.document_loaders`, reading the sample files in `resources/`:
- `1.textLoader.py` — `TextLoader` reads `cricket.txt` into a list of `Document` objects.
- `2.pdfLoader.py` — `PyPDFLoader` reads `audit-report.pdf` (one `Document` per page) and prints its metadata.
- `3.directoryLoader.py` — `DirectoryLoader` + `PyPDFLoader` loads every `*.pdf` in `resources/`, using both eager `load()` and `lazy_load()`.
- `4.webBaseLoader.py` — `WebBaseLoader` scrapes a BBC page, then answers a question from that text through an LCEL chain.
- `5.csvLoader.py` — `CSVLoader` turns each row of `countries of the world.csv` into a `Document`.

### 7.TextSplitter
Splitters from `langchain_text_splitters`, breaking long text into smaller `chunk_size` pieces with `chunk_overlap` carried into the next chunk:
- `1.characterTextSplitter.py` — `CharacterTextSplitter` splits the `audit-report.pdf` documents on a separator, printing each chunk's content and metadata.
- `2.recursiveTextSplitter.py` — `RecursiveCharacterTextSplitter` splits a raw string by trying separators (paragraph → sentence → word) in order.
- `3.pythonCodeSplitter.py` — `PythonCodeTextSplitter` splits Python source on syntax boundaries (class/function/statement) so code stays coherent.

### 8.Embedding
- `1.huggingFaceEmbedding.py` — builds `HuggingFaceEmbeddings` with the `all-MiniLM-L6-v2` model and embeds a query via `aembed_query`.

### 9.VectorStores
Each script embeds a list of strings, stores the vectors, then runs `similarity_search(query, k=2)`:
- `1.chromaDb.py` — `Chroma.from_texts` writes the vectors into a named collection (`langchain_chroma_demo`).
- `2.faissDb.py` — `FAISS.from_texts` builds an in-memory index over a handful of Bangladesh facts.

### 10.Retrievers
- `1.WikipediaRetriever.py` — `WikipediaRetriever(top_k_results=2, lang="en")` fetches Wikipedia articles for a query and prints each result's `page_content`.
- `2.vectorDbRetrievers.py` — builds a `Chroma` store from `Document`s and wraps it with `as_retriever(search_kwargs={"k": 2})`, then calls `retriever.invoke(query)`.
- `3.maximalMarginalRelevanceRetrievers.py` — a `FAISS` store retriever using `search_type="mmr"`: runs the query twice with `lambda_mult` 0.25 (diverse results) vs 1.00 (pure similarity) to show the difference.

### 11.RAG
- `1.basicRag.py` — end-to-end RAG: loads `audit-report.pdf`, splits it (`chunk_size=2000`, `chunk_overlap=300`), embeds into a `Chroma` collection, retrieves with MMR, and feeds the retrieved context + question into a Gemini prompt through an LCEL chain (`{context, question} | prompt | llm | parser`). Runs as an interactive loop until you type `exit`.
- `2.explainableRAG.py` — the same pipeline but with citation: each retrieved chunk is tagged with its page and source, the prompt is told to point at the page/chunk it used and to say "I don't know" when the answer isn't in the context, and the retrieved chunks are printed after the answer for verification.
- `3.memoryRAG.py` — conversational RAG: keeps a `chat_history` of `HumanMessage`/`AIMessage` and feeds it back through a `MessagesPlaceholder` alongside the retrieved context, so follow-up questions can resolve references like "it" or "they". Interactive loop until you type `exit`.
- `4.databaseMemoryRAG.py` — the memory RAG turned into a Streamlit chat app: chat history is persisted per `session_id` in a SQLite table (`chat_memory1.db`) instead of RAM, so past conversations survive restarts. The sidebar lists previous sessions (reload by clicking one) and starts a "New chat"; the vector store is built once and cached with `@st.cache_resource`. Run with `streamlit run 11.RAG/4.databaseMemoryRAG.py`.
- `5.multiDocumentRAG.py` — multi-document RAG: loops over every `*.pdf` in `resources/multidocsample/`, loads them all, and tags each page with a `source_document` field so chunks keep track of which file they came from. All chunks go into one `Chroma` store; the prompt is told to name the documents involved when an answer draws on more than one, and each answer prints the sources it used. Interactive loop until you type `exit`.

### 12.Agents
- `1.toolCallingAgents.py` — a basic tool-calling agent built with `create_agent`: defines two `@tool`-decorated functions (`add`, `multiply`), hands them to the agent with a system prompt, and runs an interactive loop where the agent decides whether to call a tool or answer directly (`exit` to quit).

### 13.LlmAsJudge
- `prompt.py` — shared constants: `GEMINI_VERSION` (`gemini-3.5-flash-lite`) plus two `PromptTemplate`s — `QUESTION_PROMPT` (asks the model to answer a question) and `JUDGE_PROMPT` (tells the judge to assess an answer on accuracy 0–10, hallucination true/false, and feedback, returning JSON only).
- `judge.py` — defines an `Evaluation` Pydantic model (`accuracy`, `hallucination`, `feedback`) and `evaluate_answer(question, answer)`: binds the schema with `with_structured_output(Evaluation)` and invokes the judge prompt, returning a validated `Evaluation`.
- `main.py` — runs the question → answer → judge loop through `ask_until_good_answer`: generates an answer with `generate_answer`, judges it with `evaluate_answer`, and if `accuracy` is below the threshold (8) or the judge flags a hallucination, regenerates the answer — up to `max_attempt` (2) tries — before moving on to the next question.

### 14.LangGraph
- `1.basicGraph.ipynb` — a first LangGraph state graph: defines a `MsgState` `TypedDict`, a `greetings_node` that rewrites `state["message"]`, wires it up with `StateGraph(MsgState)` + `add_node` + `add_edge(START/END)`, compiles it with `graph.compile()`, and runs `bot.invoke({"message": "Learning Hub"})`.
- `2.complexGraph.ipynb` — a multi-node graph: an `AgentState` (`name`, `age`, `skills`, `result`) flows through three nodes, each appending to `result`, with `add_edge` chaining them `START → first → second → third → END`.
- `3.conditionalEdges.ipynb` — routing with conditions: an `AgentState` (`number1`, `number2`, `operations`, `result`) goes to a pass-through `router` node, and `add_conditional_edges("router", decode_next_snode, {...})` maps the returned key (`add_op`/`subtract_op`) to the add or subtract node based on `operations`.
- `4.complexConditionalEdges.ipynb` — two-stage conditional routing: an `AgentState` with two number pairs and two operators, two pass-through routers, and `add_conditional_edges` at each stage (add/subtract per pair), so `router1 → (adder1 | subtract1) → router2 → (adder2 | subtract2) → END` computes two independent results in one run.
- `5.loopNumberGuess.ipynb` — a looping graph: a `greeting_node` initializes a `counter`, a `random_state` node appends a random number each pass, and `should_continue` routes back to the same node (`loop`) until the number matches `target_number` (or a `MAX_COUNTER` safety cap) — a graph cycle built from a conditional edge that targets its own source node.
- `6.simpleAiAgent.ipynb` — the model inside a node: `AgentState` holds a `messages` list, the `process` node invokes `ChatGoogleGenerativeAI` on those messages and appends the reply, and an interactive `while` loop feeds user input in until `exit`.
- `7.toolsCalls.ipynb` — tool-calling graph: `AgentState.messages` uses the `add_messages` reducer (`Annotated[Sequence[BaseMessage], add_messages]`), four `@tool` math functions (`add`/`sub`/`mul`/`div`) are bound with `model.bind_tools`, a `ToolNode` executes them, and `should_continue` loops back through the model while the last message still has `tool_calls`; run with `bot.stream(..., stream_mode="values")`.
- `8.emailDrafter.ipynb` — a stateful email-drafting agent: `update` and `save` tools read/write a module-level `document_content` string, the system prompt embeds the current document on every turn, and `should_continue` stops once a `ToolMessage` reports the document was saved; `save` appends `.txt` and writes the file.
- `9.ragAgent.ipynb` — agentic RAG: loads `resources/bdelection-comprehensivereportenglishbyahmadrafi.pdf` with `PyPDFLoader`, splits it (`chunk_size=1000`, `chunk_overlap=200`), embeds it with `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`) into a `Chroma` collection (`bd_election`), and wraps the store's retriever (`search_type="similarity"`, `k=3`) in a `@tool`. A hand-built agent loop (`call_llm` → `take_action` → back to `call_llm`, with `should_continue` checking whether the last message has `tool_calls`) answers questions about the report and cites its sources; interactive loop until `exit`.
- `leave_application.txt` — a sample leave-application document written by the email drafter's `save` tool.

### 15.Project
Three small agents that put the earlier pieces (tools + a loop) to work on a real task:
- `claw.ipynb` — an autonomous browser agent: a `BrowserManager` lazily launches one async Playwright browser (Firefox, non-headless) and reuses the same page for every call; async `@tool` functions (`browser_open_url`, `browser_inspect_page`, `browser_click_element`, `browser_type_into_element`, `browser_press_btn`) expose navigation, DOM inspection (visible text + a filtered list of interactive elements), clicking, typing and key presses. The model (`gemini-3.5-flash-lite`) is bound with `bind_tools`, a `ToolNode` runs the calls, and `route_after_agent` loops back to the agent while the last `AIMessage` still has `tool_calls` (an `AgentState` with the `add_messages` reducer). A detailed system prompt enforces an Understand → Plan → Observe → Act → Verify cycle and ends with the phrase "The assigned task completed". Interactive async loop (`await bot.ainvoke(..., config={"recursion_limit": 50})`) until `exit`, closing the browser on the way out.
- `sqlAgent/database.py` — generates the fixture database with `sqlite3`: drops and recreates eight tables (`customers`, `addresses`, `branches`, `accounts`, `transactions`, `loans`, `employees`, `account_managers`) and fills them with randomized banking data (50 customers, 100 accounts, ~3000 transactions, 30 loans, 10 employees), then commits to `bank.db` and closes.
- `sqlAgent/sqlAgent.py` — a natural-language SQL agent: wraps the SQLite file with `SQLDatabase.from_uri("sqlite:///bank.db")` and hands it to `create_sql_agent(agent_type="zero-shot-react-description")` with `gemini-3.5-flash-lite`, so a plain question (e.g. "how many personal loans have been assigned") is turned into SQL, executed, and answered — `verbose=True` prints the thought/action/observation steps.
- `bank.db` / `sqlAgent/bank.db` — the generated SQLite databases (regenerated each time `database.py` runs).
- `multiAgent/multiAgent.py` — a planner → worker → reviewer multi-agent system on a `StateGraph(dict)` wired with `set_entry_point`, straight edges and one conditional edge. `planner_agent` turns the user query into a short plan; `worker_agent` drafts an answer from the query + plan + any reviewer feedback (and prints an improved revision on later calls); `reviewer_agent` grades the draft against a checklist (concrete example, implementation details, tradeoffs, clarity, actionable recommendation) and returns a `Decision: approve|revise` verdict; `review_router` loops back to the worker until it approves or `revision_count` hits 2, then ends. Run interactively (one `input()` for the query), with every step logged to `logs/execution.log` and each agent's output saved to `logs/*.txt`.
- `multiAgent/workflow.png` — the compiled graph rendered via `app.get_graph().draw_mermaid_png()`.
- `multiAgent/logs/` — the run's trace: `execution.log` plus `planner_output.txt`, `worker_output_N.txt` and `reviewer_output_N.txt` for each revise round.

### 16.evaluation
LLM-app testing with **DeepEval** (`deepeval`), using `deepeval.models.GeminiModel` (`gemini-3.5-flash-lite`) as the judge:
- `LLM.py` — shared judge model: loads `.env` and exposes `GeminiModel(model="gemini-3.5-flash-lite")` as `llm`.
- `llmTestCase.py` — `LLMTestCase` (input/expected/actual) scored by two `GEval` "Correctness" metrics — one demanding an exact match, one accepting different wording with the same meaning — and asserted with `assert_test`.
- `conversationalTestCase.py` — `ConversationalTestCase` built from `Turn` objects (user/assistant), graded by `ConversationalGEval` on professionalism (`evaluation_params=[MultiTurnParams.CONTENT]`) to contrast a rude conversation with a polite one.
- `answerRelevancyMatrics.py` — the simplest end-to-end check: one `LLMTestCase` run through `AnswerRelevancyMetric(threshold=0.5, include_reason=True)` with `evaluate(...)`.
- `meetingSummarizer/meetingSummarizer.py` — a `MeetingSummarizer` class wrapping `ChatGoogleGenerativeAI` with two prompts: one turning a transcript into a 3–5 bullet summary, and one extracting action items as JSON (`task`, `responsible_person`, `deadline`, `priority`, `status`). Run directly to summarize `resources/meeting_transcript.txt`.
- `meetingSummarizer/meetingSummarizerTest.py` — the evaluation harness: loads every `*.txt` in `resources/transcripts/`, wraps each in a `Golden`, runs the summarizer on it, builds `LLMTestCase`s for the summary and the action items, then scores them with two `GEval` metrics — *Summary Concision* and *Action Item Accuracy* — before printing the result.

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
- **Document loaders** — `langchain_community.document_loaders` turns any source (file, folder, URL, CSV) into `Document` objects with `page_content` + `metadata`.
- **Eager vs lazy loading** — `load()` returns everything at once; `lazy_load()` yields documents one at a time, which is lighter for large sources.
- **Text splitters** — `langchain_text_splitters` breaks long text into `chunk_size` chunks with `chunk_overlap` so context isn't lost at the boundaries.
- **Splitter strategies** — `CharacterTextSplitter` splits on a single separator, `RecursiveCharacterTextSplitter` tries paragraphs → sentences → words in order, and `PythonCodeTextSplitter` splits on code syntax boundaries.
- **Embeddings** — `HuggingFaceEmbeddings` turns text into vectors that capture semantic meaning; `aembed_query` is the async variant.
- **Vector stores** — `Chroma` (persistent, named collections) and `FAISS` (in-memory local index) store those vectors so `similarity_search(query, k)` can retrieve the closest chunks.
- **Retrievers** — like vector stores but a simpler interface: call `invoke(query)` and get back relevant `Document`s (e.g. `WikipediaRetriever` pulls live articles, no embeddings needed).
- **Vector store retrievers** — `vector_store.as_retriever(search_kwargs={"k": 2})` turns any store into a retriever so it can plug into chains.
- **MMR (Maximal Marginal Relevance)** — `search_type="mmr"` balances relevance against diversity; `lambda_mult` near 0 favors diversity, near 1 favors pure similarity.
- **RAG (Retrieval-Augmented Generation)** — load → split → embed → store → retrieve → prompt, wiring the retriever's context into the prompt so the LLM answers from your documents instead of its own memory.
- **Explainable RAG** — carrying each chunk's `metadata` (page, source) into the prompt lets the model cite where an answer came from, and printing the retrieved chunks afterwards makes the retrieval step auditable.
- **Conversational / memory RAG** — `MessagesPlaceholder(variable_name="chat_history")` injects prior turns into the prompt, so the model can resolve pronouns and follow-ups while still grounding answers in retrieved context.
- **Persistent memory (SQLite)** — storing `(session_id, role, content)` rows in SQLite and reloading them into `HumanMessage`/`AIMessage` objects keeps conversations across restarts, and `DISTINCT session_id` powers a session picker.
- **Streamlit chat UI** — `st.chat_message`/`st.chat_input` render the conversation, `st.session_state` holds the active session, and `@st.cache_resource` builds the vector store once so it isn't rebuilt on every rerun.
- **Multi-document RAG** — loading many files into one store works as long as each chunk keeps its origin: attaching a `source_document` value to page `metadata` lets retrieval span documents and lets the answer cite which files it used.
- **Tools** — `@tool`-decorated functions expose plain Python to the model; the docstring is the description the LLM reads to decide when and how to call the tool.
- **Agents** — `create_agent(model, tools, system_prompt)` builds a loop where the model chooses tools, sees their results, and repeats until it can answer directly; `invoke({"messages": [...]})` returns the full message trace (tool calls included), not just the final text.
- **LLM as a judge** — a second LLM call (with a grading prompt) evaluates the quality of a model's answer, scoring accuracy and flagging hallucinations instead of trusting the output blindly.
- **Structured judge output** — `with_structured_output(Evaluation)` makes the judge return a validated Pydantic object (`accuracy`, `hallucination`, `feedback`), so scores are machine-readable rather than free text.
- **Self-correcting loop (judge in the loop)** — feeding the judge's verdict back into a retry condition (`accuracy >= threshold` and no hallucination) turns the judge from a passive scorer into a quality gate that regenerates weak answers before accepting them.
- **State graph** — `StateGraph(State)` builds a graph over a shared `TypedDict` state that every node reads from and writes to.
- **Nodes and edges** — `add_node(name, fn)` registers a step and `add_edge(START, ...)` / `add_edge(..., END)` connect them so the state flows from node to node in order.
- **Compile and invoke** — `graph.compile()` turns the graph into a runnable and `invoke({...})` executes it, threading the state through each node and returning the final state.
- **Multi-node graphs** — chaining more than two nodes with `add_edge` lets one state flow through a whole pipeline, with each node reading and enriching the same dict in order.
- **Router node** — a lightweight pass-through node (`lambda state: state`) gives conditional edges a stable source to branch from before any real work happens.
- **Conditional edges** — `add_conditional_edges(source, routing_fn, path_map)` decides the next node at runtime: `routing_fn` returns a key, and `path_map` translates that key into the target node name.
- **Multiple routing stages** — chaining several routers and conditional edges lets one graph branch, rejoin and branch again (`router1 → (add1 | subtract1) → router2 → (add2 | subtract2) → END`).
- **Graph loops** — a conditional edge can point back to an earlier node (even its own source), turning the graph into a cycle; a `counter` guard against a `MAX_COUNTER` cap keeps the loop from running forever.
- **Message state and reducers** — typing state as `Annotated[Sequence[BaseMessage], add_messages]` tells LangGraph how to merge new messages into `state["messages"]` (append) instead of overwriting the list.
- **ToolNode** — `langgraph.prebuilt.ToolNode(tools)` is a ready-made node that runs every tool call in the last AI message and returns the `ToolMessage` results.
- **Tool-call routing (agent loop)** — `model.bind_tools(tools)` lets the model emit `tool_calls`; a conditional edge checks `last_message.tool_calls` to route to the tools node (then back to the model) or to `END`, which is the hand-built version of the `create_agent` loop.
- **Streaming a graph** — `bot.stream(input, stream_mode="values")` yields the state after each step, so you can print each message/tool call as it happens instead of waiting for the final result.
- **Retriever as a tool** — wrapping `retriever.invoke(query)` in a `@tool` that formats the returned chunks is the bridge between a RAG pipeline and an agent loop, letting the model decide when to search.
- **Agentic RAG** — instead of always retrieving, the model chooses when to call the retriever tool (and can call it more than once) before answering, so retrieval becomes part of the agent's reasoning rather than a fixed pipeline step.
- **Google embeddings** — `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`) is an API-based alternative to the local Hugging Face embeddings, requiring no model download.
- **Browser automation tools** — wrapping Playwright actions (`page.goto`, `locator.click`/`fill`/`press`, `inner_text`) in async `@tool` functions lets the LLM drive a real browser through DOM selectors instead of screen coordinates.
- **Reusing a browser session** — a `BrowserManager` that lazily launches one browser/page and keeps handing back the same instance makes the agent stateful across tool calls, instead of opening (and losing) a browser on every call.
- **Page inspection for grounded selectors** — returning a page's visible text plus a filtered list of interactive elements (buttons, inputs, links) gives the model real selectors and state to act on rather than guessed ones.
- **Async tools and graph invocation** — async `@tool`s require the graph to run with `await bot.ainvoke(...)`, and `config={"recursion_limit": 50}` bounds the observe → act loop so a stuck agent can't spin forever.
- **SQLDatabase** — `SQLDatabase.from_uri("sqlite:///bank.db")` wires LangChain to any SQLAlchemy-supported database and exposes its tables, schema and query runner as tools for the model.
- **SQL agent** — `create_sql_agent(llm, db, agent_type="zero-shot-react-description")` packages those tools into a ReAct loop: the model lists tables, reads the schema, writes SQL, runs it, and turns the rows back into a natural-language answer (`verbose=True` shows each thought/action/observation).
- **Multi-agent system** — splitting one hard job across specialized nodes (planner, worker, reviewer) on a shared state graph, each with its own focused prompt, produces better output than one prompt trying to do everything.
- **Planner → worker → reviewer loop** — the planner decomposes the task, the worker drafts, and the reviewer critiques against explicit criteria; a conditional edge sends the draft back to the worker until the reviewer approves (or a revision cap like `revision_count >= 2` is hit) — an LLM-as-judge wired into the graph as a quality gate.
- **Self-revise (reflexion) loop** — feeding the reviewer's `reason` into the next worker prompt as feedback makes the worker explicitly fix the flagged issues, turning review into improvement instead of just a score.
- **Traceability** — writing each agent's output to `logs/*.txt` and a run log (`execution.log`), plus rendering the graph with `draw_mermaid_png()`, makes a hidden multi-step run inspectable after the fact.
- **LLM evaluation with DeepEval** — `deepeval` treats an LLM app as a system under test: build `LLMTestCase`s (input/expected/actual) and score them with metrics, either in a pytest-style `assert_test` or a batch `evaluate(...)`.
- **GEval (custom criteria)** — `GEval(name=..., criteria=..., evaluation_params=[...])` uses an LLM to grade output against free-text criteria, so you can test qualities (correctness, concision) that have no single exact-match answer.
- **Meaning vs exact match** — two almost identical GEval criteria show the range: a strict one scores 0 unless the output matches exactly, while a softer one accepts different wording with the same meaning.
- **Metric thresholds** — every metric carries a `threshold` (e.g. 0.5); a test passes when the score clears it, which is how fuzzy LLM output becomes a pass/fail.
- **Conversational test cases** — `ConversationalTestCase` + `Turn(role, content)` evaluates whole multi-turn dialogues, and `ConversationalGEval` with `MultiTurnParams.CONTENT` grades them (e.g. professionalism) instead of a single answer.
- **Reference-free metrics** — `AnswerRelevancyMetric` grades an answer against just the input (no expected output needed), which suits tasks where there is no one right answer.
- **DeepEval judge models** — `deepeval.models.GeminiModel` (or any supported LLM) is injected as the metric's `model=`, so grading runs through the same Gemini key as the app.
- **Golden datasets** — `deepeval.dataset.Golden` holds the bare inputs, and the harness expands each into `LLMTestCase`s at run time — the shape of a small custom eval set.

## Notes

- Model used: `gemini-3.6-flash` for most modules; `11.RAG/1.basicRag.py`, `11.RAG/2.explainableRAG.py` and `6.DocumentLoader/4.webBaseLoader.py` use `gemini-3.5-flash`; `13.LlmAsJudge`, the `16.evaluation` metrics/harness, and the LangGraph agent notebooks (`6.simpleAiAgent`, `7.toolsCalls`, `8.emailDrafter`, `9.ragAgent`) plus `15.Project/claw.ipynb`, `15.Project/sqlAgent/sqlAgent.py` and `15.Project/multiAgent/multiAgent.py` use `gemini-3.5-flash-lite`.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (via `langchain_huggingface`), except `14.LangGraph/9.ragAgent.ipynb`, which uses `gemini-embedding-2` via `GoogleGenerativeAIEmbeddings`.
- The SQL agent needs `sqlalchemy` and `langchain-classic` (both in `pyproject.toml`); its `sqlite:///bank.db` URI is relative, so run `15.Project/sqlAgent/sqlAgent.py` from that folder and generate the DB first with `database.py`.
- `16.evaluation` needs `deepeval` (in `pyproject.toml`); `llmTestCase.py` and `conversationalTestCase.py` are pytest-style (`assert_test` / `test_*` functions) — run them with `deepeval test run` (or `pytest`) from `16.evaluation/`, while `answerRelevancyMatrics.py` and `meetingSummarizer/meetingSummarizerTest.py` are plain scripts you run directly.
- `resources/` also holds a second report PDF (`audit-report-xorg.pdf`) alongside `audit-report.pdf`, plus `bdelection-comprehensivereportenglishbyahmadrafi.pdf` — the input for the agentic RAG notebook.
- Sample inputs for the loaders live in `resources/` (a `.txt`, three `.pdf`s and a `.csv`).
- `resources/multidocsample/` holds the multi-document RAG inputs (`audit-report.pdf`, `Customer Performance Report.pdf`, `countries of the world.csv`).
- `resources/meeting_transcript.txt` is the sample transcript summarized by `16.evaluation/meetingSummarizer/meetingSummarizer.py`, and `resources/transcripts/` holds the meeting transcripts (`meeting_transcript1.txt`, `meeting_transcript4.txt`, …) the summarizer eval runs over.
- The Streamlit RAG app needs `streamlit` (in `pyproject.toml`) and stores chat history in `chat_memory1.db` (committed at `11.RAG/chat_memory1.db`).
- Python `>=3.13`, managed with `uv`.
- `.env` is git-ignored — never commit API keys.
