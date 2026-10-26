# LangChain / LangGraph — Interview Hand-Note (detailed)

Annotated walk-through of every file in this repo. Each block is:

- **What it does** — one line of intent.
- **Code** — the real snippet, with `#` comments explaining *what each part is doing*.
- **Remember** — the point worth saying out loud in an interview.

**Repository:** `Learn-LangGraph` · Python `>=3.13` · managed with `uv` · models: `gemini-3.6-flash` / `gemini-3.5-flash*`.

---

## 0. The mental model (say this first)

```
DOCUMENTS → load → split → embed → store → retrieve ─┐
                                                     ├─→ prompt → LLM → answer
                            (the question) ──────────┘
```

- **Modules 1–12** build one piece of that pipe each; **13** grades the output; **14** turns the pipe into a state machine; **15** puts it to work.
- **The one API to remember:** **LCEL** — the `|` pipe. `runnable | runnable | runnable` composes into a single `Runnable` you call with `.invoke()`.
- **`invoke` / `stream` / `batch`** = run once / token-by-token / many inputs at once.

---

## 1. Models — `1.BasicLLM/1.geminillm.py`

**What it does:** loads the API key and makes the smallest possible LangChain call.

```python
from dotenv import load_dotenv
load_dotenv()                                    # read .env into environment (GOOGLE_API_KEY)

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")   # the model "runnable"
response = llm.invoke("Hello, langchain! Explain yourself in a single sentence")
print(response.content[0]['text'])               # Gemini wraps text in a content list
```

**Remember:**
- `llm` is a **Runnable** — anything with `invoke` can sit in a chain.
- Gemini's text is at `response.content[0]['text']` (or `response.text`).
- Chat model (messages in) vs LLM (string in) — we use chat models everywhere here.

---

## 2. Prompts — `2.Prompts/`

### `1.static_prompt.py` — a template with no variables

```python
from langchain_core.prompts import PromptTemplate
static_prompts = PromptTemplate(
    input_variables=[],                          # nothing to fill in
    template="Write a short fun fact about AI")
print(static_prompts.format())                   # format() with no args → final string
```

### `2.chat_prompt_template.py` — variables filled at call time

```python
dynamic_prompts = PromptTemplate(
    input_variables=["topic", "style"],          # placeholders the template expects
    template="Write a short paragraph about {topic} in a {style} style")
dynamic_prompts.format(topic="AI", style="humarious")        # fills {topic}/{style}
dynamic_prompts.format(topic="Block chain", style="professional")   # reusable for many inputs
```

### `3.dynamic_prompt.py` — chat prompt with roles

```python
from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

chat_prompts = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant that provides information about {subject}."),   # behavior/tone
    HumanMessagePromptTemplate.from_template(
        "can you tell me something interesting about {subject}.")])                  # user turn
prompt_text = chat_prompts.format_messages(subject="AI")   # → LIST of messages, not a string
```

### `4.project.py` — a tiny interactive blog generator with history

```python
chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
topic = input("Enter your Post topic:")

chat_prompts_template = ChatPromptTemplate.from_messages([...])   # system = "pro blog writer"
chat_history = []

while True:
    user_input = input("Ideas or interaction or type exit : ")
    if user_input.lower() == "exit": break
    message = chat_prompts_template.format_messages(topic=topic)   # base system+user messages
    for prev in chat_history:                                     # replay past turns
        message.append(prev)
    human_message = HumanMessagePromptTemplate.from_template(user_input)\
                      .format_messages(user_input=user_input)[0]   # wrap the new input
    message.append(human_message)
    chat_history.append(human_message)                             # keep history growing
    response = chat_model.invoke(message)
    chat_history.append(AIMessage(content=response.content[0]['text']))
```

**Remember:**
- `PromptTemplate` → **string**; `ChatPromptTemplate` → **list of role messages**.
- **System** = behavior/tone/constraints · **Human** = user · **AI** = past replies.
- History is just re-appending old messages — the manual version of what `MessagesPlaceholder` automates later.

---

## 3. Structured output — `3.StructuredResponse/` (same task 4 ways)

**What it does:** force the model to answer in a fixed shape instead of prose. Task = review a text.

### `1.typedDict.py` — plain TypedDict

```python
class Review(TypedDict):
    summary: str
    sentiment: str
structured_model = chat_model.with_structured_output(Review)   # binds the schema
response = structured_model.invoke(prompt)                     # → dict matching Review
```

### `2.annnotatedTypedDict.py` — TypedDict + field descriptions

```python
class Schema(TypedDict):
    key_themes: Annotated[list[str], "all key themes discussed"]     # annotation = hint to model
    summary:     Annotated[str, "brief summary"]
    sentiment:   Annotated[str, "positive or negative"]
    pros:        Annotated[Optional[list[str]], "pros list"]         # Optional → may be absent
    cons:        Annotated[Optional[list[str]], "cons list"]
```

### `3.pydanticResponse.py` — validated model with constraints

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Schema(BaseModel):
    key_themes: list[str] = Field(description="all key themes")
    summary:    str       = Field(description="brief summary")
    sentiment:  Literal["Pos", "Neg"] = Field(description="positive or negative")  # only 2 allowed
    pros:       Optional[list[str]] = Field(description="pros list")
    cons:       Optional[list[str]] = Field(description="cons list")
    name:       Optional[str] = Field(description="reviewer name", default=None)

structured_model = chat_model.with_structured_output(Schema, strict=True)   # tighter enforcement
```

### `4.jsonSchema.py` — raw JSON Schema

```python
json_schema = {
    "title": "Review", "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["Pos", "Neg"]},   # enum = allowed values
        "pros": {"type": ["array", "null"], "items": {"type": "string"}}},   # nullable array
}
structured_model = chat_model.with_structured_output(json_schema, strict=True)
```

**Remember (the classic question "compare the 4"):**

| Way | You get | Use when |
| --- | --- | --- |
| `TypedDict` | dict | quick, no validation |
| `Annotated` TypedDict | dict + field hints | guide the model cheaply |
| **Pydantic** | validated object + constraints | production (types, `Literal`, `Optional`) |
| JSON Schema | dict | language-agnostic / dynamic schema |

- `strict=True` enforces the schema more tightly. Prefer **Pydantic** when you need validation.

---

## 4. Output parsers — `4.OutputParsers/` (parse *inside* the chain)

### `1.strOutputParser.py` — model → plain string

```python
parser = StrOutputParser()
chain = template1 | chat_model | parser | template2 | chat_model | parser
#        long report  →  model  →  string  →  5-line summary prompt  →  model  →  string
```

### `2.jsonOutputParser.py` — model → dict

```python
parser = JsonOutputParser()
template = PromptTemplate(
    template="Give a fictional person. {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()})
    # partial_variables are injected at BUILD time; the parser writes the JSON instructions for us
chain = template | chat_model | parser
```

### `3.pydanticOutputParser.py` — model → validated object

```python
class Person(BaseModel):
    name: str = Field(description="full name")
    age:  int = Field(gt=18, lt=60, description="between 19 and 59")   # constraint enforced on parse
    city: str = Field(description="city")
parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(template="Give a fictional person from {place}. {format_instruction}",
                          input_variables=['place'],
                          partial_variables={"format_instruction": parser.get_format_instructions()})
chain = template | chat_model | parser     # → Person instance
```

**Remember:** parser = parsing in the chain (works with any model); `with_structured_output` = the provider's native tool-calling. `get_format_instructions()` tells the model the exact output format.

---

## 5. LCEL chains — `5.Chains/`

### `1.simpleChain.py` — one pass

```python
chain = prompt | chat_model | parser          # prompt → model → string
chain.invoke({'topic': 'Bangladesh'})
```

### `2.sequentialChain.py` — output of step 1 feeds step 2

```python
chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser
#        detailed report  → model → text   → 3-point summary prompt → model → text
```

### `3.conditionalChain.py` — branch at runtime

```python
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]     # classifier output

classifier_chain = prompt1 | chat_model | PydanticOutputParser(pydantic_object=Feedback)

branch_chain = RunnableBranch(
    (lambda x: x['classifier'].sentiment == "positive", prompt2 | chat_model | parser),   # if
    (lambda x: x['classifier'].sentiment == "negative", prompt3 | chat_model | parser),   # elif
    RunnableLambda(lambda x: "No valid sentiment found"))                                 # else/fallback

chain = RunnablePassthrough.assign(classifier=classifier_chain) | branch_chain
#       ^ runs the classifier and adds its result under key "classifier" WITHOUT dropping input
```

### `4.parallelChain.py` — fan out, then merge

```python
runnable_chain = RunnableParallel({
    "notes": prompt1 | chat_model1 | parser,     # both run on the SAME input
    "qa":    prompt2 | chat_model2 | parser})
merge_chain = prompt3 | chat_model1 | parser     # takes {"notes":..., "qa":...} → one doc
final_chain = runnable_chain | merge_chain
```

**Remember:** `RunnableParallel` = same input to many chains (returns dict of named results); `RunnablePassthrough.assign(key=chain)` = add a key, keep the rest; `RunnableBranch` = if/elif/else with a fallback.

---

## 6. Document loaders — `6.DocumentLoader/` (every source → `Document`)

A `Document` = `page_content` (str) + `metadata` (dict: source, page, …).

```python
# 1.textLoader.py — a plain text file → one Document
TextLoader(".../resources/cricket.txt", encoding="utf-8").load()

# 2.pdfLoader.py — one Document PER PAGE
PyPDFLoader(".../resources/audit-report.pdf").load()      # pdf[0].metadata has the page number

# 3.directoryLoader.py — a whole folder of PDFs
loader = DirectoryLoader(path=".../resources", glob="*.pdf", loader_cls=PyPDFLoader)
pdfs = loader.load()          # EAGER: returns everything at once
for pdf in loader.lazy_load():  # LAZY: yields one Document at a time (lighter for big sources)
    print(pdf.metadata)

# 4.webBaseLoader.py — scrape a URL, then answer a question from it (RAG in miniature)
info = WebBaseLoader("https://www.bbc.com/...").load()
chain = prompt | model | parser     # prompt = strict rules + {question} + {text}
chain.invoke({"question": "Who is the current PM of Bangladesh", "text": info[0].page_content})

# 5.csvLoader.py — one Document PER ROW
CSVLoader(file_path=".../resources/countries of the world.csv").load()
```

**Remember:** `load()` = all at once; `lazy_load()` = one at a time. Loaders standardise every source into `Document`s so the rest of the pipeline doesn't care where data came from.

---

## 7. Text splitters — `7.TextSplitter/`

```python
# 1.characterTextSplitter.py — split on a fixed separator
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="")
data = splitter.split_documents(docs)        # split_documents keeps metadata

# 2.recursiveTextSplitter.py — try separators in order (para → line → sentence → word)
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)  # → list[str]

# 3.pythonCodeSplitter.py — split on CODE boundaries (class/def/statement)
PythonCodeTextSplitter(chunk_size=500, chunk_overlap=100).split_text(code)
```

**Remember:** `chunk_size` = target size; **`chunk_overlap` carries the tail of one chunk into the next** so a sentence split across the boundary isn't lost. `RecursiveCharacterTextSplitter` is the sensible default; code splitter keeps functions whole.

---

## 8. Embeddings — `8.Embedding/1.huggingFaceEmbedding.py`

**What it does:** turn text into a vector that captures meaning.

```python
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")  # local, free
result = embedding.aembed_query("My name is Minhaz Uddin")   # async; embed_documents() for many
```

**Remember:** similar meaning ⇒ nearby vectors. Local HF model here; `GoogleGenerativeAIEmbeddings` is the API alternative (used in the RAG agent notebook).

---

## 9. Vector stores — `9.VectorStores/` (where vectors live)

```python
# 1.chromaDb.py — persistent store with named collections
Chroma.from_texts(texts=texts, embedding=embedding, collection_name="langchain_chroma_demo")
vectorStore.similarity_search("tell me more about llms", k=2)     # → closest 2 Documents

# 2.faissDb.py — fast in-memory index
FAISS.from_texts(texts=bangladesh_texts, embedding=embedding)
vectorStore.similarity_search("city of bangladesh", k=2)
```

**Remember:** `from_texts` (raw strings) vs `from_documents` (Documents → keeps metadata). Store = the vector database; it can also search directly.

---

## 10. Retrievers — `10.Retrievers/` (search-only interface)

```python
# 1.WikipediaRetriever.py — live articles, NO embeddings needed
WikipediaRetriever(top_k_results=2, lang="en").invoke("What is the Current PM of Bangladesh")

# 2.vectorDbRetrievers.py — any store → retriever
retriever = vectorStore.as_retriever(search_kwargs={"k": 2})   # k = how many to return
retriever.invoke("tell me more about llms")

# 3.maximalMarginalRelevanceRetrievers.py — relevance + diversity
vectorStore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.25})  # diverse
vectorStore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 1.00})  # similar
```

**Remember:** retriever ≈ vector store but "dumber": `invoke(query)` → Documents, plugs straight into chains. **MMR**: `lambda_mult` near 0 = diverse, near 1 = pure similarity.

---

## 11. RAG — `11.RAG/` (retrieval-augmented generation)

**The pipeline:** load → split → embed → store → retrieve → inject context into prompt → answer.
**Why:** the LLM answers **from your documents**, not its own memory (fewer hallucinations, fresh/private data).

### `1.basicRag.py` — the canonical version

```python
documents = PyPDFLoader(".../audit-report.pdf").load()
docs = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300).split_documents(documents)
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vectorDb = Chroma.from_documents(docs, embedding, collection_name="rag_collection")
retriever = vectorDb.as_retriever(search_type="mmr", search_kwargs={"k": 2})

def format_docs(similar_docs):
    return "\n\n".join(doc.page_content for doc in similar_docs)   # turn Documents into one string

rag_chain = (
    {"context": retriever | format_docs,     # retrieve + join
     "question": lambda x: x}                # pass the query through unchanged
    | prompt | llm | parser)                 # prompt = "use the context, else say I don't know"
# interactive loop until "exit"
```

### `2.explainableRAG.py` — add citations

```python
retrieved_docs = vectorDb.similarity_search(query, k=3)
for i, doc in enumerate(retrieved_docs):
    page = doc.metadata.get("page", "N/A"); source = doc.metadata.get("source", "N/A")
    context_text += f"\n chunk {i+1} : \n {doc.page_content} \n"   # tag each chunk
    sources.append({"chunk": i+1, "page": page, "content": doc.page_content})
# prompt rule: "mention which page and chunk you got the info from"
# after answering, print the retrieved chunks so retrieval is auditable
```

### `3.memoryRAG.py` — conversational (pronoun resolution)

```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="... resolve references like he/they/it using history ..."),
    MessagesPlaceholder(variable_name="chat_history"),          # ← injects prior turns here
    ("human", "context : \n {context} \n \n question: \n {input}")])
chat_history = []
# each turn: append HumanMessage, retrieve, invoke, append AIMessage
```

### `4.databaseMemoryRAG.py` — Streamlit app, history in SQLite

```python
connection = sqlite3.connect("chat_memory1.db", check_same_thread=False)
cursor.execute("create table if not exists chat_history (session_id TEXT, role TEXT, content TEXT)")

def save_message(session_id, role, content): ...        # INSERT a turn
def load_chat_history(session_id):                      # SELECT → rebuild HumanMessage/AIMessage
def get_all_session():                                  # DISTINCT session_id → session picker

@st.cache_resource
def load_vector_storage():                              # build Chroma ONCE, cached across reruns
    ...

st.session_state.session_id = str(uuid.uuid4())          # one session id per chat
# sidebar: "Now chat" button (new uuid) + buttons to reload old sessions
# chat window: st.chat_message("user"/"AI"), st.chat_input for new questions
```

### `5.multiDocumentRAG.py` — many files, one store

```python
for file in os.listdir(DOC_FOLDER_PATH):
    if file.endswith(".pdf"):
        docs = PyPDFLoader(os.path.join(DOC_FOLDER_PATH, file)).load()
        for d in docs:
            d.metadata['source_document'] = file         # ← tag origin so chunks stay traceable
        all_documents.extend(docs)
chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(all_documents)
vector_db = Chroma.from_documents(chunks, embedding)
# context tag = [Document : {source_document}] so answers can name which files were used
```

**Remember (variants):** basic = grounded answer · explainable = cites page/chunk · memory = knows pronouns via `MessagesPlaceholder` · DB memory = survives restarts via SQLite + session ids · multi-doc = one store, origin kept in metadata.

---

## 12. Agents — `12.Agents/1.toolCallingAgents.py`

**What it does:** the LLM decides when to call your Python functions.

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool()
def multiply(a: int, b: int):
    """Multiply two number"""        # ← docstring IS the description the LLM reads to pick the tool
    return a * b

@tool()
def add(a: int, b: int):
    """Add two number"""
    return a + b

agent = create_agent(model=llm, tools=[add, multiply],
                     system_prompt="Use tools when necessary, else answer directly")

response = agent.invoke({"messages": [{"role": "user", "content": "what is 6*7?"}]})
response["messages"][-1].content[-1]['text']      # final text; response holds the FULL trace
```

**Remember:** `create_agent` runs the loop model → (maybe tool) → see result → repeat → answer. The reply is the whole **message trace**, not just text.

---

## 13. LLM as a judge — `13.LlmAsJudge/`

**What it does:** a second LLM scores the first one's answer, and weak answers get regenerated.

### `prompt.py` — shared prompts + one model name

```python
GEMINI_VERSION = "gemini-3.5-flash-lite"
QUESTION_PROMPT = PromptTemplate(
    template="Answer the following Question as accurately as possible : \n {question}",
    input_variables=['question'])
JUDGE_PROMPT = PromptTemplate(
    template="""... assess the answer: 1. accuracy (0-10) 2. hallucination (true/false)
                3. feedback ... return your evaluation as json only""",
    input_variables=['question', 'answer'])
```

### `judge.py` — make the verdict structured

```python
class Evaluation(BaseModel):
    accuracy: int = Field(description="score from 0 to 10")
    hallucination: bool = Field(description="true if hallucination present, else false")
    feedback: str = Field(description="brief comments")

def evaluate_answer(question, answer):
    structured_judge_llm = ChatGoogleGenerativeAI(model=GEMINI_VERSION)\
                             .with_structured_output(Evaluation)     # machine-readable score
    return structured_judge_llm.invoke(JUDGE_PROMPT.format(question=question, answer=answer))
```

### `main.py` — self-correcting loop

```python
def generate_answer(user_input): return llm.invoke(QUESTION_PROMPT.format(question=user_input))

def ask_until_good_answer(ques, threshold=8, max_attempt=2):
    attempt = 0
    while attempt < max_attempt:
        ans = generate_answer(ques)
        _eval: Evaluation = evaluate_answer(ques, ans.text)
        if _eval.accuracy >= threshold and _eval.hallucination != True:
            break                                   # good enough → accept
        attempt += 1                                # too weak → regenerate
```

**Remember:** the judge is just another LLM call; `with_structured_output` makes its score usable in a **retry gate** (a quality gate, not a passive scorer).

---

## 14. LangGraph — `14.LangGraph/` (state machine for LLM workflows)

**Core vocabulary:** a **state** (`TypedDict`) every node reads/writes · **nodes** (functions) · **edges** (flow) · **conditional edges** (branch at runtime) · **reducers** (how to merge state) · **compile()/invoke()**.

### `1.basicGraph.ipynb` — one node

```python
class MsgState(TypedDict):
    message: str                                  # the shared state

def greetings_node(state: MsgState) -> MsgState:
    state['message'] = "Hello " + state['message'] + ". welcome to langgraph"
    return state                                  # nodes return the (updated) state

graph = StateGraph(MsgState)
graph.add_node("initial", greetings_node)
graph.add_edge(START, "initial")                  # entry
graph.add_edge("initial", END)                    # exit
bot = graph.compile()
bot.invoke({"message": "Learning Hub"})           # → {'message': 'Hello Learning Hub. welcome...'}
```

### `2.complexGraph.ipynb` — a linear pipeline of nodes

```python
def first_node(state):  state['result'] = f"{state['name']}, welcome to the system."; return state
def second_node(state): state['result'] += f"You have skills in {','.join(state['skills'])}."; return state
def third_node(state):  state['result'] += f"You are {state['age']} years old"; return state
# START → first → second → third → END : each node enriches the SAME dict in order
```

### `3.conditionalEdges.ipynb` — branch by a value

```python
def adder(state):    state['result'] = state["number1"] + state["number2"]; return state
def subtract(state): state["result"] = state["number1"] - state["number2"]; return state

def decode_next_snode(state) -> str:              # the ROUTER: returns a KEY
    if state["operations"] == "+": return "add_op"
    elif state["operations"] == "-": return "subtract_op"

graph.add_node("router", lambda state: state)     # pass-through node = stable branch source
graph.add_edge(START, "router")
graph.add_conditional_edges('router', decode_next_snode,
    {"add_op": "add_node", "subtract_op": "subtract_node"})   # path_map: key → node name
```

### `4.complexConditionalEdges.ipynb` — two routing stages

```python
START → router1 → (adder1 | subtract1) → router2 → (adder2 | subtract2) → END
# two number-pairs, two operators, two routers : computes result1 AND result2 in one run
```

### `5.loopNumberGuess.ipynb` — a graph cycle

```python
MAX_COUNTER = 1000
def greeting_node(state): state['name'] = f"Hi there , {state['name']}. we will start the game"
                          state['counter'] = 0; return state
def random_state(state):  state['number'].append(random.randint(0, 10)); state['counter'] += 1; return state

def should_continue(state) -> str:                # route back to SAME node, or exit
    if state['target_number'] != state['number'][-1] and state['counter'] < MAX_COUNTER:
        return LOOP_EDGE
    return EXIT_EDGE

graph.add_conditional_edges(RANDOM_NUM_GEN_NODE, should_continue,
    {LOOP_EDGE: RANDOM_NUM_GEN_NODE, EXIT_EDGE: END})   # edge pointing at its own source = loop
```

### `6.simpleAiAgent.ipynb` — put the model inside a node

```python
class AgentState(TypedDict):
    messages: List[str]
def process(state: AgentState) -> AgentState:
    response = model.invoke(state['messages'])     # the node calls the LLM
    state['messages'].append(response.text)
    return state
# interactive while loop feeds user input in until 'exit'
```

### `7.toolsCalls.ipynb` — tool-calling graph (hand-built agent loop)

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   # reducer: APPEND, don't overwrite

@tool
def add(a: int, b: int): """this will add two number"""; return a + b     # (+ sub, mul, div)
tools = [add, sub, mul, div]
model = model.bind_tools(tools)                     # model can now emit tool_calls

def model_call(state):
    response = model.invoke([SystemMessage("...")] + state['messages'])
    return {'messages': state['messages'] + [response]}

def should_continue(state):                          # route by presence of tool calls
    return 'end' if not state['messages'][-1].tool_calls else 'continue'

graph.add_node('tools', ToolNode(tools=tools))       # ToolNode runs the requested tools
graph.add_edge(START, 'modelcall')
graph.add_conditional_edges('modelcall', should_continue, {'end': END, 'continue': 'tools'})
graph.add_edge('tools', 'modelcall')                 # loop back → agent
bot.stream(user_input, stream_mode="values")         # see each step as it happens
```

### `8.emailDrafter.ipynb` — a stateful document agent

```python
document_content = ""                                # module-level state the tools share

@tool
def update(content: str) -> str:                     # write the whole doc
    global document_content; document_content = content; return "Document updated..."
@tool
def save(filename: str) -> str:                      # finish & persist
    ... appends ".txt" and writes document_content to disk ...

def out_agent(state):                                # asks the user, embeds current doc in system prompt
    system_prompt = SystemMessage(content=f"... The current document is : {document_content}")
    user_msg = HumanMessage(content=input("what would you like to do with this document"))
    response = model.invoke([system_prompt] + list(state['messages']) + [user_msg])
    state['messages'] = list(state['messages']) + [user_msg, response]
    return state

def should_continue(state):                          # STOP once a ToolMessage says it was saved
    for message in reversed(state['messages']):
        if isinstance(message, ToolMessage) and "saved" in message.text.lower():
            return 'end'
    return 'continue'
```

### `9.ragAgent.ipynb` — agentic RAG (retriever as a tool)

```python
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")   # API embeddings
pages = PyPDFLoader(".../bdelection-...pdf").load()                          # 7 pages
pages_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\
                .split_documents(pages)                                      # 23 chunks
vector_db = Chroma.from_documents(pages_split, embedding_model, collection_name="bd_election")
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

@tool
def retriever_tool(query: str) -> str:               # bridge RAG → agent : exposed as a tool
    docs = retriever.invoke(query)
    return "\n".join(f"source {i+1} : \n {doc.page_content}\n" for i, doc in enumerate(docs))

model = model.bind_tools([retriever_tool])
tool_dict = {t.name: t for t in tools}

def call_llm(state):                                 # LLM node (prepends system prompt)
    ...
def take_action(state):                              # executes tool_calls → ToolMessages
    ...
graph.add_conditional_edges(LLM_NODE, should_continue, {True: RETRIEVER_NODE, False: END})
graph.add_edge(RETRIEVER_NODE, LLM_NODE)             # loop until the model answers without tools
```

**Remember (the whole section):**
- **Nodes** return state; **edges** connect; **conditional edges** branch (`add_conditional_edges(src, router, path_map)`).
- **Loops** = a conditional edge pointing back to an earlier node, guarded by a counter.
- **`Annotated[..., add_messages]`** = a **reducer** telling LangGraph to *append* messages, not overwrite.
- **ToolNode / bind_tools** = the hand-built version of `create_agent`; check `last_msg.tool_calls` to route.
- **Agentic RAG** = wrap the retriever in a `@tool` so the model decides *when* to search.

---

## 15. Projects — `15.Project/`

### `claw.ipynb` — browser agent

- **`BrowserManager`** keeps **one** Playwright browser/page alive and reuses it (`get_page()` lazily launches Firefox non-headless; `close()` tears down). Stateful across tool calls.
- **Tools** (async `@tool`): `browser_open_url` (goto + status/title), `browser_inspect_page` (visible text + filtered interactive elements), `browser_click_element`, `browser_type_into_element`, `browser_press_btn`.
- **Graph**: `AgentState.messages` with `add_messages`; `out_agent` calls the model, `route_after_agent` loops to `ToolNode` while `tool_calls` exist, else `END`. Async tools ⇒ `await bot.ainvoke(..., config={"recursion_limit": 50})`.
- **System prompt** enforces Understand → Plan → Observe → Act → Verify and ends with "The assigned task completed".

### `sqlAgent/database.py` + `sqlAgent/sqlAgent.py` — SQL agent

```python
# database.py — build a fixture bank DB
conn = sqlite3.connect("bank.db")
cursor.executescript("DROP TABLE IF EXISTS customers; ...")   # idempotent rebuild
cursor.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, ...)")
# ... 8 tables; then INSERT randomized branches/customers/accounts/transactions/loans/employees
conn.commit(); conn.close()

# sqlAgent.py — natural-language → SQL → answer
db = SQLDatabase.from_uri("sqlite:///bank.db")                # wraps the DB, exposes schema+tools
agent = create_sql_agent(llm=llm, db=db,
                         agent_type='zero-shot-react-description', verbose=True)   # ReAct over SQL
agent.invoke({"input": "how many personal loans have been assigned"})
```

### `multiAgent/multiAgent.py` — planner → worker → reviewer

```python
LOG_DIR = "logs"; logging.basicConfig(filename="logs/execution.log", ...)   # full trace

def planner_agent(state):  state['plan'] = llm.invoke(plan_prompt).text;  return state
def worker_agent(state):   # drafts from query + plan + reviewer feedback
    state['worker_call'] += 1; state['draft_response'] = llm.invoke(...).text; return state
def reviewer_agent(state): # grades against a checklist, returns "Decision: approve|revise"
    decision = 'approve' if "decision: approve" in review_res.lower() else "review"; ...

def review_router(state):  # loop back to the worker until approved or 2 revisions
    if state['review_decision'] == 'approve' or state.get('revision_count', 0) >= 2:
        return END
    state['revision_count'] = state.get('revision_count', 0) + 1
    return 'worker_agent'

workflow = StateGraph(dict)
workflow.set_entry_point(PLANNER_AGENT_NODE)
workflow.add_edge(PLANNER_AGENT_NODE, WORKER_AGENT_NODE)
workflow.add_edge(WORKER_AGENT_NODE, REVIEWER_AGENT_NODE)
workflow.add_conditional_edges(REVIEWER_AGENT_NODE, review_router,
                               {WORKER_AGENT_NODE: WORKER_AGENT_NODE, END: END})
app.get_graph().draw_mermaid_png()        # export the graph picture
```

**Remember:** browser agent = tools driving a real browser, one reused session; SQL agent = `create_sql_agent` turns a question into SQL; multi-agent = specialised nodes + a judge-driven revise loop, all fully logged.

---

## Interview quickfire

| Question | One-line answer |
| --- | --- |
| What is LCEL? | LangChain Expression Language — the `\|` pipe composing runnables into one chain. |
| `PromptTemplate` vs `ChatPromptTemplate`? | String vs list of role messages (system/human/ai). |
| `with_structured_output` vs `OutputParser`? | Native tool-calling to a schema vs parsing the text inside the chain. |
| Why Pydantic over TypedDict for schemas? | Validation + constraints (`gt`/`lt`, `Literal`, `Optional`). |
| `load()` vs `lazy_load()`? | All at once vs one Document at a time. |
| Why `chunk_overlap`? | So meaning across a chunk boundary isn't lost. |
| What is `k`? | How many documents/chunks the retriever returns. |
| What is MMR? | Maximal Marginal Relevance — relevance vs diversity (`lambda_mult` 0=diverse, 1=similar). |
| How does RAG reduce hallucination? | Answers are grounded in retrieved context; prompt says "I don't know" if absent. |
| Difference store vs retriever? | Store holds vectors + can search; retriever is the search-only, chain-friendly wrapper. |
| What is an agent loop? | Model → tool calls → observe → repeat until it can answer. |
| `create_agent` vs hand-built graph agent? | Same loop; the graph gives you explicit control of nodes/edges. |
| When use LangGraph over a chain? | When you need state, branching, loops, retries, or multiple agents. |
| Why `add_messages` reducer? | Tells the graph to append new messages instead of overwriting the list. |
| Judge vs plain scoring? | A second LLM call; structured output lets it gate (regenerate) weak answers. |
| RAG failure modes? | Bad chunking, weak embeddings, low `k`, irrelevant chunks, stale source. |

---

## Gotchas collected while building

- Gemini reply text: `response.content[0]['text']` or `response.text`.
- `partial_variables` inject values at **build** time (e.g. `parser.get_format_instructions()`).
- `RunnableBranch` needs a fallback or it errors when nothing matches.
- A graph **loop** needs a guard (`counter` / `MAX_COUNTER` / `revision_count`) or it runs forever.
- `sqlite:///bank.db` is **relative to the working directory** — run the SQL agent from its folder.
- Streamlit: use `@st.cache_resource` so the vector store builds once; `check_same_thread=False` for SQLite.
- `.env` is git-ignored — never commit API keys.
