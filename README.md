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
- `1.basic_rag.py` — end-to-end RAG: loads `audit-report.pdf`, splits it (`chunk_size=2000`, `chunk_overlap=300`), embeds into a `Chroma` collection, retrieves with MMR, and feeds the retrieved context + question into a Gemini prompt through an LCEL chain (`{context, question} | prompt | llm | parser`). Runs as an interactive loop until you type `exit`.

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

## Notes

- Model used: `gemini-3.6-flash`.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (via `langchain_huggingface`).
- `resources/` also holds a second report PDF (`audit-report-xorg.pdf`) alongside `audit-report.pdf`.
- Sample inputs for the loaders live in `resources/` (a `.txt`, two `.pdf`s and a `.csv`).
- Python `>=3.13`, managed with `uv`.
- `.env` is git-ignored — never commit API keys.
