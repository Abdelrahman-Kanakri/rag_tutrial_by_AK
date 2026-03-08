# RAG From Scratch – Parts 12–14 Walkthrough

> Line-by-line explanation of [`part_12_13_14.ipynb`](part_12_13_14.ipynb)

---

## 📋 LangChain vs LangGraph Recap (Cell 1 – Markdown)

The notebook opens with the same comprehensive reference comparing **LangChain** and **LangGraph** as in all previous notebooks. This recap is repeated for quick reference.

### Key Tables Covered

| Table # | Topic | Summary |
| ------- | ----- | ------- |
| 1 | High-Level Architecture | Maps each layer (Capability, Execution, Memory, State, Context, Contracts) to its owner (LangChain vs LangGraph vs Application) |
| 2 | LangChain vs LangGraph | Compares primary role, control flow, state management, parallel execution, determinism, and production readiness |
| 3 | AgentState vs TypedDict State | Agent internal memory (LangChain) vs workflow shared state (LangGraph) |
| 4 | dataclass Context vs AgentState | Runtime configuration (static, external) vs agent memory (mutable, internal) |
| 5 | TypedDict vs BaseModel | Python typing (no runtime validation) vs Pydantic (runtime enforced) |
| 6 | State Evolution Comparison | Accumulating memory vs selective updates |
| 7 | Mental Model Summary | One-line analogies for each concept |
| 8 | Architecture Diagram | Text diagram: Application → Context → LangGraph Workflow → State → Node → LangChain Agent |

> See the [Part 1–4 walkthrough](../part_1_4/part_1_4_walkthrough.md) for a detailed explanation of each table.

---

## 📌 Notebook Title & Overview (Cells 2–3 – Markdown)

**Cell 2** — Section heading and overview:

> **RAG From Scratch: Advanced Indexing** — Explores advanced indexing strategies that go beyond simple chunk-and-embed.

**Cell 3** — Prerequisites note:

> Run environment setup and indexing cells from previous notebooks first. Chunking is covered in Part 1–4.

---

## ⚙️ Environment Initialization (Cell 4 – Code)

```python
import warnings
import os
from dotenv import load_dotenv
```

| Import                           | Role                                                 |
| -------------------------------- | ---------------------------------------------------- |
| `warnings`                       | Built-in module to control warning messages          |
| `os`                             | Built-in module for environment variable access      |
| `from dotenv import load_dotenv` | Loads variables from a `.env` file into `os.environ` |

```python
warnings.filterwarnings("ignore")
```

Suppresses all Python warnings so the output stays clean.

```python
load_dotenv()
```

Reads the `.env` file in the project root and loads its key-value pairs into environment variables.

```python
try:
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2")
    os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGSMITH_PROJECT")
    os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGSMITH_ENDPOINT")
    os.environ["MISTRAL_API_KEY"]      = os.getenv("MISTRAL_API_KEY")
    os.environ["HF_TOKEN"]             = os.getenv("HF_TOKEN")
    os.environ["GOOGLE_API_KEY"]       = os.getenv("GOOGLE_API_KEY")
    os.environ["OPEN_ROUTER_API_KEY"]  = os.getenv("OPEN_ROUTER_API_KEY")
    os.environ["USER_AGENT"]           = "MyLangChainApp/1.0"
    print("Environment variables set successfully")
except Exception as e:
    print(f"Error: {e}")
```

| Variable               | Purpose                                             |
| ---------------------- | --------------------------------------------------- |
| `LANGCHAIN_TRACING_V2` | Enables **LangSmith** tracing (observability)       |
| `LANGCHAIN_API_KEY`    | API key to authenticate with LangSmith              |
| `LANGCHAIN_PROJECT`    | LangSmith project name to group traces              |
| `LANGCHAIN_ENDPOINT`   | LangSmith API endpoint URL                          |
| `MISTRAL_API_KEY`      | API key for the **Mistral AI** LLM & embeddings     |
| `HF_TOKEN`             | Hugging Face token for model/dataset access         |
| `GOOGLE_API_KEY`       | Google Generative AI API key                        |
| `OPEN_ROUTER_API_KEY`  | OpenRouter proxy API key (for Gemini access)        |
| `USER_AGENT`           | Required header for `WebBaseLoader` HTTP requests   |

> **New in this notebook:** `GOOGLE_API_KEY` and `OPEN_ROUTER_API_KEY` are added for the RAPTOR section (Part 13), which uses Google Gemini via OpenRouter.

**Output:** `Environment variables set successfully`

---

## Cell 5 – Markdown: Environment Docstring

Short docstring cell describing the environment initialization above.

---

# Part 12: Multi-Representation Indexing

---

## Cell 6 – Markdown: Part 12 Introduction

### What is Multi-Representation Indexing?

Stores **summaries** in the vector store and **raw documents** in a docstore, linked by `doc_id`. During retrieval:
1. Summaries are matched via similarity search (they're more concise → better semantic matching).
2. The corresponding **full documents** are fetched from the docstore for generation (complete context for the LLM).

### Architecture Diagram

```text
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PHASE                       │
│                                                         │
│  Raw Doc ──► Summarize ──► Store Summary in Vectorstore │
│     │                                                   │
│     └──────────────────► Store Raw Doc in Docstore      │
│                          (linked by same doc_id)        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                      │
│                                                         │
│  User Query ──► Similarity Search on Summaries          │
│                        │                                │
│                        ▼                                │
│               Find matching doc_id                      │
│                        │                                │
│                        ▼                                │
│               Fetch Raw Doc from Docstore               │
│                        │                                │
│                        ▼                                │
│               Return Raw Doc to LLM                     │
└─────────────────────────────────────────────────────────┘
```

> **Why not just embed the full doc?** Long documents dilute their embedding — the vector tries to represent everything at once and ends up representing nothing well. Summaries are shorter, focused, and produce more precise embeddings.

---

## Cell 7 – Markdown: Detailed Explanation

Provides a prose explanation of the multi-representation indexing workflow with the architecture diagram.

---

## 📥 Load Source Documents (Cell 8 – Code)

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

| Import                          | Source                          | Role                                         |
| ------------------------------- | ------------------------------- | -------------------------------------------- |
| `WebBaseLoader`                 | `langchain_community`           | Fetches web page content as LangChain `Document` objects |
| `RecursiveCharacterTextSplitter`| `langchain_text_splitters`      | Splits text by characters with configurable chunk/overlap |

```python
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
```

Loads Lilian Weng's blog post on **LLM-powered agents** as a single `Document` object containing the full page text and metadata (URL, title, etc.).

```python
loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())
```

Loads the second blog post on **human data quality** and appends it to the `docs` list. Now `docs` contains **2 full Document objects**.

> **`extend()`** vs **`append()`**: `extend()` adds each element of the iterable individually, while `append()` would add the entire list as a single nested element. Since `loader.load()` returns a list, `extend()` is correct.

---

## Cell 9 – Markdown: Load Docs Docstring

Describes the document loading step above.

---

## 🔍 Inspect Metadata (Cell 10 – Code)

```python
docs[0].metadata
```

Displays the metadata dictionary of the first document (source URL, title, language, description). This is useful for verifying the documents loaded correctly.

> The `page_content` attribute (commented out) contains the full text. Metadata is the structured information about the document.

---

## Cell 11 – Markdown: Generate Summaries Docstring

Introduces the summarization step that creates concise representations for the vector store.

---

## 📝 Generate Summaries (Cell 12 – Code)

### Imports

```python
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
```

| Import               | Source                | Role                                                         |
| -------------------- | --------------------- | ------------------------------------------------------------ |
| `uuid`               | Python built-in       | Generates universally unique identifiers for document linking |
| `Document`           | `langchain_core`      | LangChain's document class with `page_content` + `metadata`  |
| `StrOutputParser`    | `langchain_core`      | Extracts the string content from an `AIMessage`               |
| `ChatPromptTemplate` | `langchain_core`      | Builds structured prompt templates                            |
| `ChatMistralAI`      | `langchain_mistralai` | Mistral AI chat model wrapper                                 |

### Summarization Chain

```python
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document: \n\n{doc}")
    | ChatMistralAI(model="mistral-medium-latest", max_retries=0)
    | StrOutputParser()
)
```

This LCEL chain processes one `Document` at a time:

| Step | Component | What it does |
| ---- | --------- | ------------ |
| 1 | `{"doc": lambda x: x.page_content}` | Extracts the raw text from the Document object and maps it to the `{doc}` template variable |
| 2 | `ChatPromptTemplate.from_template(...)` | Wraps the text in a summarization prompt |
| 3 | `ChatMistralAI(model="mistral-medium-latest")` | Sends the prompt to Mistral's medium model |
| 4 | `StrOutputParser()` | Extracts the summary text from the LLM response |

> **`max_retries=0`** disables automatic retries on API failure. This is useful during development to see errors immediately.

### Batch Processing

```python
summaries = chain.batch(docs, {"max_concurrency": 5})
```

| Parameter | Value | Effect |
| --------- | ----- | ------ |
| `docs` | List of 2 Documents | The inputs to process |
| `max_concurrency` | `5` | Up to 5 parallel API calls (only 2 docs here, so both run simultaneously) |

> **`.batch()`** processes multiple inputs through the chain. Each `Document` goes through the entire chain independently. This is much faster than calling `.invoke()` in a loop because API calls happen in parallel.

---

## Cell 13 – Markdown: MultiVectorRetriever Docstring

Describes the dual-store architecture that will be created next.

---

## 🏗️ Set Up MultiVectorRetriever (Cell 14 – Code)

### Imports

```python
from langchain_core.stores import InMemoryByteStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
```

| Import                  | Source               | Role                                                       |
| ----------------------- | -------------------- | ---------------------------------------------------------- |
| `InMemoryByteStore`     | `langchain_core`     | In-memory key-value store for raw documents                |
| `MistralAIEmbeddings`   | `langchain_mistralai`| Mistral's embedding model for vector similarity            |
| `Chroma`                | `langchain_chroma`   | ChromaDB vector store wrapper                              |
| `MultiVectorRetriever`  | `langchain_classic`  | Searches one store (summaries), retrieves from another (raw docs) |

> **`MultiVectorRetriever`** is the key component — it decouples what you search (summaries) from what you retrieve (raw docs). It's imported from `langchain_classic` because it was moved out of the main `langchain` package.

### Vector Store (Summaries)

```python
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=MistralAIEmbeddings(),
    persist_directory="./chroma_db",
)
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `collection_name` | `"summaries"` | Named collection within ChromaDB |
| `embedding_function` | `MistralAIEmbeddings()` | How to embed text for similarity search |
| `persist_directory` | `"./chroma_db"` | Save to disk so data survives restarts |

### Byte Store (Raw Documents)

```python
store = InMemoryByteStore()
id_key = "doc_id"
```

| Variable | Purpose |
| -------- | ------- |
| `store` | In-memory key-value store: `doc_id → raw Document` |
| `id_key` | The metadata field name used to link summaries to raw docs |

### The Retriever

```python
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
```

When you call `retriever.invoke(query)`:
1. The query is embedded and searched against the `vectorstore` (summaries)
2. The matching summary's `doc_id` metadata is extracted
3. The `doc_id` is used to look up the full raw document from the `byte_store`
4. The raw document is returned (not the summary)

### Populate Both Stores

```python
doc_ids = [str(uuid.uuid4()) for _ in docs]
```

Generates a unique UUID for each document. This ID links the summary in the vector store to the raw document in the byte store.

```python
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
```

Creates `Document` objects from the summaries, each tagged with the corresponding `doc_id`. These are what get embedded and stored in the vector store.

```python
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```

| Line | What it does |
| ---- | ------------ |
| `add_documents(summary_docs)` | Embeds the summaries and stores them in ChromaDB |
| `mset(list(zip(doc_ids, docs)))` | Stores the raw documents in the byte store, keyed by `doc_id` |

> **`mset()`** = "multi-set" — stores multiple key-value pairs at once. Each pair is `(doc_id, raw_document)`.

---

## Cell 15 – Markdown: Similarity Search Test Docstring

Introduces the direct similarity search test.

---

## 🔎 Test: Similarity Search on Summaries (Cell 16 – Code)

```python
query = "Memory in Agents"
sub_docs = vectorstore.similarity_search(query, k=1)
sub_docs
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `query` | `"Memory in Agents"` | The search query |
| `k` | `1` | Return only the top-1 most similar result |

**What this returns:** The most similar **summary** document (not the raw document). This tests that the summaries were properly indexed.

> This is a direct search on the vector store. The `MultiVectorRetriever` is not involved here — we're just verifying the index works.

---

## Cell 17 – Markdown: Full Retrieval Test Docstring

Introduces the full end-to-end retrieval test.

---

## 📄 Test: Full Retrieval via MultiVectorRetriever (Cell 18 – Code)

```python
retrieved_docs = retriever.invoke(query, n_results=1)
retrieved_docs[0].page_content[0:500]
```

| Line | What it does |
| ---- | ------------ |
| `retriever.invoke(query, n_results=1)` | Searches summaries → finds `doc_id` → fetches raw doc from byte store |
| `[0].page_content[0:500]` | Previews the first 500 characters of the raw document |

**Key difference from Cell 16:** Cell 16 returned the **summary**. This cell returns the **full raw document**. That's the whole point of Multi-Representation Indexing — you search concise summaries but generate from complete documents.

**Output:** First 500 characters of the Lilian Weng blog post on LLM-powered agents.

---

# Part 13: RAPTOR

---

## Cell 19 – Markdown: RAPTOR Introduction

### What is RAPTOR?

**RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a hierarchical tree of document summaries at multiple levels of abstraction. From the [RAPTOR paper](https://arxiv.org/pdf/2401.18059):

1. Start with **leaf documents** (raw text chunks or full docs).
2. **Embed** all leaves and **cluster** them using UMAP + GMM.
3. **Summarize** each cluster into a higher-level consolidation.
4. **Repeat**: treat the summaries as new "documents" and cluster/summarize again.
5. The result is a **tree** from raw docs (leaves) → mid-level summaries → high-level summaries.

```text
Level 3:   [   Root Summary   ]
               /       \
Level 2:  [Summary A]  [Summary B]
           / | \          / \
Level 1: [d1][d2][d3]  [d4][d5]     ← Original document chunks
```

**Why a tree?** Different questions require different levels of detail:
- "What is LCEL?" → High-level summary answers this well.
- "How do I use RunnablePassthrough?" → Needs a specific leaf-level chunk.

**Collapsed tree retrieval** (used in this notebook) flattens all levels into a single index and does kNN search across everything — the retriever naturally picks the right level of detail for each query.

---

## Cell 20 – Markdown: LCEL Docs Introduction

Short description: applying RAPTOR to LangChain's LCEL documentation. Each doc is a unique web page with context ranging from < 2k to > 10k tokens.

---

## 🌐 Load LCEL Documentation + Token Histogram (Cell 21 – Code)

### Helper Function

```python
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
```

Takes a string and a tiktoken encoding name, returns the token count. Uses `cl100k_base` encoding (the tokenizer for GPT-4 and `text-embedding-ada-002`).

> **Identical function** to the one in Part 2 (part_1_4.ipynb). Repeated here for self-containment.

### Document Loading

```python
url = "https://python.langchain.com/docs/expression_language/"
loader_1 = RecursiveUrlLoader(
    url=url,
    max_depth=20,
    extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader_1.load()
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `url` | LCEL docs root URL | Starting point for recursive crawl |
| `max_depth` | `20` | Crawl up to 20 levels deep to capture all subpages |
| `extractor` | `lambda x: Soup(x, "html.parser").text` | Strip HTML tags, keep plain text |

> **`RecursiveUrlLoader`** follows links on the page and recursively loads child pages, unlike `WebBaseLoader` which only loads the single URL.

Two additional pages are loaded separately (PydanticOutputParser and Self-Query retriever) because they're outside the main LCEL URL tree:

```python
loader_2 = RecursiveUrlLoader(url=..., max_depth=1, ...)  # PydanticOutputParser
loader_3 = RecursiveUrlLoader(url=..., max_depth=1, ...)  # Self-Query retriever
docs.extend([*docs_pydantic, *docs_sq])
```

### Token Distribution Visualization

```python
counts = [num_tokens_from_string(text, "cl100k_base") for text in docs_texts]
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
```

Plots a histogram showing how many tokens each page contains. This helps decide chunk sizes for the text splitter.

---

## 📜 Concatenate Documents (Cell 22 – Code)

```python
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])
```

| Line | What it does |
| ---- | ------------ |
| `sorted(docs, key=...)` | Sort documents alphabetically by source URL |
| `reversed(d_sorted)` | Reverse the sort order |
| `"\n\n\n --- \n\n\n".join(...)` | Concatenate all texts with visual separators between documents |

```python
print("Num of Tokens in All Context: %s" % num_tokens_from_string(concatenated_content, "cl100k_base"))
```

Reports the total token count for the entire concatenated corpus. This determines whether the corpus fits within a single LLM context window.

---

## ✂️ Split Into Chunks (Cell 23 – Code)

```python
chunk_size_toks = 2000
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_toks, chunk_overlap=0
)
texts_split = text_splitter.split_text(concatenated_content)
```

| Parameter | Value | Comparison to Earlier Notebooks |
| --------- | ----- | ------------------------------ |
| `chunk_size` | `2000` | Part 1–4 and Part 5–9 used `300` |
| `chunk_overlap` | `0` | Parts 1–4 and 5–9 used `50` overlap |

> **Why larger chunks with no overlap for RAPTOR?** RAPTOR relies on clustering to find relationships between chunks rather than overlapping text windows. Larger chunks preserve more context per leaf node, and the clustering algorithm handles coherence across boundaries.

> **Same splitter class** as Parts 1–4 and 5–9 (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`), just different parameters.

---

## Cell 24 – Markdown: Models Introduction

Describes the models available for RAPTOR. This notebook uses HuggingFace BGE for embeddings and Google Gemini (via OpenRouter) for generation.

---

## 🤖 Model Setup: Embeddings + LLM (Cell 25 – Code)

### Imports

```python
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
```

| Import                    | Source | Role |
| ------------------------- | ------ | ---- |
| `MistralAIEmbeddings`     | `langchain_mistralai` | Mistral embedding model (imported but not used here) |
| `ChatMistralAI`           | `langchain_mistralai` | Mistral chat model (imported but not used here) |
| `HuggingFaceBgeEmbeddings`| `langchain_community` | Free, open-source multilingual embedding model |
| `ChatOpenAI`              | `langchain_openai` | OpenAI-compatible chat wrapper (used with OpenRouter) |

### Embedding Model

```python
embd = HuggingFaceBgeEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)
```

| Setting | Value | Why |
| ------- | ----- | --- |
| Model | `intfloat/multilingual-e5-large` | Free to use, strong multilingual support, 1024-dim output |

> **Why not MistralAIEmbeddings?** This notebook uses a free local model to avoid API costs during the many embedding calls RAPTOR requires (embedding every leaf + every cluster summary at every level).

### LLM

```python
model = ChatOpenAI(
    model="google/gemini-3.1-flash-lite-preview",
    openai_api_key=os.environ["OPEN_ROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "RAG Tutorial",
    }
)
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `model` | `google/gemini-3.1-flash-lite-preview` | Google Gemini with 1M context window |
| `openai_api_key` | OpenRouter API key | Authentication (not an OpenAI key) |
| `openai_api_base` | `https://openrouter.ai/api/v1` | Routes to OpenRouter instead of OpenAI |
| `HTTP-Referer` | `http://localhost:3000` | Required by some OpenRouter models |
| `X-Title` | `"RAG Tutorial"` | Optional identification header |

> **OpenRouter** provides a unified API (OpenAI-compatible) for accessing models from many providers (Google, Anthropic, Meta, etc.). Using `ChatOpenAI` with a custom base URL is the standard pattern. This replaces the `ChatMistralAI` used in Parts 1–12.

---

## ⚠️ Duplicate Cell (Cell 26 – Code)

This cell is an **exact duplicate** of Cell 25's model initialization. It exists from iterative development/testing and can be safely removed.

---

## Cell 27 – Markdown: Tree Construction Introduction

Describes the algorithms used in RAPTOR's tree construction:

| Algorithm | Role |
| --------- | ---- |
| **GMM** (Gaussian Mixture Model) | Clusters data points; finds optimal cluster count via BIC |
| **UMAP** (Uniform Manifold Approximation and Projection) | Reduces high-dimensional embeddings to lower dimensions for clustering |
| **Local + Global Clustering** | Captures both fine-grained and broad patterns |
| **Thresholding** | Assigns data points to ≥ 1 cluster based on probability |

---

## 🌳 RAPTOR Functions: Clustering + Summarization Pipeline (Cell 28 – Code)

This is the largest cell in the notebook — it defines the entire RAPTOR pipeline. Let's walk through each section.

### Imports

```python
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import umap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
```

| Import | Role |
| ------ | ---- |
| `numpy` | Numerical operations on embedding arrays |
| `pandas` | DataFrame-based data management for clusters |
| `umap` | Dimensionality reduction (UMAP algorithm) |
| `GaussianMixture` | Probabilistic clustering via GMM |

### Global Seed

```python
RANDOM_SEED = 224
```

Fixed seed for reproducibility across all clustering operations.

### 1. UMAP Dimensionality Reduction

#### `global_cluster_embeddings(embeddings, dim, n_neighbors, metric)`

Reduces high-dimensional embeddings to `dim` dimensions using UMAP at the **global** level (all data at once).

```python
if n_neighbors is None:
    n_neighbors = int((len(embeddings) - 1) ** 0.5)  # sqrt(n-1) heuristic
return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
```

> The `sqrt(n-1)` heuristic for `n_neighbors` balances local and global structure preservation. Too few neighbors → overfitting to local noise. Too many → losing fine-grained structure.

#### `local_cluster_embeddings(embeddings, dim, num_neighbors, metric)`

Same as global, but with a fixed `num_neighbors=10` — used for fine-grained clustering **within** each global cluster.

### 2. GMM Clustering

#### `get_optimal_clusters(embeddings, max_clusters, random_state)`

Finds the optimal number of clusters by fitting GMMs with 1 to `max_clusters` components and selecting the one with the **lowest BIC** (Bayesian Information Criterion).

```python
max_clusters = min(max_clusters, len(embeddings))  # Can't have more clusters than data points
n_clusters = np.arange(1, max_clusters)
bics = []
for n in n_clusters:
    gm = GaussianMixture(n_components=n, random_state=random_state)
    gm.fit(embeddings)
    bics.append(gm.bic(embeddings))
return n_clusters[np.argmin(bics)]
```

> **BIC** penalizes model complexity. Lower BIC = better balance between fit quality and number of parameters. This avoids overfitting by choosing the simplest model that explains the data well.

#### `GMM_cluster(embeddings, threshold, random_state)`

Clusters embeddings using the optimal number of clusters from above, then assigns each point to **all clusters where its probability exceeds `threshold`**.

```python
probs = gm.predict_proba(embeddings)
labels = [np.where(prob > threshold)[0] for prob in probs]
```

> **Soft clustering**: Unlike k-means where each point belongs to exactly one cluster, GMM with thresholding allows **overlapping membership**. A document about "Python data analysis" might belong to both a "Python" cluster and a "data science" cluster.

### 3. Main Clustering Pipeline

#### `perform_clustering(embeddings, dim, threshold)`

Orchestrates the full 3-step clustering:

1. **Global UMAP** → reduce dimensionality across all embeddings.
2. **Global GMM** → find broad clusters.
3. **Local UMAP + GMM** → refine clusters within each global cluster.

```python
if len(embeddings) <= dim + 1:
    return [np.array([0]) for _ in range(len(embeddings))]
```

> Edge case: if there are too few embeddings for UMAP (need at least `dim + 2` points), assign everything to cluster 0.

The local clustering loop:
```python
for i in range(n_global_clusters):
    global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]
    # ... local UMAP + GMM within this global cluster ...
    for idx in indices:
        all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)
```

> Local cluster IDs are offset by `total_clusters` to ensure globally unique IDs across all global clusters.

### 4. RAPTOR Helper Functions

#### `embed(texts)`

Wraps the global `embd` model (HuggingFace BGE) to embed a list of texts. Returns a numpy array.

#### `embed_cluster_texts(texts)`

Embeds texts and clusters them in one step:
```python
text_embeddings_np = embed(texts)
cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
```

Parameters: `dim=10` (reduce to 10 dimensions for clustering), `threshold=0.1` (10% probability threshold for cluster membership).

Returns a DataFrame with columns: `text`, `embd`, `cluster`.

#### `fmt_txt(df)`

Joins all texts in a DataFrame cluster into a single string separated by `"--- --- \n --- --- "`. Used to feed cluster content to the summarization LLM.

#### `embed_cluster_summarize_texts(texts, level)`

One complete iteration of RAPTOR tree-building:

1. Embed and cluster all texts.
2. Expand multi-cluster assignments (one row per text-cluster pair).
3. For each cluster: concatenate all member texts → summarize with LLM.
4. Return two DataFrames: `df_clusters` (texts + embeddings + clusters) and `df_summary` (one summary per cluster).

```python
template = """Here is a sub-set of LangChain Expression Language doc. 
Give a detailed summary of the documentation provided.
Documentation:
{context}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()
```

> The summarization chain uses the global `model` (Gemini via OpenRouter), not Mistral.

#### `recursive_embed_cluster_summarize(texts, level, n_levels)`

The recursive entry point — calls `embed_cluster_summarize_texts` at each level, then feeds the resulting summaries as input to the next level:

```python
results = {}
df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
results[level] = (df_clusters, df_summary)
if level < n_levels and unique_clusters > 1:
    new_texts = df_summary["summaries"].tolist()
    next_level_results = recursive_embed_cluster_summarize(new_texts, level + 1, n_levels)
    results.update(next_level_results)
return results
```

**Recursion stops when:**
- Maximum depth (`n_levels`) is reached, OR
- Only 1 cluster remains (the tree has converged to a single root summary).

---

## 🔨 Build the RAPTOR Tree (Cell 29 – Code)

```python
leaf_texts = docs_texts
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
```

Kicks off the recursive tree-building process:
- **Level 1**: Embed and cluster the original document texts → produce summaries.
- **Level 2**: Embed and cluster Level 1 summaries → produce higher-level summaries.
- **Level 3**: Embed and cluster Level 2 summaries → produce root-level summaries.

`results` is a dict: `{1: (df_clusters_1, df_summary_1), 2: (...), 3: (...)}`.

---

## Cell 30 – Markdown: Collapsed Tree Retrieval Introduction

Explains that the RAPTOR paper reports best performance from **collapsed tree retrieval** — flattening the tree into a single layer and performing kNN search across all nodes simultaneously.

---

## 🌲 Collapsed Tree Retrieval (Cell 31 – Code)

```python
all_texts = leaf_texts.copy()

for level in sorted(results.keys()):
    summaries = results[level][1]["summaries"].tolist()
    all_texts.extend(summaries)
```

| Line | What it does |
| ---- | ------------ |
| `leaf_texts.copy()` | Start with the original document chunks |
| `results[level][1]` | Access the summary DataFrame for this level |
| `["summaries"].tolist()` | Extract the summary texts as a list |
| `all_texts.extend(summaries)` | Add summaries from every level to the flat list |

The result: `all_texts` contains leaf chunks + Level 1 summaries + Level 2 summaries + Level 3 summaries — the entire tree flattened.

```python
vectorstore = Chroma.from_texts(
    texts=all_texts,
    collection_name="all_texts_collection",
    embedding=embd,
    persist_directory="./chroma_all_texts_db",
)
retriever_ = vectorstore.as_retriever()
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `texts` | `all_texts` | All tree nodes (every level) |
| `collection_name` | `"all_texts_collection"` | ChromaDB collection name |
| `embedding` | `embd` | HuggingFace BGE model |
| `persist_directory` | `"./chroma_all_texts_db"` | On-disk persistence |

> **`Chroma.from_texts()`** (class method) creates a new collection and immediately adds all texts — different from `Chroma()` (constructor) used in Part 12 which creates an empty collection.

---

## Cell 32 – Markdown: RAG Chain Introduction

Short bridge text: "Now we can use our flattened, indexed tree in a RAG chain."

---

## 🔗 Final RAG Chain with RAPTOR Retriever (Cell 33 – Code)

```python
from langsmith import Client
from langchain_core.runnables import RunnablePassthrough

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
```

Pulls the community RAG prompt template from LangSmith Hub — same prompt used in Part 4 (part_1_4.ipynb).

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

> **Identical helper** to the one in Parts 1–4 and Parts 5–9. Joins retrieved documents into a single context string.

```python
rag_chain = (
    {
        "context": retriever_ | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)
```

The canonical LCEL RAG pattern:

| Step | Component | What it does |
| ---- | --------- | ------------ |
| 1 | `retriever_ \| format_docs` | Retrieve from the flattened RAPTOR tree → format as string |
| 2 | `RunnablePassthrough()` | Pass the question through unchanged |
| 3 | `prompt` | Fill the `rlm/rag-prompt` template with context + question |
| 4 | `model` | Generate answer (Gemini via OpenRouter) |
| 5 | `StrOutputParser()` | Extract the string response |

> **Same chain structure** as Part 4 (part_1_4.ipynb) and the RAG chains in Parts 5–9. The only differences are the retriever (RAPTOR instead of basic Chroma) and the model (Gemini instead of Mistral).

```python
rag_chain.invoke("How to define a RAG chain? Give me a specific code example.")
```

Test query asking for a code-specific answer from the LCEL documentation.

---

# Part 14: ColBERT

---

## Cell 34 – Markdown: ColBERT Introduction

### What is ColBERT?

**ColBERT** (Contextualized Late Interaction over BERT) is a retrieval model that generates **per-token embeddings** instead of a single vector per document.

**How it works:**

1. **Indexing**: Each passage is split into tokens, and a contextually influenced vector is generated for **each token** (not one vector for the whole passage).
2. **Querying**: Each query token also gets its own vector.
3. **Scoring**: The score for a passage = sum of the **maximum similarity** between each query token embedding and any passage token embedding.

```text
Query:  "What studio did Miyazaki found?"
         ↓      ↓     ↓       ↓     ↓
        [q1]   [q2]   [q3]   [q4]  [q5]    ← Per-token query embeddings

Passage: "Miyazaki founded Studio Ghibli in 1985"
          ↓         ↓       ↓      ↓     ↓    ↓
         [d1]      [d2]    [d3]   [d4]  [d5]  [d6]  ← Per-token doc embeddings

Score = MaxSim(q1, d1..d6) + MaxSim(q2, d1..d6) + ... + MaxSim(q5, d1..d6)
```

> **Why per-token?** Traditional dense retrieval (one vector per doc) loses fine-grained token-level semantics. ColBERT's late interaction preserves token-level matching while still being efficient (the heavy BERT computation happens at indexing time, not query time).

**RAGatouille** is a Python library that wraps ColBERT and makes it easy to use with just a few lines of code.

---

## Cell 35 – Markdown: RAGatouille Compatibility Fix

Detailed explanation of the monkey-patching needed to use RAGatouille with LangChain v0.3+:

1. **Problem**: RAGatouille imports from `langchain.retrievers` which was moved to `langchain_core` in v0.3.
2. **Solution**: Module aliasing via `sys.modules` — intercept the old import path and redirect to the new location.
3. **Additional fix**: The `mark_tied_weights_as_initialized` method in newer `transformers` expects `all_tied_weights_keys` which may not exist on all models.

---

## 🔧 Load ColBERT via RAGatouille (Cell 36 – Code)

```python
from transformers import PreTrainedModel
_original_mark_tied_weights = PreTrainedModel.mark_tied_weights_as_initialized
```

Saves the original HuggingFace method before patching.

```python
def _patched_mark_tied_weights(self, *args, **kwargs):
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
    return _original_mark_tied_weights(self, *args, **kwargs)
```

Patches the method to safely inject the missing `all_tied_weights_keys` attribute. If it already exists, the original method runs unchanged.

```python
PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights
```

Applies the patch globally so all model loads go through the safe version.

```python
from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
```

| Line | What it does |
| ---- | ------------ |
| `RAGPretrainedModel.from_pretrained(...)` | Downloads and initializes ColBERTv2 from HuggingFace Hub |
| `"colbert-ir/colbertv2.0"` | The official ColBERTv2 model checkpoint |

> **ColBERTv2** improves on ColBERTv1 with better compression and training. It's the state-of-the-art for late-interaction retrieval models.

---

## 📖 Fetch Wikipedia Document (Cell 37 – Code)

```python
def get_wikipedia_page(title: str):
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None
```

| Parameter | Purpose |
| --------- | ------- |
| `action=query` | Use the query API action |
| `format=json` | Return JSON response |
| `titles=title` | Which Wikipedia page to fetch |
| `prop=extracts` | Request page content extracts |
| `explaintext=True` | Return plain text (no HTML markup) |
| `User-Agent` | Required by Wikipedia API policy to identify callers |

> **`next(iter(data["query"]["pages"].values()))`** — The Wikipedia API returns pages as a dict keyed by page ID (e.g., `{"12345": {...}}`). Since we requested one page, we take the first (and only) value.

```python
full_document = get_wikipedia_page("Hayao_Miyazaki")
```

Fetches the complete Wikipedia article on Hayao Miyazaki — a substantial document to test ColBERT's per-token indexing and retrieval.

---

## 📦 Index with ColBERT (Cell 38 – Code)

```python
RAG.index(
    Collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True
)
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `Collection` | `[full_document]` | List of documents to index |
| `index_name` | `"Miyazaki-123"` | Name for the on-disk index directory |
| `max_document_length` | `180` | Maximum tokens per passage (ColBERT splits internally) |
| `split_documents` | `True` | Auto-split long documents into passages |

**What happens under the hood:**
1. The document is split into passages of ≤ 180 tokens.
2. Each passage is tokenized with ColBERT's BERT tokenizer.
3. A contextual embedding is generated for **every token** in every passage.
4. The embeddings are compressed and stored in an on-disk index.

> Unlike vector stores like Chroma where you store one vector per chunk, ColBERT stores **hundreds of vectors per passage** (one per token). This is more expensive to store but enables much more precise retrieval.

---

## 🔍 ColBERT Search (Cell 39 – Code)

```python
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
results
```

| Parameter | Value | Purpose |
| --------- | ----- | ------- |
| `query` | `"What animation studio did Miyazaki found?"` | The search query |
| `k` | `3` | Return top-3 passages |

**Output:** A list of dictionaries, each containing:
- `content` — The matched passage text
- `score` — ColBERT MaxSim score (sum of max similarities per query token)
- `rank` — Position in the result list (1, 2, 3)
- `document_id` — Internal document identifier

> **MaxSim scoring**: For each query token, find the document token with the highest cosine similarity. Sum all these maximum similarities to get the passage score. This captures fine-grained token-level matching.

---

## 🔗 ColBERT as LangChain Retriever (Cell 40 – Code)

```python
retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")
```

Wraps the ColBERT model as a LangChain-compatible retriever, returning standard `Document` objects with `page_content` and `metadata`.

> **Same `.as_retriever()` pattern** used throughout the series — Chroma in Parts 1–4 and 5–9, the flattened RAPTOR tree above, and now ColBERT. LangChain's retriever abstraction means you can swap ColBERT into any existing RAG chain.

This retriever can be plugged into the same LCEL RAG chain pattern used in every notebook:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
```

---

# 🔄 Cross-Notebook Similarities

Throughout this notebook, several patterns are reused from earlier parts:

| Pattern | Where it appears | Notes |
| ------- | --------------- | ----- |
| `num_tokens_from_string()` | Part 2 (part_1_4.ipynb), Part 13 (this notebook) | Identical function |
| `format_docs()` | Parts 1–4, 5–9, 13 (this notebook) | Identical function |
| `RecursiveCharacterTextSplitter.from_tiktoken_encoder` | Parts 1–4, 5–9, 13 | Same class, different params (2000/0 vs 300/50) |
| LCEL RAG chain pattern | Parts 1–4, 5–9, 13 | `retriever \| format_docs → prompt → LLM → StrOutputParser` |
| Environment setup cell | All 5 notebooks | Same structure, this one adds OpenRouter + Google keys |
| `.as_retriever()` pattern | Parts 1–4, 5–9, 12, 13, 14 | Chroma, MultiVector, RAPTOR tree, ColBERT |

---

# 📊 Summary

| Part | Technique | Key Idea | Trade-off |
| ---- | --------- | -------- | --------- |
| **12** | Multi-Representation Indexing | Search summaries, retrieve raw docs | Better retrieval precision, but requires pre-summarization |
| **13** | RAPTOR | Recursive hierarchical tree of summaries | Multi-granularity retrieval, but expensive to build |
| **14** | ColBERT | Per-token late-interaction scoring | Best precision for token-level matching, but larger index size |

All three techniques are **complementary** — they can be combined in a production system:
- Use **ColBERT** for passages requiring precise token-level matching.
- Use **RAPTOR** for navigating large document collections at varying abstraction levels.
- Use **Multi-Representation Indexing** when retrieval of full documents (not chunks) is needed.
