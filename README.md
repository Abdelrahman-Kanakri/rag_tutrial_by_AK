# RAG From Scratch

A hands-on tutorial series that builds **Retrieval-Augmented Generation (RAG)** systems from the ground up using [LangChain](https://www.langchain.com/), progressing from a basic pipeline to advanced techniques like RAPTOR, ColBERT, and query routing.

Each notebook is **self-contained** with:
- Markdown documentation explaining every concept before the code
- Inline code comments on every line
- A companion **walkthrough** (`.md`) with line-by-line explanations, import tables, and architecture diagrams

---

## Table of Contents

| Part | Topic | Notebook | Walkthrough |
| ---- | ----- | -------- | ----------- |
| **1** | Overview of RAG | [part_1_4.ipynb](src/part_1_4/part_1_4.ipynb) | [walkthrough](src/part_1_4/part_1_4_walkthrough.md) |
| **2** | Indexing (Embeddings + Similarity) | ↑ same notebook | ↑ |
| **3** | Retrieval | ↑ same notebook | ↑ |
| **4** | Generation | ↑ same notebook | ↑ |
| **5** | Multi-Query | [part_5_9.ipynb](src/part_5_9/part_5_9.ipynb) | [walkthrough](src/part_5_9/part_5_9_walkthrough.md) |
| **6** | RAG Fusion (RRF) | ↑ same notebook | ↑ |
| **7** | Decomposition | ↑ same notebook | ↑ |
| **8** | Step-Back Prompting | ↑ same notebook | ↑ |
| **9** | HyDE | ↑ same notebook | ↑ |
| **10** | Logical & Semantic Routing | [part_10.ipynb](src/part_10_and_11/part_10.ipynb) | [walkthrough](src/part_10_and_11/part_10_walkthrough.md) |
| **11** | Query Construction | [part_11.ipynb](src/part_10_and_11/part_11.ipynb) | [walkthrough](src/part_10_and_11/part_11_walkthrough.md) |
| **12** | Multi-Representation Indexing | [part_12_13_14.ipynb](src/part_12_13_14/part_12_13_14.ipynb) | [walkthrough](src/part_12_13_14/part_12_13_14_walkthrough.md) |
| **13** | RAPTOR | ↑ same notebook | ↑ |
| **14** | ColBERT | ↑ same notebook | ↑ |

---

## Recommended Learning Path

```text
Start Here
    │
    ▼
┌──────────────────────────────────────────────┐
│  Parts 1–4: The Foundation                   │
│  Build a complete RAG pipeline from scratch  │
│  (Indexing → Retrieval → Generation)         │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Parts 5–9: Query Translation                │
│  Transform user queries to improve retrieval │
│  (Multi-Query, RAG Fusion, Decomposition,    │
│   Step-Back Prompting, HyDE)                 │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Part 10: Routing                            │
│  Direct queries to the right data source     │
│  (Logical Routing, Semantic Routing)         │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Part 11: Query Construction                 │
│  Convert natural language to structured      │
│  metadata filters                            │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Parts 12–14: Advanced Indexing              │
│  Multi-Representation, RAPTOR, ColBERT       │
│  (Can be explored in any order)              │
└──────────────────────────────────────────────┘
```

> **Parts 1–4 are essential** — every later notebook builds on the concepts and code patterns established there. Parts 5–14 can be explored more flexibly, though they reference earlier material.

---

## Project Structure

```
rag_tutorial/
├── src/                          # Jupyter notebooks (the tutorials)
│   ├── part_1_4/                 #   Parts 1–4: Full RAG pipeline
│   │   ├── part_1_4.ipynb
│   │   └── part_1_4_walkthrough.md
│   ├── part_5_9/                 #   Parts 5–9: Query translation techniques
│   │   ├── part_5_9.ipynb
│   │   └── part_5_9_walkthrough.md
│   ├── part_10_and_11/           #   Parts 10–11: Routing & query construction
│   │   ├── part_10.ipynb
│   │   ├── part_10_walkthrough.md
│   │   ├── part_11.ipynb
│   │   └── part_11_walkthrough.md
│   └── part_12_13_14/            #   Parts 12–14: Advanced indexing
│       ├── part_12_13_14.ipynb
│       └── part_12_13_14_walkthrough.md
├── assets/                       # Diagrams and images referenced by notebooks
├── docs/                         # PDF reference guides (by RAG pipeline stage)
│   ├── 0 Pipeline Building/
│   ├── 1 Basic Indexing & Advanced Indexing/
│   ├── 2 Retrieval/
│   ├── 3 Generation/
│   ├── 4 Query Translation/
│   ├── 5 Routing/
│   ├── 6 Query Constructing/
│   └── 7, 8 Agentic Rag (Advanced Retrieval)/
├── pyproject.toml                # Project metadata & dependencies (uv)
├── requirements.txt              # Pip-compatible dependency lock
├── SETUP_GUIDE.md                # uv package manager setup guide
└── .gitignore
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip
- API keys for:
  - [Mistral AI](https://console.mistral.ai/) — LLM & embeddings (Parts 1–12)
  - [LangSmith](https://smith.langchain.com/) — tracing & observability
  - [HuggingFace](https://huggingface.co/settings/tokens) — model access (Part 13)
  - [OpenRouter](https://openrouter.ai/) — Gemini access for RAPTOR (Part 13)

### 1. Clone the Repository

```bash
git clone https://github.com/Abdelrahman-Kanakri/rag_tutrial_by_AK.git
cd rag_tutrial_by_AK
```

### 2. Install Dependencies

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> See [SETUP_GUIDE.md](SETUP_GUIDE.md) for a detailed guide on using `uv`.

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# LangSmith (tracing & observability)
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-tutorial
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Mistral AI (LLM + Embeddings — Parts 1–12)
MISTRAL_API_KEY=your_mistral_api_key

# HuggingFace (BGE embeddings — Part 13)
HF_TOKEN=your_huggingface_token

# OpenRouter (Gemini access — Part 13 RAPTOR)
OPEN_ROUTER_API_KEY=your_openrouter_api_key

# Google (optional, if using Google AI directly)
GOOGLE_API_KEY=your_google_api_key
```

### 4. Open and Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or open in VS Code (recommended) — just open any .ipynb file
```

Start with **[Part 1–4](src/part_1_4/part_1_4.ipynb)** and run cells top to bottom.

---

## What You'll Learn

### The Full RAG Pipeline (Parts 1–4)

| Part | Topic | What You'll Build |
| ---- | ----- | ----------------- |
| 1 | Overview | Understand what RAG is and why it matters |
| 2 | Indexing | Split documents → embed with Mistral → store in ChromaDB |
| 3 | Retrieval | Similarity search to find relevant documents for a query |
| 4 | Generation | Feed retrieved context + question to an LLM for an answer |

### Query Translation Techniques (Parts 5–9)

| Part | Technique | Key Idea |
| ---- | --------- | -------- |
| 5 | Multi-Query | Generate multiple reformulations of the query → retrieve for each → union results |
| 6 | RAG Fusion | Multi-query + Reciprocal Rank Fusion (RRF) to re-rank |
| 7 | Decomposition | Break complex questions into sub-questions → answer recursively or individually |
| 8 | Step-Back Prompting | Ask a more abstract version of the question first |
| 9 | HyDE | Generate a hypothetical answer → use it as the search query |

### Routing & Query Construction (Parts 10–11)

| Part | Technique | Key Idea |
| ---- | --------- | -------- |
| 10 | Logical Routing | LLM classifies query → routes to the right datasource |
| 10 | Semantic Routing | Cosine similarity between query and prompt embeddings |
| 11 | Query Construction | Convert natural language to structured metadata filters (Pydantic) |

### Advanced Indexing (Parts 12–14)

| Part | Technique | Key Idea |
| ---- | --------- | -------- |
| 12 | Multi-Representation | Search summaries, retrieve full documents |
| 13 | RAPTOR | Build a hierarchical tree of summaries (UMAP + GMM clustering) |
| 14 | ColBERT | Per-token late-interaction retrieval (via RAGatouille) |

---

## Key Technologies

| Technology | Role | Used In |
| ---------- | ---- | ------- |
| [LangChain](https://python.langchain.com/) | RAG framework — chains, prompts, retrievers | All parts |
| [ChromaDB](https://www.trychroma.com/) | Vector store for embeddings | Parts 1–4, 5–9, 12–13 |
| [Mistral AI](https://mistral.ai/) | LLM (`mistral-medium-latest`) & embeddings | Parts 1–12 |
| [HuggingFace BGE](https://huggingface.co/intfloat/multilingual-e5-large) | Free multilingual embeddings | Part 13 |
| [OpenRouter](https://openrouter.ai/) / Gemini | LLM with 1M context window | Part 13 |
| [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) / ColBERT | Per-token retrieval model | Part 14 |
| [LangSmith](https://smith.langchain.com/) | Tracing & observability | All parts |
| [UMAP](https://umap-learn.readthedocs.io/) + [scikit-learn](https://scikit-learn.org/) | Dimensionality reduction & GMM clustering | Part 13 |
| [Pydantic](https://docs.pydantic.dev/) | Structured output & data validation | Parts 10–11 |

---

## Reference Materials

The `docs/` folder contains PDF guides organized by RAG pipeline stage:

| Stage | PDF Guide |
| ----- | --------- |
| Pipeline Architecture | `docs/0 Pipeline Building/` |
| Indexing (Basic + Advanced) | `docs/1 Basic Indexing & Advanced Indexing/` |
| Retrieval | `docs/2 Retrieval/` |
| Generation | `docs/3 Generation/` |
| Query Translation | `docs/4 Query Translation/` |
| Routing | `docs/5 Routing/` |
| Query Construction | `docs/6 Query Constructing/` |
| Agentic RAG | `docs/7, 8 Agentic Rag (Advanced Retrieval)/` |

---

## License

This project is for educational purposes. Built following the [RAG From Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) series by Lance Martin / LangChain.