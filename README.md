# think-insights-rag-poc

A local, offline Retrieval-Augmented Generation (RAG) proof of concept. Ingests a corpus of text, embeds it into a persistent vector store, and answers questions against it using a locally-hosted Llama 3 model — with streaming responses and source citations.

No external API calls. Everything runs on your machine.

## Features

- **Local LLM inference** via `llama-cpp-python` (Meta Llama 3 8B Instruct, GGUF-quantized)
- **Persistent vector store** using ChromaDB with `all-MiniLM-L6-v2` embeddings from HuggingFace
- **Source citations** — shows which chunks were retrieved before generating an answer
- **Streaming responses** via LangChain's `rag_chain.stream()`
- **Interactive REPL** for asking questions in a loop
- **Configurable via environment variables** — no code edits needed to point at your own data/model

## Tech stack

- Python 3.10+
- [LangChain](https://www.langchain.com/) (core, community, chroma, huggingface, text-splitters)
- [ChromaDB](https://www.trychroma.com/) for the vector store
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for local inference

## Getting started

### 1. Prerequisites

- Python 3.10+
- A GGUF-format LLM (this repo defaults to Llama 3 8B Instruct Q4_K_M — download from [Hugging Face](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF))
- A plain-text corpus to ingest

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure paths

The pipeline reads three paths from environment variables, with sensible defaults:

| Variable              | Default                                                    |
| --------------------- | ---------------------------------------------------------- |
| `CLEANED_DATA_FILE`   | `./data/cleaned_data.txt`                                  |
| `MODEL_FILE_PATH`     | `./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`            |
| `CHROMA_DIRECTORY`    | `./chroma_db`                                              |
| `MODEL_N_CTX`         | `2048`                                                     |
| `MODEL_TEMP`          | `0.3`                                                      |

Either drop your files at the default paths, or export overrides:

```bash
export CLEANED_DATA_FILE=/path/to/your/corpus.txt
export MODEL_FILE_PATH=/path/to/your/model.gguf
```

### 4. Ingest

```bash
python ingest_data.py
```

This chunks the text (1000 chars, 200 overlap), embeds it with `all-MiniLM-L6-v2`, and persists the vector store to `CHROMA_DIRECTORY`. The script wipes the old store on each run so you can re-ingest idempotently.

### 5. Query

```bash
python query_data.py
```

You'll get an interactive prompt. For each question, the script:

1. Retrieves the top 3 chunks from Chroma and prints their sources
2. Builds a RAG prompt with the retrieved context
3. Streams the model's answer token-by-token

Type `exit`, `quit`, or `q` to leave.

## Project structure

```
.
├── config.py          # Env-var-driven paths and model params
├── ingest_data.py     # Chunk → embed → persist to Chroma
├── query_data.py      # Retrieve → prompt → stream answer
├── requirements.txt
└── README.md
```

## How it works

**Ingestion** (`ingest_data.py`):
`RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings(all-MiniLM-L6-v2)` → `Chroma.from_documents(persist_directory=...)`.

**Querying** (`query_data.py`):
`Chroma(as_retriever, k=3)` → `PromptTemplate` → `LlamaCpp(n_gpu_layers=0)` → `StrOutputParser`, wired into a LangChain Runnable chain and consumed with `.stream()`.

## Notes

- The vector store (`chroma_db/`) and model weights (`*.gguf`) are gitignored — regenerate the store with `ingest_data.py` after cloning.
- `n_gpu_layers=0` means CPU-only inference; increase this if you have a compatible GPU build of `llama-cpp-python`.
