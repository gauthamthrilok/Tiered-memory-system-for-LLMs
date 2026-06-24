# Tiered memory system for LLMs

A RAG based memory system for LLMs. Enables a locally running language model to persist, retrieve, and reason over memories across sessions — with conflict resolution, hybrid retrieval, decay-weighted reranking, and structured observability.

---

## Architecture

```
User message
    └─► Hybrid Retrieval (BM25 + Vector Search → RRF merge → Decay rerank)
          └─► Prompt augmentation (semantic + episodic + procedural context)
                └─► LLM response
                      └─► Memory extraction (LLM-powered, atomic facts)
                            └─► Conflict resolution (detect contradictions, retire stale facts)
                                  └─► ChromaDB write (3-tier persistent store)
```

---

## Features

**Three-tier memory store**
Memories are classified and stored in separate ChromaDB collections:
- `semantic` — stable facts about the user
- `episodic` — events and things discussed in past sessions
- `procedural` — user preferences and communication style

**Hybrid retrieval with RRF**
Every query runs both dense vector search (ChromaDB) and sparse keyword search (BM25). Results are merged using Reciprocal Rank Fusion — a rank-based merging algorithm that rewards memories surfaced by both methods. Final ranking applies decay scoring: similarity (0.6) + recency (0.25) + access frequency (0.15).

**Conflict resolution**
Before writing any new fact, the system retrieves semantically similar existing memories and runs a contradiction detection step. Contradicted facts are marked `superseded` with an audit trail rather than deleted. Superseded facts are excluded from retrieval but preserved in the database.

**Intelligent extraction**
After every LLM response, a second LLM call extracts 1-3 atomic facts from the exchange using few-shot prompting. Facts are classified into tiers before storage.

**Structured observability**
Every pipeline stage (retrieval, extraction, storage, generation) is instrumented with a `log_stage` context manager that records latency, status, and stage-specific metadata to `logs/pipeline.jsonl` in JSON Lines format.

**Evaluation harness**
A 5-case eval suite measures retrieval recall and answer faithfulness against ground truth. Recall measures whether the right memories were retrieved; faithfulness measures whether the LLM response is grounded in retrieved context.

---

## Stack

| Component | Technology |
|---|---|
| LLM | Ollama (llama3.1:8b)|
| Vector store | ChromaDB (local persistent) |
| Embeddings | ChromaDB DefaultEmbeddingFunction|
| Keyword search | rank-bm25 |
| Logging | JSON Lines (logs/pipeline.jsonl) |

---

## Project Structure

```
rag-mem/
├── memory/
│   ├── store.py          # ChromaDB read/write, hybrid retrieval, decay reranking
│   ├── extractor.py      # LLM-powered fact extraction with tier classification
│   └── conflict.py       # Contradiction detection and fact versioning
├── llm/
│   └── client.py         # LLM client wrapper (swap model here)
├── observability/
│   └── logger.py         # log_stage context manager, JSON Lines writer
├── eval/
│   ├── eval.py           # Evaluation harness (recall + faithfulness)
│   └── test_cases.py     # Ground truth test cases
├── logs/
│   └── pipeline.jsonl    # Structured pipeline logs (auto-generated)
├── chat.py               # Main conversation loop
├── requirements.txt
└── .env                  # API keys (not committed)
```

---

## Setup

```bash
git clone <repo-url>
cd rag-mem
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Install and start Ollama:
```bash
# Install from https://ollama.com
ollama serve &
ollama pull llama3.1:8b
```

Create `.env` in the project root:
```
# Only needed if using a hosted LLM instead of Ollama
GEMINI_API_KEY=your-key-here
```

---

## Usage

**Start a conversation:**
```bash
python3 chat.py
```

**Run the evaluation harness:**
```bash
python3 eval/eval.py
```

**Inspect stored memories:**
```bash
python3 -c "
import chromadb
from chromadb.utils import embedding_functions
embed_fn = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path='./chroma_db')
for tier in ['semantic', 'episodic', 'procedural']:
    col = client.get_or_create_collection(name=tier, embedding_function=embed_fn)
    print(f'=== {tier} ({col.count()} entries) ===')
    results = col.get(include=['documents', 'metadatas'])
    for doc, meta in zip(results['documents'], results['metadatas']):
        status = 'SUPERSEDED' if meta.get('superseded') else 'active'
        print(f'  [{status}] [{meta.get(\"access_count\", 0)} hits] {doc}')
    print()
"
```

**Inspect pipeline logs:**
```bash
python3 -c "
import json
with open('logs/pipeline.jsonl') as f:
    for line in f:
        e = json.loads(line)
        print(f\"{e['stage']:12} {e['latency_ms']:>8.1f}ms  {e.get('status','')}\")"
```

---

## Evaluation Baseline

Measured on llama3.1:8b (CPU, Intel i5):

| Metric | Score |
|---|---|
| Average Recall | 0.90 |
| Average Faithfulness | 0.90 |
| Test cases | 5 |

---
