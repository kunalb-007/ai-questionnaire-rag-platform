# Enterprise Document Intelligence — RAG System

A focused, interview-defensible Retrieval-Augmented Generation (RAG) pipeline
for enterprise security and compliance documents.

## Architecture

```
Document (PDF / DOCX / TXT)
    │
    ▼
┌─────────────┐
│   Parser    │  PyMuPDF · python-docx · plain text
│  (parser.py)│  → list of {text, page, filename}
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Cleaner   │  Normalize whitespace, rejoin hyphenated words
│ (cleaner.py)│  → same structure, cleaner text
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunker   │  RecursiveCharacterTextSplitter (LangChain)
│ (chunker.py)│  → list of Chunk(text, filename, page, chunk_index)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ EmbeddingService │  sentence-transformers/all-MiniLM-L6-v2
│(embedding_svc.py)│  → list of 384-dim float vectors
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│    Qdrant    │  Collection + HNSW index (Cosine distance)
│(qdrant_svc.py│  → stored points with payload metadata
└──────────────┘

          ↑  INGESTION complete  ↑

User Query
    │
    ▼
┌──────────────────┐
│ EmbeddingService │  Same model → query vector (384-dim)
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   Retriever  │  Top-K similarity search in Qdrant
│(retriever.py)│  → list of RetrievedChunk with scores
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│  AnswerGenerator  │  LangChain ChatOpenAI + grounded prompt
│(answer_generator) │  → answer text + source citations
└───────────────────┘
```

## Project Structure

```
rag_project/
├── app/
│   ├── config.py                    # All settings from env vars
│   ├── ingestion/
│   │   ├── parser.py                # PDF, DOCX, TXT text extraction
│   │   ├── cleaner.py               # Whitespace normalization
│   │   └── chunker.py               # RecursiveCharacterTextSplitter
│   ├── embeddings/
│   │   └── embedding_service.py     # SentenceTransformers wrapper
│   ├── vectorstore/
│   │   └── qdrant_service.py        # Collection CRUD + similarity search
│   ├── retrieval/
│   │   └── retriever.py             # Query → embed → search → results
│   └── generation/
│       └── answer_generator.py      # Prompt builder + LLM call
├── pipeline.py                      # CLI entry point (ingest / query)
├── sample_docs/                     # Test documents
├── requirements.txt
├── docker-compose.yml               # Qdrant local instance
├── .env.example                     # Config template
└── README.md
```

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d
# Qdrant UI: http://localhost:6333/dashboard
```

### 2. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY (and optionally LLM_BASE_URL for Ollama)
```

### 4. Ingest a Document

```bash
python pipeline.py ingest sample_docs/password_policy.txt
python pipeline.py ingest sample_docs/incident_response.txt
```

### 5. Query

```bash
python pipeline.py query "What is the minimum password length?"
python pipeline.py query "How quickly must a Severity 1 incident be contained?"
python pipeline.py query "What MFA methods are approved?"
```

### 6. Ingest + Query in One Step

```bash
python pipeline.py ingest sample_docs/password_policy.txt \
    --query "What are the password complexity requirements?"
```

## Using Ollama (Local LLM — No API Key Required)

```bash
# Install and start Ollama
ollama pull llama3.2

# In .env:
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
```

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_COLLECTION_NAME` | `enterprise_docs` | Collection name |
| `LLM_API_KEY` | _(required)_ | API key |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_BASE_URL` | OpenAI URL | Override for Ollama/vLLM |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |

---

## Interview Talking Points

### Why RAG instead of fine-tuning?
RAG retrieves *current* information at query time. Fine-tuning bakes knowledge
into model weights — expensive, slow, and stale the moment documents change.
RAG lets you update the knowledge base by simply re-ingesting documents.

### Why chunk at all?
Embedding models have token limits (MiniLM: 256 tokens). Long documents must
be split. Even with unlimited context, embedding a 50-page document as one
vector loses specificity — the vector becomes an "average" of all topics,
making targeted retrieval unreliable.

### Why RecursiveCharacterTextSplitter?
It splits on paragraph breaks first (`\n\n`), then sentences (`\n`), then
words, then characters. This produces more semantically coherent chunks
than a naive character split. The `chunk_overlap` ensures sentences at
boundaries appear in at least one complete chunk.

### What does cosine similarity actually measure?
The angle between two vectors in embedding space. cos(0°) = 1.0 (identical
direction = identical meaning). cos(90°) = 0 (orthogonal = unrelated).
For L2-normalised vectors, cosine similarity equals the dot product, which
Qdrant can compute extremely efficiently using HNSW indexing.

### Why can RAG still hallucinate?
1. The retrieved chunks may not contain the answer — the LLM fills the gap
   from its training data (retrieval failure → generation failure).
2. The LLM may misread or misattribute content between retrieved chunks.
3. The LLM may blend retrieved context with training knowledge.
   Prompt instructions reduce but don't eliminate this. Production systems add
   confidence scoring, human review, or retrieval validation.

### What would you improve next?
1. **Reranking**: pass top-20 through a cross-encoder, return top-5.
2. **Hybrid search**: combine dense (semantic) + sparse (BM25) retrieval.
3. **Metadata filtering**: restrict search to specific documents or date ranges.
4. **Semantic chunking**: split on topic boundaries, not character count.
5. **Evaluation**: RAGAS metrics (faithfulness, answer relevance, context recall).

---

## Extending with FastAPI

```python
# app/api.py — minimal integration point (not implemented in MVP)
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import query_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(req: QueryRequest):
    # Re-use the same retriever + generator logic
    ...
```