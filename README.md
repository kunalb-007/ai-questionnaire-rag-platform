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