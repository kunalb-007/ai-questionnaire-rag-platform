# Enterprise Document Intelligence — RAG System
Production-style **Retrieval-Augmented Generation (RAG)** API for querying enterprise security and compliance documents with grounded, cited answers.

### Stack
**Python · FastAPI · LangChain · Pinecone · OpenAI Embeddings · BM25 · Cross-Encoder · Docling · PyMuPDF**

📖 **Swagger API Docs:**   
https://ai-questionnaire-rag-platform.onrender.com/docs

---

## Architecture

```
Document (PDF / DOCX / TXT)
    │
    ▼
┌─────────────┐
│   Parser    │  PyMuPDF · Docling · plain text
│             │  → list of {text, page, filename}
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Cleaner   │  Normalize whitespace, rejoin hyphenated words
│             │  → same structure, cleaner text
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunker   │  RecursiveCharacterTextSplitter (LangChain)
│             │  → list of Chunk(text, filename, page, chunk_index)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ EmbeddingService │  nemotron-3-embed-1b
│                  │  → 2048 dimensions
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│    Pinecone  │  Dense + BM25 Sparse
│              │  → stored points with payload metadata
└──────────────┘

          ↑  INGESTION complete  ↑


User Query
    │
    ▼
┌──────────────────┐
│ EmbeddingService │  Same model → query vector (2048-dim)
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│   Hybrid Retrieval   │  Top-20 chunks
│     (Top-20)         │  → list of RetrievedChunk with scores
└──────┬───────────────┘
       │
       ▼
┌──────────────────┐
│   Cross Encoder  │  Top-K similarity search
│   Reranking      │  → list of RetrievedChunk with scores
└──────┬───────────┘
       │
       ▼
┌───────────────────┐
│   LLM + Grounded  │  Top-K similarity search
│   Prompt          │  → list of RetrievedChunk with scores
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│  AnswerGenerator  │  LangChain ChatOpenAI + grounded prompt
│(answer_generator) │  → answer text + source citations
└───────────────────┘
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

### 4. Query

```bash
python pipeline.py query "What is the minimum password length?"
python pipeline.py query "How quickly must a Severity 1 incident be contained?"
python pipeline.py query "What MFA methods are approved?"
```
## Evaluation

The project includes a small curated evaluation set measuring:

* **Recall@K** — expected source retrieved in top-K
* **MRR** — ranking quality of the first relevant source
* **Faithfulness** — keyword heuristic or optional LLM judge
