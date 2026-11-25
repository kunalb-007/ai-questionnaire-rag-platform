# Dockerfile
#
# Multi-stage build keeps the final image lean by installing build tools
# only in the builder stage.
#
# Stage 1 — builder: install Python dependencies into a venv.
# Stage 2 — runtime: copy only the venv and application source.
#
# The optional heavy dependencies (pinecone-text, sentence-transformers)
# are NOT installed here. Enable them by uncommenting the relevant lines
# if USE_HYBRID_SEARCH=true or USE_RERANKING=true in your .env.

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for packages that compile C extensions
# (PyMuPDF requires libmupdf headers at build time).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY pipeline.py .

# Create a non-root user for the container process
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# --workers 1 is intentional for this deployment:
#   - The ChatOpenAI singleton (_get_llm) is process-local; multiple workers
#     each hold their own connection pool — still correct, just less efficient.
#   - For higher throughput, increase workers or switch to gunicorn + uvicorn workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]