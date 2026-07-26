# BINUSAssist Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for BINUS School of Computer Science that answers questions about academic programs (curricula, learning outcomes, career prospects, etc.) by retrieving from official program-guide documents rather than relying on a model's general knowledge.

## Architecture

- **Backend**: FastAPI + LlamaIndex. Documents (PDF/DOCX program guides) are parsed with Docling, chunked with a parent-child split, and embedded with `BAAI/bge-m3` into a ChromaDB vector store.
- **Retrieval**: Hybrid dense + BM25 search fused via `QueryFusionRetriever` (top 20 each → reciprocal-rank fusion → top 15), then reranked with a `BAAI/bge-reranker-v2-m3` cross-encoder down to the top 5 chunks actually sent to the LLM.
- **Confidence gate**: if the top reranked chunk scores below a threshold, the query is retried with LLM-generated paraphrases; if it still scores too low, the chatbot returns a fallback message with contact info instead of guessing.
- **Generation**: Groq-hosted `llama-3.1-8b-instant` (temperature 0.0, pinned in `config.py`), streamed token-by-token over SSE.
- **Frontend**: React 19 + Vite + Tailwind. Chat UI with streaming responses, source citations panel, starter questions, and an admin panel (document management, starter questions, fallback contacts, account settings) gated behind HTTP Basic auth.

```
backend/
  main.py            FastAPI app, /chat, /feedback, /health, /config, /documents, /avatar
  admin/             Admin-only routes (documents, reindex, starter questions, fallback contacts, profile) + auth
  rag/               ingestion (parsing/chunking/indexing), retrieval (fusion + rerank), generation (prompting/streaming)
  config.py          Settings, fallback/error message templates
frontend/
  src/components/    ChatPanel, SourcePanel, AdminPanel, AdminLogin, Profile, Header
  src/hooks/useChat.js
scripts/
  seed_kb.py         Initial ingestion of backend/documents/ into the vector store
  manage_users.py    CLI for creating/listing/removing admin accounts (no web-facing signup)
  eval.py            Automated eval harness (fallback accuracy, latency, relevance/precision grading)
```

## Setup (local development)

**Prerequisites**: Python 3.12, Node 18+, an NVIDIA GPU + CUDA for `EMBEDDING_DEVICE=cuda`/`RERANKER_DEVICE=cuda` (or set both to `cpu` if you don't have one), and a [Groq API key](https://console.groq.com).

1. **Backend**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
   Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`.

2. **Create an admin account** (there's no web signup by design):
   ```
   python scripts/manage_users.py add <username> admin
   ```

3. **Ingest the knowledge base** — put PDF/DOCX program guides in `backend/documents/`, then:
   ```
   python scripts/seed_kb.py
   ```

4. **Frontend**
   ```
   cd frontend
   npm install
   ```

5. **Run both**: `start.bat` launches the backend (`uvicorn backend.main:app --reload --port 8000`) and frontend (`npm run dev`, served at `http://localhost:5173`) in separate windows.

## Running with Docker

```
docker compose up --build
```
Builds and runs `backend` (GPU-enabled, port 8000) and `frontend` (port 5173) per [`docker-compose.yml`](docker-compose.yml). Requires `.env` with `GROQ_API_KEY` and the NVIDIA Container Toolkit for GPU passthrough.

**The knowledge base is seeded automatically on first boot.** The container's entrypoint runs
[`scripts/seed_if_empty.py`](scripts/seed_if_empty.py) before starting the API: if the mounted
vectorstore has no index yet, it builds one from the program documents, the recorded scraped
URLs, and the faculty snapshot — so a fresh `docker compose up` answers out of the box instead
of booting empty. It's idempotent (a persisted vectorstore volume is reused, so restarts don't
re-seed or re-crawl).

Two things to have in place before the first boot, since both are runtime data, not code:
- **Program documents** in `backend/documents/` on the host (PDF/DOCX) — mounted into the
  container. Without them the KB seeds empty and every query falls back until documents are
  added (via the mount or the admin panel).
- **`.env`** with `GROQ_API_KEY` (and `DOMAIN` for the prod HTTPS setup below).

For a public deployment without a GPU, [`docker-compose.prod.yml`](docker-compose.prod.yml) adds a Caddy reverse proxy (automatic HTTPS via `DOMAIN` in `.env`) and uses the CPU-only backend image (`backend/Dockerfile.cpu`):
```
DOMAIN=chat.example.edu docker compose -f docker-compose.prod.yml up --build
```
Note the CPU reranker adds ~10–15s/query latency (the confidence gate is calibrated to reranker scores, so it can't simply be dropped) — a GPU host is recommended for the PRD's sub-3s target.

## Evaluation

`scripts/eval.py` runs a fixed set of in-scope and out-of-scope questions (English + Indonesian) against the live pipeline and reports:
- **Fallback accuracy** — out-of-scope questions correctly triggering the fallback message, and in-scope questions that incorrectly fall back.
- **First-token latency** against the PRD's <3s target.
- A results JSON for manual relevance/retrieval-precision grading.

Run with the backend's models already loaded in-process (no server needed):
```
python scripts/eval.py
```
Each run writes a timestamped `eval_results_*.json` with every question's answer and sources for manual review.

## Unit tests & regression eval

`tests/` covers the pure, deterministic helpers behind retrieval/chunking/caching (smalltalk detection, chunk-boilerplate filters, the semantic cache's safety gates, etc.) plus a **labeled regression eval** ([`tests/regression_cases.py`](tests/regression_cases.py)) over the two behaviours that repeatedly regressed — program routing and language detection. Both are deterministic (program routing is decided by literal matching, not an LLM — see the classifier in [`backend/rag/generation.py`](backend/rag/generation.py)), so the real bug cases are guarded with no GPU, Groq, or network, and it all runs in a few seconds:
```
pip install pytest
pytest
```
Runs automatically on every push/PR via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), alongside a frontend lint + build check. The full, model-dependent eval (retrieval and answer quality) lives in `scripts/eval.py` (see [Evaluation](#evaluation) above) and is run manually, since it needs the GPU models and a real Groq key.

## Notes

- `.env` is gitignored; only `.env.example` (no real keys) is checked in.
- `socs_documents/` holds the source program catalogs the knowledge base is built from.
