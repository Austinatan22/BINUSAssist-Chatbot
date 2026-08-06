# BINUSAssist Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for BINUS School of Computer Science. It answers questions about academic programs (curricula, learning outcomes, career prospects and so on) from the official program-guide documents rather than from a model's general knowledge.

## Architecture

- **Backend**: FastAPI and LlamaIndex. PDF and DOCX program guides are parsed with Docling, split with a parent-child chunker, embedded with `BAAI/bge-m3`, and stored in a ChromaDB vector store.
- **Retrieval**: dense and BM25 search run together through `QueryFusionRetriever`, top 20 from each. Reciprocal-rank fusion narrows that to 15, then a `BAAI/bge-reranker-v2-m3` cross-encoder narrows it to the 5 chunks actually sent to the LLM.
- **Confidence gate**: if the top reranked chunk scores below a threshold, the query is retried with LLM-generated paraphrases. If the score is still too low, the bot returns a fallback message with contact details instead of guessing.
- **Generation**: OpenAI `gpt-4o-mini` at temperature 0.0, pinned in `config.py`, streamed token by token over SSE. `LLM_PROVIDER` switches the provider. Groq (`llama-3.1-8b-instant`) and Gemini are also wired up behind the same LlamaIndex `Settings.llm`.
- **Frontend**: React 19, Vite, Tailwind. Streaming chat UI, source citations panel, starter questions, and an admin panel covering document management, starter questions, fallback contacts, and account settings, gated behind HTTP Basic auth.

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

You need Python 3.12, Node 18+, and an [OpenAI API key](https://platform.openai.com/api-keys). OpenAI is the default provider and costs roughly $3/month at about 5k requests on `gpt-4o-mini`. An NVIDIA GPU with CUDA lets you run `EMBEDDING_DEVICE=cuda` and `RERANKER_DEVICE=cuda`; set both to `cpu` if you don't have one.

1. **Backend**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
   Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`.

2. **Create an admin account.** There is no web signup, by design:
   ```
   python scripts/manage_users.py add <username> admin
   ```

3. **Ingest the knowledge base.** Put PDF or DOCX program guides in `backend/documents/`, then:
   ```
   python scripts/seed_kb.py
   ```

4. **Frontend**
   ```
   cd frontend
   npm install
   ```

5. **Run both**: `start.bat` launches the backend (`uvicorn backend.main:app --reload --port 8000`) and the frontend (`npm run dev`, served at `http://localhost:5173`) in separate windows.

## Running with Docker

```
docker compose up --build
```

This builds and runs `backend` (GPU-enabled, port 8000) and `frontend` (port 5173) per [`docker-compose.yml`](docker-compose.yml). It needs `.env` with `OPENAI_API_KEY`, and the NVIDIA Container Toolkit for GPU passthrough.

The knowledge base is seeded automatically on first boot. The container's entrypoint runs [`scripts/seed_if_empty.py`](scripts/seed_if_empty.py) before starting the API. If the mounted vectorstore has no index yet, the script builds one from the program documents, the recorded scraped URLs, and the faculty snapshot, so a fresh `docker compose up` answers questions instead of booting empty. It is idempotent: a persisted vectorstore volume is reused, so restarts don't re-seed or re-crawl.

Two things have to be in place before the first boot, since both are runtime data rather than code:

- **Program documents** in `backend/documents/` on the host, in PDF or DOCX, mounted into the container. Without them the KB seeds empty and every query falls back until documents are added through the mount or the admin panel.
- **`.env`** with `OPENAI_API_KEY`, plus `DOMAIN` for the production HTTPS setup below.

For a public deployment without a GPU, [`docker-compose.prod.yml`](docker-compose.prod.yml) adds a Caddy reverse proxy with automatic HTTPS from `DOMAIN` in `.env`, and uses the CPU-only backend image ([`backend/Dockerfile.cpu`](backend/Dockerfile.cpu)):

```
DOMAIN=chat.example.edu docker compose -f docker-compose.prod.yml up --build
```

The CPU reranker adds 10-15s per query. It cannot simply be dropped, because the confidence gate is calibrated against reranker scores. Use a GPU host if you need the PRD's sub-3s target.

## Evaluation

`scripts/eval.py` runs a fixed set of in-scope and out-of-scope questions, in English and Indonesian, against the live pipeline. It reports:

- **Fallback accuracy**: out-of-scope questions that correctly trigger the fallback message, and in-scope questions that incorrectly fall back.
- **First-token latency** against the PRD's 3-second target.
- A results JSON for manual relevance and retrieval-precision grading.

Run it with the backend's models loaded in-process, so no server is needed:

```
python scripts/eval.py
```

Each run writes a timestamped `eval_results_*.json` holding every question's answer and sources for manual review.

## Unit tests and regression eval

`tests/` covers the pure, deterministic helpers behind retrieval, chunking and caching: smalltalk detection, chunk-boilerplate filters, the semantic cache's safety gates, and the rest. It also holds a labeled regression eval, [`tests/regression_cases.py`](tests/regression_cases.py), over the two behaviours that repeatedly regressed: program routing and language detection.

Both are deterministic. Program routing is decided by literal matching rather than an LLM (see the classifier in [`backend/rag/generation.py`](backend/rag/generation.py)), so the real bug cases are guarded with no GPU, no LLM API key and no network, and the whole thing runs in a few seconds.

```
pip install pytest
pytest
```

This runs on every push and PR through GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), alongside a frontend lint and build check. The full model-dependent eval of retrieval and answer quality lives in `scripts/eval.py` (see [Evaluation](#evaluation) above) and is run by hand, since it needs the GPU models and a real OpenAI key.

## Notes

- `.env` is gitignored. Only `.env.example`, which holds no real keys, is checked in.
- `socs_documents/` holds the source program catalogs the knowledge base is built from.
