# BINUSAssist — Codebase Walkthrough

A file-by-file guide to what each important file does and how it works. Written for
presenting the technical implementation to a supervisor/reviewer who wants to
understand the system beyond the high-level architecture in [README.md](README.md).

## The one-sentence pitch

BINUSAssist is a RAG (Retrieval-Augmented Generation) chatbot for BINUS School of
Computer Science: it answers questions about programs/tuition/faculty by retrieving
from real BINUS documents rather than letting an LLM improvise, with a
citation-grounded UI and an admin panel to manage the knowledge base.

---

## 1. Backend entry & orchestration

**[backend/main.py](backend/main.py)** — the FastAPI app. On startup (`lifespan`) it
validates config, loads the embedding/LLM models, loads the vector index, and builds
the retriever/reranker — all stored in a shared `app_state` dict so every request
reuses them instead of reloading. Exposes `/chat` (rate-limited 30/hour per IP),
`/feedback`, `/health` (reports index/GPU/API-key status; `?deep=true` adds a real
LLM-provider reachability probe, kept opt-in so a monitoring tool polling it doesn't
burn API quota), `/config/starter-questions`, and file-serving routes for
documents/avatars (with path-traversal guards via `Path(...).name`).

**[backend/chat_service.py](backend/chat_service.py)** — the brain of a single chat
turn, deliberately split from `main.py` so routing logic is unit-testable without a
live model. Split into two phases:
- `_plan()` — decides what to do: condense a follow-up into a standalone query, check
  the semantic cache, classify which program(s)/campus/faculty-person the question is
  about, and pick a retrieval route (comparison mode, single-program-scoped,
  faculty-scoped, campus-programs-scoped, or open retrieval with a low-confidence
  retry).
- `stream()` — turns that plan into the actual SSE response, applying the daily
  token-budget gate, attaching follow-up suggestions, and wrapping the stream to
  populate the cache and write the query log afterward.

This is the most "designed" file in the codebase — there are ~8 distinct routing
branches, each targeting a real failure mode found in production (e.g., a program's
own catalog PDF has no tuition data, so a program-scoped query that comes up empty
retries against scraped tuition pages specifically).

**[backend/config.py](backend/config.py)** — all tunable settings (Pydantic
`BaseSettings`, loaded from `.env`): which LLM provider (OpenAI/Groq/Gemini), model
names, `confidence_threshold=0.5` (the reranker-score cutoff below which the bot
refuses to answer rather than guess — empirically calibrated, documented in detail),
chunk sizes, cache/budget parameters. Also holds `validate_startup_config()` (fail
fast if the active provider's API key is missing) and the fallback/service-error
message templates.

**[backend/state.py](backend/state.py)** — trivial but important: one shared
`app_state = {}` dict holding the live index/retriever so admin actions
(upload/delete/reindex) mutate the same object every request sees, with no reload.

---

## 2. The RAG pipeline (`backend/rag/`)

**[models.py](backend/rag/models.py)** — loads the embedding model
(`BAAI/bge-m3`, fp16 on GPU — halves VRAM with negligible accuracy loss, measured)
and builds the LLM client for whichever provider is configured
(OpenAI/Groq/Gemini all funnel into one LlamaIndex `Settings.llm`, so the rest of the
code never needs to know which vendor is active).

**[ingestion.py](backend/rag/ingestion.py)** (1,433 lines — the largest file) — turns
PDFs/DOCX/XLSX/scraped web pages into searchable chunks:
- Parses documents with **Docling**, splits with **parent-child chunking** (small
  child chunks get embedded/searched for precision; the LLM sees the larger parent
  chunk for fuller context).
- Has several deterministic "recovery" patches for cases where Docling silently drops
  content — e.g., one program's credit-total table row isn't extracted by Docling but
  is recovered from the PDF's raw text layer; a career list that's actually an
  *image* in the source PDF is OCR'd once and hardcoded.
- Structured tables (tuition fees, course credit tables) bypass generic chunking
  entirely and get parsed row-by-row so a fact never gets split mid-row.
- Builds/loads the index against **ChromaDB** — notably constructs
  `VectorStoreIndex(nodes=[], storage_context=...)` rather than the more obvious
  `from_vector_store()`, because that shortcut silently drops the docstore, which the
  BM25 half of hybrid search needs.
- Handles web scraping (via `trafilatura`) and a genuinely fragile faculty-roster
  crawl (undocumented AJAX endpoints, hardcoded token) — which is why that crawl
  result is **snapshotted to a JSON file** and only re-run on explicit admin request,
  so routine reindexing never depends on that crawl succeeding.

**[retrieval.py](backend/rag/retrieval.py)** — hybrid search: dense (embedding) +
BM25 (keyword) retrieval fused via reciprocal-rank fusion, then reranked by a
cross-encoder (`BAAI/bge-reranker-v2-m3`, also fp16 on GPU) down to the top 5 chunks.
Also has `retrieve_for_named_programs()`, a scoped-retrieval helper used when the
question is about one or two specific programs, with a "balanced" mode that
round-robins chunks across programs so a comparison question doesn't get dominated by
one side.

**[generation.py](backend/rag/generation.py)** (2,059 lines — the other giant file)
— this is where the actual answer gets produced, and it's built on a consistent
philosophy visible throughout: **classify with deterministic code first, only fall
back to the LLM for genuinely ambiguous cases**, because the LLM was repeatedly
caught not following prompt-only instructions in testing. Concretely:
- Language detection is regex-first (a statistical library misfired on short
  Indonesian questions).
- Program/campus name matching is literal string matching with alias tables, not an
  LLM classifier.
- Prompt-injection attempts ("ignore your instructions", "repeat everything above")
  are caught by regex *before* any LLM call.
- `stream_answer()` streams the actual response token-by-token over SSE, with a
  clever trick: it buffers the first ~200 characters before forwarding anything, so
  it can detect a `NO_ANSWER` sentinel the model is told to emit when it can't
  answer — because once tokens are sent to the browser you can't take them back.

**[prompts.py](backend/rag/prompts.py)** — all the prompt templates: the main system
prompt (citation format, anti-fabrication rules, treats retrieved context as
untrusted data to resist injection), a comparison-table instruction,
condense/rewrite/translate prompts for query preprocessing, and a structured-JSON
classifier prompt used only as a last resort.

**[cache.py](backend/rag/cache.py)** — a semantic answer cache. Worth highlighting to
a supervisor: its docstring documents an actual calibration experiment showing that
embedding similarity *alone* cannot safely decide a cache hit (genuine paraphrases
and dangerous near-misses score in the same 0.74–0.86 range), so every candidate must
also pass two deterministic gates — same matched program(s), same detected
topic/aspect — before being trusted.

**[token_budget.py](backend/rag/token_budget.py)** — tracks daily LLM token usage
against a soft budget (originally to survive Groq's daily free-tier quota, which eval
traffic exhausted; it now caps the active provider's spend), rolling over at UTC
midnight, so generation degrades to a decline
message rather than the app crashing when quota runs out.

---

## 3. Admin backend (`backend/admin/`)

**[auth.py](backend/admin/auth.py)** — HTTP Basic auth, hand-parsed (not FastAPI's
built-in, to avoid the browser's native login popup since the frontend has its own
form), bcrypt password checks with a constant-time dummy-hash comparison for unknown
usernames (prevents username enumeration), plus an in-memory brute-force lockout (5
failed attempts → 15-minute IP lockout).

**[routes.py](backend/admin/routes.py)** — all `/admin/*` endpoints: document
upload/list/delete, URL scraping, full reindex, faculty-roster refresh,
starter-questions editor, fallback-contacts editor, and self-service profile/avatar
management. Every document-mutating route calls a shared `_sync_index()` that
rebuilds the BM25 retriever and clears the semantic cache, since a cached answer is
only valid for the KB state it was generated against.

**[users.py](backend/admin/users.py)** — simple JSON-file-backed admin account
store. Deliberately has no "create account" API route — accounts are provisioned only
via the [scripts/manage_users.py](scripts/manage_users.py) CLI, so there's no
web-facing signup surface.

---

## 4. Frontend (`frontend/src/`)

**[App.jsx](frontend/src/App.jsx)** — top-level layout and a hand-rolled router (no
react-router) switching between chat/login/admin/profile views based on the URL path;
owns the citation-click ↔ source-panel wiring.

**[hooks/useChat.js](frontend/src/hooks/useChat.js)** +
**[lib/api.js](frontend/src/lib/api.js)** — the streaming chat logic: POSTs the
message + trimmed history to `/chat`, reads the response as a raw SSE stream, and
appends tokens to the last message as they arrive; messages persist to
`localStorage` across reloads.

**[components/ChatPanel.jsx](frontend/src/components/ChatPanel.jsx)** (1,142 lines)
— the main chat UI: message list with smart auto-scroll, a small markdown renderer
with inline citation pills, feedback (thumbs up/down + regenerate), starter
questions, and follow-up suggestion chips.

**[components/AdminPanel.jsx](frontend/src/components/AdminPanel.jsx)** — document
management dashboard (upload/URL-scrape/list/delete/reindex), plus starter-questions
and fallback-contacts editors.

**[components/SourcePanel.jsx](frontend/src/components/SourcePanel.jsx)**,
**[AdminLogin.jsx](frontend/src/components/AdminLogin.jsx)**,
**[Header.jsx](frontend/src/components/Header.jsx)**,
**[Profile.jsx](frontend/src/components/Profile.jsx)** — citation source list with
relevance bars; the login form; the shared top bar with auth-aware menu;
self-service account settings.

---

## 5. Scripts & tests

- **[scripts/seed_kb.py](scripts/seed_kb.py)** /
  **[seed_if_empty.py](scripts/seed_if_empty.py)** — initial/first-boot ingestion.
- **[scripts/eval.py](scripts/eval.py)** — runs a fixed question set (English +
  Indonesian, in-scope + out-of-scope) against the live pipeline, reporting fallback
  accuracy and latency against the PRD's targets.
- **[scripts/probe_confidence.py](scripts/probe_confidence.py)** — the tool used to
  empirically calibrate `confidence_threshold`.
- **[scripts/manage_users.py](scripts/manage_users.py)**,
  **[log_analytics.py](scripts/log_analytics.py)** — admin CLI, query-log analysis.
- **[tests/](tests/)** — pytest suite covering the deterministic pieces
  (classifiers, cache safety gates, chunking filters) plus a labeled regression suite
  (`tests/regression_cases.py`) targeting the two behaviors that repeatedly broke in
  the past: program routing and language detection. Runs in CI on every push, no GPU/
  API key needed.

---

## What's worth emphasizing to a supervisor

1. **"Ask, don't guess" is enforced at multiple layers, not just prompting** — a
   confidence gate on retrieval scores, a `NO_ANSWER` sentinel the model must emit,
   and deterministic clarification questions when a query names an unresolvable
   campus/program.
2. **Deterministic code does the classification, the LLM only handles genuinely
   semantic judgment** — this is the throughline in `generation.py` and shows up as a
   direct response to observed LLM failures (documented in comments with real
   measured scores).
3. **Cost-consciousness is architectural**: semantic caching, a daily token budget,
   template-based (not LLM-generated) follow-ups.
4. **Almost every non-obvious piece of logic has a comment citing a specific
   production incident** (a query that scored 0.016 vs 0.997, a GPU VRAM contention
   bug, a docling table-extraction gap) — this is a codebase hardened against real
   failures, not built speculatively.
