# PRD: BINUS School of Computer Science — RAG Chatbot

**Version:** 1.0 — Prototype  
**Date:** 2026-04-03  
**Author:** Henry (Deputy, Data Science Program)  
**Status:** Draft

---

## 1. Problem Statement

Students, prospective students, and visitors to the BINUS School of Computer Science (SoCS) have administrative, academic, and campus-life questions that currently require direct human contact — email, WhatsApp, or in-person visits to faculty/staff. Response times are inconsistent, information is scattered across PDFs, spreadsheets, and web pages, and staff time is consumed by repetitive queries.

There is no centralized, always-available information system that can answer questions grounded in authoritative program documents.

---

## 2. Product Definition

A web-based, mobile-friendly chatbot that answers questions about BINUS SoCS using Retrieval-Augmented Generation (RAG) over a curated document knowledge base. It is publicly accessible (no student login required), and provides source citations for every answer.

**What this is:**
- A question-answering system grounded in uploaded documents
- A self-service tool to reduce repetitive admin/faculty queries
- A prototype to validate demand and accuracy before institutional investment

**What this is NOT:**
- A general-purpose AI assistant (no freeform generation outside document scope)
- A replacement for academic advising
- A system that handles personally identifiable student data
- A system where end-users upload documents

---

## 3. Users

| User | Need | Access |
|------|------|--------|
| **Current students** | Academic regulations, course info, schedules, procedures, forms | Public chat interface |
| **Prospective students** | Program details, admission requirements, curriculum overview | Public chat interface |
| **Parents / guardians** | Tuition, facilities, program credibility | Public chat interface |
| **Faculty / staff** | Quick reference to policies they don't have memorized | Public chat interface |
| **Knowledge base admins** | Upload, update, delete documents in the knowledge base | Password-protected admin panel |

---

## 4. Core Requirements

### 4.1 Chat Interface (Public-Facing)

| ID | Requirement | Priority |
|----|-------------|----------|
| C-01 | Single-page chat UI, mobile-first responsive design | P0 |
| C-02 | Streaming response display (token-by-token) | P0 |
| C-03 | Inline source citations with document name + page/section reference | P0 |
| C-04 | Source panel: clickable citations expand to show the retrieved chunk | P0 |
| C-05 | Fallback message with WhatsApp/email contact when confidence is low | P0 |
| C-06 | Bilingual support: accept and respond in Indonesian or English, matching user language | P0 |
| C-07 | Conversation history within session (cleared on page refresh) | P1 |
| C-08 | Suggested starter questions on first load | P1 |
| C-09 | "Was this helpful?" feedback thumbs on each response | P1 |
| C-10 | Rate limiting: max 30 messages per IP per hour | P0 |
| C-11 | Disclaimer banner: "AI-generated answers — verify critical information with faculty" | P0 |

### 4.2 Knowledge Base Admin Panel

| ID | Requirement | Priority |
|----|-------------|----------|
| A-01 | Password-protected access (single shared password, no user accounts) | P0 |
| A-02 | Upload documents: PDF, DOCX, XLSX, CSV | P0 |
| A-03 | Add web page by URL (scrape and index) | P1 |
| A-04 | View all indexed documents with metadata (filename, upload date, chunk count) | P0 |
| A-05 | Delete documents (removes from vector store) | P0 |
| A-06 | Re-index: trigger re-processing of all documents | P1 |
| A-07 | Upload size limit: 20MB per file | P0 |

### 4.3 RAG Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| R-01 | Document-aware parsing: extract text preserving structure from PDF, DOCX, XLSX | P0 |
| R-02 | Semantic chunking with parent-child strategy (small chunks for retrieval, large chunks for context) | P0 |
| R-03 | Hybrid retrieval: dense vector search + BM25 sparse search with Reciprocal Rank Fusion | P0 |
| R-04 | Cross-encoder reranking on top-k retrieved chunks | P0 |
| R-05 | Confidence scoring: if top reranked chunk score < threshold, trigger fallback | P0 |
| R-06 | System prompt enforcing grounded answers only — no hallucination, no out-of-scope answers | P0 |
| R-07 | Metadata preservation: source filename, page number, section title attached to every chunk | P0 |
| R-08 | Query rewriting: rephrase ambiguous queries before retrieval | P1 |

---

## 5. Technical Architecture

### 5.1 Stack Decision Matrix

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Backend framework** | FastAPI (Python) | Async-native, lightweight, strong ecosystem for ML/NLP workloads |
| **RAG framework** | LlamaIndex | Purpose-built for RAG; superior document parsing, chunking, and retrieval abstractions vs. LangChain |
| **LLM (generation)** | Groq API — `llama-3.3-70b-versatile` | Best free-tier model for instruction following and multilingual generation. 30 RPM / 14,400 RPD on free tier. Sufficient for prototype traffic. |
| **Embedding model** | `BAAI/bge-m3` (local) | State-of-the-art multilingual embeddings. Supports dense + sparse + ColBERT in a single model. Critical for Indonesian + English mixed-language retrieval. Runs locally — no API cost. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (local) | Multilingual cross-encoder reranker. Significant precision improvement over embedding-only retrieval. Runs locally. |
| **Vector store** | ChromaDB (prototype) → Qdrant (production) | ChromaDB: zero-config, embedded, file-based. Qdrant: production-grade, persistent, filterable. Migration path is clean. |
| **Document parsing** | `docling` (IBM) | Best-in-class PDF/DOCX table and layout extraction. Handles complex academic documents (tables, multi-column, headers) better than PyPDF or Unstructured. |
| **Web scraping** | `trafilatura` | Clean article extraction from web pages, handles Indonesian content well. |
| **Frontend** | React (Vite) + Tailwind CSS + shadcn/ui | Fast build, mobile-first, component library matches NotebookLM aesthetic. |
| **Deployment (prototype)** | Local machine via Docker Compose | Single `docker compose up` for full stack. |
| **Deployment (production)** | BINUS server with Docker Compose + Nginx reverse proxy | Same containers, add SSL and domain. |

### 5.2 Architecture Diagram (Textual)

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Vite)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Chat Panel   │  │ Source Panel  │  │ Admin Panel       │  │
│  │  - Messages   │  │ - Citations   │  │ - Upload docs     │  │
│  │  - Streaming  │  │ - Chunk view  │  │ - Manage KB       │  │
│  │  - Feedback   │  │ - Doc name    │  │ - Password gate   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / WebSocket
┌────────────────────────────┴────────────────────────────────┐
│                   BACKEND (FastAPI)                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               RAG Pipeline (LlamaIndex)              │    │
│  │                                                     │    │
│  │  Query → Rewrite → Hybrid Retrieval → Rerank →      │    │
│  │  Context Assembly → LLM Generation → Citations      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Ingestion    │  │ ChromaDB     │  │ Groq API          │  │
│  │ Pipeline     │  │ (vectors +   │  │ (Llama 3.3 70B)   │  │
│  │ (docling +   │  │  BM25 index) │  │                   │  │
│  │ trafilatura) │  │              │  │                   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Local Models (loaded once at startup)                │   │
│  │  - BAAI/bge-m3 (embeddings + sparse)                 │   │
│  │  - BAAI/bge-reranker-v2-m3 (cross-encoder)           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 RAG Pipeline Detail

**Ingestion (on document upload):**

1. File received via admin panel → format detection
2. Parsing via `docling` (PDF, DOCX, XLSX) or `trafilatura` (web URL)
3. Structural chunking:
   - Primary chunks: ~512 tokens, split at section/paragraph boundaries
   - Parent chunks: ~1536 tokens (3x primary), used as context window when a primary chunk is retrieved
   - Metadata attached: `{source_file, page_number, section_title, upload_date}`
4. Embed primary chunks with `bge-m3` (dense vectors)
5. Generate BM25 sparse representations from `bge-m3`
6. Store in ChromaDB with metadata

**Retrieval (on user query):**

1. **Language detection** on user query (simple heuristic or `langdetect`)
2. **Query rewriting** (P1): LLM rewrites ambiguous query into a clear retrieval query
3. **Hybrid search**: 
   - Dense vector search → top 20 candidates
   - BM25 sparse search → top 20 candidates
   - Reciprocal Rank Fusion (RRF) to merge → top 15 candidates
4. **Cross-encoder reranking** with `bge-reranker-v2-m3` → top 5 chunks
5. **Confidence check**: if top reranked score < 0.35 (calibrate empirically), trigger fallback
6. **Parent expansion**: replace primary chunks with their parent chunks for richer context
7. **Context assembly**: format top 3-5 parent chunks with source metadata into LLM prompt

**Generation:**

1. System prompt (see Section 6)
2. Retrieved context injected with clear source markers: `[Source: filename.pdf, Page 3]`
3. LLM generates answer in user's detected language
4. Post-processing: extract citation markers, map to source metadata
5. Stream response to frontend

### 5.4 System Prompt (Generation)

```
You are the BINUS School of Computer Science information assistant. 
You answer questions ONLY based on the provided context documents. 

RULES:
1. If the context contains the answer, provide it clearly and cite the source using [Source: filename, Page X].
2. If the context does NOT contain the answer, say: "I don't have information about that in my current documents. Please contact [FALLBACK_CONTACT] for help."
3. NEVER fabricate information. NEVER answer from general knowledge.
4. Match the language of the user's question. If they ask in Indonesian, respond in Indonesian. If in English, respond in English.
5. For ambiguous questions, ask a clarifying question before answering.
6. Keep answers concise. Use bullet points for lists. Cite every factual claim.

CONTEXT:
{retrieved_chunks_with_source_markers}

USER QUESTION: {user_query}
```

### 5.5 Fallback Configuration

```json
{
  "fallback_message": {
    "id": "Maaf, saya belum memiliki informasi tersebut. Silakan hubungi:",
    "en": "Sorry, I don't have that information. Please contact:"
  },
  "contacts": [
    {
      "role": "Academic Administration",
      "name": "SoCS Academic Office",
      "whatsapp": "https://wa.me/62XXXXXXXXXX",
      "email": "socs@binus.edu"
    },
    {
      "role": "Student Affairs",
      "name": "Student Services",
      "whatsapp": "https://wa.me/62XXXXXXXXXX",
      "email": "studentservices@binus.edu"
    }
  ]
}
```

This is a config file, not hardcoded. Admin can update contacts without code changes.

---

## 6. UI/UX Specification

### 6.1 Layout — NotebookLM-Adapted

**Desktop (>768px):**
```
┌────────────────────────────────────────────────────────┐
│  BINUS SoCS Assistant                    [Admin ⚙]     │
├───────────────────────────────┬────────────────────────┤
│                               │                        │
│       CHAT PANEL              │    SOURCES PANEL       │
│                               │                        │
│  [Starter questions]          │  Referenced documents  │
│                               │  appear here when      │
│  User: ...                    │  the bot responds.     │
│  Bot: ... [1] [2]             │                        │
│                               │  [1] handbook.pdf p.3  │
│                               │  [2] curriculum.xlsx   │
│                               │  → expandable chunk    │
│                               │    preview             │
│  ┌──────────────────────┐     │                        │
│  │ Type your question...│     │                        │
│  └──────────────────────┘     │                        │
├───────────────────────────────┴────────────────────────┤
│  ⚠ AI-generated. Verify critical info with faculty.    │
└────────────────────────────────────────────────────────┘
```

**Mobile (<768px):**
- Sources panel collapses into a bottom sheet triggered by tapping citation badges
- Chat takes full width
- Input bar fixed at bottom with safe area padding

### 6.2 Visual Identity

| Element | Specification |
|---------|---------------|
| Primary color | BINUS blue `#00529B` |
| Font | Inter (system fallback: -apple-system, sans-serif) |
| Bot avatar | BINUS SoCS logo (small, 32px) |
| Message bubbles | Left-aligned bot (light gray bg), right-aligned user (blue bg, white text) |
| Citation badges | Inline pill `[1]` in bot messages, blue outline, tappable |
| Dark mode | Not in prototype scope |

### 6.3 Starter Questions (Configurable)

Displayed as tappable chips on first load:
- "Apa saja program studi di SoCS?" / "What programs does SoCS offer?"
- "Bagaimana cara mengajukan cuti akademik?" / "How do I apply for academic leave?"
- "Apa syarat kelulusan program Data Science?" / "What are the graduation requirements for Data Science?"
- "Kapan jadwal UTS semester ini?" / "When are midterm exams this semester?"

Stored in a config file. Admin-editable without code changes.

---

## 7. Non-Functional Requirements

| Requirement | Target | Notes |
|-------------|--------|-------|
| Response latency (first token) | < 3 seconds | Groq inference is fast (~200 tok/s). Bottleneck is reranker. |
| Response latency (complete) | < 10 seconds | For typical 200-token responses. |
| Concurrent users | 5-10 (prototype) | Groq free tier is the binding constraint at 30 RPM. |
| Availability | Best-effort (local machine) | No SLA for prototype. |
| Data privacy | No PII collected. No user accounts. Session-only conversation. | IP-based rate limiting stores IPs in memory only, not persisted. |
| Document ingestion time | < 60 seconds per document | For typical 20-page PDF. |
| Embedding model memory | ~2-3 GB VRAM or RAM | bge-m3 runs on CPU if no GPU, slower but functional. |
| Reranker memory | ~1-2 GB | Same: CPU-capable. |

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Groq free tier rate limit exhaustion** | High | IP-based rate limiting (C-10). Queue requests with backoff. Display "busy" message. For production, budget for paid tier or self-host Llama 3.3. |
| **Hallucination on out-of-scope questions** | High | Strict system prompt grounding (R-06). Confidence threshold triggers fallback (R-05). Disclaimer banner (C-11). |
| **Poor retrieval on Indonesian queries** | Medium | bge-m3 is multilingual-native. Test empirically with real student questions. Adjust chunk sizes if needed. |
| **Knowledge base poisoning via admin panel** | Medium | Password protection (A-01). Prototype risk only — production needs proper auth. |
| **Complex tables in PDFs not parsed correctly** | Medium | docling handles tables better than alternatives. Manual review of parsed output for critical documents. |
| **bge-m3 + reranker too slow on CPU** | Medium | Quantize models (ONNX int8). Or run on a machine with a GPU. Prototype can tolerate 2-3s retrieval latency. |

---

## 9. Success Criteria (Prototype)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Answer relevance | >80% of test queries answered correctly from documents | Manual evaluation: 50 test questions, graded by admin |
| Fallback accuracy | >90% of out-of-scope queries correctly trigger fallback | Same test set, 20 out-of-scope questions |
| Retrieval precision | >70% of retrieved chunks are relevant to the query | Spot-check top-5 chunks for 30 queries |
| User feedback | >60% positive (thumbs up) | In-app feedback widget (C-09) |
| Latency | First token < 3s for 90% of queries | Server-side logging |

---

## 10. Project Structure

```
binus-socs-chatbot/
├── docker-compose.yml
├── .env.example                  # GROQ_API_KEY, ADMIN_PASSWORD
│
├── backend/
│   ├── main.py                   # FastAPI app, routes
│   ├── config.py                 # Settings, fallback contacts, starter questions
│   ├── rag/
│   │   ├── ingestion.py          # Document parsing, chunking, embedding, indexing
│   │   ├── retrieval.py          # Hybrid search, reranking, confidence scoring
│   │   ├── generation.py         # LLM prompt assembly, streaming, citation extraction
│   │   └── models.py             # Load bge-m3, bge-reranker at startup
│   ├── admin/
│   │   ├── routes.py             # Upload, delete, list documents
│   │   └── auth.py               # Password middleware
│   ├── documents/                # Uploaded raw files stored here
│   └── vectorstore/              # ChromaDB persistent storage
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatPanel.jsx     # Message list, input bar, streaming
│   │   │   ├── SourcePanel.jsx   # Citation display, chunk preview
│   │   │   ├── AdminPanel.jsx    # Upload, manage, password gate
│   │   │   └── StarterChips.jsx  # Suggested questions
│   │   └── hooks/
│   │       └── useChat.js        # WebSocket/SSE connection, message state
│   ├── tailwind.config.js
│   └── vite.config.js
│
└── scripts/
    ├── seed_kb.py                # Bulk-load initial documents
    └── eval.py                   # Run test questions, measure accuracy
```

---

## 11. Implementation Phases

### Phase 1: Core RAG (Week 1-2)

- Set up FastAPI backend with LlamaIndex
- Implement ingestion pipeline: PDF + DOCX parsing with docling
- Configure bge-m3 embeddings + ChromaDB
- Basic retrieval (dense only, no reranker yet)
- Groq integration with streaming
- Minimal chat UI (no source panel yet)
- Test with 5-10 real BINUS SoCS documents

### Phase 2: Quality + UI (Week 3)

- Add BM25 hybrid search + RRF
- Add bge-reranker-v2-m3 cross-encoder reranking
- Implement confidence-based fallback
- Build source panel with citation mapping
- Add admin panel (upload, delete, list)
- Add XLSX + web scraping support
- Starter questions + feedback widget

### Phase 3: Hardening (Week 4)

- IP-based rate limiting
- Docker Compose setup for reproducible deployment
- Evaluation script with 50 test questions
- Performance tuning (quantization if needed)
- Documentation: setup guide, admin guide
- Prepare migration notes for BINUS server deployment

---

## 12. Dependencies and Prerequisites

| Dependency | Action Required |
|------------|-----------------|
| Groq API key (free tier) | Register at console.groq.com |
| Python 3.11+ | Install locally or use Docker |
| Node.js 20+ | For frontend build |
| ~8GB RAM minimum | For bge-m3 + reranker on CPU |
| GPU (optional, recommended) | NVIDIA with 6GB+ VRAM significantly speeds embedding and reranking |
| 5-10 real SoCS documents for initial KB | Henry to supply: student handbook, curriculum docs, academic calendar, FAQ sheets |
| Fallback contact details | WhatsApp numbers and email addresses for SoCS admin and student services |
| BINUS SoCS logo | For bot avatar and branding |

---

## 13. Future Considerations (Out of Prototype Scope)

These are explicitly **not** in the prototype. Listed to prevent scope creep.

- User authentication (BINUS SSO integration)
- Role-based access control for admin panel
- Conversation persistence across sessions
- Analytics dashboard (query volume, common topics, fallback rate)
- Multi-turn conversation memory with context window management
- Automated document re-ingestion on file change detection
- Integration with BINUS academic systems (SAT, student portal)
- Voice input (speech-to-text)
- Proactive notifications (e.g., "registration deadline in 3 days")
- Fine-tuned embedding model on BINUS-specific corpus
- Self-hosted LLM to eliminate API dependency

---

## 14. Open Questions

| # | Question | Owner | Deadline |
|---|----------|-------|----------|
| 1 | Which specific documents should seed the initial KB? | Henry | Before Phase 1 |
| 2 | Exact WhatsApp numbers and emails for fallback contacts? | Henry | Before Phase 2 |
| 3 | Is there a BINUS brand guide with exact colors/logo assets? | Henry | Before Phase 2 |
| 4 | Will a GPU-equipped machine be available for local development? | Henry | Before Phase 1 |
| 5 | Who else on the team will test and evaluate prototype quality? | Henry | Before Phase 3 |
