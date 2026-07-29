import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = ""
    embedding_device: str = "cuda"
    # CPU reranking added 10-15s of CPU-bound latency per query (568M-param cross-encoder).
    # GPU + fp16 keeps latency sub-second while using ~half the VRAM of the original fp32
    # GPU setup, reducing (without eliminating) contention with other GPU processes on the host.
    reranker_device: str = "cuda"

    embedding_model_name: str = "BAAI/bge-m3"
    # llama-index's HuggingFaceEmbedding defaults to 10, which under-uses a CUDA GPU's
    # batching parallelism during a from-scratch reseed (hundreds/thousands of chunks
    # embedded sequentially in small batches). Benchmarked directly on this project's
    # GPU (RTX 3080, 10GB VRAM) with bge-reranker-v2-m3 also resident (the real
    # /admin/documents-time condition, not an idle GPU): 32 gave ~238->308 chunks/sec,
    # a ~29% gain over the default 10, using well under half the available VRAM (peak
    # ~3.3GB). Pushing further to 64/128 didn't reliably improve on that in repeated
    # runs -- retune if the model, GPU, or reranker co-residency changes.
    embedding_batch_size: int = 32
    # Temporarily pinned to llama-3.1-8b-instant (2026-07-09) while
    # meta-llama/llama-4-scout-17b-16e-instruct's daily quota is exhausted -- llama-4-scout
    # remains the better-verified default (see PROJECT_LOG's 2026-07-09 entries: it's the
    # only candidate that reliably followed the program-name-conflation guard, SYSTEM_PROMPT
    # rule 8, in eval; 8b-instant was observed to miss it). Switch back once quota resets or
    # the user says so explicitly -- don't revert this unilaterally.
    llm_model: str = "llama-3.1-8b-instant"
    # Pinned EXPLICITLY rather than relying on llama-index's Groq default (currently 0.1),
    # which a library upgrade could silently change. For a grounded, no-fabrication RAG
    # assistant, deterministic/faithful generation matters more than creative variety, so
    # 0.0: the answer should be the same reading of the same context every time, and the
    # model shouldn't sample its way into plausible-but-unsupported additions (the mild
    # career-list over-generation observed in the 2026-07-27 grading is exactly that risk).
    llm_temperature: float = 0.0
    # Forwarded as an extra request-body field (Groq's OpenAI-compatible endpoint), not
    # a llama-index-native param -- see backend/rag/models.py. Only meaningful for
    # reasoning models (gpt-oss-*, qwen3.6-*); ignored by plain instruct models like the
    # llama-3.3/3.1 family above. None means "don't send the field at all."
    llm_reasoning_effort: str | None = None
    # Uncapped before this -- a smaller instruct model like the currently-pinned
    # llama-3.1-8b-instant is more prone than a larger model to a repetition/runaway
    # generation on a longer answer (e.g. a multi-program comparison table). 1024 is
    # generous relative to the prompt's own "keep answers concise" rule (comfortably
    # covers even a detailed comparison table with citations) while bounding worst-case
    # latency and per-response token spend.
    llm_max_tokens: int = 1024
    # Mild repetition guard for the same reason as llm_max_tokens above -- smaller
    # instruct models are more prone to repetition loops on longer generations. Verified
    # directly against this model on the two most repetition-heavy answer shapes this
    # project produces (a multi-row comparison table, and a bullet list citing the same
    # source on every line) before picking this value: 0.3 left both completely
    # unchanged, and even 0.8 didn't corrupt table syntax or drop citation markers --
    # chose 0.3 anyway as the smallest value that showed any effect, since there was no
    # observed upside to going further and it costs nothing to stay conservative.
    llm_frequency_penalty: float = 0.3

    # Generation provider (see backend/rag/models.py). Only the generation model differs --
    # the embedding/retrieval/rerank stack is identical for all three:
    #   "openai" (default) -- OpenAI's own models (gpt-4o-mini); chosen as the primary
    #            after a 2026-07-29 head-to-head: best out-of-the-box fallback accuracy
    #            (incl. Indonesian), reliable, cheap (~$3/mo at ~5k req).
    #   "groq"   -- the pinned open Llama model via Groq's OpenAI-compatible endpoint.
    #            RETAINED but dormant: fastest + cheapest per token, so kept fully working
    #            for future reactivation (set LLM_PROVIDER=groq); parked in mid-2026 because
    #            the account was 403-blocked and its pro tier was unavailable.
    #   "gemini" -- Google's Gemini Flash (see gemini_model). Also selectable.
    llm_provider: str = "openai"
    openai_api_key: str = ""
    # OpenAI generation model when llm_provider="openai". gpt-4o-mini is the cheap,
    # well-established mini tier; gpt-4.1-mini / a current gpt-5-mini are stronger if
    # available on the account -- verify with the models endpoint before pinning.
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    # Gemini generation model when llm_provider="gemini". flash-lite-latest chosen
    # empirically (2026-07-29) as the fast, available, clean-output option on a new
    # free-tier key: thinking is OFF by default (so no token-starvation, ~1.7s responses),
    # and it's a big step up from the 8B model. The alternatives all had blockers that day:
    # gemini-2.5-flash is retired for new keys (404); gemini-3.5-flash accepts thinking off
    # but was 503-congested; gemini-3.6-flash / flash-latest keep thinking ON and (under the
    # 1024 token cap) let it eat the answer -> garbled fragments. Google rotates ids and
    # capacity, so re-verify with models.list and a timed call before pinning a full-flash tier.
    gemini_model: str = "gemini-flash-lite-latest"
    # Gemini "thinking" budget. None = don't send a thinking_config at all -- correct for
    # flash-lite (thinking already off; it 400s if the field is sent) and any model that
    # rejects the field. Set to 0 to explicitly DISABLE thinking on a full-flash tier that
    # supports it (e.g. gemini-3.5-flash), which is essential there for chatbot latency.
    gemini_thinking_budget: int | None = None

    documents_dir: Path = BASE_DIR / "backend" / "documents"
    avatar_dir: Path = BASE_DIR / "backend" / "avatar"
    users_path: Path = BASE_DIR / "backend" / "users.json"
    chroma_persist_dir: Path = BASE_DIR / "backend" / "vectorstore"
    chroma_collection_name: str = "binus_socs_kb"
    feedback_log_path: Path = BASE_DIR / "backend" / "feedback.jsonl"
    # Query/fallback log (IMPROVEMENTS.md #6.1) -- distinct from feedback.jsonl (explicit
    # thumbs up/down on a shown answer): this logs every /chat request, most valuably
    # which ones fell back, since that's a direct signal of KB content gaps that
    # otherwise only surfaced in ad-hoc eval runs.
    query_log_path: Path = BASE_DIR / "backend" / "query_log.jsonl"
    # Whether the query log also records the ASSISTANT'S RESPONSE (answer text + cited
    # source files), not just the input/routing metadata. OFF by default: answers are long
    # free text, so logging them grows the file fast and stores generated content that has
    # a real privacy/retention cost the short-query fields don't. Turn on (LOG_RESPONSES=true
    # in .env) when you want to audit answer QUALITY from the log -- catch a confidently-wrong
    # answer or a hallucination, which the `fallback`/`top_score` fields alone can't reveal.
    # Answer text is truncated to log_response_max_chars to bound growth.
    log_responses: bool = False
    log_response_max_chars: int = 2000
    starter_questions_path: Path = BASE_DIR / "backend" / "starter_questions.json"
    fallback_contacts_path: Path = BASE_DIR / "backend" / "fallback_contacts.json"
    # Persisted record of every URL scraped via /admin/documents/url (IMPROVEMENTS.md
    # #5.1) -- /admin/reindex only rebuilds from documents_dir on disk by nature, so
    # without this list a full reindex would silently drop every previously-scraped
    # web page.
    scraped_urls_path: Path = BASE_DIR / "backend" / "scraped_urls.json"
    # Cached snapshot of the scraped+API-enriched faculty roster (name/rank/courses/campus
    # per lecturer). The faculty source is a fragile crawl (undocumented AJAX + a hardcoded
    # token + ~70 profile fetches); once captured here, /admin/reindex rebuilds the faculty
    # nodes from this file OFFLINE instead of re-crawling, so a routine reindex can't
    # silently degrade the roster when BINUS changes their site. Refreshed only by an
    # explicit /admin/faculty/refresh.
    faculty_snapshot_path: Path = BASE_DIR / "backend" / "faculty_snapshot.json"
    # Last-known-good cache of the chunks each scraped URL last produced (serialized nodes,
    # keyed by URL). On /admin/reindex a URL that fails to re-fetch (page moved, network
    # blip, restyled HTML, rate-limit) would otherwise vanish from the KB entirely; instead
    # its cached chunks are reused, so a transient/partial fetch failure degrades to
    # stale-but-present content rather than silent loss. Refreshed automatically on every
    # SUCCESSFUL scrape (add-url and reindex).
    url_cache_path: Path = BASE_DIR / "backend" / "url_cache.json"

    # Parent-child chunking (R-02): retrieval/reranking runs on small child chunks for
    # precision; the LLM gets the larger parent chunk a child belongs to for fuller context.
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 100
    child_chunk_size: int = 256
    child_chunk_overlap: int = 50

    # Multi-turn: how many prior messages (user+assistant combined) to include in the
    # generation prompt and in the follow-up condensing step. Capped to bound both LLM
    # token cost and the risk of stale earlier turns crowding out the current question.
    max_history_messages: int = 6

    retrieval_top_k: int = 20  # dense and BM25 each retrieve this many before fusion
    fusion_top_k: int = 15  # candidates kept after reciprocal-rank fusion, pre-rerank
    rerank_top_n: int = 5  # final chunks kept after cross-encoder reranking
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    # Re-calibrated 2026-07-09 against the current 10-doc SOCS KB using the full 66-
    # question eval set's real top_scores (scripts/eval.py), not just probe_confidence.py's
    # 6 -- the original 0.15 was set against the old 87-doc corpus, where in-scope queries
    # scored as low as 0.34; on the current KB the in-scope floor has moved up to 0.72
    # (smaller, more topically-homogeneous corpus -> higher-confidence genuine matches),
    # leaving 0.15 far more conservative than it needs to be.
    #
    # The should-answer (0.72-0.995) and should-decline (0.0004-0.947) score ranges
    # overlap and can NOT be cleanly separated by any single threshold: a genuinely
    # unrelated query ("recommend a pizza topping" style) can score 0.79 -- inside the
    # in-scope range -- because it superficially matches unrelated catalog boilerplate,
    # while "Computer Science International" (an archived, unindexed program) scores
    # 0.947 by matching the real "Computer Science" catalog. Those two cases are NOT
    # threshold problems -- they're caught by generation-side rules (SYSTEM_PROMPT rule 2
    # "if context doesn't answer, decline" and rule 8 "don't conflate similarly-named
    # programs"), verified 100% correct in eval regardless of this threshold.
    #
    # What IS a real, safely-exploitable gap: every should-decline score below 0.72 is
    # 0.54 or under, and every in-scope score is 0.72 or over -- a genuine ~0.18-wide
    # empirical gap. Moved the threshold from 0.15 to 0.5, comfortably inside that gap
    # (0.22 margin below the observed in-scope floor). This reclassifies several
    # should-decline queries that previously passed the gate and relied on the LLM's own
    # judgment (Industrial Engineering 0.16, Animation 0.23, "theory of relativity" 0.25,
    # Business Management 0.29, Visual Communication Design 0.34, "best smartphone" 0.46)
    # into ones the gate itself now correctly rejects -- fewer wasted generation calls,
    # and one less thing resting on the LLM following instructions instead of a hard gate.
    # Verify against a fresh eval run if the KB or reranker ever changes materially.
    confidence_threshold: float = 0.5

    # Semantic cache (IMPROVEMENTS.md #3.1). Deliberately loose -- this only narrows
    # candidates down to "plausibly the same question" before the real safety gates
    # (program-entity match + aspect match) run. See backend/rag/cache.py's module
    # docstring: a bare embedding-similarity threshold is NOT safe here on its own --
    # calibrated 2026-07-10, genuine paraphrases and dangerous near-misses (different
    # program, or same program/different aspect) score in a completely overlapping
    # 0.74-0.86 range on bge-m3.
    semantic_cache_prefilter_threshold: float = 0.85
    semantic_cache_max_entries: int = 200

    # Soft daily token budget (IMPROVEMENTS.md #3.2) -- Groq's daily token limit (TPD)
    # was exhausted from eval traffic alone during development; nothing stops real
    # traffic (or abuse) from doing the same in production. 400_000 sits comfortably
    # under llama-3.1-8b-instant's 500K TPD, leaving margin for eval/dev traffic sharing
    # the same key -- retune if the pinned model changes (see llm_model's comment above).
    # 0 disables the check entirely. See backend/rag/token_budget.py.
    daily_token_budget: int = 400_000

    # Knowledge-base staleness (IMPROVEMENTS.md #5.2): tuition/admission info changes
    # yearly and catalogs are swapped manually with no re-scrape schedule -- flag any
    # source older than this in the admin UI as a nudge to check for a newer version,
    # rather than letting it silently go out of date.
    staleness_threshold_days: int = 365

    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()


def validate_startup_config() -> None:
    """Fail fast on critical misconfiguration at server boot, rather than at the first
    request that happens to need it (IMPROVEMENTS.md #4.3).

    groq_api_key defaults to "" and is accepted silently by the Groq client at
    construction time (backend/rag/models.py) -- with no validation, a missing key
    previously surfaced only as a cryptic auth error deep inside the first /chat call's
    LLM request, well after the server had already reported itself running. Deliberately
    scoped to backend/main.py's server startup only (called from lifespan(), not from
    Settings itself) -- several scripts (probe_confidence.py and others) call
    init_models() and legitimately never invoke the LLM, so a blanket pydantic-level
    validator on Settings would break those for no reason.
    """
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key.strip():
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set. Add it to .env "
                "(GEMINI_API_KEY=...) or switch LLM_PROVIDER back to groq."
            )
        return
    if settings.llm_provider == "openai":
        if not settings.openai_api_key.strip():
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. Add it to .env "
                "(OPENAI_API_KEY=sk-...) or switch LLM_PROVIDER back to groq."
            )
        return
    if not settings.groq_api_key.strip():
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env (GROQ_API_KEY=gsk_...) or set it as "
            "an environment variable before starting the server -- see README.md's setup "
            "section."
        )


# Model-facing prompts (system/user templates, classifier, condense, etc.) live in
# backend/rag/prompts.py. The contact-driven, user-facing message templates below stay
# here, since they're content an admin effectively edits (via fallback_contacts), not
# prompt engineering.

# Per PRD §5.5: contacts live in a config file, not hardcoded, so admins can update them
# without code changes. Read fresh on every call (not cached) so an admin edit takes effect
# immediately, with no server restart.
def load_fallback_contacts() -> list[dict]:
    return json.loads(settings.fallback_contacts_path.read_text(encoding="utf-8"))


def _format_contacts() -> str:
    blocks = []
    for c in load_fallback_contacts():
        blocks.append(
            f"{c['name']} ({c['role']})\nEmail: {c['email']}\nWhatsApp: {c['whatsapp']}"
        )
    return "\n\n".join(blocks)


# Deliberately does NOT embed the contact block (unlike SERVICE_ERROR_* below): contacts
# now ride along in the SSE 'done' event as structured data so the frontend can render a
# proper contact card instead of a wall of text inside the message bubble -- the way
# support bots normally do a human handoff. Also deliberately avoids naming the retrieval
# internals ("in my current documents"), which leaked implementation detail no user cares
# about; a support bot just says it doesn't have the answer.
FALLBACK_MESSAGE_TEMPLATES = {
    "id": (
        "Saya tidak menemukan jawaban untuk pertanyaan tersebut. Coba ajukan pertanyaan "
        "dengan cara lain, atau hubungi tim kami:"
    ),
    "en": (
        "I couldn't find an answer to that. Try rephrasing your question, or reach out to "
        "our team:"
    ),
}


def get_fallback_message(lang: str) -> str:
    """The fallback copy alone -- see FALLBACK_MESSAGE_TEMPLATES for why contacts are not
    included here. Callers that need the contacts send them separately (see
    generation.stream_answer's 'done' event)."""
    return FALLBACK_MESSAGE_TEMPLATES[lang]


# Distinct from get_fallback_message: this is for when the LLM call itself fails (rate
# limit, network, provider outage) rather than the documents lacking an answer -- telling
# a user "I don't have that information" when the real issue is a service hiccup is
# misleading, since rephrasing won't help.
SERVICE_ERROR_MESSAGE_TEMPLATES = {
    "id": (
        "Maaf, saya sedang mengalami gangguan teknis dan tidak dapat menjawab saat ini. "
        "Silakan coba lagi dalam beberapa menit, atau hubungi kami langsung:\n{contacts}"
    ),
    "en": (
        "Sorry, I'm having technical trouble answering right now. Please try again in a "
        "few minutes, or contact us directly:\n{contacts}"
    ),
}


def get_service_error_message(lang: str) -> str:
    return SERVICE_ERROR_MESSAGE_TEMPLATES[lang].format(contacts=_format_contacts())
