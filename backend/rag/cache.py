"""In-memory semantic cache for the /chat endpoint (IMPROVEMENTS.md #3.1).

A public FAQ bot gets the same handful of questions constantly. Every one currently runs
the full retrieve -> rerank -> LLM pipeline even though the answer is deterministic given
the same KB. Serving a genuine near-duplicate straight from cache skips retrieval AND
generation entirely -- real latency and token savings, directly mitigating the daily-quota
exhaustion this project has repeatedly hit.

SAFETY -- read before touching the thresholds below. Naive whole-query embedding
similarity is NOT safe on its own for this KB. Direct calibration (2026-07-10, bge-m3)
found genuine paraphrases and dangerous near-misses (same template, different program, or
same program, different question aspect) score in COMPLETELY OVERLAPPING cosine-similarity
ranges:
    genuine paraphrases:  0.74, 0.92, 0.98
    dangerous near-misses: 0.74, 0.74, 0.80, 0.80, 0.81, 0.85, 0.86
No threshold separates these. A false hit here is worse than a bad retrieval score: it
returns a fully-formed, confident answer about the WRONG program or WRONG aspect, with no
LLM judgment in the loop to catch it (unlike retrieval, which still has generation as a
backstop). Every cache hit therefore requires TWO deterministic gates in addition to the
embedding pre-filter -- both must pass, see is_safe_cache_hit:
  1. Program-entity gate: the same named program(s), via detect_named_programs
     (backend/rag/generation.py) -- reuses the classifier call the normal pipeline needs
     anyway, so this isn't a wasted extra LLM call except on what would otherwise have
     been a trusted hit.
  2. Aspect gate: overlapping coarse topic tags (career/curriculum/tuition/admission/
     outcome/scholarship) via detect_aspects below -- keyword-based, no LLM call.
     Ambiguous aspect (empty set) on either side means "don't trust it," not "wildcard
     match" -- conservative by design.

Further scope, also deliberately conservative (enforced by the caller, backend/main.py):
  - Only applied when the conversation has no prior history (a fresh, first-turn
    question) -- what's contextually appropriate in a multi-turn conversation can depend
    on more than the resolved standalone question's literal meaning.
  - Never applied to smalltalk (that path is already cheap and meant to read as varied,
    natural small talk, not a canned reply).
  - Partitioned by detected language, so a cross-lingual embedding match can never surface
    an answer in the wrong language (SYSTEM_PROMPT requires matching the question's
    language).
  - In-memory, single-process, bounded size with FIFO eviction -- same "fine for a
    single-process prototype" philosophy as the rate limiter (backend/main.py). Cleared
    entirely on any KB mutation or fallback-contacts update (backend/admin/routes.py),
    since a cached answer -- including a cached FALLBACK message, which embeds the current
    contact list -- is only valid for the state it was generated against.
"""
import numpy as np

from backend.config import settings

_cache: list[dict] = []

_ASPECT_KEYWORDS = {
    "career": ["career", "job", "prospect", "karir", "kerja", "peluang", "profesi"],
    "curriculum": ["curriculum", "course", "subject", "kurikulum", "mata kuliah", "mata pelajaran"],
    # "harga"/"bayar" are the everyday Indonesian words a prospective student actually
    # types ("berapa harga jurusan X"); without them detect_aspects returned an empty set
    # for a plainly-tuition question, which both skipped the campus-balanced tuition retry
    # (chat_service._retry_with_supplementary_sources gates on "tuition" in plan.aspects)
    # and made every such question un-cacheable via the aspect gate.
    "tuition": [
        "tuition", "fee", "cost", "price", "biaya", "spp", "uang kuliah", "harga", "bayar",
    ],
    "admission": ["admission", "apply", "application", "requirement", "pendaftaran", "syarat", "masuk"],
    "outcome": ["outcome", "competenc", "capaian", "kompetensi"],
    "scholarship": ["scholarship", "beasiswa"],
}


def detect_aspects(query: str) -> set[str]:
    """Coarse, keyword-based topic tags (e.g. {"career"}) for a query. Deterministic, no
    LLM call -- see module docstring for why this exists as a cache-safety gate. Empty
    set ("ambiguous") if no known aspect keyword is found; the caller treats that as "not
    safe to trust," never as a wildcard.
    """
    query_lower = query.lower()
    return {
        aspect
        for aspect, keywords in _ASPECT_KEYWORDS.items()
        if any(kw in query_lower for kw in keywords)
    }


def clear_semantic_cache() -> None:
    _cache.clear()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_cache_candidate(query_embedding, language: str) -> dict | None:
    """Returns the closest cached entry in the same language, if it clears the loose
    embedding pre-filter (settings.semantic_cache_prefilter_threshold), else None.

    NOT sufficient on its own to trust a hit -- deliberately loose, just narrows down to
    "plausibly the same question" before the real safety gates run. The caller MUST
    additionally call is_safe_cache_hit before ever serving this candidate's answer.
    """
    best_entry = None
    best_score = -1.0
    for entry in _cache:
        if entry["language"] != language:
            continue
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry is not None and best_score >= settings.semantic_cache_prefilter_threshold:
        return best_entry
    return None


def is_safe_cache_hit(new_matched_programs: list[str], new_aspects: set[str], candidate: dict) -> bool:
    """The two deterministic gates a cache candidate must clear before being served --
    see module docstring for why the embedding pre-filter alone is not enough."""
    if sorted(new_matched_programs) != sorted(candidate["matched_programs"]):
        return False
    if not new_aspects or not candidate["aspects"] or not (new_aspects & candidate["aspects"]):
        return False
    return True


def store_answer(
    query_embedding,
    language: str,
    matched_programs: list[str],
    aspects: set[str],
    answer: str,
    sources: list,
    is_comparison: bool,
    is_fallback: bool = False,
    follow_ups: list[str] | None = None,
) -> None:
    # is_fallback/follow_ups (IMPROVEMENTS.md #9.3/#9.4): replayed as-is on a cache hit
    # (stream_cached_answer) rather than recomputed, since they describe properties of
    # THIS specific cached answer at the time it was generated.
    _cache.append(
        {
            "embedding": query_embedding,
            "language": language,
            "matched_programs": matched_programs,
            "aspects": aspects,
            "answer": answer,
            "sources": sources,
            "is_comparison": is_comparison,
            "is_fallback": is_fallback,
            "follow_ups": follow_ups or [],
        }
    )
    if len(_cache) > settings.semantic_cache_max_entries:
        _cache.pop(0)  # FIFO eviction, oldest first
