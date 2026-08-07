"""Orchestration for a single /chat turn, extracted from the FastAPI route so the route
stays a thin adapter (parse request -> ChatService.stream -> StreamingResponse) and the
RAG decision logic lives in one testable place.

The turn is split into two phases:
  * ChatService._plan() -- side-effect-free: condense the follow-up, check the semantic
    cache, classify named programs, and route retrieval (comparison / single-program /
    out-of-catalog fallback / open + rewrite retry). Returns a Plan describing what to do
    but performs no streaming, logging, or cache writes.
  * ChatService.stream() -- turns a Plan into the SSE token stream, applying the daily
    token budget gate, attaching follow-up suggestions, and wrapping the stream to
    populate the semantic cache and append the query log as post-stream side effects.

Splitting decision from I/O is what makes the routing unit-testable without a live
model, and keeps each method small enough to read top to bottom.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Mapping

from llama_index.core import Settings as LlamaSettings
from llama_index.core.schema import NodeWithScore

from backend.config import get_service_error_message, settings
from backend.rag.cache import (
    detect_aspects,
    get_cache_candidate,
    is_safe_cache_hit,
    store_answer,
)
from backend.rag.generation import (
    comparison_attribute_query,
    condense_question,
    detect_language,
    detect_named_programs,
    detect_unresolved_campus_mention,
    detect_unresolved_program_mention,
    has_domain_vocabulary,
    is_campus_programs_query,
    is_leadership_query,
    is_prompt_extraction_attempt,
    is_smalltalk,
    is_who_teaches_query,
    normalize_campus_aliases,
    resolve_named_campus,
    rank_clarification_suggestions,
    resembles_known_entity,
    rewrite_query,
    strip_retrieval_filler,
    stream_answer,
    stream_budget_exceeded,
    stream_cached_answer,
    stream_clarification,
    stream_prompt_extraction_refusal,
    stream_smalltalk_reply,
    suggest_follow_ups,
)
from backend.rag.ingestion import (
    admission_requirement_url_for_campus,
    FACULTY_ROSTER_URL,
    known_campus_names,
    load_scraped_urls,
)
from backend.rag.retrieval import (
    get_program_catalog,
    retrieve_and_rerank,
    retrieve_for_named_programs,
)
from backend.rag.token_budget import is_budget_exceeded

logger = logging.getLogger(__name__)

# Substring shared by every BINUS per-campus tuition-fee page URL (see ingestion.py's
# _TUITION_FEE_URL_RE) -- used to pick the "campus family" out of load_scraped_urls()
# for the balanced tuition retry below, without importing ingestion's full regex just
# for a membership check.
_TUITION_FEE_URL_SUBSTR = "gabung.binus.ac.id/tuition-fee/"

# Lower confidence gate for the faculty-scoped who-teaches path than the default 0.5. Safe
# BECAUSE that retrieval is already restricted to the faculty roster: a real subject's top
# lecturer reranks ~0.4-0.9 while a subject nobody teaches reranks ~0.0 (measured), so this
# cleanly separates "weak but real" from "nonsense" without the higher gate rejecting real
# English who-teaches queries (which rerank lower than their Indonesian equivalents).
_WHO_TEACHES_GATE = 0.25
# Hand the model only the top few lecturers so its answer stays a short "2-3 + others" list
# -- deterministic backstop, since the prompt's "at most 3" alone doesn't hold on this model.
_WHO_TEACHES_MAX_LECTURERS = 3


@dataclass
class Plan:
    """The outcome of the decision phase: everything ChatService.stream() needs to
    produce the response, decided before any generation call."""

    standalone_query: str
    nodes: list[NodeWithScore] = field(default_factory=list)
    is_comparison: bool = False
    matched_programs: list[str] = field(default_factory=list)
    aspects: set[str] = field(default_factory=set)
    # Full catalog program-name list, captured in _plan() so stream() can rank program
    # "did you mean" suggestions without re-reading the index.
    program_names: list[str] = field(default_factory=list)
    # A "who teaches X" question -> route to the faculty roster (their courses), never X's
    # program catalog (see _route_retrieval). leadership -> same faculty scoping but for a
    # "who is the head of X / dean" question (their structural role), named singly, no cap.
    who_teaches: bool = False
    leadership: bool = False
    # "Ask, don't guess": a campus/program token the query names but retrieval couldn't
    # resolve, set only when retrieval also came up empty. Non-None routes stream() to a
    # clarifying question instead of a silent fallback (see _route_retrieval's tail).
    unresolved_campus_mention: str | None = None
    unresolved_program_mention: str | None = None
    # Overrides the raw user message as the question shown to the model at generation, for the
    # one case where the message names something by an alias the retrieved context never uses
    # verbatim: the campus-programs route resolves e.g. "anggrek" -> Kemanggisan and scopes to
    # Kemanggisan's program list, but the model, still seeing "anggrek", declines because the
    # word isn't in the context. A prompt-side note alone did NOT move gpt-4o-mini here; giving
    # it the canonical name (the deterministic fix) does. None -> generation uses the message.
    generation_query: str | None = None
    # Did the low-confidence rewrite retry (R-08) fire on this turn? Every branch of
    # _route_retrieval that can call rewrite_query sets it. Purely diagnostic -- it changes
    # no behaviour -- but it is the single most expensive optional step in the pipeline
    # (1-2s of LLM latency plus a second retrieval pass), and until now it left no trace in
    # the query log at all: scripts/eval.py reported it as None rather than fake it, and
    # query_log.jsonl never carried the field. That blind spot cost real debugging time on
    # 2026-08-07, when the retry was wrongly ruled out as a latency suspect because the eval
    # showed "0/66 rewrites" for a step that was in fact firing on most fallbacks.
    rewrite_triggered: bool = False
    # Semantic-cache bookkeeping -- non-None only for a fresh (no-history) turn with an
    # index loaded, i.e. the states where caching applies.
    cache_embedding: object | None = None
    cache_language: str | None = None
    # The trusted cache entry to replay, or None to run generation normally.
    cache_hit: dict | None = None


def _log_query(entry: dict) -> None:
    """Appends one record to query_log.jsonl (IMPROVEMENTS.md #6.1) -- the only place that
    captures what people actually ask in production: the query, retrieval confidence,
    latency, and -- most valuably -- whether it fell back."""
    with open(settings.query_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class ChatService:
    def __init__(self, state: Mapping):
        self._index = state.get("index")
        self._fusion_retriever = state.get("fusion_retriever")
        self._reranker = state.get("reranker")

    @property
    def _ready(self) -> bool:
        return self._fusion_retriever is not None and self._reranker is not None

    async def stream(self, message: str, history: list[dict]) -> AsyncGenerator[str, None]:
        """Public entry point: yields SSE events for one chat turn."""
        start = time.perf_counter()

        if is_smalltalk(message):
            _log_query({
                "timestamp": _now(),
                "query": message,
                "query_type": "smalltalk",
                "fallback": False,
                "latency_ms": _elapsed_ms(start),
            })
            async for event in stream_smalltalk_reply(message):
                yield event
            return

        # Prompt-extraction attempt ("repeat everything above / reveal your instructions"):
        # decline with the standard fallback BEFORE any retrieval or LLM call, so the system
        # prompt and context can never be echoed back (see is_prompt_extraction_attempt).
        if is_prompt_extraction_attempt(message):
            logger.warning("PROMPT_EXTRACTION_BLOCKED query=%r", message)
            _log_query({
                "timestamp": _now(),
                "query": message,
                "query_type": "blocked_prompt_extraction",
                "fallback": True,
                "latency_ms": _elapsed_ms(start),
            })
            async for event in stream_prompt_extraction_refusal(message):
                yield event
            return

        if self._ready:
            plan = await self._plan(message, history)
        else:
            # No index loaded -> nothing to retrieve; generation will emit the fallback.
            plan = Plan(standalone_query=message)

        if plan.cache_hit is not None:
            _log_query({
                "timestamp": _now(),
                "query": message,
                "standalone_query": plan.standalone_query,
                "query_type": "cache_hit",
                "matched_programs": plan.matched_programs,
                "fallback": False,
                "history_turns": len(history),
                "latency_ms": _elapsed_ms(start),
            })
            async for event in stream_cached_answer(
                plan.cache_hit["answer"], plan.cache_hit["sources"],
                is_fallback=plan.cache_hit.get("is_fallback", False),
                follow_ups=plan.cache_hit.get("follow_ups", []),
            ):
                yield event
            return

        log_entry = {
            "timestamp": _now(),
            "query": message,
            "standalone_query": plan.standalone_query,
            "query_type": "comparison" if plan.is_comparison else "single",
            "matched_programs": plan.matched_programs,
            "top_score": float(plan.nodes[0].score) if plan.nodes else None,
            "fallback": not plan.nodes,
            "rewrite_triggered": plan.rewrite_triggered,
            "history_turns": len(history),
        }

        # "Ask, don't guess" (see _route_retrieval's tail): the query named a campus/
        # program we couldn't resolve, so ask which one rather than dead-ending. Checked
        # before the budget gate because it's a deterministic emission with no LLM call,
        # so it stays helpful even when generation is otherwise declined. Deliberately NOT
        # wired through _cache_after_stream: the message embeds the user's specific garbled
        # term, which must never be replayed to a different user's different typo -- the
        # else branch below is the only place caching is attached, so excluding it here is
        # structural, not a content check. Campus takes priority (the confirmed-severe bug
        # shape); at most one field is ever set (see _route_retrieval).
        if plan.unresolved_campus_mention or plan.unresolved_program_mention:
            if plan.unresolved_campus_mention:
                term, known, kind = plan.unresolved_campus_mention, known_campus_names(), "campus"
                log_entry["query_type"] = "clarification_campus"
            else:
                term, known, kind = plan.unresolved_program_mention, plan.program_names, "program"
                log_entry["query_type"] = "clarification_program"
            log_entry["unresolved_term"] = term
            log_entry["fallback"] = False
            suggestions = rank_clarification_suggestions(term, known)
            answer_stream = stream_clarification(
                term, suggestions, known, kind, detect_language(plan.standalone_query)
            )
        # Soft daily token budget (IMPROVEMENTS.md #3.2): decline generation before it
        # ever reaches the LLM provider once today's usage crosses the cap. A cache hit
        # already returned above and never reaches here, so it keeps working regardless.
        elif is_budget_exceeded():
            logger.warning("DAILY_TOKEN_BUDGET_EXCEEDED query=%r", message)
            log_entry["query_type"] = "budget_exceeded"
            answer_stream = stream_budget_exceeded(plan.standalone_query)
        else:
            # Follow-up suggestions (IMPROVEMENTS.md #9.3): only once there's something to
            # answer with (an empty node list becomes a fallback, where #9.4 shows starter
            # questions instead).
            follow_ups = (
                suggest_follow_ups(
                    plan.matched_programs, plan.aspects, detect_language(plan.standalone_query)
                )
                if plan.nodes
                else []
            )
            answer_stream = stream_answer(
                plan.generation_query or message, plan.nodes, history=history,
                is_comparison=plan.is_comparison, follow_ups=follow_ups,
            )
            if plan.cache_embedding is not None:
                answer_stream = self._cache_after_stream(answer_stream, plan)

        answer_stream = self._log_after_stream(answer_stream, log_entry, start)
        async for event in answer_stream:
            yield event

    async def _plan(self, message: str, history: list[dict]) -> Plan:
        """Decision phase (no streaming, logging, or cache writes): condense -> cache
        lookup -> classify -> route retrieval. See the module docstring."""
        # Condense a context-dependent follow-up ("tell me more about it") into a
        # standalone question for retrieval only -- generation still gets the original
        # message. bge-m3 / bge-reranker are multilingual, so there's no translate step.
        program_catalog = get_program_catalog(self._index)
        standalone_query = await condense_question(history, message, list(program_catalog))
        # normalize_campus_aliases before filler-stripping: an informal campus name (e.g.
        # "alsut" for Alam Sutera) scores far below the confidence gate in raw retrieval
        # even when the campus IS in the KB, since that's not the term its own pages use --
        # see generation._CAMPUS_ALIASES for the measured before/after. Retrieval-only,
        # same scoping as strip_retrieval_filler -- generation still sees the user's
        # original wording.
        # A "who teaches X" question is routed differently (see _route_retrieval): its
        # answer is in the faculty roster, not X's program catalog, and -- measured -- its
        # retrieval query must keep the "siapa/dosen/who/lecturer" framing rather than have
        # it filler-stripped, since that framing is the signal that lifts faculty-node
        # reranker scores from below the gate to ~0.75-0.93. So skip strip_retrieval_filler
        # for this intent.
        # Both "who teaches X" and "who is the head of X / dean" are answered from the
        # faculty roster (never X's program catalog) and keep their framing (no filler-strip).
        who_teaches = is_who_teaches_query(standalone_query)
        leadership = is_leadership_query(standalone_query)
        faculty_scoped = who_teaches or leadership
        normalized_query = normalize_campus_aliases(standalone_query)
        retrieval_query = normalized_query if faculty_scoped else strip_retrieval_filler(normalized_query)
        # Aspect tags (career/curriculum/tuition/...) -- keyword-based, no LLM/embedding
        # cost, and used both by the cache safety gate and by follow-up suggestions.
        aspects = detect_aspects(standalone_query)

        plan = Plan(
            standalone_query=standalone_query, aspects=aspects,
            program_names=list(program_catalog), who_teaches=who_teaches,
            leadership=leadership,
        )

        # Semantic cache: only for a fresh (no-history) conversation -- see cache.py's
        # module docstring for why a bare embedding threshold is unsafe on its own and the
        # two deterministic gates (program-entity, aspect) every candidate must clear.
        cache_candidate = None
        if not history:
            plan.cache_language = detect_language(standalone_query)
            plan.cache_embedding = await LlamaSettings.embed_model.aget_text_embedding(
                standalone_query
            )
            cache_candidate = get_cache_candidate(plan.cache_embedding, plan.cache_language)

        # detect_named_programs is needed either way -- for the cache's entity gate, or
        # for routing below. When there's a cache candidate, compute it alone first (a
        # trusted hit skips retrieval, so gathering retrieval in parallel would waste it);
        # otherwise run it concurrently with the default retrieval.
        if cache_candidate is not None:
            program_match = await detect_named_programs(standalone_query, list(program_catalog))
            plan.matched_programs = program_match.matched
            if is_safe_cache_hit(program_match.matched, aspects, cache_candidate):
                logger.info("SEMANTIC_CACHE_HIT query=%r", standalone_query)
                plan.cache_hit = cache_candidate
                return plan
            logger.info(
                "SEMANTIC_CACHE_REJECTED query=%r candidate_programs=%r new_programs=%r",
                standalone_query, cache_candidate["matched_programs"], program_match.matched,
            )
            nodes = await retrieve_and_rerank(
                self._fusion_retriever, self._reranker, retrieval_query
            )
        else:
            program_match, nodes = await asyncio.gather(
                detect_named_programs(standalone_query, list(program_catalog)),
                retrieve_and_rerank(self._fusion_retriever, self._reranker, retrieval_query),
            )
            plan.matched_programs = program_match.matched

        await self._route_retrieval(plan, program_match, program_catalog, nodes, retrieval_query)
        return plan

    async def _route_retrieval(
        self, plan, program_match, program_catalog, default_nodes, retrieval_query
    ) -> None:
        """Fills plan.nodes / plan.is_comparison from the program classification.

        - 2-3 named programs -> comparison mode (retrieve each program's own document).
        - exactly 1 -> single-program-scoped retrieval (#2.4 leak prevention).
        - a specific program NOT in the KB -> empty (caller falls back).
        - otherwise -> the default open retrieval, with a low-confidence rewrite retry.
        Named-program paths retry against supplementary URL sources when the program's own
        document comes up empty (tuition/admission fees live in scraped pages, not the
        catalogs). Each path re-applies the confidence gate as a backstop.
        """
        gate = settings.confidence_threshold

        # A "who teaches X" / "who is the head of X" query is answered ONLY from the faculty
        # roster -- never a program catalog, which outranks faculty for a strong program name
        # (e.g. "Computer Science") even in open retrieval. Its own scoped branch guarantees that.
        if plan.who_teaches or plan.leadership:
            await self._route_faculty(plan)
            return

        # "What programs are offered at campus X" -> scope to that campus's admission-
        # requirement page (the only source that lists them). Open retrieval otherwise lets a
        # faculty bio outrank the program list, and the model declines on the faculty-dominated
        # context -- confirmed live for "anggrek" (alias -> Kemanggisan) and "kemanggisan".
        # Gated on BOTH the enumeration intent AND a resolvable campus, so a program-named
        # ("...di Computer Science") or campus-less ("what programs does BINUS offer") query
        # falls through to normal routing untouched.
        if is_campus_programs_query(plan.standalone_query):
            campus = resolve_named_campus(plan.standalone_query, known_campus_names())
            campus_url = admission_requirement_url_for_campus(campus) if campus else None
            if campus_url:
                # retrieval_query, not standalone_query: it's alias-normalized (anggrek ->
                # Kemanggisan), so the reranker scores the campus's program-list nodes against
                # the campus's real name -- with the raw "anggrek" they fell below the gate and
                # dead-ended even though the right page was scoped in.
                nodes = await retrieve_for_named_programs(
                    self._index, self._reranker, retrieval_query, [campus_url],
                    per_program_top_n=6,
                )
                plan.nodes = nodes if (nodes and nodes[0].score >= gate) else []
                if plan.nodes:
                    # Show the model the campus's canonical name so it doesn't decline on an
                    # alias absent from the context (e.g. "anggrek"). Normalizes only the
                    # campus alias; the rest of the user's wording/language is preserved.
                    plan.generation_query = normalize_campus_aliases(plan.standalone_query)
                return

        matched = program_match.matched
        named_unmatched = program_match.named_unmatched

        if len(matched) >= 2:
            plan.is_comparison = True
            source_files = [program_catalog[p] for p in matched]
            # Retrieve each program's document with the bare attribute being compared
            # ("total credits"), not the full "Compare X and Y" prose, and balance the
            # final context evenly across programs -- see comparison_attribute_query and
            # retrieve_for_named_programs(balanced=True) for why each is load-bearing.
            attribute_query = comparison_attribute_query(plan.standalone_query, matched)
            nodes = await retrieve_for_named_programs(
                self._index, self._reranker, attribute_query, source_files, balanced=True
            )
            # Vocabulary-mismatch retry (R-08), same reasoning as the single-match branch
            # below -- a program-scoped comparison can suffer the identical failure (e.g.
            # comparing in Indonesian against English-only catalogs).
            extra_queries: list[str] = []
            if (not nodes or nodes[0].score < gate) and not is_budget_exceeded():
                plan.rewrite_triggered = True
                extra_queries = await rewrite_query(attribute_query)
                if extra_queries:
                    nodes = await retrieve_for_named_programs(
                        self._index, self._reranker, attribute_query, source_files,
                        balanced=True, extra_queries=extra_queries,
                    )
            nodes = await self._retry_with_supplementary_sources(
                plan, attribute_query, nodes, source_files, extra_queries
            )
            if not nodes or nodes[0].score < gate:
                nodes, plan.is_comparison = [], False
            plan.nodes = nodes
        elif len(matched) == 1:
            # A genuine single match drives program-scoped retrieval; named_unmatched is
            # always False here (it's only set when nothing was literally matched). Use the
            # filler-stripped retrieval_query, same as the open branch below: a weak
            # conversational verb ("Ceritakan program Data Science", "tell me about...")
            # otherwise dominates the short query's embedding and drops it below the gate
            # against the program's own doc -- measured 0.265 raw vs 0.992 stripped. Stripping
            # is semantics-preserving (only scaffolding, never a topic word), so a genuinely
            # unanswerable in-scope question still scores low and falls back.
            source_files = [program_catalog[matched[0]]]

            # Tuition first, catalog second. A program's own catalog cannot answer a
            # tuition question -- it reranks 0.004 even spelled correctly (see
            # _retry_with_supplementary_sources' docstring) -- yet the default order pays
            # for that doomed retrieval AND the LLM rewrite its low score triggers, every
            # time, before the supplementary retry finally reaches the per-campus pages that
            # hold the answer. Measured on "Berapa biaya kuliah program CS di Kemanggisan?":
            # ~2-4s of the ~10s TTFT is spent proving what 1a7f009 already established.
            #
            # Purely a reordering: when the tuition pages don't clear the gate, the normal
            # program-scoped path below runs untouched, so nothing that answers today stops
            # answering. Single-program only, for the same reason the retry itself is
            # (comparison mode can't tell which program a campus row belongs to).
            if "tuition" in plan.aspects:
                campus_nodes = await self._retry_tuition_across_campuses(
                    retrieval_query, load_scraped_urls()
                )
                if campus_nodes and campus_nodes[0].score >= gate:
                    # Same reasoning as the retry path: multiple campuses of one program is
                    # structurally a comparison, so reuse COMPARISON_NOTE's table format.
                    plan.is_comparison = True
                    plan.nodes = campus_nodes
                    return

            nodes = await retrieve_for_named_programs(
                self._index, self._reranker, retrieval_query, source_files
            )
            # Vocabulary-mismatch retry (R-08): the open-retrieval branch already had this
            # for unscoped queries, but a program-SCOPED query can suffer the exact same
            # failure -- confirmed live, "Apa saja capaian pembelajaran program studi Ilmu
            # Komputer?" scored 0.016 against Computer Science's own (English-only) catalog
            # PDF even though the same-topic ENGLISH question scored 0.997 against the
            # identical document. Skipped once the token budget is spent, same as the open
            # path, since generation is about to be declined anyway in that case.
            extra_queries: list[str] = []
            if (not nodes or nodes[0].score < gate) and not is_budget_exceeded():
                plan.rewrite_triggered = True
                extra_queries = await rewrite_query(retrieval_query)
                if extra_queries:
                    nodes = await retrieve_for_named_programs(
                        self._index, self._reranker, retrieval_query, source_files,
                        extra_queries=extra_queries,
                    )
            nodes = await self._retry_with_supplementary_sources(
                plan, retrieval_query, nodes, source_files, extra_queries
            )
            plan.nodes = nodes if (nodes and nodes[0].score >= gate) else []
        elif named_unmatched:
            # Names a specific program not in the KB (e.g. "Information Systems") -- fall
            # back rather than let an open search surface a wrong-program chunk.
            plan.nodes = []
        else:
            # Open retrieval. Retry with LLM-rewritten paraphrases if the first pass was
            # too weak to clear the gate -- skipped once the token budget is spent, since
            # generation is about to be declined anyway.
            #
            # Also skipped when the query contains no academic vocabulary at all. A
            # paraphrase of "spaghetti carbonara" is still about spaghetti carbonara, so the
            # rewrite plus its second retrieval spent ~1.2s to reach the same fallback --
            # the largest remaining cost on the out-of-scope tail (p50 5.56s vs in-scope's
            # 1.80s) once the contextual-fallback call was gated the same way.
            #
            # Deliberately NOT a score floor: one transposed character drops the reranker
            # from 0.709 to 0.119 on identical chunks, so the misspelled-tuition queries this
            # retry exists to rescue sit exactly where a floor would cut. Those queries
            # literal-match their program name and take the single-program branch above, not
            # this one, so they never reach this gate at all. A misspelled query that reaches
            # HERE (no literal match) is an out-of-catalog near-miss like "Data Enginering",
            # which must fall back regardless.
            nodes = default_nodes
            # Vocabulary alone is not enough: "Cybr Security" and "kemanggisan" contain no
            # academic word yet are exactly the misspelling/alias cases the retry rescues.
            rewrite_worthwhile = has_domain_vocabulary(retrieval_query) or resembles_known_entity(
                plan.standalone_query, plan.program_names, known_campus_names()
            )
            if (not nodes or nodes[0].score < gate) and rewrite_worthwhile and not is_budget_exceeded():
                plan.rewrite_triggered = True
                extra_queries = await rewrite_query(retrieval_query)
                if extra_queries:
                    nodes = await retrieve_and_rerank(
                        self._fusion_retriever, self._reranker, retrieval_query, extra_queries
                    )
            plan.nodes = nodes if (nodes and nodes[0].score >= gate) else []

        # "Ask, don't guess": only once retrieval has definitively come up empty above
        # (every branch's own retries already exhausted). If the query still LOOKS like it
        # names a specific campus/program we couldn't resolve, prefer a clarifying question
        # over the silent fallback. One check point, not per-branch: the detectors no-op
        # whenever the entity was already recognized, so they're safe to call regardless of
        # which branch emptied plan.nodes. (A who-teaches query already returned above.)
        if not plan.nodes:
            plan.unresolved_campus_mention = detect_unresolved_campus_mention(
                plan.standalone_query, known_campus_names()
            )
            if not plan.unresolved_campus_mention:
                token = detect_unresolved_program_mention(
                    plan.standalone_query, plan.program_names
                )
                # Clarify a program token when it's either a near-miss of a real catalog
                # program (a recoverable typo/abbreviation like "Cybersec" -> Cyber
                # Security, confirmed live to otherwise dead-end at "contact us") OR one the
                # LLM did NOT flag as a genuine out-of-catalog program. Only a token that's
                # BOTH unlike any catalog name AND flagged out-of-catalog by the LLM (e.g.
                # "Kedokteran") stays on the existing plain-fallback path -- rank_clarifi-
                # cation_suggestions returning [] is exactly that "nothing close" signal, so
                # this defers to the LLM's judgment precisely when there's nothing better to
                # offer, and overrides it when there demonstrably is.
                if token and (
                    rank_clarification_suggestions(token, plan.program_names)
                    or not program_match.named_unmatched
                ):
                    plan.unresolved_program_mention = token

    async def _route_faculty(self, plan) -> None:
        """Fills plan.nodes for a faculty-person query (who-teaches OR leadership): retrieve
        ONLY from the faculty roster (retrieve_for_named_programs scoped to the one faculty
        source), rerank among lecturers, apply the lower _WHO_TEACHES_GATE (see its comment)
        and the same low-confidence rewrite retry as the program-scoped paths. A who-teaches
        query is then capped to a short "2-3 + others" list; a leadership query names one
        specific person, so it is NOT capped."""
        nodes = await retrieve_for_named_programs(
            self._index, self._reranker, plan.standalone_query, [FACULTY_ROSTER_URL],
            per_program_top_n=5,
        )
        if (not nodes or nodes[0].score < _WHO_TEACHES_GATE) and not is_budget_exceeded():
            plan.rewrite_triggered = True
            extra_queries = await rewrite_query(plan.standalone_query)
            if extra_queries:
                nodes = await retrieve_for_named_programs(
                    self._index, self._reranker, plan.standalone_query, [FACULTY_ROSTER_URL],
                    per_program_top_n=5, extra_queries=extra_queries,
                )
        nodes = nodes if (nodes and nodes[0].score >= _WHO_TEACHES_GATE) else []
        plan.nodes = nodes[:_WHO_TEACHES_MAX_LECTURERS] if plan.who_teaches else nodes

    async def _retry_with_supplementary_sources(
        self, plan, standalone_query, nodes, source_files, extra_queries=None
    ):
        """A named program's own catalog has no tuition/admission-fee content -- that
        lives in separately-scraped reference pages. When program-scoped retrieval comes
        up empty/low-confidence, retry with every scraped URL added, but only on that
        already-failing path so the common case pays no extra cost.

        `extra_queries` are the paraphrases the caller's own rewrite retry (R-08) already
        paid for -- they are forwarded here rather than dropped, because the caller spent
        them against the program's CATALOG, which is exactly the document that cannot
        answer the questions that reach this retry. Confirmed live from query_log.jsonl:
        "berapa harga jurusna computer science?" (a one-character transposition of
        "jurusan") reranked 0.119 against the tuition pages and fell back, while the
        correctly-spelled question scored 0.709 -- the right chunks were in the candidate
        pool both times, the cross-encoder just collapses on an out-of-vocabulary subword
        split. The catalog leg of that same request scored 0.004 even spelled correctly
        (tuition simply isn't in it), so the entire rewrite budget was being spent where it
        could never help. Forwarding the paraphrases here takes those same typo'd queries
        to 0.980. No extra LLM call -- the rewrite already happened.
        """
        if nodes and nodes[0].score >= settings.confidence_threshold:
            return nodes
        supplementary = load_scraped_urls()
        if not supplementary:
            return nodes

        # Scoped to exactly one named program: with 2-3 programs (comparison mode),
        # _retry_tuition_across_campuses has no way to tell which program a campus's
        # top-matching row belongs to (it retrieves per CAMPUS, not per program), so
        # applying it there could silently collapse a program-vs-program comparison
        # into one program's own campus breakdown.
        if "tuition" in plan.aspects and len(source_files) == 1:
            campus_nodes = await self._retry_tuition_across_campuses(
                standalone_query, supplementary, extra_queries
            )
            if campus_nodes and campus_nodes[0].score >= settings.confidence_threshold:
                # Multiple campuses of the SAME program is structurally identical to
                # comparing multiple programs (one column per item, one row per metric)
                # -- reuse the existing table-formatting instruction (COMPARISON_NOTE)
                # rather than teaching the model a second "when to use a table" trigger.
                plan.is_comparison = True
                return campus_nodes

        # Paraphrases last, not first. retrieve_for_named_programs builds one
        # metadata-filtered retriever PER source file and runs every query against each, so
        # cost scales with sources x queries -- and this is the widest source set in the
        # pipeline (the program's own doc plus every scraped URL, 48 today). Measured on
        # "Berapa biaya kuliah program CS di Kemanggisan?": 48 sources x 1 query = 3.29s
        # scoring 0.966, versus 48 x 4 = 16.21s scoring 0.995. Both clear the 0.5 gate, so
        # the paraphrase legs bought 0.03 of headroom for 13 seconds.
        #
        # They still run when the original query is the problem, which is the case they were
        # forwarded here for: a one-character transposition collapses the reranker to 0.119
        # (see the docstring above), and paraphrases take it back to 0.980. Gating them on
        # the original having failed keeps that recovery intact and charges for it only when
        # it is needed. Worst case (original fails, paraphrases also needed) is ~20% slower
        # than before on a query that was already heading for a fallback.
        # Narrowing this set by aspect (tuition -> only the tuition pages, etc.) was tried
        # and reverted: measured 16.21s -> 4.86s in isolation with an identical top score,
        # but no reproducible effect end to end, because tuition queries return early from
        # _retry_tuition_across_campuses above and rarely reach this branch at all. The
        # mapping table would have needed keeping in sync with the scraped URL families for
        # a benefit that could not be demonstrated.
        widened = source_files + supplementary

        nodes = await retrieve_for_named_programs(
            self._index, self._reranker, standalone_query, widened,
        )
        if not extra_queries or (nodes and nodes[0].score >= settings.confidence_threshold):
            return nodes
        return await retrieve_for_named_programs(
            self._index, self._reranker, standalone_query, widened,
            extra_queries=extra_queries,
        )

    async def _retry_tuition_across_campuses(
        self, standalone_query, supplementary, extra_queries=None
    ):
        """BINUS publishes tuition/fees as one page PER CAMPUS -- the generic
        supplementary-source retry above pools all of them (plus every other scraped
        page) into one unbalanced global top-N, so 1-2 campuses that happen to rerank
        highest crowd out every other campus offering the exact same program. Restructured
        tuition chunks are now one program/campus/year row each (~65 tokens -- see
        ingestion.py's _tuition_fee_row_nodes), so retrieving every campus's row and
        raising the cap to cover them all (confirmed live: up to 12 rows for a program
        offered at every campus) still costs LESS total context than the unbalanced
        default cap used to spend on far fewer, far noisier multi-program table
        fragments.
        """
        campus_urls = [u for u in supplementary if _TUITION_FEE_URL_SUBSTR in u]
        if not campus_urls:
            return []
        return await retrieve_for_named_programs(
            self._index, self._reranker, standalone_query, campus_urls,
            balanced=True, per_program_top_n=2, max_nodes=16, extra_queries=extra_queries,
        )

    async def _log_after_stream(self, stream, entry: dict, start_time: float):
        """Wraps the answer stream to record total latency (including the LLM call) into the
        query log once the stream completes, and -- when settings.log_responses is on -- the
        assistant's response too (answer text, truncated, plus the cited source files).

        This is the outermost stream wrapper (see stream()), so it observes every SSE event
        after any inner wrapper (_cache_after_stream) has passed it through. Capturing the
        response only matters when logging is enabled, so the token accumulation is guarded
        to add nothing for the default (off) path beyond the loop that already runs."""
        capture = settings.log_responses
        answer_parts: list[str] = []
        sources: list = []
        async for event in stream:
            yield event
            if capture and event.startswith("data: "):
                try:
                    data = json.loads(event[len("data: "):].strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                if data.get("type") == "token":
                    answer_parts.append(data.get("content", ""))
                elif data.get("type") == "done":
                    sources = data.get("sources", [])
        entry["latency_ms"] = _elapsed_ms(start_time)
        if capture:
            answer = "".join(answer_parts)
            limit = settings.log_response_max_chars
            entry["response"] = answer[:limit]
            entry["response_truncated"] = len(answer) > limit
            # Just the distinct source files, in first-seen order -- the full source dicts
            # (snippets, pages) would bloat the log; the file identity is what an audit needs.
            entry["response_sources"] = list(
                dict.fromkeys(
                    s.get("source_file") for s in sources if isinstance(s, dict) and s.get("source_file")
                )
            )
        _log_query(entry)

    async def _cache_after_stream(self, stream, plan: Plan):
        """Populates the semantic cache once the full answer is known, without buffering
        (tokens still reach the client immediately). Skips a transient service-error
        response (an LLM-provider failure), which has nothing to do with the question or KB
        state and would otherwise be replayed to identical questions long after it recovers."""
        service_error_texts = {get_service_error_message("id"), get_service_error_message("en")}

        accumulated = ""
        sources: list = []
        is_fallback = False
        follow_ups: list[str] = []
        async for event in stream:
            yield event
            if not event.startswith("data: "):
                continue
            data = json.loads(event[len("data: "):].strip())
            if data["type"] == "token":
                accumulated += data["content"]
            elif data["type"] == "done":
                sources = data["sources"]
                is_fallback = data.get("fallback", False)
                follow_ups = data.get("follow_ups", [])

        if accumulated and accumulated.strip() not in service_error_texts:
            store_answer(
                plan.cache_embedding, plan.cache_language, plan.matched_programs,
                plan.aspects, accumulated, sources, plan.is_comparison,
                is_fallback=is_fallback, follow_ups=follow_ups,
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)
