"""Unit tests for backend/rag/cache.py -- especially is_safe_cache_hit, the
deterministic safety gate that exists because naive embedding-similarity caching was
found UNSAFE for this KB (see that module's docstring: direct calibration found genuine
paraphrases and dangerous near-misses score in a completely overlapping cosine-
similarity range). No GPU, no Groq -- pure logic and small numpy arrays.
"""
import numpy as np
import pytest

from backend.rag.cache import (
    _cache,
    _cosine_similarity,
    clear_semantic_cache,
    detect_aspects,
    get_cache_candidate,
    is_safe_cache_hit,
    store_answer,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_semantic_cache()
    yield
    clear_semantic_cache()


class TestDetectAspects:
    def test_career_keyword(self):
        assert "career" in detect_aspects("What are the career prospects for CS graduates?")

    def test_tuition_keyword_indonesian(self):
        assert "tuition" in detect_aspects("Berapa biaya kuliah Computer Science?")

    @pytest.mark.parametrize("query", [
        "berapa harga jurusan computer science?",
        "berapa bayar kuliah di Computer Science?",
    ])
    def test_colloquial_indonesian_price_words_are_tuition(self, query):
        # "biaya"/"uang kuliah" is the formal register; "harga"/"bayar" is what students
        # actually type. Found live in query_log.jsonl -- a plainly-tuition question tagged
        # as ambiguous, which skipped the campus-balanced tuition retry
        # (chat_service._retry_with_supplementary_sources gates on this tag) and returned
        # an unbalanced subset of campuses.
        assert "tuition" in detect_aspects(query)

    def test_ambiguous_query_returns_empty_set(self):
        # Empty means "ambiguous" to the caller, never a wildcard match.
        assert detect_aspects("Tell me more about it") == set()

    def test_multiple_aspects_can_match_at_once(self):
        aspects = detect_aspects("What is the tuition and curriculum for this program?")
        assert "tuition" in aspects
        assert "curriculum" in aspects


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_not_a_crash(self):
        a, b = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestIsSafeCacheHit:
    """The core safety gate. A false hit here is worse than a bad retrieval score: it
    returns a fully-formed, confident answer about the WRONG program or WRONG aspect
    with no LLM judgment in the loop to catch it."""

    @staticmethod
    def _candidate(programs, aspects):
        return {"matched_programs": programs, "aspects": aspects}

    def test_same_program_same_aspect_is_a_safe_hit(self):
        candidate = self._candidate(["Cyber Security"], {"career"})
        assert is_safe_cache_hit(["Cyber Security"], {"career"}, candidate) is True

    def test_different_program_is_rejected(self):
        # The exact disqualifying near-miss found during calibration: "Computer
        # Science" vs "Data Science" career-prospects questions scored 0.86 cosine
        # similarity on bge-m3 -- high enough to clear a naive threshold, but a
        # completely different correct answer.
        candidate = self._candidate(["Computer Science"], {"career"})
        assert is_safe_cache_hit(["Data Science"], {"career"}, candidate) is False

    def test_same_program_different_aspect_is_rejected(self):
        # The other disqualifying near-miss: same program, different question type
        # (tuition vs curriculum) also scored dangerously high on embedding similarity
        # alone.
        candidate = self._candidate(["Computer Science"], {"tuition"})
        assert is_safe_cache_hit(["Computer Science"], {"curriculum"}, candidate) is False

    def test_ambiguous_new_query_aspect_is_rejected(self):
        candidate = self._candidate(["Computer Science"], {"career"})
        assert is_safe_cache_hit(["Computer Science"], set(), candidate) is False

    def test_ambiguous_cached_aspect_is_rejected(self):
        candidate = self._candidate(["Computer Science"], set())
        assert is_safe_cache_hit(["Computer Science"], {"career"}, candidate) is False

    def test_overlapping_but_nonidentical_aspects_still_pass(self):
        candidate = self._candidate(["Computer Science"], {"career", "outcome"})
        assert is_safe_cache_hit(["Computer Science"], {"career"}, candidate) is True

    def test_comparison_mode_is_order_independent(self):
        candidate = self._candidate(["Computer Science", "Software Engineering"], {"curriculum"})
        assert (
            is_safe_cache_hit(
                ["Software Engineering", "Computer Science"], {"curriculum"}, candidate
            )
            is True
        )


class TestGetCacheCandidateAndStore:
    def test_no_candidate_when_cache_is_empty(self):
        assert get_cache_candidate(np.array([1.0, 0.0, 0.0]), "en") is None

    def test_stores_and_retrieves_a_close_match(self):
        embedding = np.array([1.0, 0.0, 0.0])
        store_answer(embedding, "en", ["Cyber Security"], {"career"}, "answer text", [], False)
        candidate = get_cache_candidate(embedding, "en")
        assert candidate is not None
        assert candidate["answer"] == "answer text"

    def test_different_language_is_never_matched(self):
        embedding = np.array([1.0, 0.0, 0.0])
        store_answer(embedding, "id", ["Cyber Security"], {"career"}, "jawaban", [], False)
        assert get_cache_candidate(embedding, "en") is None

    def test_dissimilar_embedding_is_not_matched(self):
        store_answer(np.array([1.0, 0.0, 0.0]), "en", ["Cyber Security"], {"career"}, "a", [], False)
        orthogonal = np.array([0.0, 1.0, 0.0])
        assert get_cache_candidate(orthogonal, "en") is None

    def test_fifo_eviction_keeps_cache_at_max_size(self):
        from backend.config import settings

        max_entries = settings.semantic_cache_max_entries
        for i in range(max_entries + 3):
            store_answer(np.array([float(i), 0.0, 0.0]), "en", [], set(), f"answer {i}", [], False)
        assert len(_cache) == max_entries
        # The 3 oldest entries should have been evicted first (FIFO, not LRU/random).
        assert _cache[0]["answer"] == "answer 3"
