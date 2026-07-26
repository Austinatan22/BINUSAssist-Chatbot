"""Runs the labeled regression cases in tests/regression_cases.py -- program routing and
language detection, the two behaviours that repeatedly regressed. Deterministic, so this
is part of the normal `pytest` run and therefore CI (no GPU, no Groq). See
regression_cases.py for the split between this and scripts/eval.py's live eval.
"""
import pytest

from backend.rag.generation import (
    _literal_program_matches,
    comparison_attribute_query,
    detect_language,
    is_career_outcome_query,
    is_leadership_query,
    is_prompt_extraction_attempt,
    is_who_teaches_query,
)
from tests.regression_cases import (
    CATALOG,
    COMPARISON_ATTRIBUTE_CASES,
    INTENT_DETECTOR_CASES,
    LANGUAGE_CASES,
    ROUTING_CASES,
)


@pytest.mark.parametrize("query, expected", ROUTING_CASES)
def test_program_routing(query, expected):
    assert sorted(_literal_program_matches(query, CATALOG)) == sorted(expected)


@pytest.mark.parametrize("query, expected", LANGUAGE_CASES)
def test_language_detection(query, expected):
    assert detect_language(query) == expected


@pytest.mark.parametrize("query, matched, must_contain", COMPARISON_ATTRIBUTE_CASES)
def test_comparison_attribute_query(query, matched, must_contain):
    result = comparison_attribute_query(query, matched).lower()
    for token in must_contain:
        assert token.lower() in result, f"{token!r} missing from {result!r}"


_INTENT_DETECTORS = {
    "career": is_career_outcome_query,
    "who_teaches": is_who_teaches_query,
    "leadership": is_leadership_query,
    "extraction": is_prompt_extraction_attempt,
}


@pytest.mark.parametrize("query, detector, expected", INTENT_DETECTOR_CASES)
def test_intent_detectors(query, detector, expected):
    assert _INTENT_DETECTORS[detector](query) is expected
