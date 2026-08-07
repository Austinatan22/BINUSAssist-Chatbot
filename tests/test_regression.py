"""Runs the labeled regression cases in tests/regression_cases.py -- program routing and
language detection, the two behaviours that repeatedly regressed. Deterministic, so this
is part of the normal `pytest` run and therefore CI (no GPU, no Groq). See
regression_cases.py for the split between this and scripts/eval.py's live eval.
"""
import pytest

from backend.rag.generation import (
    _AMBIGUOUS_PROGRAM_ABBREVIATIONS,
    _CAMPUS_ALIASES,
    _INDONESIAN_PROGRAM_ALIASES,
    _PROGRAM_NICKNAMES,
    _literal_program_matches,
    comparison_attribute_query,
    detect_language,
    is_career_outcome_query,
    is_leadership_query,
    is_prompt_extraction_attempt,
    is_who_teaches_query,
)
from backend.rag.ingestion import known_campus_names
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


# --- alias-table consistency -------------------------------------------------------
# Every alias table maps a REAL program/campus name to informal spellings of it. If a key
# is not a real name, the whole entry is dead: it can never match, and it takes its aliases
# with it, silently. That is not hypothetical -- _INDONESIAN_PROGRAM_ALIASES was keyed on
# "Mobile Application and Technology" while the catalog says "Mobile Application
# Technology" (the document is Mobile_Application___Technology_2023.pdf), so that entry and
# its two Indonesian aliases matched nothing for an unknown length of time. It surfaced
# only by accident, from a flaky eval row on 2026-08-07.
#
# These tables are hand-maintained and there are dozens of them across the codebase; a key
# typo is invisible at runtime because a dead entry looks exactly like "no alias defined".
# Cheap to assert, so assert it. No GPU or index needed: program names come from CATALOG
# (kept in sync with the KB, see regression_cases) and campus names from
# known_campus_names(), which derives them from the tracked scraped_urls.json.
_PROGRAM_ALIAS_TABLES = {
    "_INDONESIAN_PROGRAM_ALIASES": _INDONESIAN_PROGRAM_ALIASES,
    "_PROGRAM_NICKNAMES": _PROGRAM_NICKNAMES,
    "_AMBIGUOUS_PROGRAM_ABBREVIATIONS": _AMBIGUOUS_PROGRAM_ABBREVIATIONS,
}


@pytest.mark.parametrize("table_name", sorted(_PROGRAM_ALIAS_TABLES))
def test_program_alias_keys_are_real_catalog_programs(table_name):
    unknown = sorted(set(_PROGRAM_ALIAS_TABLES[table_name]) - set(CATALOG))
    assert not unknown, (
        f"{table_name} keys are not in the catalog: {unknown}. A key that isn't a real "
        f"program name makes the whole entry dead -- it can never match, and its aliases "
        f"are lost with it. Either fix the key's spelling or update CATALOG if the KB "
        f"genuinely changed."
    )


def test_campus_alias_keys_are_real_campuses():
    known = known_campus_names()
    unknown = sorted(set(_CAMPUS_ALIASES) - known)
    assert not unknown, (
        f"_CAMPUS_ALIASES keys are not real campuses: {unknown}. Known: {sorted(known)}"
    )


@pytest.mark.parametrize("table_name", sorted(_PROGRAM_ALIAS_TABLES))
def test_program_aliases_do_not_collide_across_tables(table_name):
    # The same alias string mapping to two different programs would make routing depend on
    # table iteration order, which is not something to leave to chance.
    seen: dict[str, str] = {}
    for name, table in _PROGRAM_ALIAS_TABLES.items():
        for program, aliases in table.items():
            for alias in aliases:
                key = alias.lower()
                assert key not in seen or seen[key] == program, (
                    f"alias {alias!r} maps to both {seen.get(key)!r} and {program!r}"
                )
                seen[key] = program


def test_an_alias_is_never_its_own_program_name():
    # A self-referential alias is redundant at best; at worst it hides a copy-paste error
    # where the alias list was never filled in.
    for name, table in _PROGRAM_ALIAS_TABLES.items():
        for program, aliases in table.items():
            for alias in aliases:
                assert alias.lower() != program.lower(), (
                    f"{name}[{program!r}] lists its own name as an alias"
                )


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
