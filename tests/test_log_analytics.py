"""Tests for scripts/log_analytics.py -- the query-log analytics report (IMPROVEMENTS.md
#6.1's analytics half). All aggregation is pure functions over record dicts, so these run
with no filesystem, models, GPU, or network. Records are hand-built to mirror the real log
schema, including the schema drift the parser must tolerate (older rows lack `top_score` /
`unresolved_term`)."""
from datetime import datetime, timedelta, timezone

import pytest

from scripts.log_analytics import (
    build_report,
    cited_sources,
    fallback_queries,
    filter_since,
    format_report,
    load_records,
    query_type_breakdown,
    response_coverage,
    summarize,
    top_programs,
    unresolved_terms,
    _percentile,
)


def _rec(**kw):
    """A log record with sensible defaults; override any field."""
    base = {
        "timestamp": "2026-07-15T10:00:00+00:00",
        "query": "What is Computer Science?",
        "standalone_query": "What is Computer Science?",
        "query_type": "single",
        "matched_programs": ["Computer Science"],
        "top_score": 0.9,
        "fallback": False,
        "history_turns": 0,
        "latency_ms": 1500,
    }
    base.update(kw)
    return base


class TestSummarize:
    def test_counts_and_fallback_rate(self):
        records = [_rec(fallback=False), _rec(fallback=True), _rec(fallback=True), _rec(fallback=False)]
        s = summarize(records)
        assert s["total"] == 4
        assert s["fallbacks"] == 2
        assert s["fallback_rate"] == 0.5

    def test_empty_records_do_not_divide_by_zero(self):
        s = summarize([])
        assert s["total"] == 0
        assert s["fallback_rate"] == 0.0
        assert s["latency_ms"]["within_3s_rate"] == 0.0
        assert s["date_range"] == (None, None)

    def test_latency_within_3s_counts_end_to_end(self):
        records = [_rec(latency_ms=1000), _rec(latency_ms=2999), _rec(latency_ms=3000), _rec(latency_ms=8000)]
        lat = summarize(records)["latency_ms"]
        assert lat["within_3s"] == 2  # 1000 and 2999 are under 3000; 3000 is not (<, not <=)
        assert lat["within_3s_rate"] == 0.5
        assert lat["max"] == 8000

    def test_missing_optional_fields_are_tolerated(self):
        # Older records predate top_score; summarize must not choke on their absence.
        records = [
            {"timestamp": "2026-07-13T10:00:00+00:00", "query": "x", "fallback": False, "latency_ms": 500},
            {"timestamp": "2026-07-13T11:00:00+00:00", "query": "y", "fallback": True, "latency_ms": 600},
        ]
        s = summarize(records)
        assert s["total"] == 2
        assert s["top_score"]["count"] == 0  # neither had top_score
        assert s["fallbacks"] == 1

    def test_date_range_uses_min_and_max_timestamps(self):
        records = [
            _rec(timestamp="2026-07-15T10:00:00+00:00"),
            _rec(timestamp="2026-07-13T10:00:00+00:00"),
            _rec(timestamp="2026-07-19T10:00:00+00:00"),
        ]
        lo, hi = summarize(records)["date_range"]
        assert lo.startswith("2026-07-13")
        assert hi.startswith("2026-07-19")


class TestPercentile:
    def test_p50_is_median_ish(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p90_and_max(self):
        vals = list(range(1, 101))  # 1..100, indices 0..99
        # nearest-rank: index = round(0.90 * 99) = 89 -> value 90; round(0.99*99)=98 -> 99
        assert _percentile(vals, 90) == 90
        assert _percentile(vals, 99) == 99
        assert _percentile(vals, 100) == 100

    def test_empty_is_zero(self):
        assert _percentile([], 50) == 0.0


class TestFilterSince:
    def test_keeps_only_records_at_or_after(self):
        records = [
            _rec(timestamp="2026-07-13T10:00:00+00:00", query="old"),
            _rec(timestamp="2026-07-18T10:00:00+00:00", query="new"),
        ]
        since = datetime(2026, 7, 15, tzinfo=timezone.utc)
        kept = filter_since(records, since)
        assert [r["query"] for r in kept] == ["new"]

    def test_records_without_timestamp_are_dropped_when_filtering(self):
        records = [_rec(query="ok"), {"query": "no-ts", "fallback": False}]
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        kept = filter_since(records, since)
        assert [r["query"] for r in kept] == ["ok"]


class TestBreakdowns:
    def test_query_type_breakdown_most_common_first(self):
        records = [_rec(query_type="single"), _rec(query_type="single"), _rec(query_type="comparison")]
        assert query_type_breakdown(records) == [("single", 2), ("comparison", 1)]

    def test_missing_query_type_is_unknown(self):
        records = [{"query": "x", "fallback": False}]
        assert query_type_breakdown(records) == [("unknown", 1)]

    def test_top_programs_aggregates_across_matched_lists(self):
        records = [
            _rec(matched_programs=["Computer Science", "Software Engineering"]),
            _rec(matched_programs=["Computer Science"]),
            _rec(matched_programs=[]),
            _rec(matched_programs=None),  # some records legitimately have no match
        ]
        assert top_programs(records, limit=10) == [("Computer Science", 2), ("Software Engineering", 1)]

    def test_top_programs_respects_limit(self):
        records = [_rec(matched_programs=[p]) for p in ["A", "B", "C", "D"]]
        assert len(top_programs(records, limit=2)) == 2


class TestFallbackQueries:
    def test_only_fallbacks_grouped_by_normalized_text(self):
        records = [
            _rec(query="What about housing?", fallback=True),
            _rec(query="what about   housing?", fallback=True),  # spacing/case variant -> same group
            _rec(query="What is Computer Science?", fallback=False),  # not a fallback
        ]
        result = fallback_queries(records, limit=10)
        assert result == [("What about housing?", 2)]  # first-seen casing shown, count 2

    def test_ignores_blank_queries(self):
        records = [_rec(query="   ", fallback=True), _rec(query="Nursing program?", fallback=True)]
        assert fallback_queries(records, limit=10) == [("Nursing program?", 1)]

    def test_respects_limit_and_orders_by_frequency(self):
        records = (
            [_rec(query="a", fallback=True)] * 3
            + [_rec(query="b", fallback=True)] * 2
            + [_rec(query="c", fallback=True)]
        )
        result = fallback_queries(records, limit=2)
        assert result == [("a", 3), ("b", 2)]


class TestUnresolvedTerms:
    def test_counts_unresolved_terms(self):
        records = [
            _rec(unresolved_term="cybersec"),
            _rec(unresolved_term="cybersec"),
            _rec(unresolved_term="alsut"),
            _rec(),  # no unresolved_term
        ]
        assert unresolved_terms(records, limit=10) == [("cybersec", 2), ("alsut", 1)]

    def test_empty_when_no_terms(self):
        assert unresolved_terms([_rec()], limit=10) == []


class TestBuildAndFormatReport:
    def test_build_report_has_all_sections(self):
        report = build_report([_rec(fallback=True, query="Nursing?")], top=15)
        assert set(report) == {
            "summary", "by_query_type", "top_programs", "top_fallback_queries",
            "unresolved_terms", "response_coverage", "cited_sources",
        }

    def test_format_report_is_text_and_mentions_key_numbers(self):
        records = [_rec(fallback=True, query="Nursing program?"), _rec(fallback=False)]
        text = format_report(build_report(records, top=15), top=15)
        assert "QUERY LOG ANALYTICS" in text
        assert "Fallbacks:" in text
        assert "Nursing program?" in text  # the fallback query is surfaced
        assert "TOP MATCHED PROGRAMS" in text

    def test_format_report_handles_no_fallbacks(self):
        text = format_report(build_report([_rec(fallback=False)], top=15), top=15)
        assert "(none)" in text  # the fallback section says (none), not a crash


class TestResponseLogging:
    """The opt-in response fields (settings.log_responses): analytics must surface them when
    present and stay silent when absent (older/default logs don't carry them)."""

    def test_response_coverage_counts_records_with_a_response(self):
        records = [
            _rec(response="Computer Science teaches...", response_sources=["Computer_Science_2026.pdf"]),
            _rec(),  # no response field
        ]
        assert response_coverage(records) == 1

    def test_response_coverage_is_zero_when_logging_was_off(self):
        assert response_coverage([_rec(), _rec()]) == 0

    def test_cited_sources_aggregates_across_response_sources(self):
        records = [
            _rec(response="a", response_sources=["Computer_Science_2026.pdf", "Data_Science_2026.pdf"]),
            _rec(response="b", response_sources=["Computer_Science_2026.pdf"]),
            _rec(),  # no response_sources
        ]
        assert cited_sources(records, limit=10) == [
            ("Computer_Science_2026.pdf", 2), ("Data_Science_2026.pdf", 1),
        ]

    def test_cited_sources_empty_without_response_logging(self):
        assert cited_sources([_rec(), _rec()], limit=10) == []

    def test_report_shows_response_section_when_present(self):
        records = [_rec(response="answer text", response_sources=["Cyber_Security_2025.pdf"])]
        text = format_report(build_report(records, top=15), top=15)
        assert "RESPONSES LOGGED" in text
        assert "Cyber_Security_2025.pdf" in text

    def test_report_hides_response_section_when_absent(self):
        text = format_report(build_report([_rec(), _rec()], top=15), top=15)
        assert "RESPONSES LOGGED" not in text


class TestLoadRecords:
    def test_parses_jsonl_and_skips_blank_and_corrupt_lines(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text(
            '{"query": "a", "fallback": false}\n'
            "\n"                                   # blank line
            "{not valid json\n"                    # corrupt line (e.g. crash mid-append)
            '{"query": "b", "fallback": true}\n',
            encoding="utf-8",
        )
        records = load_records(p)
        assert [r["query"] for r in records] == ["a", "b"]
