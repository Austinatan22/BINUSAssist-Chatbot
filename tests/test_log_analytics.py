"""Tests for scripts/log_analytics.py -- the query-log analytics report (IMPROVEMENTS.md
#6.1's analytics half). All aggregation is pure functions over record dicts, so these run
with no filesystem, models, GPU, or network. Records are hand-built to mirror the real log
schema, including the schema drift the parser must tolerate (older rows lack `top_score` /
`unresolved_term`, and rows before 2026-08-08 lack `route` / `ttft_ms` / `language` / `cache`)."""
from datetime import datetime, timedelta, timezone

import pytest

from scripts.log_analytics import (
    build_report,
    cache_breakdown,
    cited_sources,
    fallback_queries,
    filter_since,
    format_report,
    language_breakdown,
    load_records,
    query_type_breakdown,
    response_coverage,
    route_breakdown,
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
        "language": "en",
        "route": "program_scoped",
        "node_count": 5,
        "cache": "miss",
        "rewrite_triggered": False,
        "plan_ms": 900,
        "ttft_ms": 1200,
        "latency_ms": 1500,
    }
    base.update(kw)
    return base


def _old_rec(**kw):
    """A record in the pre-2026-08-08 shape: no route, ttft_ms, language, cache or
    rewrite_triggered. Every metric derived from those fields must exclude it rather than read
    a missing value as zero."""
    rec = _rec(**kw)
    for gone in ("route", "ttft_ms", "plan_ms", "language", "cache", "rewrite_triggered",
                 "node_count"):
        rec.pop(gone, None)
    return rec


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

    def test_ttft_is_scored_against_the_prd_target_separately_from_end_to_end(self):
        # ttft_ms is the PRD criterion; latency_ms includes the whole generation. A turn can
        # pass one and fail the other, so they must not be conflated.
        records = [_rec(ttft_ms=1200, latency_ms=4000), _rec(ttft_ms=5000, latency_ms=6000)]
        s = summarize(records)
        assert s["ttft_ms"]["within_3s"] == 1
        assert s["latency_ms"]["within_3s"] == 0

    def test_pre_ttft_records_are_excluded_not_counted_as_zero(self):
        # The trap: a missing ttft_ms read as 0 would count as "under 3s" and flatter the
        # metric. Old records must drop out of the TTFT scope entirely while still counting
        # towards totals and fallbacks.
        records = [_rec(ttft_ms=1000), _old_rec(), _old_rec(fallback=True)]
        s = summarize(records)
        assert s["total"] == 3
        assert s["fallbacks"] == 1
        assert s["ttft_ms"]["count"] == 1
        assert s["ttft_ms"]["within_3s_rate"] == 1.0

    def test_rewrite_rate_is_over_records_that_carry_the_field(self):
        # Same trap for rewrite_triggered: diluting the rate with records written before the
        # field existed is how the 2026-08-07 "0/66 rewrites" reading went wrong.
        records = [_rec(rewrite_triggered=True), _rec(rewrite_triggered=False), _old_rec()]
        rw = summarize(records)["rewrites"]
        assert (rw["triggered"], rw["scope"]) == (1, 2)

    def test_plan_ms_p50_is_reported(self):
        records = [_rec(plan_ms=500), _rec(plan_ms=900), _rec(plan_ms=1300)]
        assert summarize(records)["plan_ms_p50"] == 900

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

    def test_route_breakdown_counts_fallbacks_and_rewrites_per_route(self):
        records = [
            _rec(route="program_scoped", rewrite_triggered=True, ttft_ms=6000),
            _rec(route="program_scoped", rewrite_triggered=False, ttft_ms=1000),
            _rec(route="program_scoped", rewrite_triggered=False, ttft_ms=1400),
            _rec(route="open", rewrite_triggered=True, fallback=True, ttft_ms=2000),
        ]
        rows = {row["route"]: row for row in route_breakdown(records)}
        assert rows["program_scoped"]["count"] == 3
        assert (rows["program_scoped"]["rewrites"], rows["program_scoped"]["rewrite_scope"]) == (1, 3)
        assert rows["program_scoped"]["fallbacks"] == 0
        assert rows["program_scoped"]["ttft_p50"] == 1400
        assert rows["open"]["fallbacks"] == 1

    def test_route_breakdown_is_ordered_by_count(self):
        records = [_rec(route="open")] + [_rec(route="program_scoped")] * 3
        assert [row["route"] for row in route_breakdown(records)] == ["program_scoped", "open"]

    def test_route_breakdown_excludes_records_with_no_route(self):
        # Pre-2026-08-08 records would otherwise form the biggest row in any window spanning
        # the change, and an "unknown" row that large says nothing.
        assert route_breakdown([_old_rec(), _old_rec()]) == []

    def test_language_and_cache_breakdowns(self):
        records = [
            _rec(language="id", cache="hit"),
            _rec(language="id", cache="miss"),
            _rec(language="en", cache="rejected"),
            _old_rec(),  # carries neither field
        ]
        assert language_breakdown(records) == [("id", 2), ("en", 1)]
        # Fixed hit -> rejected -> miss order, not count order: reading a cache column means
        # comparing the same positions between two runs.
        assert cache_breakdown(records) == [("hit", 1), ("rejected", 1), ("miss", 1)]

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
            "summary", "by_query_type", "by_route", "by_language", "by_cache",
            "top_programs", "top_fallback_queries", "unresolved_terms",
            "response_coverage", "cited_sources",
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
    """The log holds two record formats: compact one-per-line (everything written before
    2026-08-08) and pretty-printed multi-line (everything since). Both must parse from the same
    file, since the 667 compact records are the source of eval.py's in_scope_traffic questions
    and are not going to be rewritten."""

    def test_parses_compact_records_and_skips_blank_and_corrupt_lines(self, tmp_path):
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

    def test_parses_pretty_printed_records(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text(
            '{\n  "query": "a",\n  "matched_programs": [\n    "Computer Science"\n  ],\n'
            '  "fallback": false\n}\n'
            '{\n  "query": "b",\n  "fallback": true\n}\n',
            encoding="utf-8",
        )
        records = load_records(p)
        assert [r["query"] for r in records] == ["a", "b"]
        assert records[0]["matched_programs"] == ["Computer Science"]

    def test_parses_a_file_holding_both_formats(self, tmp_path):
        # What the real log looks like from 2026-08-08 on: old compact rows, then new ones.
        p = tmp_path / "log.jsonl"
        p.write_text(
            '{"query": "old", "fallback": false}\n'
            '{\n  "query": "new",\n  "route": "open",\n  "fallback": true\n}\n',
            encoding="utf-8",
        )
        assert [r["query"] for r in load_records(p)] == ["old", "new"]

    def test_a_truncated_final_record_does_not_lose_the_rest(self, tmp_path):
        # A crash mid-append leaves a half-written object. Recovery resyncs on the next "{" at
        # column 0, which is a record boundary in both formats and, since JSON escapes newlines
        # inside strings, can never occur inside one.
        p = tmp_path / "log.jsonl"
        p.write_text(
            '{\n  "query": "a",\n  "fallback": false\n}\n'
            '{\n  "query": "trunc",\n  "resp'          # cut off mid-string
            '\n{\n  "query": "c",\n  "fallback": true\n}\n',
            encoding="utf-8",
        )
        assert [r["query"] for r in load_records(p)] == ["a", "c"]

    def test_non_object_json_is_ignored(self, tmp_path):
        # Guards the aggregation functions, which all call .get() on each record.
        p = tmp_path / "log.jsonl"
        p.write_text('[1, 2]\n"a string"\n{"query": "a", "fallback": false}\n', encoding="utf-8")
        assert [r["query"] for r in load_records(p)] == ["a"]

    def test_empty_file_yields_no_records(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_records(p) == []
