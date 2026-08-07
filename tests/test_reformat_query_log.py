"""Tests for scripts/reformat_query_log.py -- the one-time migration that pretty-prints the
records a deployment's query log already holds.

This script rewrites 667 records of real production traffic that are gitignored (so there is no
diff to review) and are the source of scripts/eval.py's in_scope_traffic questions. The property
that matters is therefore not "it produces nice output" but "it loses nothing", which is what
these assert. Pure string in, string out -- no filesystem, models or network.
"""
import json

from scripts.reformat_query_log import _parse, compact_record_count, reformat

_COMPACT = (
    '{"timestamp": "2026-07-13T17:37:06+00:00", "query": "What are the career prospects '
    'for Computer Science graduates?", "query_type": "single", "matched_programs": '
    '["Computer Science"], "top_score": 0.775, "fallback": false, "history_turns": 0, '
    '"latency_ms": 3722}\n'
    '{"timestamp": "2026-07-13T17:43:11+00:00", "query": "Berapa biaya kuliah Teknik '
    'Informatika?", "query_type": "comparison", "matched_programs": [], "fallback": true, '
    '"latency_ms": 1048}\n'
)


class TestReformatLosesNothing:
    def test_records_survive_unchanged(self):
        before = _parse(_COMPACT)
        after = _parse(reformat(_COMPACT))
        assert after == before

    def test_key_order_is_preserved_not_normalized(self):
        # It's a log. Re-serializing is defensible; reordering what was written is a second
        # transformation with no reader asking for it.
        assert list(_parse(reformat(_COMPACT))[0]) == list(_parse(_COMPACT)[0])

    def test_output_is_pretty_printed(self):
        out = reformat(_COMPACT)
        assert out.count("\n") > 2 * _COMPACT.count("\n")
        assert '\n  "query": ' in out

    def test_idempotent(self):
        # Anyone can run the script twice; the second run must be a no-op.
        once = reformat(_COMPACT)
        assert reformat(once) == once

    def test_indonesian_text_is_not_escaped(self):
        out = reformat(_COMPACT)
        assert "Teknik Informatika" in out
        assert "\\u" not in out

    def test_a_record_holding_a_newline_is_not_split_by_it(self):
        # The real log's `response` field holds markdown tables full of "\n". Those are JSON
        # escapes, not line breaks, but a formatter that reasoned about lines rather than objects
        # would mangle them -- and the resync rule ("{" at column 0") depends on it.
        record = {"query": "tuition?", "response": "| Fee | Amount |\n|---|---|\n| DP3 | Rp48m |"}
        text = json.dumps(record) + "\n"
        assert _parse(reformat(text)) == [record]

    def test_empty_log_produces_empty_output(self):
        assert reformat("") == ""


class TestCompactRecordCount:
    """Drives the script's "nothing to do" exit, so it has to tell the two formats apart."""

    def test_counts_one_line_records(self):
        assert compact_record_count(_COMPACT) == 2

    def test_pretty_printed_records_count_as_zero(self):
        assert compact_record_count(reformat(_COMPACT)) == 0

    def test_a_mixed_log_counts_only_the_compact_half(self):
        # What every deployment's log looks like from 2026-08-08 until it is migrated.
        assert compact_record_count(_COMPACT + reformat(_COMPACT)) == 2
