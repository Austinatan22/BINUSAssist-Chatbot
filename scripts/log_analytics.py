"""Analytics view over the query log (IMPROVEMENTS.md #6.1's un-done half).

`backend/query_log.jsonl` captures one record per /chat request -- the raw and condensed
query, query_type, matched program(s), top rerank score, `fallback`, history-turn count, and
latency. This turns that raw log into a report, so the bot becomes a feedback loop for the KB:
the headline signal is FALLBACKS grouped by query ("15 people asked about housing this week;
we have nothing on it"), which are a direct pointer at content gaps.

Pure JSONL parsing -- no models, GPU, Groq, or network, so it's fast and runs anywhere. All
the aggregation lives in pure functions (summarize / *_breakdown / top_*) that take a list of
record dicts, so they're unit-tested directly (tests/test_log_analytics.py) without touching
the filesystem. Tolerates schema drift: every field is read with .get(), since older records
predate later additions (e.g. `top_score`, `unresolved_term`).

Usage:
  python scripts/log_analytics.py                 # full report over the whole log
  python scripts/log_analytics.py --days 7        # only the last 7 days (PRD's "this week")
  python scripts/log_analytics.py --since 2026-07-15
  python scripts/log_analytics.py --top 20        # show more fallback queries / programs
  python scripts/log_analytics.py --json          # machine-readable summary instead of text
  python scripts/log_analytics.py --log PATH      # a different log file
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings

# PRD §9 latency success criterion: this is END-TO-END latency (includes the LLM call), not
# time-to-first-token, so it's a looser cousin of the PRD's <3s TTFT target -- reported as
# context, not as the PRD metric itself (which eval.py measures on TTFT).
_LATENCY_TARGET_MS = 3000


def load_records(path: Path) -> list[dict]:
    """Parse a JSONL query log into a list of record dicts, skipping blank/corrupt lines
    (a partially-written final line from a crash mid-append shouldn't sink the whole report)."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _parse_ts(record: dict) -> datetime | None:
    ts = record.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def filter_since(records: list[dict], since: datetime) -> list[dict]:
    """Records at or after `since`. Records with no/unparseable timestamp are dropped when
    filtering (they can't be placed on the timeline), so an explicit window means what it says."""
    kept = []
    for r in records:
        ts = _parse_ts(r)
        if ts is not None and ts >= since:
            kept.append(r)
    return kept


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation) -- enough for a latency report, and avoids
    a numpy dependency for a script that otherwise needs nothing but the stdlib."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


def summarize(records: list[dict]) -> dict:
    """Headline metrics over a record list. Everything is derived, never assumed: fallback
    rate is the PRD #6.1 signal; latency percentiles are reported against the 3s target."""
    total = len(records)
    fallbacks = [r for r in records if r.get("fallback")]
    latencies = [r["latency_ms"] for r in records if isinstance(r.get("latency_ms"), (int, float))]
    scores = [r["top_score"] for r in records if isinstance(r.get("top_score"), (int, float))]
    timestamps = [ts for ts in (_parse_ts(r) for r in records) if ts is not None]
    within_target = sum(1 for ms in latencies if ms < _LATENCY_TARGET_MS)
    return {
        "total": total,
        "fallbacks": len(fallbacks),
        "fallback_rate": (len(fallbacks) / total) if total else 0.0,
        "latency_ms": {
            "count": len(latencies),
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p99": _percentile(latencies, 99),
            "max": max(latencies) if latencies else 0,
            "within_3s": within_target,
            "within_3s_rate": (within_target / len(latencies)) if latencies else 0.0,
        },
        "top_score": {
            "count": len(scores),
            "mean": (sum(scores) / len(scores)) if scores else 0.0,
        },
        "date_range": (
            (min(timestamps).isoformat(), max(timestamps).isoformat()) if timestamps else (None, None)
        ),
    }


def query_type_breakdown(records: list[dict]) -> list[tuple[str, int]]:
    """(query_type, count) most-common first. Missing type -> 'unknown'."""
    return Counter(r.get("query_type") or "unknown" for r in records).most_common()


def top_programs(records: list[dict], limit: int) -> list[tuple[str, int]]:
    """Most-frequently matched catalog programs -- what people actually ask about."""
    counter: Counter = Counter()
    for r in records:
        for p in r.get("matched_programs") or []:
            counter[p] += 1
    return counter.most_common(limit)


def fallback_queries(records: list[dict], limit: int) -> list[tuple[str, int]]:
    """Fallback queries grouped by (case-folded, whitespace-collapsed) text, most-frequent
    first -- the #6.1 headline: a repeated fallback is a repeated unmet need = a content gap.
    Grouping is on a normalized key so trivial spacing/case variants collapse, but the ORIGINAL
    (first-seen) casing is what's shown back."""
    counter: Counter = Counter()
    display: dict[str, str] = {}
    for r in records:
        if not r.get("fallback"):
            continue
        raw = (r.get("query") or "").strip()
        if not raw:
            continue
        key = " ".join(raw.lower().split())
        counter[key] += 1
        display.setdefault(key, raw)
    return [(display[key], n) for key, n in counter.most_common(limit)]


def unresolved_terms(records: list[dict], limit: int) -> list[tuple[str, int]]:
    """Terms the clarification flow couldn't resolve (a garbled campus/program the user named)
    -- a second content/alias-gap signal, distinct from a plain fallback."""
    counter = Counter(
        r["unresolved_term"] for r in records if r.get("unresolved_term")
    )
    return counter.most_common(limit)


def cited_sources(records: list[dict], limit: int) -> list[tuple[str, int]]:
    """Most-frequently cited source files across answers -- only present when response logging
    is on (settings.log_responses), so this is empty on logs captured with it off. Shows which
    documents/URLs are actually carrying the answers (and which never get cited)."""
    counter: Counter = Counter()
    for r in records:
        for sf in r.get("response_sources") or []:
            counter[sf] += 1
    return counter.most_common(limit)


def response_coverage(records: list[dict]) -> int:
    """How many records carry a logged assistant response (response logging on). 0 when off."""
    return sum(1 for r in records if r.get("response") is not None)


def build_report(records: list[dict], top: int) -> dict:
    """The whole report as plain data (also what --json emits)."""
    return {
        "summary": summarize(records),
        "by_query_type": query_type_breakdown(records),
        "top_programs": top_programs(records, top),
        "top_fallback_queries": fallback_queries(records, top),
        "unresolved_terms": unresolved_terms(records, top),
        "response_coverage": response_coverage(records),
        "cited_sources": cited_sources(records, top),
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_report(report: dict, top: int) -> str:
    """Human-readable text rendering of build_report()'s data."""
    s = report["summary"]
    lo, hi = s["date_range"]
    lat = s["latency_ms"]
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("QUERY LOG ANALYTICS")
    lines.append("=" * 72)
    if lo and hi:
        lines.append(f"Window:        {lo[:19]}  ->  {hi[:19]}")
    lines.append(f"Total queries: {s['total']}")
    lines.append(
        f"Fallbacks:     {s['fallbacks']}  ({_fmt_pct(s['fallback_rate'])})   "
        "<- content-gap signal (#6.1)"
    )
    if lat["count"]:
        lines.append(
            f"Latency (end-to-end, ms):  p50={lat['p50']:.0f}  p90={lat['p90']:.0f}  "
            f"p99={lat['p99']:.0f}  max={lat['max']:.0f}"
        )
        lines.append(
            f"  under 3s:    {lat['within_3s']}/{lat['count']}  ({_fmt_pct(lat['within_3s_rate'])})"
        )
    if report["summary"]["top_score"]["count"]:
        lines.append(f"Mean top rerank score: {report['summary']['top_score']['mean']:.3f}")

    lines.append("")
    lines.append("BY QUERY TYPE")
    for qtype, n in report["by_query_type"]:
        lines.append(f"  {n:5}  {qtype}")

    lines.append("")
    lines.append(f"TOP FALLBACK QUERIES (up to {top}) -- likely content gaps")
    fbq = report["top_fallback_queries"]
    if not fbq:
        lines.append("  (none)")
    for query, n in fbq:
        marker = f"{n}x " if n > 1 else "   "
        lines.append(f"  {marker}{query[:90]}")

    lines.append("")
    lines.append(f"TOP MATCHED PROGRAMS (up to {top})")
    tp = report["top_programs"]
    if not tp:
        lines.append("  (none)")
    for program, n in tp:
        lines.append(f"  {n:5}  {program}")

    if report["unresolved_terms"]:
        lines.append("")
        lines.append("UNRESOLVED CLARIFICATION TERMS -- alias/content gaps")
        for term, n in report["unresolved_terms"]:
            lines.append(f"  {n:5}  {term}")

    # Only shown when response logging (settings.log_responses) was on for these records.
    if report.get("response_coverage"):
        lines.append("")
        lines.append(
            f"RESPONSES LOGGED: {report['response_coverage']}/{s['total']}  "
            "(LOG_RESPONSES=true -- answer text + cited sources captured)"
        )
        cs = report.get("cited_sources") or []
        if cs:
            lines.append(f"TOP CITED SOURCES (up to {top})")
            for source, n in cs:
                lines.append(f"  {n:5}  {source}")

    lines.append("=" * 72)
    return "\n".join(lines)


def _resolve_since(args) -> datetime | None:
    if args.since:
        dt = datetime.fromisoformat(args.since)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if args.days is not None:
        return datetime.now(timezone.utc) - timedelta(days=args.days)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analytics report over the query log (#6.1).")
    parser.add_argument("--log", type=Path, default=settings.query_log_path,
                        help="path to query_log.jsonl (default: the configured one)")
    parser.add_argument("--days", type=int, default=None,
                        help="only include the last N days")
    parser.add_argument("--since", type=str, default=None,
                        help="only include records at/after this ISO date (e.g. 2026-07-15)")
    parser.add_argument("--top", type=int, default=15,
                        help="how many fallback queries / programs to list (default 15)")
    parser.add_argument("--json", action="store_true",
                        help="emit the report as JSON instead of text")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"No query log at {args.log} -- nothing to report yet.", file=sys.stderr)
        return 1

    records = load_records(args.log)
    since = _resolve_since(args)
    if since is not None:
        records = filter_since(records, since)

    if not records:
        print("No records in the selected window.", file=sys.stderr)
        return 0

    report = build_report(records, args.top)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
