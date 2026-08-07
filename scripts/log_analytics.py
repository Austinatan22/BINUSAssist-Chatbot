"""Analytics view over the query log (IMPROVEMENTS.md #6.1's un-done half).

`backend/query_log.jsonl` captures one record per /chat request -- the raw and condensed
query, its language, which retrieval route answered it, matched program(s), top rerank score,
`fallback`, whether the rewrite retry fired, time-to-first-token and end-to-end latency. This
turns that raw log into a report, so the bot becomes a feedback loop for the KB: the headline
signal is FALLBACKS grouped by query ("15 people asked about housing this week; we have nothing
on it"), which are a direct pointer at content gaps.

Pure JSON parsing -- no models, GPU, LLM API, or network, so it's fast and runs anywhere. All
the aggregation lives in pure functions (summarize / *_breakdown / top_*) that take a list of
record dicts, so they're unit-tested directly (tests/test_log_analytics.py) without touching
the filesystem. Tolerates schema drift: every field is read with .get(), since older records
predate later additions (`top_score`, `unresolved_term`, and the whole 2026-08-08 batch --
`route`, `ttft_ms`, `plan_ms`, `language`, `aspects`, `node_count`, `cache`). A missing field
means "not measured" and is excluded from its metric rather than counted as zero, so a window
spanning the change reports honestly on both halves.

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

# PRD §9's latency criterion: <3s for 90% of queries, measured on time-to-first-token. Records
# written from 2026-08-08 carry `ttft_ms` and are scored against it directly; older records have
# only end-to-end `latency_ms` (the whole generation included), a looser cousin reported as
# context. Both are shown, TTFT first, since that is the criterion itself.
_LATENCY_TARGET_MS = 3000


def load_records(path: Path) -> list[dict]:
    """Parse the query log into a list of record dicts.

    Handles both record formats in one pass. chat_service._log_query used to write one compact
    record per line; it now pretty-prints each record over several lines, because a record with
    ten cited tuition URLs runs past 1,500 characters and nobody can read that in an editor.
    Splitting on newlines would break on the new format and reading a JSON array would break on
    the old, so scan with raw_decode instead: it consumes one object at a time and reports where
    it stopped, which works whatever the whitespace between objects looks like.

    A corrupt or half-written record (a crash mid-append) is skipped rather than sinking the
    rest of the file: resync on the next line beginning with "{" at column 0, which is where
    every record starts and, since JSON escapes newlines inside strings, can never occur inside
    one. Reads the whole file into memory, which is fine at this scale (369KB / 667 records) for
    a report script that is run by hand.
    """
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            boundary = text.find("\n{", i)
            if boundary == -1:
                break
            i = boundary + 1
            continue
        if isinstance(obj, dict):
            records.append(obj)
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


def _numbers(records: list[dict], field: str) -> list[float]:
    """The numeric values of `field` across records, skipping records that lack it or carry
    null. Older records predate several fields, so a missing value means "not measured", never 0
    -- averaging a missing latency as zero would quietly flatter every percentile."""
    return [r[field] for r in records if isinstance(r.get(field), (int, float))]


def _timing(values: list[float]) -> dict:
    """Percentiles plus the share under the 3s target, for one timing field."""
    within = sum(1 for ms in values if ms < _LATENCY_TARGET_MS)
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p99": _percentile(values, 99),
        "max": max(values) if values else 0,
        "within_3s": within,
        "within_3s_rate": (within / len(values)) if values else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    """Headline metrics over a record list. Everything is derived, never assumed: fallback
    rate is the PRD #6.1 signal; TTFT percentiles are the PRD §9 latency criterion, with
    end-to-end latency alongside for the records that predate ttft_ms."""
    total = len(records)
    fallbacks = [r for r in records if r.get("fallback")]
    scores = _numbers(records, "top_score")
    timestamps = [ts for ts in (_parse_ts(r) for r in records) if ts is not None]
    rewrite_scope = [r for r in records if r.get("rewrite_triggered") is not None]
    return {
        "total": total,
        "fallbacks": len(fallbacks),
        "fallback_rate": (len(fallbacks) / total) if total else 0.0,
        "ttft_ms": _timing(_numbers(records, "ttft_ms")),
        "latency_ms": _timing(_numbers(records, "latency_ms")),
        # Median time spent in ChatService._plan (condense, cache lookup, classification,
        # retrieval) against median TTFT: the difference is the LLM provider's, and this split
        # is what says whether a slow turn is ours to fix.
        "plan_ms_p50": _percentile(_numbers(records, "plan_ms"), 50),
        "top_score": {
            "count": len(scores),
            "mean": (sum(scores) / len(scores)) if scores else 0.0,
        },
        # The rewrite retry is the most expensive optional step in the pipeline (an LLM call
        # plus a second retrieval pass). Counted only over records that carry the field at all,
        # so the rate isn't diluted by the records written before it existed.
        "rewrites": {
            "scope": len(rewrite_scope),
            "triggered": sum(1 for r in rewrite_scope if r.get("rewrite_triggered")),
        },
        "date_range": (
            (min(timestamps).isoformat(), max(timestamps).isoformat()) if timestamps else (None, None)
        ),
    }


def query_type_breakdown(records: list[dict]) -> list[tuple[str, int]]:
    """(query_type, count) most-common first. Missing type -> 'unknown'."""
    return Counter(r.get("query_type") or "unknown" for r in records).most_common()


def route_breakdown(records: list[dict]) -> list[dict]:
    """Per-route counts, most-used first: how often each branch of _route_retrieval ran, how
    often it fell back, how often it needed the rewrite retry to rescue it, and its median TTFT.

    This is the table `query_type` could never give. query_type says how many programs were
    named; `route` says which branch actually answered, and the two disagree (a tuition query
    names one program and is logged "comparison" because it renders as a table). The rewrite
    column is the one to read: a route that needs the retry on a large share of its traffic is
    a route whose first retrieval is aimed at the wrong sources.

    Records with no `route` (written before 2026-08-08, or turns where no routing ran) are
    excluded rather than lumped into an "unknown" row, which would be the largest row in any
    window spanning the change and would say nothing.
    """
    by_route: dict[str, list[dict]] = {}
    for r in records:
        route = r.get("route")
        if route:
            by_route.setdefault(route, []).append(r)
    rows = []
    for route, rs in by_route.items():
        rewrite_scope = [r for r in rs if r.get("rewrite_triggered") is not None]
        rows.append({
            "route": route,
            "count": len(rs),
            "fallbacks": sum(1 for r in rs if r.get("fallback")),
            "rewrites": sum(1 for r in rewrite_scope if r.get("rewrite_triggered")),
            "rewrite_scope": len(rewrite_scope),
            "ttft_p50": _percentile(_numbers(rs, "ttft_ms"), 50),
        })
    return sorted(rows, key=lambda row: -row["count"])


def language_breakdown(records: list[dict]) -> list[tuple[str, int]]:
    """(language, count) most-common first -- the split between Indonesian and English traffic,
    which the fallback and latency numbers are worth reading against separately (an Indonesian
    question against an English-only catalog PDF is the exact failure the rewrite retry exists
    for). Absent on records predating the field."""
    return Counter(r["language"] for r in records if r.get("language")).most_common()


def cache_breakdown(records: list[dict]) -> list[tuple[str, int]]:
    """(hit / rejected / miss, count). "rejected" means the embedding was close enough but a
    deterministic gate vetoed the replay (see rag/cache.py); it used to appear only as a log
    line nothing aggregated. Absent on records predating the field."""
    order = {"hit": 0, "rejected": 1, "miss": 2}
    counts = Counter(r["cache"] for r in records if r.get("cache"))
    return sorted(counts.items(), key=lambda kv: order.get(kv[0], 99))


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
        "by_route": route_breakdown(records),
        "by_language": language_breakdown(records),
        "by_cache": cache_breakdown(records),
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
    ttft = s["ttft_ms"]
    if ttft["count"]:
        lines.append(
            f"TTFT (ms):     p50={ttft['p50']:.0f}  p90={ttft['p90']:.0f}  "
            f"p99={ttft['p99']:.0f}  max={ttft['max']:.0f}"
        )
        lines.append(
            f"  under 3s:    {ttft['within_3s']}/{ttft['count']}  "
            f"({_fmt_pct(ttft['within_3s_rate'])})   <- PRD 9 target: 90%"
        )
        if s["plan_ms_p50"]:
            lines.append(
                f"  of which p50 retrieval+routing: {s['plan_ms_p50']:.0f}ms  "
                f"(the rest is the LLM)"
            )
    if lat["count"]:
        lines.append(
            f"End-to-end (ms):  p50={lat['p50']:.0f}  p90={lat['p90']:.0f}  "
            f"p99={lat['p99']:.0f}  max={lat['max']:.0f}"
        )
    if s["top_score"]["count"]:
        lines.append(f"Mean top rerank score: {s['top_score']['mean']:.3f}")
    rw = s["rewrites"]
    if rw["scope"]:
        lines.append(
            f"Rewrite retry fired: {rw['triggered']}/{rw['scope']}  "
            f"({_fmt_pct(rw['triggered'] / rw['scope'])})   <- +1 LLM call, +1 retrieval pass"
        )
    if report["by_cache"]:
        lines.append(
            "Semantic cache:  "
            + "  ".join(f"{state}={n}" for state, n in report["by_cache"])
        )
    if report["by_language"]:
        lines.append(
            "Language:        "
            + "  ".join(f"{lang}={n}" for lang, n in report["by_language"])
        )

    lines.append("")
    lines.append("BY QUERY TYPE")
    for qtype, n in report["by_query_type"]:
        lines.append(f"  {n:5}  {qtype}")

    if report["by_route"]:
        lines.append("")
        lines.append("BY RETRIEVAL ROUTE -- which branch answered, and what it cost")
        lines.append(f"  {'count':>5} {'fallback':>9} {'rewrite':>8} {'ttft p50':>9}  route")
        for row in report["by_route"]:
            rewrite = (
                f"{row['rewrites']}/{row['rewrite_scope']}" if row["rewrite_scope"] else "-"
            )
            ttft_p50 = f"{row['ttft_p50']:.0f}ms" if row["ttft_p50"] else "-"
            lines.append(
                f"  {row['count']:5} {row['fallbacks']:9} {rewrite:>8} {ttft_p50:>9}  {row['route']}"
            )

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
