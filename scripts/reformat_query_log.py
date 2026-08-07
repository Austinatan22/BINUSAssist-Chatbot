"""Reformats an existing query log's records from one-per-line to pretty-printed.

chat_service._log_query wrote one compact record per line until 2026-08-08 and pretty-prints
them since. Both formats parse (log_analytics.load_records scans with raw_decode), so nothing is
broken by leaving a log mixed -- this just makes the older half as readable as the newer half,
which is the whole point of the format change. One-time per deployment, and idempotent: running
it twice changes nothing the second time.

Whitespace only. It re-serializes the same objects with the same keys in the same order, and does
NOT add the fields introduced alongside the format change (`route`, `ttft_ms`, `plan_ms`,
`language`, `aspects`, `node_count`, `cache`, `rewrite_queries`). Most of those were never
measured and are simply gone; the two that look recomputable are the dangerous ones. `route` can
be guessed from matched_programs + query_type + aspects, but that guess is precisely what the
field exists to replace, and the routing changed on 2026-08-07 (tuition now bypasses the program
catalog), so an older query would be labelled with a branch that did not exist when it ran.
`language` and `aspects` come from pure functions, but both have been edited since, so today's
value is not the value that turn used. A backfilled field is worse than a missing one: the report
excludes a missing field and says so, while a reconstructed one is indistinguishable from a
measured one.

Safety: the rewrite is verified before it is applied. The reformatted text is parsed back and
compared record-for-record against the original, and the file is only replaced if they are
identical. The original is copied to `<name>.bak-YYYYMMDD` first (already gitignored, see
.gitignore's `backend/query_log.jsonl.bak-*`, the same pattern used for the 2026-07-18 probe-query
cleanup). This matters because the log is gitignored runtime data: there is no diff to review, and
the 667 records in it are the source of scripts/eval.py's in_scope_traffic questions.

Usage:
  python scripts/reformat_query_log.py --dry-run     # report what would change
  python scripts/reformat_query_log.py               # back up, verify, rewrite
  python scripts/reformat_query_log.py --log PATH
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from scripts.log_analytics import load_records


def reformat(text: str) -> str:
    """Every record in `text`, pretty-printed. Pure, so it's the part worth testing directly."""
    records = _parse(text)
    return "".join(json.dumps(r, indent=2, ensure_ascii=False) + "\n" for r in records)


def _parse(text: str) -> list[dict]:
    """load_records over a string rather than a path, so reformat() stays filesystem-free."""
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


def compact_record_count(text: str) -> int:
    """Records still written as a single line -- i.e. how many this would actually change. A
    pretty-printed record's opening brace is alone on its line, so a line that starts with "{"
    and also ends with "}" is a compact one."""
    return sum(
        1 for line in text.splitlines()
        if line.startswith("{") and line.rstrip().endswith("}")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=settings.query_log_path)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and exit without writing")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"No query log at {args.log} -- nothing to reformat.", file=sys.stderr)
        return 1

    original_text = args.log.read_text(encoding="utf-8")
    original = _parse(original_text)
    compact = compact_record_count(original_text)
    print(f"{args.log}: {len(original)} records, {compact} still on one line")
    if not compact:
        print("Already reformatted -- nothing to do.")
        return 0

    reformatted_text = reformat(original_text)

    # The whole safety argument: parse the new text and require it to hold exactly the same
    # records, in the same order, with the same values. Anything less and we don't write.
    roundtripped = _parse(reformatted_text)
    if roundtripped != original:
        print(
            f"REFUSING TO WRITE: round-trip check failed "
            f"({len(original)} records in, {len(roundtripped)} back out). The log is unchanged.",
            file=sys.stderr,
        )
        return 1
    print(f"Round-trip verified: {len(roundtripped)}/{len(original)} records identical.")

    if args.dry_run:
        print(f"--dry-run: would rewrite {args.log} "
              f"({len(original_text):,} chars -> {len(reformatted_text):,}).")
        return 0

    backup = args.log.with_suffix(args.log.suffix + f".bak-{datetime.now():%Y%m%d}")
    shutil.copy2(args.log, backup)
    print(f"Backed up to {backup}")
    # Write to a sibling temp file then replace, so an interrupted write can't truncate the log.
    tmp = args.log.with_suffix(args.log.suffix + ".tmp")
    tmp.write_text(reformatted_text, encoding="utf-8")
    tmp.replace(args.log)
    print(f"Rewrote {args.log}: {compact} records reformatted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
