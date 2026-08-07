"""Grade an eval run against the PRD's two manual success criteria.

PRD §9 specifies both by hand: answer relevance (>80% of 50 test questions answered
correctly from the documents) and retrieval precision (>70% of retrieved chunks relevant,
spot-checked on 30 queries). scripts/eval.py measures everything mechanical and leaves a
`relevant` field null for a human. Every eval run in this repo has that field null on every
row, because reading escaped JSON is a bad grading interface.

This is that interface. It walks the ungraded rows one at a time, prints the question, the
retrieved sources and the answer as readable text, and takes 1/0. Progress is written back
to the same JSON after every answer, so quitting halfway loses nothing and re-running
resumes where you stopped.

    python scripts/grade_eval.py                # grade the newest eval_results_*.json
    python scripts/grade_eval.py --report       # score what has been graded so far
    python scripts/grade_eval.py <path.json>    # grade a specific run

Deliberately NOT auto-graded by an LLM: the PRD assigns this to an admin, and a model
scoring the output of its own pipeline measures agreement with itself, not correctness.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The PRD's "50 test questions" for relevance = the curated benchmark plus the real-traffic
# set added 2026-08-08. Other categories (out_of_scope, adversarial, ...) are pass/fail on
# mechanical checks eval.py already makes and need no human judgement.
GRADEABLE = {"in_scope", "in_scope_traffic"}
PRECISION_TARGET_QUERIES = 30  # PRD: spot-check top-5 chunks for 30 queries
RELEVANCE_TARGET = 0.80
PRECISION_TARGET = 0.70


def newest_run() -> Path:
    runs = sorted(ROOT.glob("eval_results_*.json"))
    if not runs:
        sys.exit("No eval_results_*.json found. Run scripts/eval.py first.")
    return runs[-1]


def wrap(text: str, indent: str = "    ", width: int = 96) -> str:
    out = []
    for para in (text or "").split("\n"):
        out.append(textwrap.fill(para, width=width, initial_indent=indent,
                                 subsequent_indent=indent) if para.strip() else "")
    return "\n".join(out)


def save(path: Path, rows: list) -> None:
    # Write through a temp file: a Ctrl-C mid-write must not truncate a run that already
    # holds grading work.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def report(rows: list) -> None:
    pool = [r for r in rows if r["category"] in GRADEABLE]
    answered = [r for r in pool if not r.get("fallback_triggered")]
    fell_back = [r for r in pool if r.get("fallback_triggered")]
    graded = [r for r in answered if r.get("relevant") is not None]
    good = [r for r in graded if r["relevant"]]
    prec = [r for r in answered if r.get("sources_relevant") is not None]
    prec_good = [r for r in prec if r["sources_relevant"]]

    print()
    print("=" * 78)
    print("PRD section 9 manual criteria")
    print("=" * 78)
    print(f"  Relevance pool     : {len(pool)} questions "
          f"({len(answered)} answered, {len(fell_back)} fell back)")
    print(f"  Graded             : {len(graded)}/{len(answered)}")
    if graded:
        pct = len(good) / len(graded)
        verdict = "PASS" if pct >= RELEVANCE_TARGET else "FAIL"
        print(f"  Answer relevance   : {len(good)}/{len(graded)} = {pct:.0%}  "
              f"(target >{RELEVANCE_TARGET:.0%})  {verdict}")
    else:
        print("  Answer relevance   : not yet graded")
    if prec:
        pct = len(prec_good) / len(prec)
        verdict = "PASS" if pct >= PRECISION_TARGET else "FAIL"
        short = "" if len(prec) >= PRECISION_TARGET_QUERIES else \
            f"  [only {len(prec)}/{PRECISION_TARGET_QUERIES} spot-checked]"
        print(f"  Retrieval precision: {len(prec_good)}/{len(prec)} = {pct:.0%}  "
              f"(target >{PRECISION_TARGET:.0%})  {verdict}{short}")
    else:
        print("  Retrieval precision: not yet spot-checked")

    if fell_back:
        print(f"\n  {len(fell_back)} in-scope question(s) fell back and cannot be graded for")
        print("  relevance. These are counted as false fallbacks by eval.py:")
        for r in fell_back:
            print(f"    - [{r['category']}] {r['question'][:74]}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="eval_results_*.json (default: newest)")
    ap.add_argument("--report", action="store_true", help="score only, don't grade")
    args = ap.parse_args()

    path = Path(args.path) if args.path else newest_run()
    rows = json.loads(path.read_text(encoding="utf-8"))
    print(f"Run: {path.name}")

    if args.report:
        report(rows)
        return

    todo = [r for r in rows
            if r["category"] in GRADEABLE
            and not r.get("fallback_triggered")
            and r.get("relevant") is None]
    if not todo:
        print("Nothing left to grade.")
        report(rows)
        return

    already_prec = sum(1 for r in rows if r.get("sources_relevant") is not None)
    print(f"{len(todo)} row(s) to grade.  1 = correct, 0 = wrong, s = skip, q = save and quit.\n")

    for i, row in enumerate(todo, 1):
        print("=" * 96)
        print(f"[{i}/{len(todo)}]  ({row['category']})  top_score={row.get('top_score')}")
        print(f"Q: {row['question']}")
        srcs = row.get("sources") or []
        if srcs:
            print(f"\nSources ({len(srcs)}):")
            for s in srcs:
                name = s.get("source_file") or s.get("file") or "?"
                print(f"    [{s.get('score', '?')}] {str(name)[-72:]}")
        print("\nAnswer:")
        print(wrap(row.get("answer") or "(empty)"))
        print()

        while True:
            try:
                ans = input("  relevant? [1/0/s/q] ").strip().lower()
            except EOFError:
                print("\n(no input available -- saving and exiting)")
                save(path, rows)
                report(rows)
                return
            if ans in ("1", "0", "s", "q"):
                break
        if ans == "q":
            break
        if ans != "s":
            row["relevant"] = int(ans)
            # Retrieval precision is a separate PRD metric over its own sample: only ask
            # while we still need spot-checks, and only where there are chunks to judge.
            if srcs and already_prec < PRECISION_TARGET_QUERIES:
                while True:
                    p = input("  are the retrieved sources relevant? [1/0/s] ").strip().lower()
                    if p in ("1", "0", "s"):
                        break
                if p != "s":
                    row["sources_relevant"] = int(p)
                    already_prec += 1
        save(path, rows)

    save(path, rows)
    report(rows)


if __name__ == "__main__":
    main()
