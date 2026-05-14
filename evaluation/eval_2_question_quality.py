#!/usr/bin/env python3
"""
eval_2_question_quality.py — Measure question quality across the dataset.

METHOD
------
Three checks, all deterministic and model-independent:
  1. Generic question detection — catches "according to the text", etc.
  2. Length distribution — too short = lacks specificity.
  3. Question type distribution — what/how/where/why mix.

WHY
---
Even a 100% faithful answer is useless if the question is generic, vague, or
meta-textual. A fine-tuned model trained on bad questions will produce bad
questions at inference time. This metric catches that pre-training.

USAGE
-----
    python eval_2_question_quality.py --pairs qa_pairs.jsonl --output-dir eval_results/
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

# Patterns that indicate generic / meta-textual questions
GENERIC_PATTERNS = [
    r"^what is described",
    r"^what does (this|the) (text|section|passage|chunk)",
    r"^what (is|are) the (main )?(point|topic|subject|content)",
    r"^what does it say",
    r"^summarize",
    r"^tell me about (this|the section)",
    r"\baccording to the (text|section|passage|chunk)\b",
    r"\bin (this|the) (text|section|passage|chunk)\b",
    r"\b(the|this) (text|section|passage) (states|describes|mentions|says|provides)\b",
]

QUESTION_TYPE_PATTERNS = {
    "what":   r"^what\b",
    "how":    r"^how\b",
    "where":  r"^where\b",
    "when":   r"^when\b",
    "why":    r"^why\b",
    "which":  r"^which\b",
    "who":    r"^who\b",
    "yes/no": r"^(is|are|does|do|can|will|should|may|has|have|did)\b",
}

# Targets — flag if exceeded
TARGET_MAX_GENERIC_PCT = 5.0
TARGET_MIN_WORDS = 8


def extract_q(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def is_generic(q: str) -> bool:
    ql = q.lower()
    return any(re.search(p, ql) for p in GENERIC_PATTERNS)


def question_type(q: str) -> str:
    ql = q.strip().lower()
    for label, pat in QUESTION_TYPE_PATTERNS.items():
        if re.match(pat, ql):
            return label
    return "other"


def word_count(s: str) -> int:
    return len(s.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    pairs = []
    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[info] Loaded {len(pairs)} pairs from {args.pairs}")

    # Score every question
    rows = []
    for i, p in enumerate(pairs):
        q = extract_q(p)
        rows.append({
            "idx": i,
            "page": p.get("provenance", {}).get("page", ""),
            "category": p.get("provenance", {}).get("category", ""),
            "word_count": word_count(q),
            "is_generic": int(is_generic(q)),
            "type": question_type(q),
            "is_short": int(word_count(q) < TARGET_MIN_WORDS),
            "question": q,
        })

    # Write per-pair CSV
    csv_path = args.output_dir / "question_quality_per_pair.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate
    n = len(rows)
    words = [r["word_count"] for r in rows]
    n_generic = sum(r["is_generic"] for r in rows)
    n_short = sum(r["is_short"] for r in rows)
    type_dist = Counter(r["type"] for r in rows)

    summary = {
        "metric": "question_quality",
        "n_pairs": n,
        "length": {
            "mean_words": round(statistics.mean(words), 1),
            "median_words": int(statistics.median(words)),
            "min_words": min(words),
            "max_words": max(words),
            "short_count": n_short,
            "short_pct": round(100 * n_short / n, 2),
        },
        "generic": {
            "count": n_generic,
            "pct": round(100 * n_generic / n, 2),
            "target_max_pct": TARGET_MAX_GENERIC_PCT,
            "passes": (100 * n_generic / n) <= TARGET_MAX_GENERIC_PCT,
        },
        "type_distribution": {
            k: {"count": v, "pct": round(100 * v / n, 2)}
            for k, v in type_dist.most_common()
        },
    }

    summary_path = args.output_dir / "question_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print
    print(f"\n=== QUESTION QUALITY ===")
    print(f"  Pairs evaluated:    {n}")
    print(f"  Length (words):     mean {summary['length']['mean_words']}, "
          f"median {summary['length']['median_words']}, "
          f"range {summary['length']['min_words']}-{summary['length']['max_words']}")
    print(f"  Short (<{TARGET_MIN_WORDS} words): {n_short}  ({summary['length']['short_pct']}%)")
    print(f"  Generic questions:  {n_generic}  ({summary['generic']['pct']}%)   "
          f"target <= {TARGET_MAX_GENERIC_PCT}%   "
          f"{'PASS' if summary['generic']['passes'] else 'FAIL'}")
    print(f"  Type distribution:")
    for t, v in type_dist.most_common():
        print(f"     {t:8s}  {v:4d}  ({100*v/n:5.1f}%)")
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
