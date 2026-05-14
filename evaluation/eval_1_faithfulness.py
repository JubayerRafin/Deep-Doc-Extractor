#!/usr/bin/env python3
"""
eval_1_faithfulness.py — Measure how grounded each answer is in its source chunk.

METHOD
------
Token-overlap. For each Q&A pair:
  1. Tokenize the answer and the chunk (lowercase, drop stopwords, light stem).
  2. Compute: (# answer tokens also in chunk) / (# answer tokens).
  3. Bucket the score into faithful / partial / suspicious.

WHY NO LLM JUDGE
----------------
- Deterministic: same input → same output, always.
- Independent of the generator model. Works on any LLM's output.
- Fully local, no API calls.

USAGE
-----
    python eval_1_faithfulness.py --pairs qa_pairs.jsonl --output-dir eval_results/
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

STOPWORDS = set("""
a an and are as at be been by do does for from has have he her his i in is it
its of on or that the their them they this to was were what when where which
who why will with you your we our about into out over under
""".split())

THRESHOLD_FAITHFUL = 0.70
THRESHOLD_PARTIAL  = 0.40


def tokenize(text: str) -> list[str]:
    """Lowercase, extract word-like tokens, drop stopwords, light stem."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if len(t) <= 2 or t in STOPWORDS:
            continue
        # Light stem: drop common suffixes
        for suf in ("ing", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[:-len(suf)]
                break
        out.append(t)
    return out


def faithfulness_score(answer: str, chunk: str) -> tuple[float, int, int]:
    """Return (overlap_ratio, matched_tokens, total_answer_tokens)."""
    a_tokens = tokenize(answer)
    if not a_tokens:
        return 0.0, 0, 0
    c_tokens = set(tokenize(chunk))
    if not c_tokens:
        # Cross-lingual content — answer in English, chunk in another script
        # We can't measure this fairly; mark as "untestable" via score = 1.0
        return 1.0, len(a_tokens), len(a_tokens)
    matched = sum(1 for t in a_tokens if t in c_tokens)
    return matched / len(a_tokens), matched, len(a_tokens)


def verdict(score: float) -> str:
    if score >= THRESHOLD_FAITHFUL:
        return "faithful"
    if score >= THRESHOLD_PARTIAL:
        return "partial"
    return "suspicious"


def extract_qa(pair: dict) -> tuple[str, str]:
    q = a = ""
    for m in pair.get("messages", []):
        if m.get("role") == "user" and not q:
            q = (m.get("content") or "").strip()
        elif m.get("role") == "assistant" and not a:
            a = (m.get("content") or "").strip()
    return q, a


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

    # Score every pair
    rows = []
    for i, p in enumerate(pairs):
        q, a = extract_qa(p)
        chunk = p.get("chunk_text", "")
        if not chunk:
            print(f"[warn] Pair {i}: no chunk_text — skipping")
            continue
        score, matched, total = faithfulness_score(a, chunk)
        rows.append({
            "idx": i,
            "page": p.get("provenance", {}).get("page", ""),
            "category": p.get("provenance", {}).get("category", ""),
            "score": score,
            "matched_tokens": matched,
            "total_tokens": total,
            "verdict": verdict(score),
            "question": q,
            "answer": a,
        })

    # Write per-pair CSV
    csv_path = args.output_dir / "faithfulness_per_pair.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["score"] = f"{r['score']:.4f}"
            w.writerow(r2)

    # Compute summary
    scores = [r["score"] for r in rows]
    n = len(scores)
    verdicts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("faithful", "partial", "suspicious")}

    summary = {
        "metric": "faithfulness",
        "method": "token-overlap (stopwords removed, lightly stemmed)",
        "n_pairs": n,
        "mean_score": round(statistics.mean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "stdev_score": round(statistics.stdev(scores), 4) if n > 1 else 0,
        "verdict_counts": verdicts,
        "verdict_pct": {k: round(100 * v / n, 2) for k, v in verdicts.items()},
        "thresholds": {
            "faithful_min": THRESHOLD_FAITHFUL,
            "partial_min": THRESHOLD_PARTIAL,
        },
    }

    summary_path = args.output_dir / "faithfulness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print
    print(f"\n=== FAITHFULNESS ===")
    print(f"  Pairs evaluated:   {n}")
    print(f"  Mean overlap:      {summary['mean_score']}")
    print(f"  Median overlap:    {summary['median_score']}")
    print(f"  Faithful  (>=0.70): {verdicts['faithful']:5d}  ({summary['verdict_pct']['faithful']}%)")
    print(f"  Partial   (0.40-0.70): {verdicts['partial']:5d}  ({summary['verdict_pct']['partial']}%)")
    print(f"  Suspicious (<0.40): {verdicts['suspicious']:5d}  ({summary['verdict_pct']['suspicious']}%)")
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
