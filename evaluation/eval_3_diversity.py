#!/usr/bin/env python3
"""
eval_3_diversity.py — Measure dataset diversity (near-duplicate detection).

METHOD
------
Embedding-based nearest-neighbor analysis:
  1. Embed every question with sentence-transformers (all-MiniLM-L6-v2).
  2. For each question, find its nearest neighbor (excluding self).
  3. Count near-duplicates (cosine > 0.95) and very-close pairs (cosine > 0.90).

ALSO MEASURES
-------------
Q-A relevance: cosine similarity between each question and its own answer.
Low relevance = the LLM produced an answer that doesn't address the question.

WHY EMBEDDINGS, NOT LLM JUDGE
-----------------------------
- The sentence-transformer model is fixed (all-MiniLM-L6-v2, ~90 MB).
- Different model family than any generation LLM.
- Deterministic and reproducible.

PREREQUISITES
-------------
    pip install sentence-transformers numpy

USAGE
-----
    python eval_3_diversity.py --pairs qa_pairs.jsonl --output-dir eval_results/
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Thresholds
TARGET_MAX_NEAR_DUP_PCT = 5.0   # % of questions whose nearest neighbor cosine > 0.95
TARGET_MEAN_QA_COSINE = 0.50    # Q-A cosine should be above this on average


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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — only needed for this script
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n[ERROR] Required packages missing. Install with:")
        print("    pip install sentence-transformers numpy\n")
        raise

    # Load pairs
    pairs = []
    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[info] Loaded {len(pairs)} pairs from {args.pairs}")

    questions = [extract_qa(p)[0] for p in pairs]
    answers = [extract_qa(p)[1] for p in pairs]

    # Embed
    print(f"[info] Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"[info] Embedding questions...")
    q_emb = model.encode(questions, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True)
    print(f"[info] Embedding answers...")
    a_emb = model.encode(answers, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True)

    # Q-A cosine (element-wise dot product of normalized vectors)
    qa_cos = (q_emb * a_emb).sum(axis=1)

    # Question-to-question: full cosine matrix, then nearest neighbor (excluding self)
    print(f"[info] Computing question diversity...")
    sim_qq = q_emb @ q_emb.T
    np.fill_diagonal(sim_qq, -np.inf)
    nn_cos = sim_qq.max(axis=1)
    nn_idx = sim_qq.argmax(axis=1)

    # Write per-pair CSV
    rows = []
    for i, p in enumerate(pairs):
        rows.append({
            "idx": i,
            "page": p.get("provenance", {}).get("page", ""),
            "category": p.get("provenance", {}).get("category", ""),
            "qa_cosine": round(float(qa_cos[i]), 4),
            "nn_cosine": round(float(nn_cos[i]), 4),
            "nearest_neighbor_idx": int(nn_idx[i]),
            "is_near_duplicate": int(nn_cos[i] > 0.95),
            "question": questions[i],
        })

    csv_path = args.output_dir / "diversity_per_pair.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary
    n = len(pairs)
    qa_scores = qa_cos.tolist()
    nn_scores = nn_cos.tolist()
    near_dup = sum(1 for s in nn_scores if s > 0.95)
    very_close = sum(1 for s in nn_scores if s > 0.90)
    low_qa = sum(1 for s in qa_scores if s < 0.30)

    summary = {
        "metric": "diversity_and_relevance",
        "embedding_model": args.model,
        "n_pairs": n,
        "qa_relevance": {
            "mean_cosine": round(statistics.mean(qa_scores), 4),
            "median_cosine": round(statistics.median(qa_scores), 4),
            "pct_above_0.7": round(100 * sum(1 for s in qa_scores if s > 0.7) / n, 2),
            "pct_above_0.5": round(100 * sum(1 for s in qa_scores if s > 0.5) / n, 2),
            "pct_below_0.3": round(100 * low_qa / n, 2),
            "passes_mean_target": statistics.mean(qa_scores) >= TARGET_MEAN_QA_COSINE,
        },
        "question_diversity": {
            "mean_nn_cosine": round(statistics.mean(nn_scores), 4),
            "near_duplicates_0.95": near_dup,
            "near_duplicates_0.95_pct": round(100 * near_dup / n, 2),
            "very_close_0.90": very_close,
            "passes_near_dup_target": (100 * near_dup / n) <= TARGET_MAX_NEAR_DUP_PCT,
        },
        "thresholds": {
            "near_duplicate_max_pct": TARGET_MAX_NEAR_DUP_PCT,
            "qa_cosine_min_mean": TARGET_MEAN_QA_COSINE,
        },
    }

    summary_path = args.output_dir / "diversity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    qr = summary["qa_relevance"]
    qd = summary["question_diversity"]
    print(f"\n=== DIVERSITY & RELEVANCE ===")
    print(f"  Pairs evaluated:       {n}")
    print(f"\n  Q-A RELEVANCE:")
    print(f"    Mean cosine:         {qr['mean_cosine']}      target >= {TARGET_MEAN_QA_COSINE}   "
          f"{'PASS' if qr['passes_mean_target'] else 'FAIL'}")
    print(f"    Median cosine:       {qr['median_cosine']}")
    print(f"    % above 0.70:        {qr['pct_above_0.7']}%")
    print(f"    % below 0.30:        {qr['pct_below_0.3']}%  (potentially off-topic)")
    print(f"\n  QUESTION DIVERSITY:")
    print(f"    Mean nearest-neighbor cosine: {qd['mean_nn_cosine']}")
    print(f"    Near-duplicates (>0.95): {near_dup}  ({qd['near_duplicates_0.95_pct']}%)   "
          f"target <= {TARGET_MAX_NEAR_DUP_PCT}%   "
          f"{'PASS' if qd['passes_near_dup_target'] else 'FAIL'}")
    print(f"    Very close (>0.90):      {very_close}")
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
