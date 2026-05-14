#!/usr/bin/env python3
"""
eval_1b_faithfulness_nli.py — Gold-standard faithfulness via Natural Language Inference.

METHOD
------
For each Q&A pair:
  1. Split the answer into sentence-level claims.
  2. For each claim, run NLI: (chunk, claim) -> entailment | neutral | contradiction
  3. Aggregate per-pair: score = entailed / total, verdict = unfaithful if any contradiction.

This is the STRONGEST faithfulness signal available without using a generative LLM:
  - Catches "chunk says 16 GB, answer says 8 GB" — contradiction
  - Catches "chunk doesn't mention X, answer asserts X" — neutral (not entailed)
  - Catches subtle hallucinations that token-overlap misses

NLI model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  - ~1.4 GB download on first run (cached locally after that)
  - Different model family from any generation LLM (Qwen, Llama, GPT-*)
  - Deterministic given the same input

PREREQUISITES
-------------
    pip install transformers torch sentence-transformers

PERFORMANCE
-----------
  ~500 pairs, avg 3 claims/answer = ~1500 NLI calls
  CPU (i5-13420H):  ~25 minutes
  GPU (any modern): ~2 minutes

USAGE
-----
    python eval_1b_faithfulness_nli.py --pairs qa_pairs.jsonl --output-dir eval_results/

    # Smoke test on first 20 pairs:
    python eval_1b_faithfulness_nli.py --pairs qa_pairs.jsonl --limit 20
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

# Model choice. This specific checkpoint is trained on MNLI + FEVER + ANLI + Wanli,
# giving it strong performance on technical / factual text.
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Verdict thresholds
THRESHOLD_FAITHFUL = 0.80   # >= 80% of claims entailed AND no contradictions
THRESHOLD_PARTIAL  = 0.50   # 50-80% entailed, no contradictions

# Practical limits
MIN_CLAIM_CHARS = 6         # claims shorter than this are noise
MAX_PREMISE_CHARS = 4000    # NLI models have ~512 token limit; truncate long chunks


# ---------- Helpers ----------
def extract_qa(pair: dict) -> tuple[str, str]:
    q = a = ""
    for m in pair.get("messages", []):
        if m.get("role") == "user" and not q:
            q = (m.get("content") or "").strip()
        elif m.get("role") == "assistant" and not a:
            a = (m.get("content") or "").strip()
    return q, a


def split_claims(text: str) -> list[str]:
    """Sentence-level claim splitter. Each sentence = one claim to verify."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    claims = [p.strip() for p in parts if len(p.strip()) >= MIN_CLAIM_CHARS]
    return claims if claims else ([text] if len(text) >= MIN_CLAIM_CHARS else [])


def truncate(text: str, max_chars: int = MAX_PREMISE_CHARS) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def verdict(score: float, n_contra: int) -> str:
    if n_contra > 0:
        return "unfaithful"
    if score >= THRESHOLD_FAITHFUL:
        return "faithful"
    if score >= THRESHOLD_PARTIAL:
        return "partial"
    return "unfaithful"


# ---------- NLI inference ----------
def run_nli_batch(pairs_list, model_name: str, batch_size: int = 8):
    """
    Run NLI on a list of (premise, hypothesis) tuples.
    Returns list of dicts: [{label, p_entail, p_neutral, p_contradict}, ...]
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"[info] Loading NLI model: {model_name}")
    print("       (first run downloads ~1.4 GB; subsequent runs use local cache)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[info] Device: {device}")

    # Map model's label IDs to canonical names
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    # Find which ID corresponds to each canonical label
    def find_idx(keyword):
        for k, v in id2label.items():
            if keyword in v:
                return k
        return None

    idx_entail = find_idx("entail")
    idx_neutral = find_idx("neutral")
    idx_contra = find_idx("contradict")

    results = []
    total = len(pairs_list)
    for i in range(0, total, batch_size):
        batch = pairs_list[i:i + batch_size]
        premises = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]

        enc = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            top_idx = int(row.argmax())
            label_raw = id2label[top_idx]
            label = (
                "entailment" if "entail" in label_raw else
                "contradiction" if "contradict" in label_raw else
                "neutral"
            )
            results.append({
                "label": label,
                "p_entail": float(row[idx_entail]) if idx_entail is not None else 0.0,
                "p_neutral": float(row[idx_neutral]) if idx_neutral is not None else 0.0,
                "p_contradict": float(row[idx_contra]) if idx_contra is not None else 0.0,
            })

        # Progress
        done = min(i + batch_size, total)
        if (i // batch_size) % 10 == 0 or done == total:
            print(f"  [progress] {done}/{total} claims scored")

    return results


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only first N pairs (for smoke test)")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: require transformers
    try:
        import transformers, torch  # noqa: F401
    except ImportError:
        print("\n[ERROR] Required packages missing. Install with:")
        print("    pip install transformers torch\n")
        raise

    # Load pairs
    pairs = []
    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[info] Loaded {len(pairs)} pairs from {args.pairs}")

    # Build flat (premise, claim) list, remember which pair each came from
    flat = []
    pair_claim_idxs = []  # for each pair, list of indices into `flat`
    for p in pairs:
        _, answer = extract_qa(p)
        chunk = truncate(p.get("chunk_text", ""))
        claims = split_claims(answer)
        idxs = []
        for claim in claims:
            idxs.append(len(flat))
            flat.append((chunk, claim))
        pair_claim_idxs.append(idxs)

    print(f"[info] Total claims to score: {len(flat)}")
    if not flat:
        print("[error] No claims to score. Check that pairs have chunk_text and non-empty answers.")
        return

    # Run NLI
    nli_results = run_nli_batch(flat, args.model, batch_size=args.batch_size)

    # Aggregate per pair
    rows = []
    for pair_i, p in enumerate(pairs):
        q, a = extract_qa(p)
        prov = p.get("provenance", {})
        idxs = pair_claim_idxs[pair_i]

        if not idxs:
            rows.append({
                "idx": pair_i,
                "page": prov.get("page", ""),
                "category": prov.get("category", ""),
                "n_claims": 0, "n_entailed": 0, "n_neutral": 0, "n_contradicted": 0,
                "score": 0.0, "verdict": "no_claims",
                "question": q, "answer": a,
            })
            continue

        labels = [nli_results[i]["label"] for i in idxs]
        n_entail = labels.count("entailment")
        n_neutral = labels.count("neutral")
        n_contra = labels.count("contradiction")
        n_total = len(labels)
        score = n_entail / n_total

        rows.append({
            "idx": pair_i,
            "page": prov.get("page", ""),
            "category": prov.get("category", ""),
            "n_claims": n_total,
            "n_entailed": n_entail,
            "n_neutral": n_neutral,
            "n_contradicted": n_contra,
            "score": round(score, 4),
            "verdict": verdict(score, n_contra),
            "question": q,
            "answer": a,
        })

    # Write per-pair CSV
    csv_path = args.output_dir / "faithfulness_nli_per_pair.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Compute summary
    n = len(rows)
    scores = [r["score"] for r in rows if r["n_claims"] > 0]
    verdicts = Counter(r["verdict"] for r in rows)
    total_claims = sum(r["n_claims"] for r in rows)
    total_entail = sum(r["n_entailed"] for r in rows)
    total_contra = sum(r["n_contradicted"] for r in rows)

    summary = {
        "metric": "faithfulness_nli",
        "method": "NLI (DeBERTa-v3-MNLI) on sentence-level claims",
        "model": args.model,
        "n_pairs": n,
        "claim_level": {
            "total_claims": total_claims,
            "entailed": total_entail,
            "contradicted": total_contra,
            "entailment_rate": round(total_entail / total_claims, 4) if total_claims else 0.0,
            "contradiction_rate": round(total_contra / total_claims, 4) if total_claims else 0.0,
        },
        "pair_level": {
            "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "median_score": round(statistics.median(scores), 4) if scores else 0.0,
            "verdict_counts": dict(verdicts),
            "verdict_pct": {k: round(100 * v / n, 2) for k, v in verdicts.items()},
        },
        "thresholds": {
            "faithful_min_score": THRESHOLD_FAITHFUL,
            "partial_min_score": THRESHOLD_PARTIAL,
            "any_contradiction_fails": True,
        },
    }

    # Compatibility shim: write fields evaluate_all.py expects to read
    # so the orchestrator can include this metric in the combined report.
    summary["verdict_counts"] = summary["pair_level"]["verdict_counts"]
    summary["verdict_pct"] = summary["pair_level"]["verdict_pct"]
    summary["mean_score"] = summary["pair_level"]["mean_score"]
    summary["median_score"] = summary["pair_level"]["median_score"]

    summary_path = args.output_dir / "faithfulness_nli_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print
    pl = summary["pair_level"]
    cl = summary["claim_level"]
    print(f"\n=== FAITHFULNESS (NLI gold standard) ===")
    print(f"  Pairs evaluated:    {n}")
    print(f"  Total claims:       {cl['total_claims']}")
    print(f"  Entailed claims:    {cl['entailed']}  ({100*cl['entailment_rate']:.1f}%)")
    print(f"  Contradicted claims: {cl['contradicted']}  ({100*cl['contradiction_rate']:.2f}%)")
    print(f"  Mean per-pair score: {pl['mean_score']}")
    print(f"  Median per-pair score: {pl['median_score']}")
    print(f"\n  Verdicts:")
    for k in ("faithful", "partial", "unfaithful", "no_claims"):
        v = pl["verdict_counts"].get(k, 0)
        if v == 0:
            continue
        pct = pl["verdict_pct"].get(k, 0.0)
        print(f"     {k:11s} {v:5d}  ({pct}%)")
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
