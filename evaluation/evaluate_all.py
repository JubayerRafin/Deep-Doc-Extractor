#!/usr/bin/env python3
"""
evaluate_all.py — Run all 3 evaluators and produce a single Stage 2 quality report.

Runs in sequence:
  1. eval_1_faithfulness.py   — token-overlap faithfulness
  2. eval_2_question_quality.py — generic-Q detection, length, type distribution
  3. eval_3_diversity.py       — Q-A relevance + near-duplicate detection

Output: eval_results/stage2_evaluation_report.md   (mentor-ready, paste-able)

USAGE
-----
    python evaluate_all.py --pairs output/stage2/qa_pairs.jsonl --output-dir eval_results/
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_script(script: Path, pairs: Path, out_dir: Path) -> bool:
    """Run a sub-script. Returns True on success."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {script.name}")
    print('='*70)
    result = subprocess.run(
        [sys.executable, str(script), "--pairs", str(pairs), "--output-dir", str(out_dir)],
        capture_output=False
    )
    return result.returncode == 0


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(faith, qq, div, source_file, total_raw=None, faith_nli=None):
    n = faith["n_pairs"] if faith else (qq["n_pairs"] if qq else 0)

    # Headline pass/fail
    faith_pct = faith["verdict_pct"]["faithful"] if faith else None
    nli_pct = faith_nli["verdict_pct"].get("faithful", 0) if faith_nli else None
    generic_pct = qq["generic"]["pct"] if qq else None
    near_dup_pct = div["question_diversity"]["near_duplicates_0.95_pct"] if div else None
    qa_mean = div["qa_relevance"]["mean_cosine"] if div else None

    def pf(passes):
        return "PASS" if passes else "FAIL" if passes is False else "-"

    lines = []
    lines.append(f"# Stage 2 Q&A Dataset — Evaluation Report\n")
    lines.append(f"**Source:** `{source_file}`")
    lines.append(f"**Pairs evaluated:** {n}")
    if total_raw:
        lines.append(f"**Raw pairs (pre-filter):** {total_raw}")
    lines.append(f"**Evaluation method:** independent of generation LLM (token-overlap + NLI + sentence-transformers)\n")
    lines.append("---\n")

    # Headline table
    lines.append("## Headline Numbers\n")
    lines.append("| Metric | Result | Target | Status |")
    lines.append("|---|---:|---:|:---:|")
    if faith_nli:
        lines.append(f"| **Faithful pairs (NLI, gold standard)** | **{faith_nli['verdict_pct'].get('faithful', 0)}%** | >= 80% | {pf(faith_nli['verdict_pct'].get('faithful', 0) >= 80)} |")
        lines.append(f"| Contradicted claims (NLI) | {round(100*faith_nli['claim_level']['contradiction_rate'], 2)}% | <= 1% | {pf(faith_nli['claim_level']['contradiction_rate'] <= 0.01)} |")
    if faith:
        lines.append(f"| Faithful pairs (token-overlap, fast check) | {faith['verdict_pct']['faithful']}% | >= 80% | {pf(faith['verdict_pct']['faithful'] >= 80)} |")
        lines.append(f"| Suspicious pairs (token-overlap) | {faith['verdict_pct']['suspicious']}% | <= 5% | {pf(faith['verdict_pct']['suspicious'] <= 5)} |")
    if qq:
        lines.append(f"| Generic questions | {qq['generic']['pct']}% | <= {qq['generic']['target_max_pct']}% | {pf(qq['generic']['passes'])} |")
        lines.append(f"| Short questions | {qq['length']['short_pct']}% | <= 10% | {pf(qq['length']['short_pct'] <= 10)} |")
    if div:
        lines.append(f"| Q-A mean cosine | {div['qa_relevance']['mean_cosine']} | >= {div['thresholds']['qa_cosine_min_mean']} | {pf(div['qa_relevance']['passes_mean_target'])} |")
        lines.append(f"| Near-duplicate Qs | {div['question_diversity']['near_duplicates_0.95_pct']}% | <= {div['thresholds']['near_duplicate_max_pct']}% | {pf(div['question_diversity']['passes_near_dup_target'])} |")
    lines.append("")

    # NLI gold-standard detail (the primary faithfulness number)
    if faith_nli:
        cl = faith_nli["claim_level"]
        pl = faith_nli["pair_level"]
        lines.append("## 1a. Faithfulness — NLI Gold Standard\n")
        lines.append(f"Method: {faith_nli['method']}")
        lines.append(f"Model: `{faith_nli['model']}` (independent of generation LLM, deterministic)\n")
        lines.append(f"**Claim level** ({cl['total_claims']} sentences across all answers):")
        lines.append(f"- Entailed by chunk: **{cl['entailed']}** ({round(100*cl['entailment_rate'], 2)}%)")
        lines.append(f"- Contradicted: {cl['contradicted']} ({round(100*cl['contradiction_rate'], 2)}%)")
        lines.append(f"- Neutral (chunk doesn't say): {cl['total_claims'] - cl['entailed'] - cl['contradicted']}\n")
        lines.append(f"**Pair level**:")
        lines.append(f"- Mean per-pair score: **{pl['mean_score']}**")
        lines.append(f"- Median per-pair score: {pl['median_score']}")
        for k in ("faithful", "partial", "unfaithful", "no_claims"):
            v = pl["verdict_counts"].get(k, 0)
            if v > 0:
                lines.append(f"- {k.capitalize()}: {v} ({pl['verdict_pct'].get(k, 0)}%)")
        lines.append("")

    # Token-overlap detail (the fast cross-check)
    if faith:
        lines.append("## 1b. Faithfulness — Token-Overlap (Fast Cross-Check)\n")
        lines.append(f"Method: {faith['method']}")
        lines.append(f"- Mean overlap score: **{faith['mean_score']}**")
        lines.append(f"- Median overlap score: {faith['median_score']}")
        lines.append(f"- Faithful (>= 0.70 overlap): **{faith['verdict_counts']['faithful']}** ({faith['verdict_pct']['faithful']}%)")
        lines.append(f"- Partial (0.40-0.70 overlap): {faith['verdict_counts']['partial']} ({faith['verdict_pct']['partial']}%)")
        lines.append(f"- Suspicious (< 0.40 overlap): {faith['verdict_counts']['suspicious']} ({faith['verdict_pct']['suspicious']}%)")
        lines.append("")
        if faith_nli:
            # Comment on agreement between the two methods
            lines.append(f"_Both faithfulness methods are reported. NLI is the primary metric. "
                         f"Token-overlap is a fast sanity check — methods should broadly agree._\n")

    # Question quality detail
    if qq:
        lines.append("## 2. Question Quality\n")
        lines.append(f"- Average question length: **{qq['length']['mean_words']}** words "
                     f"(median {qq['length']['median_words']}, range {qq['length']['min_words']}-{qq['length']['max_words']})")
        lines.append(f"- Generic / meta-textual questions: **{qq['generic']['count']}** ({qq['generic']['pct']}%)")
        lines.append(f"- Type distribution:")
        for t, info in qq["type_distribution"].items():
            lines.append(f"   - {t}: {info['count']} ({info['pct']}%)")
        lines.append("")

    # Diversity & relevance detail
    if div:
        lines.append("## 3. Diversity and Q-A Relevance\n")
        lines.append(f"Embedding model: `{div['embedding_model']}`")
        lines.append(f"\n**Q-A relevance** (does the answer address the question?):")
        lines.append(f"- Mean cosine: **{div['qa_relevance']['mean_cosine']}**")
        lines.append(f"- % above 0.70 (highly relevant): {div['qa_relevance']['pct_above_0.7']}%")
        lines.append(f"- % below 0.30 (off-topic): {div['qa_relevance']['pct_below_0.3']}%")
        lines.append(f"\n**Question diversity** (are questions different from each other?):")
        lines.append(f"- Mean nearest-neighbor cosine: {div['question_diversity']['mean_nn_cosine']}")
        lines.append(f"- Near-duplicates (> 0.95): **{div['question_diversity']['near_duplicates_0.95']}** ({div['question_diversity']['near_duplicates_0.95_pct']}%)")
        lines.append(f"- Very-close pairs (> 0.90): {div['question_diversity']['very_close_0.90']}")
        lines.append("")

    # Methodology block
    lines.append("---\n")
    lines.append("## Methodology Notes\n")
    lines.append("- **Faithfulness (NLI)** uses DeBERTa-v3-MNLI to check whether each answer sentence is entailed by the source chunk. "
                 "This is the gold-standard signal — catches numerical contradictions and subtle hallucinations that token-overlap misses.")
    lines.append("- **Faithfulness (token-overlap)** is a fast sanity check using set intersection of stop-word-removed tokens. "
                 "Used as a cross-check against the NLI result.")
    lines.append("- **Question quality** uses regex patterns and length statistics. Deterministic.")
    lines.append("- **Diversity** uses sentence-transformer embeddings (`all-MiniLM-L6-v2`).")
    lines.append("")
    lines.append("**Why no LLM-as-judge?** Using a generative LLM to score another LLM's output introduces evaluator bias, "
                 "requires API calls (breaks the offline requirement), and produces non-deterministic scores. "
                 "All metrics in this report are deterministic and run fully locally.")
    lines.append("")
    lines.append("**Comparing across generation LLMs:** Run this same pipeline on each LLM's output. Differences in scores "
                 "reflect differences in the generators, not differences in the evaluator. The evaluator is the constant.")
    lines.append("")
    lines.append("## Per-Pair Detail\n")
    lines.append("Individual scores for every pair are in:")
    lines.append("- `faithfulness_nli_per_pair.csv` (gold standard)")
    lines.append("- `faithfulness_per_pair.csv` (token-overlap)")
    lines.append("- `question_quality_per_pair.csv`")
    lines.append("- `diversity_per_pair.csv`")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    scripts = [
        here / "eval_1_faithfulness.py",
        here / "eval_1b_faithfulness_nli.py",   # gold-standard NLI faithfulness
        here / "eval_2_question_quality.py",
        here / "eval_3_diversity.py",
    ]

    # Run each evaluator
    for s in scripts:
        if not s.exists():
            print(f"[ERROR] Missing script: {s}")
            sys.exit(1)
        ok = run_script(s, args.pairs, args.output_dir)
        if not ok:
            print(f"[ERROR] Script failed: {s.name}. Stopping.")
            sys.exit(1)

    # Load summaries
    faith = load_summary(args.output_dir / "faithfulness_summary.json")
    faith_nli = load_summary(args.output_dir / "faithfulness_nli_summary.json")
    qq = load_summary(args.output_dir / "question_quality_summary.json")
    div = load_summary(args.output_dir / "diversity_summary.json")

    # Render combined report
    report = render_report(faith, qq, div, source_file=args.pairs.name, faith_nli=faith_nli)
    report_path = args.output_dir / "stage2_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print('='*70)
    print(f"\n[saved] {report_path}")
    print(f"\nOpen the report:")
    print(f"   {report_path.resolve()}")


if __name__ == "__main__":
    main()
