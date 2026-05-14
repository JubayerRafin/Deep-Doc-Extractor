# Stage 2 Evaluation Pipeline

Four scripts that measure Q&A dataset quality. All scoring is independent of
the generation LLM — so scores stay comparable when you swap Qwen 3.5:9B for
any larger model.

## Files

| File | Measures | Needs |
|---|---|---|
| `eval_1_faithfulness.py` | Hallucinations — token-overlap (fast cross-check) | Python stdlib only |
| `eval_1b_faithfulness_nli.py` | **Hallucinations — NLI (gold standard)** | `transformers`, `torch` |
| `eval_2_question_quality.py` | Generic questions, length, type mix | Python stdlib only |
| `eval_3_diversity.py` | Q–A relevance + near-duplicate questions | `sentence-transformers`, `numpy` |
| `evaluate_all.py` | Runs all four, writes one report | Same as above |

## Install

```bash
pip install transformers torch sentence-transformers numpy
```

First run downloads the NLI model (~1.4 GB, cached locally after that).
Stdlib packages (`argparse`, `csv`, `json`, `re`, `statistics`) are already in Python.

## Run

```bash
# One command does everything (NLI run takes ~25 min on CPU):
python evaluate_all.py \
    --pairs output/stage2/qa_pairs.jsonl \
    --output-dir eval_results/
```

Or run any single evaluator on its own:

```bash
# Quick metrics (seconds):
python eval_1_faithfulness.py    --pairs output/stage2/qa_pairs.jsonl --output-dir eval_results/
python eval_2_question_quality.py --pairs output/stage2/qa_pairs.jsonl --output-dir eval_results/
python eval_3_diversity.py        --pairs output/stage2/qa_pairs.jsonl --output-dir eval_results/

# NLI gold standard (slow on CPU):
python eval_1b_faithfulness_nli.py --pairs output/stage2/qa_pairs.jsonl --output-dir eval_results/

# Smoke test the NLI script on first 20 pairs (1 min):
python eval_1b_faithfulness_nli.py --pairs output/stage2/qa_pairs.jsonl --output-dir eval_results/ --limit 20
```

## Output

In `eval_results/`:

```
stage2_evaluation_report.md       <-- mentor-ready, paste-able
faithfulness_per_pair.csv         <-- every pair scored
faithfulness_summary.json
question_quality_per_pair.csv
question_quality_summary.json
diversity_per_pair.csv
diversity_summary.json
```

The `.md` report is the file to share. The CSVs are for drilling into specific
pairs if a metric flags something.

## What Each Metric Catches

### Faithfulness (eval_1)
Hallucinations. Token-overlap between answer and source chunk:
- ≥ 0.70 → faithful (answer is essentially paraphrased from chunk)
- 0.40–0.70 → partial (some shared content, worth flagging)
- < 0.40 → suspicious (likely hallucination)

**Known limitation:** can't see across languages. If the chunk is Spanish and
the answer is a correct English translation, this score will be low even
though the answer is faithful. Flag those manually.

### Question Quality (eval_2)
Bad questions ruin fine-tuning data even when answers are perfect. Catches:
- Generic phrasing ("according to the text", "what does this section say")
- Too-short questions (< 8 words)
- Skewed type distribution (all-what, no how/why/where)

### Diversity (eval_3)
Two questions in one script:
1. **Q–A relevance**: cosine similarity between each question and its answer.
   Low score = the answer doesn't address the question.
2. **Question diversity**: nearest-neighbor cosine among all questions.
   High = your dataset has many near-duplicate questions, low training value.

## Methodology to Defend to Your Mentor

### Why token-overlap, not LLM-as-judge

Using an LLM to evaluate another LLM's output introduces three problems:

1. **Evaluator bias**: A GPT-4 judge tends to prefer GPT-4-style outputs.
   When you compare two LLMs, the judge's bias contaminates the comparison.
2. **API dependency**: Costs money, requires internet, breaks the offline
   execution requirement (FR028).
3. **Reproducibility**: LLM judges produce slightly different scores on
   re-runs. Deterministic methods don't.

Token-overlap catches the most common failure mode (vocabulary made up out
of thin air) without any of those problems.

### Why these thresholds

| Threshold | Value | Source |
|---|---:|---|
| Faithful (overlap) | ≥ 0.70 | Manual review of v2/v3 borderline pairs showed 0.70 separates clear paraphrases from drift |
| Near-duplicate question | cosine > 0.95 | Standard threshold in dedup literature |
| Q-A relevance min mean | 0.50 | Empirical floor for well-formed Q-A pairs |
| Generic question max pct | 5% | Industry standard for instruction-tuning datasets |

### How comparison across LLMs works

When the mentor runs the same pipeline on a bigger LLM's output:

- **Higher faithfulness score** → new LLM hallucinates less
- **Lower generic % ** → new LLM asks better questions
- **Higher Q-A cosine** → new LLM answers more on-topic
- **Lower near-duplicate %** → new LLM produces more diverse questions

The evaluator is the constant. The LLM is the variable. That's how
the comparison stays clean.

## Re-Using This Pipeline

Drop-in compatible with any JSONL where each line has:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "the question"},
    {"role": "assistant", "content": "the answer"}
  ],
  "provenance": {
    "source_file": "...",
    "category": "...",
    "page": 0
  },
  "chunk_text": "the source chunk text (required for faithfulness)"
}
```

If your future pipeline uses a different format, only `extract_qa()` in each
script needs to change.
