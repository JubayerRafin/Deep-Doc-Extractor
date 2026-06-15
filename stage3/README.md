# Stage 3 — Dataset Augmentation

Multiplies the accepted Stage 2 Q&A dataset by generating paraphrased
question variations. The **original answer is preserved unchanged** on every
variation — only the question text is rewritten.

Implements FR016, FR018–FR021 from the FR specification.

## Modules

| File | Purpose |
|------|---------|
| `llm_rephraser.py`    | Sends each question to Ollama, parses JSON array of paraphrases |
| `augment_filter.py`   | Drops near-duplicates (sim > 0.97) and semantic drift (sim < 0.70) |
| `final_exporter.py`   | Writes `augmented.jsonl`, CSV, mentor_batch, stats |
| `stage3_pipeline.py`  | **Main orchestrator** |
| `test_stage3.py`      | Offline smoke test (no LLM required) |

## Installation

Stage 3 uses the same dependencies as Stage 2 — no new installs needed.
If running standalone:

```bash
pip install pyyaml requests sentence-transformers numpy
```

`sentence-transformers` is optional; the filter falls back to a token-overlap
heuristic if it's missing (you'll get a `[WARN]` in the log).

## Usage

### 1. Offline smoke test (recommended first)

```bash
cd stage3
python test_stage3.py
```

Runs 5 checks: imports, variation parser, filter logic, exporter schema,
pipeline dry-run. No Ollama required.

### 2. Small demo run (for the midterm presentation)

Process only the first 3 accepted pairs to verify end-to-end flow with the LLM:

```bash
python stage3_pipeline.py --limit 3
```

### 3. Dry run (skip LLM, write originals-only)

```bash
python stage3_pipeline.py --dry-run
```

Useful to see what the output directory will look like without waiting for Ollama.

### 4. Full run

```bash
python stage3_pipeline.py
```

Or from the repo root using the top-level pipeline:

```bash
python pipeline.py --stage 3 --config config.yaml
python pipeline.py --stage all --config config.yaml      # 1 + 2 + 3
```

## Inputs and outputs

**Input**: `output/stage2/qa_pairs.jsonl` (from Stage 2)

**Output** (in `output/stage3/`):

| File                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| `augmented.jsonl`     | Originals + accepted variations in Stage 2's JSONL schema |
| `augmented.csv`       | CSV mirror, one row per record                           |
| `mentor_batch.jsonl`  | Originals only — package for mentor GPU (FR021)          |
| `stats.json`          | Run statistics (counts, multiplier, failure modes)       |

## Output schema

Each line of `augmented.jsonl` preserves your Stage 2 schema and adds a
`provenance.stage3` block:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful HP printer technician."},
    {"role": "user",      "content": "<question — original OR rephrased>"},
    {"role": "assistant", "content": "<original answer, always unchanged>"}
  ],
  "provenance": {
    "source_file": "hp-e877-series-user-guide.md",
    "chunk": "Replace toner cartridge (chunk 0)",
    "category": "procedure",
    "page": 35,
    "stage3": {
      "status": "original",            // or "augmented"
      "augmented_from": null,          // hash of original Q for variations
      "variation_index": 0             // 0 = original; 1..N = variations
    }
  }
}
```

## Configuration

All settings in `config.yaml` under the `stage3:` key:

```yaml
stage3:
  output_dir: "output/stage3"
  llm:
    model: "qwen2.5:3b"           # smaller OK — task is simpler than Q&A gen
    temperature: 0.7              # higher than Stage 2 (encourages variety)
    max_tokens: 512
    timeout: 600
  augmentation:
    variations_per_qa: 3          # ask for 3, filter usually keeps ~2
  filter:
    similarity_max: 0.97          # drop near-duplicates
    similarity_min: 0.70          # drop semantic drift
    min_variation_length: 10
    embedding_model: "all-MiniLM-L6-v2"
  export:
    jsonl: true
    csv: true
    mentor_batch: true            # FR021 — mentor GPU handoff
```

## How it fits the midterm demo

The vertical-slice presentation plan shown in the proposal works like this:

1. Pick one page you've already run Stage 2 on (e.g. page 35 — toner cartridge).
2. Filter the Stage 2 JSONL down to just that page's pairs.
3. Run `python stage3_pipeline.py --limit 5` to get a fast augmented sample.
4. In the presentation, show a table like:

   | Stage | Question | Answer |
   |-------|----------|--------|
   | 2 (original) | "How do I replace the toner cartridge?" | ... |
   | 3 (variation 1) | "What is the procedure for replacing the toner?" | *same answer* |
   | 3 (variation 2) | "Steps to install a new toner cartridge?" | *same answer* |

This is a static walkthrough — no live LLM required during the presentation.

## Troubleshooting

**"Cannot connect to Ollama at localhost:11434"**
→ Run `ollama serve` in a separate terminal first.

**"model 'qwen2.5:3b' not found"**
→ `ollama pull qwen2.5:3b` (or change `stage3.llm.model` in config.yaml).

**"sentence-transformers not installed — using token-overlap heuristic"**
→ This is fine for a demo, but install `sentence-transformers` for
    production-grade filtering: `pip install sentence-transformers`.

**"Stage 2 output not found"**
→ Run Stage 2 first: `python pipeline.py --stage 2`.
