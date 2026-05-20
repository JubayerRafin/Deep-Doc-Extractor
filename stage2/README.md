# Stage 2: Dataset Generation

Converts Stage 1 Markdown → JSONL Q&A pairs for LLM fine-tuning.

## Modules

| File | Purpose |
|------|---------|
| `markdown_parser.py` | Parses `.md` → `ContentBlock` objects |
| `semantic_classifier.py` | Classifies: `procedure` / `spec` / `rule_error` / `figure` |
| `chunker.py` | Splits blocks into `Chunk` objects (size from `config.yaml`) |
| `qa_generator.py` | Sends chunks to Ollama, parses Q&A JSON response |
| `quality_filters.py` | Length → Weak → Hallucination → Dedup filter chain |
| `stage2_pipeline.py` | **Main orchestrator** |

## Config

All settings read from the unified `config.yaml` under `stage2:` key.
Stage 1 output paths are auto-detected from `stage1:` and `input:` keys.

## Usage

```bash
pip install -r requirements.txt
ollama serve                          # start Ollama
python stage2_pipeline.py --dry-run   # test parse/classify/chunk (no LLM)
python stage2_pipeline.py             # full run
```
