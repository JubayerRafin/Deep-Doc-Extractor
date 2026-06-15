# Deep Doc Extractor

**Automated conversion of unstructured HP technical manuals into LLM-ready training datasets and a fully offline RAG chatbot.**

Team DocuForge · Woosong University Capstone 2026 · Industry Partner: HP Printing Korea

---

## Overview

Technical knowledge in large organizations is often locked inside unstructured documents — printer service manuals, user guides — in formats that a Large Language Model cannot consume directly. **Deep Doc Extractor** is a four-stage, fully offline pipeline that converts these manuals (PDF) into structured, LLM-ready Q&A datasets and serves a grounded Retrieval-Augmented Generation (RAG) chatbot over them.

The pipeline was developed and validated on the 241-page HP E877 user guide. Every stage runs locally — no cloud APIs, no external network calls.

```
PDF manual
   │
   ▼
[Stage 1]  Extraction      →  structured Markdown + JSON tables + images
   │
   ▼
[Stage 2]  Q&A Generation  →  qa_pairs.jsonl   (local LLM + 4-gate quality filter)
   │
   ▼
[Stage 3]  Augmentation    →  augmented.jsonl  (paraphrased, deduplicated)
   │
   ▼
[Stage 4]  RAG Chatbot      →  grounded answers with citations (Streamlit, offline)
```

---

## Key Results

Validated on the HP E877 manual. Final production run: Stage 2 base = Mistral 24B, Stage 3 rephraser = Qwen3.6 27B.

| Stage | Output | NLI Faithfulness | Near-Duplicates |
|-------|--------|-----------------:|----------------:|
| Stage 1 — Extraction | Markdown + tables + images | Table F1 **0.94**, Image F1 **0.90** | — |
| Stage 2 — Q&A base | 698 pairs | **88.8%** | 0.0% |
| Stage 3 — Augmented | 1,739 pairs | **89.1%** | **0.58%** |

- **Five local LLMs benchmarked** at each stage; all cleared the ≥70% quality-filter target.
- Tightening the augmentation similarity filter (0.97 → 0.92) cut near-duplicates from **26% to 0.58%**.
- Render-and-mask image extraction (F1 0.90) outperforms gap-based detection (0.62) and DocLayout-YOLO (0.72) — without a GPU.

---

## Quick Start

### 1. Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) (local LLM runtime)
- ~16 GB RAM minimum; a GPU is strongly recommended for Stage 2–3 (CPU runs are slow)

### 2. Install

```bash
git clone https://github.com/JubayerRafin/Deep-Doc-Extractor.git
cd Deep-Doc-Extractor
pip install -r requirements.txt
```

### 3. Pull a model in Ollama

```bash
ollama pull qwen3.5:9b      # default; any Ollama model works
```

### 4. Add your source PDF

The HP manual is **not** included (copyright). Place your own PDF and point the config at it:

```yaml
# config.yaml
input:
  pdf_path: "data/source_pdfs/your-manual.pdf"
```

### 5. Run the pipeline

```bash
# Run Stages 1 + 2 + 3 end-to-end in one command:
python pipeline.py --stage all --config config.yaml

# Then build the RAG index and launch the chatbot:
python pipeline.py --stage 4 --config config.yaml
streamlit run stage4/streamlit_app.py
```

The chatbot opens at `http://localhost:8501`.

---

## Command Reference

| Command | Description |
|---------|-------------|
| `python pipeline.py --stage 1 --config config.yaml` | Extraction only (PDF → Markdown) |
| `python pipeline.py --stage 2 --config config.yaml` | Q&A generation only |
| `python pipeline.py --stage 3 --config config.yaml` | Augmentation only |
| `python pipeline.py --stage all --config config.yaml` | **Stages 1 + 2 + 3 (full generation pipeline)** |
| `python pipeline.py --stage both --config config.yaml` | Stages 1 + 2 |
| `python pipeline.py --stage 4 --config config.yaml` | Build the RAG vector index |
| `python pipeline.py --stage 4 --config config.yaml --query "..."` | One-shot CLI query |

Helper flags: `--test-pages N` (Stage 1, process a single page), `--limit N` (Stage 3, first N pairs), `--rebuild` (Stage 4, rebuild index).

---

## Changing the Model

All model selection is in `config.yaml`. To run a different LLM, pull it in Ollama and set it per stage:

```yaml
stage2:
  llm:
    model: "mistral-small3.2:24b"     # Q&A generation model
stage3:
  llm:
    model: "qwen3.6:27b"              # augmentation rephraser
    fallback_model: "qwen3.6:27b"
stage4:
  llm:
    model: "qwen3.5:9b"              # chat-serving model
```

Then re-run `python pipeline.py --stage all --config config.yaml`.

---

## Architecture

The pipeline uses a **file-based handoff** between stages: each stage writes its output to disk, and the next stage reads it. This means any stage can be re-run independently after a fix without re-executing the earlier stages.

```
Deep-Doc-Extractor/
├── pipeline.py              # orchestrator — runs any stage or all of them
├── config.yaml             # single control plane (models, thresholds, paths)
├── requirements.txt
│
├── stage1/                 # Extraction
│   ├── file_validator.py
│   ├── text_extractor.py
│   ├── image_extractor.py      # render-and-mask image extraction
│   ├── table_extractor.py      # Camelot + pdfplumber fallback
│   ├── extracted_element.py
│   └── markdown_compiler.py
│
├── stage2/                 # Q&A dataset construction
│   ├── markdown_parser.py
│   ├── chunker.py
│   ├── semantic_classifier.py
│   ├── qa_generator.py
│   ├── quality_filters.py      # 4-gate: length, weak-answer, hallucination, dedup
│   └── stage2_pipeline.py
│
├── stage3/                 # Augmentation
│   ├── llm_rephraser.py
│   ├── augment_filter.py       # diversity-band filter (0.70–0.92)
│   ├── final_exporter.py
│   └── stage3_pipeline.py
│
├── stage4/                 # RAG chat service
│   ├── embedding_engine.py
│   ├── chroma_indexer.py
│   ├── answer_generator.py
│   ├── offline_guard.py        # blocks all non-localhost calls
│   └── streamlit_app.py
│
├── evaluation/             # independent quality evaluators
│   ├── eval_1_faithfulness.py        # token-overlap
│   ├── eval_1b_faithfulness_nli.py   # NLI (DeBERTa) — gold standard
│   ├── eval_2_question_quality.py
│   ├── eval_3_diversity.py           # near-duplicate / relevance
│   └── evaluate_all.py
│
└── scripts/                # multi-model benchmarking helpers
```

---

## Evaluation

Each generated dataset is scored independently of the generation model:

```bash
python evaluation/evaluate_all.py \
  --pairs output/stage3/augmented.jsonl \
  --output-dir eval_results/
```

This produces faithfulness (NLI + token-overlap), question-quality, and diversity reports plus a paste-ready summary.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11 |
| PDF extraction | pdfplumber, Camelot-py |
| LLM runtime | Ollama (Qwen / Mistral / Gemma, local) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (embedded, local) |
| Chat UI | Streamlit |
| Evaluation | DeBERTa-v3-MNLI (NLI faithfulness) |

---

## Design Notes

- **Fully offline.** An `OfflineGuard` blocks any non-localhost network call at runtime; the entire pipeline runs without internet.
- **One model per run.** The pipeline uses the model set in `config.yaml`. The five-model comparison in the project report was a separate benchmarking experiment (see `scripts/`), not something the production pipeline does on every run.
- **Automated transformation with a quality checkpoint.** Stages 1→2→3 chain automatically. The four-gate automatic filter runs on every pair; a human review of a dataset sample is recommended before considering the dataset final.

---

## Out of Scope (Future Work)

- Model fine-tuning on the generated dataset
- OCR for scanned / image-only PDFs
- Multi-language manuals
- Resume-on-crash and watched-folder / scheduled triggering

---

## Team

**Team DocuForge** — Woosong University, Department of AI & Big Data

| Member | Role |
|--------|------|
| MD Jubayer Hossain | Team Lead · Pipeline Architect |
| Arlen Kasymbaev | Stage 1 · Extraction |
| Sangwook Yoo | Stage 2 · NLP & Chunking |
| Inyoung Park | Stage 2 · Dataset & QA |
| Sungho Choi | Stage 3 & 4 · LLM & RAG |
| Sherlet | Stage 3 & 4 · LLM & Demo |

Industry Partner: **HP Printing Korea** · Supervising Professor: **Kim Young-Il**

## License

MIT License — see [LICENSE](LICENSE).
