# Deep Doc Extractor

> **Automated PDF-to-Dataset pipeline for Local LLM fine-tuning.**
> Industry Partner: **HP Printing Korea** — Data Science Team.
> Woosong University · Capstone Design 2026.

---

## What It Does

HP's on-premise chatbot is limited by training data. Critical knowledge is locked inside hundreds of technical manuals — PDFs that no fine-tuning pipeline can read directly. Manual dataset creation takes **weeks of human labour** per document.

Deep Doc Extractor solves this with a **four-stage automated pipeline** that converts unstructured HP manuals into Local LLM-ready Q&A training datasets. The entire system runs **100% locally** with zero cloud dependencies.

---

## Key Results

| Metric | Value |
|---|---:|
| NLI Faithfulness (answers grounded in source) | **95%** |
| Q&A pairs generated (from one 241-page manual) | **2,100+** |
| Speedup (CPU baseline → GPU) | **28×** |
| Models compared (Qwen / Mistral / Gemma) | **4** |
| RAG chatbot response time | **~1 s** |

---

## Architecture

Four sequential stages:

| Stage | Input | Output | Tools |
|---|---|---|---|
| **1. Document Extraction** | PDF / DOCX / XLSX | Structured Markdown + JSON tables | pdfplumber, Camelot, render-and-mask |
| **2. Q&A Generation** | Markdown | JSONL training pairs with provenance | Local LLM via Ollama |
| **3. Dataset Augmentation** | JSONL | Augmented JSONL (3× variety) | LLM paraphrasing + similarity filter |
| **4. RAG Validation** | JSONL | Interactive chatbot at localhost:8501 | ChromaDB + Streamlit + Ollama |

See `docs/HLD_v1_6.docx` for the full design document.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/Deep-Doc-Extractor.git
cd Deep-Doc-Extractor
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

```bash
# Install Ollama (no sudo required)
curl -fsSL https://ollama.com/install.sh | sh

# Pull your model of choice
ollama pull qwen3.5:9b
```

### 3. Place your source PDF

```bash
mkdir -p input
cp /path/to/your-manual.pdf input/
```

### 4. Edit `config.yaml`

Set the input PDF path and the model you pulled.

### 5. Run end-to-end

```bash
# Each stage is independent — you can run them separately:
python3 stage1/stage1_pipeline.py    # PDF → Markdown
python3 stage2/stage2_pipeline.py    # Markdown → Q&A pairs
python3 stage3/stage3_pipeline.py    # Augment dataset
streamlit run stage4/streamlit_app.py  # Launch RAG chatbot
```

---

## Configuration

All pipeline behavior is controlled through `config.yaml`. Key fields:

```yaml
input:
  pdf_path: "input/your-manual.pdf"

stage2:
  llm:
    model: "qwen3.5:9b"

stage3:
  augmentation:
    variations_per_qa: 3
```

To switch experiments quickly (different models, same code):

```bash
python3 scripts/switch_experiment.py --tag qwen35_9b --model qwen3.5:9b
```

---

## For Team Members

- **Running on the GPU server**: see `docs/runbooks/stage2_gpu_runbook.md`
- **Multi-model comparison**: each member runs one model with the same code — see runbook
- **Repo workflow**: see `docs/runbooks/github_setup.md`

---

## Project Structure

```
.
├── stage1/          PDF → Markdown extraction
├── stage2/          Markdown → Q&A pair generation
├── stage3/          Question paraphrasing & augmentation
├── stage4/          RAG chatbot (Streamlit)
├── evaluation/      Three-layer evaluator (NLI faithfulness + diversity)
├── scripts/         Helper scripts (experiment switcher, backfill, etc.)
├── docs/            HLD, LLD, FR specifications, runbooks
├── examples/        Sample inputs and outputs (for judges)
├── results/         Per-model evaluation results
└── config.yaml      Single source of truth
```

---

## Team

**Team DocuForge** — Department of AI & Big Data

| Role | Member |
|---|---|
| Team Leader | MD Jubayer Hossain |
| Members | Arlen Kasymbaev, Sangwook Yoo, Inyoung Park, Sungho Choi |
| Professor | Kim Young-Il (김영일) |
| Industry Mentor | Mintae Kim — HP Printing Korea |

---

## Acknowledgements

This project was developed in collaboration with **HP Printing Korea — Data Science Team** as part of the Woosong University Capstone Design 2026 program.

## License

MIT — see `LICENSE`.
