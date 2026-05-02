# Deep Doc Extractor

**Automated document processing pipeline for Local LLM training dataset generation.**

Capstone 2026 · Team DocuForge · Woosong University
Industry partner: **HP Printing Korea — Data Science Team**

Converts unstructured HP technical manuals (PDF) into structured, fine-tuning-ready datasets for local LLM training, with a Retrieval-Augmented Generation chatbot for end-to-end validation.

---

## Architecture

A four-layer pipeline, fully local, zero cloud dependencies.

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Layer 1        │   │   Layer 2        │   │   Layer 3        │   │   Layer 4        │
│   Extract        │ → │   Build Q&A      │ → │   Augment        │ → │   RAG Chat       │
│                  │   │                  │   │                  │   │                  │
│   PDF → MD       │   │   MD → JSONL     │   │   3× variations  │   │   ChromaDB +     │
│   + images +     │   │   (Qwen3.5:9B    │   │   per question   │   │   Streamlit UI   │
│   JSON tables    │   │    via Ollama)   │   │                  │   │                  │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
```

| Layer | Purpose | FRs |
|---|---|---|
| 1 — Ingestion & Extraction | PDF/DOCX/XLSX → structured Markdown with linked images and JSON tables | FR001–FR008 |
| 2 — Semantic Structuring | Markdown → categorised, deduplicated Q&A JSONL with provenance | FR009–FR017 |
| 3 — Augmentation | Per-question paraphrasing (3× multiplier) + similarity-bounded filter | FR018–FR021 |
| 4 — RAG Chat Service | Embed → ChromaDB → retrieval → grounded local-LLM answer with citations | FR022–FR029 |

See `docs/HLD.docx` and `docs/LLD.docx` for the full design.

---

## Tech stack

- **Python** 3.11
- **Extraction:** pdfplumber, Camelot-py, Pillow
- **LLM runtime:** Ollama (Qwen3.5:9B local model)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB (embedded, local)
- **RAG orchestration:** LangChain
- **UI:** Streamlit (localhost:8501)

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-org>/deep-doc-extractor.git
cd deep-doc-extractor

# 2. Install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt

# 3. Install Ollama and pull the model (one-time)
#    https://ollama.com/download
ollama pull qwen3.5:9b

# 4. Place your input PDF in inputs/  (or edit config.yaml)
#    Primary test document: HP E877 Series User Guide

# 5. Run the pipeline (layers can be run individually)
python pipeline.py --config config.yaml --layer 1
python pipeline.py --config config.yaml --layer 2
python pipeline.py --config config.yaml --layer 3
python pipeline.py --config config.yaml --layer 4

# 6. Launch the RAG chat
streamlit run layer4/streamlit_app.py
```

---

## Configuration

All pipeline parameters live in `config.yaml` — chunk size, overlap, similarity thresholds, LLM model, retrieval top-k, and I/O paths. Change parameters without touching code.

---

## Repository layout

```
deep-doc-extractor/
├── config.yaml
├── pipeline.py
├── stage1/             # extraction (FR001–FR008)
├── stage2/             # Q&A construction (FR009–FR017)
├── stage3/             # augmentation (FR018–FR021)
├── layer4/             # RAG service (FR022–FR029)
├── common/             # shared Ollama client, utilities
├── tests/              # unit + integration tests
├── output/
│   └── stage3/         # final augmented dataset (JSONL + CSV)
├── docs/               # HLD, LLD, FR specifications
└── requirements.txt
```

---

## Pipeline metrics (HP E877 manual, 241 pages)

| Metric | Target (HLD §11.2) | Achieved |
|---|---|---|
| Stage 2 quality-filter pass rate | ≥ 70% | **80.2%** |
| Stage 3 multiplier (post-filter) | 3× pre-filter | 2.6× post-filter (100% in [0.70, 0.97]) |
| Stage 3 near-clones | 0 | **0** |
| Stage 3 mean similarity to original | — | 0.91 |
| Final dataset size | — | 2,121 records |
| Image extraction F1 | — | 0.900 |
| Table extraction F1 | — | 0.938 |

---

## Status

- [x] Layer 1 — Document Ingestion & Extraction
- [x] Layer 2 — Semantic Structuring & Q&A Dataset
- [x] Layer 3 — Augmentation & Export
- [ ] Layer 4 — RAG Chat Service *(in progress)*
- [ ] End-to-end integration test on HP E877

---

## License

MIT — see [LICENSE](LICENSE). Per FR031.

---

## Team

**Team DocuForge — Woosong University, Department of AI & Big Data**

| Name | Role |
|---|---|
| MD Jubayer Hossain | All stages, Architecture, Layer 1, Quality Audit |
| Arlen Kasymbaev | Layer 1 Extraction |
| Sangwook Yoo | Layer 2 Q&A Generation |
| Inyoung Park | Layer 2 Quality Filtering |
| Sungho Choi | Layers 3 & 4, Augmentation, RAG |

**Industry mentor:** Mintae Kim — HP Printing Korea Data Science Team
**Supervising professor:** Kim Young-Il (김영일) — Woosong University
