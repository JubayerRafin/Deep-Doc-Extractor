# Deep Doc Extractor

**Team DocuForge | Woosong University Capstone 2026**
**Mentor: Mintae Kim (HP Korea)**

Automated 3-stage document processing pipeline that converts HP technical
manuals (PDF) into structured Markdown, LLM-ready JSONL datasets, and
DLI Knowledge Base snapshots for local LLM fine-tuning.

---

## Project Structure

```
deep_doc_extractor/
│
├── config.yaml              ← Single control file for all settings
├── pipeline.py              ← Main entry point
├── requirements.txt
│
├── stage1/                  ← Stage 1: Extraction & Markdown
│   ├── __init__.py
│   ├── extracted_element.py ← Shared data structure
│   ├── text_extractor.py    ← Text blocks via pdfplumber
│   ├── image_extractor.py   ← Images via gap detection → PNG files
│   ├── table_extractor.py   ← Tables via Camelot-py → JSON files
│   ├── file_validator.py    ← Input PDF validation
│   └── markdown_compiler.py ← Merges all elements → .md output
│
├── stage2/                  ← Stage 2: Dataset Construction
│   ├── __init__.py
│   ├── markdown_parser.py   ← Parse .md back into typed blocks
│   ├── semantic_chunker.py  ← Verb-phase semantic chunking
│   ├── schema_classifier.py ← Classify chunk types
│   ├── qa_generator.py      ← Q&A pair generation via Qwen 2.5
│   ├── dataset_validator.py ← Length/hallucination/dedup filters
│   └── dataset_exporter.py  ← Export to JSONL, CSV, DLI KB
│
├── stage3/                  ← Stage 3: Q&A Augmentation (planned)
│
└── output/
    ├── stage1/
    │   ├── images/          ← Extracted PNG files
    │   ├── tables/          ← Extracted JSON table files
    │   └── *.md             ← Final Markdown output
    └── stage2/
        ├── *.jsonl          ← Q&A dataset
        ├── *.csv            ← Spreadsheet export
        └── *.txt            ← DLI Knowledge Base snapshot
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your PDF in the project folder
cp /path/to/hp-e877-series-user-guide.pdf .

# 4. (Stage 2+) Start Ollama with Qwen 2.5
ollama pull qwen2.5:3b
ollama serve
```

---

## Running the Pipeline

```bash
# Validate the PDF first
python pipeline.py --stage 1 --validate-only

# Test with mentor sample pages (35-36)
python pipeline.py --stage 1 --test-pages 35,36

# Process the full 240-page PDF
python pipeline.py --stage 1

# Run Stage 2 (requires Stage 1 output)
python pipeline.py --stage 2

# Run both stages end-to-end
python pipeline.py --stage both
```

---

## How Stage 1 Works

```
PDF File
   │
   ├──► TextExtractor    (pdfplumber)  → text blocks + Y-coordinates
   │      • Char-level font size via max() for heading detection
   │      • Bold detection via fontname analysis
   │      • Footer/header filtering
   │
   ├──► ImageExtractor   (pdfplumber)  → PNG files + Y-coordinates
   │      • Gap-detection approach (HP uses vector graphics)
   │      • Width-filtered text mask (30% page width threshold)
   │      • Blank/white image filtering
   │      • Table-as-image rejection
   │
   └──► TableExtractor   (Camelot-py)  → JSON files + Y-coordinates
          • Lattice mode with stream fallback
          • False-positive rejection (CAUTION boxes, layout boxes)
              │
              ▼
        MarkdownCompiler
        (sort by page → Y-coordinate → write .md)
              │
              ▼
        output/stage1/document.md
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Tables as **JSON** (not Markdown pipes) | Better for LLM processing in Stage 2 |
| Images as **separate PNGs** | Markdown references with descriptive alt text |
| Heading detection via **max font size** | pdfplumber duplicates words; average fails |
| Image extraction via **text-gap detection** | HP E877 uses vector Form XObjects |
| Semantic chunking by **verb phase** | Better context than token-count splitting |
| Dedup threshold **≥ 0.92** | Avoids over-filtering distinct Q&A pairs |

---

## config.yaml Quick Reference

| Key | Description | Default |
|-----|-------------|---------|
| `input.pdf_path` | Path to the PDF file | `hp-e877-series-user-guide.pdf` |
| `input.test_pages` | Pages to test (list or null) | `null` |
| `stage1.text.heading_font_threshold` | Font ratio for heading detection | `1.2` |
| `stage1.images.min_width` | Minimum image width in pixels | `50` |
| `stage1.tables.flavor` | Camelot mode | `lattice` |
| `stage2.llm.model` | Ollama model for Q&A generation | `qwen2.5:3b` |

---

## Troubleshooting

**`__pycache__` interference:** If code changes don't take effect, delete
`__pycache__` folders before re-running:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

**Camelot requires Ghostscript:** Install via your package manager:
```bash
# Ubuntu/Debian
sudo apt-get install ghostscript

# macOS
brew install ghostscript

# Windows — download from https://ghostscript.com/
```
