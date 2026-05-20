# Stage 4 — RAG Chat Service · Step-by-Step Guide

Builds the retrieval-augmented chatbot that lets users ask questions about
the HP E877 manual and get answers grounded in the Stage 1 markdown and
the Stage 2/3 Q&A dataset.

This document walks you through the build **one command at a time**. Every
step has a verification check — if the check doesn't pass, stop and fix
before moving on.

---

## Architecture in 30 seconds

```
  Stage 1 *.md              Stage 3 augmented.jsonl  (or Stage 2 qa_pairs.jsonl)
        │                              │
        ▼                              ▼
  MarkdownChunker              load_qa_pairs_as_retrieval_chunks()
        │                              │
        │  chunks (kind='chunk')       │  QA pairs (kind='qa_pair')
        └──────────────┬───────────────┘
                       ▼
              EmbeddingEngine   (all-MiniLM-L6-v2, 384-d)
                       │
                       ▼
                ChromaIndexer   (persistent, cosine distance)
                       │
  QUERY  ──────────────┘
    │         top-k ANN → score filter → promote confident QA match
    ▼
  RAGRetriever  →  List[Hit]
    │
    ▼
  AnswerGenerator  →  grounded prompt  →  Ollama Qwen3.5:9B  →  Answer
    │
    ▼
  streamlit_app     ←  full grounding panel  (page, score, kind, text)
```

**Hybrid index** — one Chroma collection, two kinds of rows:
- `kind='chunk'`     — re-chunked Stage 1 markdown (retrieval context)
- `kind='qa_pair'`   — Stage 2/3 Q&A pairs (direct FAQ match for high-similarity queries)

---

## Step 0 — Prerequisites

You should already have Stages 1–3 working. Your `output/` directory should contain at least:

```
output/stage1/*.md
output/stage2/qa_pairs.jsonl          (at minimum)
output/stage3/augmented.jsonl         (preferred; Stage 4 uses this if present)
```

If you don't, build them first:

```bash
python pipeline.py --stage 1 --test-pages 14
python pipeline.py --stage 2
python pipeline.py --stage 3
```

Also make sure **Ollama is running** in a separate terminal:

```bash
ollama serve
ollama list | grep qwen3.5:9b    # confirm the model is pulled
```

---

## Step 1 — Unpack and install (5 min)

Unzip `stage4.zip` into your project root. Your layout afterwards:

```
your-project/
├── stage1/
├── stage2/
├── stage3/
├── stage4/                      ← NEW
│   ├── __init__.py
│   ├── offline_guard.py
│   ├── embedding_engine.py
│   ├── markdown_chunker.py
│   ├── chroma_indexer.py
│   ├── rag_retriever.py
│   ├── answer_generator.py
│   ├── stage4_pipeline.py
│   ├── streamlit_app.py
│   ├── test_stage4.py
│   ├── requirements.txt
│   └── README.md
├── config.yaml                  ← needs stage4 block added
├── config_stage4_block.yaml     ← paste this block into config.yaml
└── pipeline.py                  ← OVERWRITTEN (adds --stage 4 / --stage all-rag)
```

Install dependencies:

```bash
pip install -r stage4/requirements.txt
```

This installs `chromadb==0.5.x`, `sentence-transformers`, `streamlit`, `numpy`.
First import of sentence-transformers will download `all-MiniLM-L6-v2` (~90 MB).

---

## Step 2 — Add the Stage 4 block to config.yaml

Open `config_stage4_block.yaml` and paste its contents at the bottom of your
`config.yaml`. The new block starts with `stage4:`. Everything under
`input:`, `stage1:`, `stage2:`, `stage3:` stays unchanged.

**Verification check:** run

```bash
python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(list(c.keys()))"
```

You should see `stage4` in the output.

---

## Step 3 — Run the offline smoke test (30 s)

This runs 6 checks **without** touching Ollama or downloading the embedding
model. If any of these fail, stop and tell me.

```bash
cd stage4
python test_stage4.py
cd ..
```

Expected output ends with:

```
ALL TESTS PASSED ✓
```

What it verified:
1. All 7 modules import cleanly
2. `offline_guard` correctly blocks non-loopback connections
3. `markdown_chunker` splits a test MD file into chunks with correct page numbers
4. `load_qa_pairs_as_retrieval_chunks` parses a Stage 2/3 JSONL line
5. `AnswerGenerator` refuses (without calling Ollama) when no hits are passed
6. The grounded prompt builder includes page numbers and both sources

---

## Step 4 — Build the Chroma index (2–5 min)

This is the one-time data-prep step. It reads your Stage 1 markdown, re-chunks
it for retrieval, loads the Stage 3 JSONL (or Stage 2 if no Stage 3), embeds
every row with `all-MiniLM-L6-v2`, and upserts into a persistent Chroma collection.

```bash
python pipeline.py --stage 4
```

or equivalently:

```bash
python -m stage4.stage4_pipeline index
```

What you should see:

```
============================================================
STAGE 4 — INDEX PIPELINE
============================================================
[1/4] Sources:
      MD    : output/stage1/hp-e877-series-user-guide_pages_14.md
      JSONL : output/stage3/augmented.jsonl

[2/4] Re-chunking markdown for retrieval...
      hp-e877..._pages_14.md                   →    2 chunks
      Total markdown chunks: 2

[3/4] Loading Q&A pairs as retrieval rows...
      augmented.jsonl                          →    8 pairs
      Total QA-pair rows: 8

[4/4] Embedding + upserting into Chroma...

      Chroma collection: hp_e877_manual
      Total rows       : 10
      → chunks         : 2
      → qa_pairs       : 8
      Elapsed          : 12.4s
```

**Verification check:**

```bash
python -m stage4.stage4_pipeline status
```

Should show your collection with `Total rows > 0`.

If it shows 0: your Stage 1/2/3 outputs don't exist where expected. Check your
config's `output_dir` paths.

---

## Step 5 — One-shot CLI query (verify retrieval works)

Before launching the UI, confirm the retrieval and generation pipeline works end-to-end:

```bash
python pipeline.py --stage 4 --query "What is item 7 on the printer?"
```

Expected output:

```
Q: What is item 7 on the printer?

A: Item 7 is the easy-access USB port, which allows users to insert a
   USB flash drive for printing or scanning without a computer, or for
   firmware updates [p.14].

   (model=qwen3.5:9b  ·  2.8s  ·  3 hits  ·  refused=False)

SOURCES:
  [1] p.14  ·  score=0.871  ·  qa_pair  ·  Printer front view (chunk 0)
  [2] p.14  ·  score=0.722  ·  chunk    ·  Printer front view
  [3] p.14  ·  score=0.418  ·  chunk    ·  Printer views
```

Two signs this worked:
- The answer **cites a page number** inline (`[p.14]`)
- The sources list shows **both chunk and qa_pair rows** retrieved

**If you see "I don't have that information in the HP E877 manual"** — this
means either: (a) your page 14 content wasn't indexed, or (b) the similarity
threshold is too high. Drop `stage4.retrieval.min_score` to `0.2` in config
and retry.

**If Ollama timed out** — make sure `ollama serve` is running in another
terminal and `qwen3.5:9b` is pulled.

---

## Step 6 — Test the refusal path (hallucination guard)

Ask something NOT in the manual and verify the system refuses instead of making
things up:

```bash
python pipeline.py --stage 4 --query "What is the capital of France?"
```

Expected:

```
A: I don't have that information in the HP E877 manual.
   Please consult a different source or rephrase the question.

   (refused=True)
```

This is **the central guarantee of grounded RAG**. If the model answers "Paris"
here, your min_score is too low or your prompt needs tightening — tell me and
we'll fix it.

---

## Step 7 — Launch the Streamlit chat UI (5 s)

```bash
streamlit run stage4/streamlit_app.py
```

Your browser will open to `http://localhost:8501`. You should see:

- **Sidebar**: index status (chunks + Q&A pair counts), runtime info, session history
- **Main**: chat input at the bottom, 3 example questions as shortcuts
- **When you ask**: streamed answer with citation like `[p.14]`, plus an
  expandable "📑 Grounding" panel showing each source's **page, similarity
  score, kind (chunk / qa_pair), section**, and **excerpt**

Try these three queries in order to show off all three paths:

1. `"What is item 7?"`               — should hit the QA pair (fast path)
2. `"How do I replace the toner?"`   — should hit a chunk with procedure content
3. `"What's the weather today?"`     — should refuse

The grounding panel on each turn is what makes this presentation-ready. Open it,
screenshot it, drop it on a slide — it's your proof that the dataset grounds
the answers.

---

## Step 8 — (optional) Full run with one command

Once you're happy with the individual stages, you can run everything end-to-end:

```bash
python pipeline.py --stage all-rag --test-pages 14
```

This runs 1 → 2 → 3 → 4 (index) in sequence. UI still launches separately.

---

## Troubleshooting

**`OfflineGuard: refused connection to huggingface.co`**

First-time embedding model download needs the internet. Temporarily turn off
the guard: set `stage4.offline.enforce: false` in config, run the index build
once (it caches the model), then turn it back on for the UI. Alternatively,
pre-download with:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")   # one-time; cached under ~/.cache/
```

**`RuntimeError: attempt to write a readonly database`**

ChromaDB is holding a write lock from an aborted earlier run. Delete
`output/stage4/chroma/` and re-run Step 4.

**`Collection hp_e877_manual does not exist`**

You ran a query before building the index. Go back to Step 4.

**Answer is plausible but no page citation**

Qwen sometimes forgets the citation instruction. Either:
- lower `stage4.llm.temperature` to `0.1`
- add `"Always end with the page number in square brackets."` to the system prompt
  in `answer_generator.py`

**Index seems to have the right rows but similarity is always low**

The embedder expects reasonable-length text. Check your Stage 1 markdown —
if chunks are 3–5 words each, try raising `stage4.chunking.min_chunk_chars` to `200`.

---

## Files you'll reference in the demo

| What to show                          | File                                  |
|---------------------------------------|---------------------------------------|
| The hybrid collection                 | `output/stage4/chroma/`               |
| Offline-guard log (FR028 evidence)    | `output/stage4/offline_guard.log`     |
| Screenshot of grounding panel         | (capture in-browser)                  |
| CLI query transcript                  | (Step 5 stdout)                       |

---

## What's in each module (one-liner each)

| Module                 | Responsibility                                               |
|------------------------|--------------------------------------------------------------|
| `offline_guard.py`     | Monkey-patches sockets to reject non-loopback (FR028).       |
| `embedding_engine.py`  | `all-MiniLM-L6-v2` wrapper; 384-d normalised vectors.        |
| `markdown_chunker.py`  | Re-chunks Stage 1 MD; tracks page + section; classifies.     |
| `chroma_indexer.py`    | Persistent Chroma client with hybrid upsert + stats.         |
| `rag_retriever.py`     | Embed query → ANN → score filter → promote QA match.         |
| `answer_generator.py`  | Grounded Ollama prompt + refusal when no hits.               |
| `stage4_pipeline.py`   | CLI: `index`, `query`, `status` subcommands.                 |
| `streamlit_app.py`     | Chat UI with full grounding panel.                           |

---

## FR coverage

| FR    | Covered by                                  |
|-------|---------------------------------------------|
| FR022 | `embedding_engine.py`                       |
| FR023 | `chroma_indexer.py`                         |
| FR024 | `rag_retriever.py`                          |
| FR025 | `answer_generator.py` (grounded prompt)     |
| FR026 | `answer_generator.py` (citation instruction)|
| FR027 | `streamlit_app.py`                          |
| FR028 | `offline_guard.py`                          |
| FR029 | `streamlit_app.py` (history in sidebar)     |
