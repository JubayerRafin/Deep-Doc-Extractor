"""
stage4_pipeline.py — Stage 4 orchestrators.

Two entry points:

  1. run_index_pipeline(config, rebuild=False)
     One-time data prep. Reads Stage 1 markdown + Stage 2/3 JSONL, re-chunks
     the markdown, loads the QA pairs, embeds everything, upserts into Chroma.
     Set rebuild=True to drop and rebuild from scratch.

  2. answer_query(query, config)
     Per-query: retrieve → generate → return Answer dict (JSON-serialisable).
     Used by streamlit_app.py and by the CLI.

Usage:
    python stage4_pipeline.py index                 # build the index
    python stage4_pipeline.py index --rebuild       # rebuild from scratch
    python stage4_pipeline.py query "how do I..."   # one-shot query
    python stage4_pipeline.py status                # print index stats
"""
import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, List, Any

import yaml

from markdown_chunker import MarkdownChunker, load_qa_pairs_as_retrieval_chunks
from chroma_indexer   import ChromaIndexer
from rag_retriever    import RAGRetriever
from answer_generator import AnswerGenerator


# ── Path helpers ─────────────────────────────────────────────────────

def _resolve_sources(config: Dict):
    """Return (list of .md paths, list of .jsonl paths to ingest)."""
    s1_dir = config.get("stage1", {}).get("output_dir", "output/stage1")
    s2_dir = config.get("stage2", {}).get("output_dir", "output/stage2")
    s3_dir = config.get("stage3", {}).get("output_dir", "output/stage3")
    s4_src = config.get("stage4", {}).get("sources", {})

    # Markdown files from Stage 1
    md_glob   = s4_src.get("markdown_glob") or os.path.join(s1_dir, "*.md")
    md_files  = sorted(glob.glob(md_glob))

    # JSONL files — prefer augmented (Stage 3), fall back to Stage 2
    jsonl_paths: List[str] = []
    aug = s4_src.get("augmented_jsonl") or os.path.join(s3_dir, "augmented.jsonl")
    qa  = s4_src.get("qa_jsonl")        or os.path.join(s2_dir, "qa_pairs.jsonl")
    if os.path.exists(aug):
        jsonl_paths.append(aug)
    elif os.path.exists(qa):
        jsonl_paths.append(qa)
    return md_files, jsonl_paths


# ── Index pipeline ───────────────────────────────────────────────────

def run_index_pipeline(config: Dict, rebuild: bool = False) -> Dict[str, Any]:
    print("=" * 60)
    print("STAGE 4 — INDEX PIPELINE")
    print("=" * 60)

    indexer = ChromaIndexer(config)
    chunker = MarkdownChunker(config)

    if rebuild:
        print("[0/4] --rebuild set: dropping existing collection...")
        indexer.reset()

    md_files, jsonl_paths = _resolve_sources(config)
    print(f"[1/4] Sources:")
    for f in md_files:    print(f"      MD    : {f}")
    for f in jsonl_paths: print(f"      JSONL : {f}")
    if not md_files and not jsonl_paths:
        print("\n[Stage 4] No source files found. Run Stages 1–3 first.")
        return {"indexed": 0, "error": "no_sources"}

    t0 = time.time()

    # --- 2. Re-chunk markdown
    print(f"\n[2/4] Re-chunking markdown for retrieval...")
    md_chunks = []
    for md in md_files:
        cs = chunker.chunk_file(md)
        md_chunks.extend(cs)
        print(f"      {os.path.basename(md):40s} → {len(cs):4d} chunks")
    print(f"      Total markdown chunks: {len(md_chunks)}")

    # --- 3. Load QA pairs
    print(f"\n[3/4] Loading Q&A pairs as retrieval rows...")
    qa_chunks = []
    for j in jsonl_paths:
        cs = load_qa_pairs_as_retrieval_chunks(j)
        qa_chunks.extend(cs)
        print(f"      {os.path.basename(j):40s} → {len(cs):4d} pairs")
    print(f"      Total QA-pair rows: {len(qa_chunks)}")

    # --- 4. Embed and upsert
    print(f"\n[4/4] Embedding + upserting into Chroma...")
    n_md = indexer.index(md_chunks)
    n_qa = indexer.index(qa_chunks)
    elapsed = time.time() - t0

    stats = indexer.stats()
    print(f"\n      Chroma collection: {indexer.collection_name}")
    print(f"      Total rows       : {stats.get('total', '?')}")
    print(f"      → chunks         : {stats.get('chunks', '?')}")
    print(f"      → qa_pairs       : {stats.get('qa_pairs', '?')}")
    print(f"      Elapsed          : {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"INDEX BUILD COMPLETE  |  {n_md + n_qa} rows  |  {elapsed:.1f}s")
    print("=" * 60)
    return {
        "indexed_markdown_chunks": n_md,
        "indexed_qa_pairs":        n_qa,
        "total_rows":              stats.get("total", n_md + n_qa),
        "elapsed_seconds":         round(elapsed, 2),
    }


# ── Query pipeline ───────────────────────────────────────────────────

def answer_query(query: str, config: Dict) -> Dict[str, Any]:
    """Per-query: retrieve → answer. Returns a JSON-ready dict."""
    retriever = RAGRetriever(config)
    generator = AnswerGenerator(config)
    hits = retriever.retrieve(query)
    ans  = generator.answer(query, hits)
    return {
        "query":   query,
        "answer":  ans.text,
        "refused": ans.refused,
        "elapsed_s": ans.elapsed_s,
        "model":   ans.model,
        "sources": ans.sources,
        "n_hits":  len(hits),
    }


# ── CLI ──────────────────────────────────────────────────────────────

def _load_config(path: str) -> Dict:
    if not os.path.exists(path):
        print(f"[ERROR] Config not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Stage 4: RAG Chat Service")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_idx = sub.add_parser("index",   help="Build (or rebuild) the Chroma index.")
    ap_idx.add_argument("--config",    default="config.yaml")
    ap_idx.add_argument("--rebuild",   action="store_true")

    ap_q = sub.add_parser("query",     help="One-shot query from the CLI.")
    ap_q.add_argument("text",          help="The question.")
    ap_q.add_argument("--config",      default="config.yaml")

    ap_s = sub.add_parser("status",    help="Print index stats.")
    ap_s.add_argument("--config",      default="config.yaml")

    args = ap.parse_args()
    config = _load_config(args.config)

    if args.cmd == "index":
        run_index_pipeline(config, rebuild=args.rebuild)

    elif args.cmd == "query":
        out = answer_query(args.text, config)
        print(f"\nQ: {out['query']}")
        print(f"\nA: {out['answer']}\n")
        print(f"   (model={out['model']}  ·  {out['elapsed_s']}s  ·  "
              f"{out['n_hits']} source hits  ·  refused={out['refused']})")
        if out["sources"]:
            print("\nSOURCES:")
            for i, src in enumerate(out["sources"], 1):
                print(f"  [{i}] p.{src.get('page')}  ·  "
                      f"score={src.get('score'):.3f}  ·  "
                      f"{src.get('kind')}  ·  {src.get('section') or 'n/a'}")

    elif args.cmd == "status":
        idx = ChromaIndexer(config)
        s = idx.stats()
        print(f"Collection: {idx.collection_name}")
        print(f"Persist   : {idx.persist_dir}")
        print(f"Total rows: {s.get('total', 0)}")
        print(f"  chunks  : {s.get('chunks', 0)}")
        print(f"  qa_pairs: {s.get('qa_pairs', 0)}")


if __name__ == "__main__":
    main()
