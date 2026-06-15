"""
pipeline.py — Deep Doc Extractor
---------------------------------
Main entry point for the 4-stage document processing pipeline.

Usage:
  python pipeline.py --stage 1 --config config.yaml
  python pipeline.py --stage 1 --config config.yaml --test-pages 14
  python pipeline.py --stage 2 --config config.yaml
  python pipeline.py --stage 3 --config config.yaml
  python pipeline.py --stage 3 --config config.yaml --limit 5
  python pipeline.py --stage 4 --config config.yaml                 # build RAG index
  python pipeline.py --stage 4 --config config.yaml --rebuild       # rebuild index
  python pipeline.py --stage 4 --config config.yaml --query "..."   # one-shot query
  python pipeline.py --stage both --config config.yaml              # stages 1+2
  python pipeline.py --stage all --config config.yaml               # stages 1+2+3
  python pipeline.py --stage all-rag --config config.yaml           # stages 1+2+3+4 index
  python pipeline.py --validate-only --config config.yaml

  # The chat UI is a separate process — launch after --stage 4 finishes:
  #   streamlit run stage4/streamlit_app.py

Team DocuForge | Woosong University Capstone 2026
Mentor: Mintae Kim (HP Korea)
"""

import os
import sys
import argparse
import yaml
import time


def parse_args():
    p = argparse.ArgumentParser(description="Deep Doc Extractor Pipeline")
    p.add_argument("--stage",
                   choices=["1", "2", "3", "4", "both", "all", "all-rag"],
                   required=True,
                   help="'both' = stages 1+2; 'all' = 1+2+3; 'all-rag' = 1+2+3+4 index")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--test-pages", type=str, default=None,
                   help="Comma-separated page numbers, e.g. --test-pages 14")
    p.add_argument("--limit", type=int, default=None,
                   help="Stage 3 only: process first N accepted pairs (debug/demo)")
    p.add_argument("--dry-run", action="store_true",
                   help="Stage 2/3 only: skip LLM calls")
    p.add_argument("--rebuild", action="store_true",
                   help="Stage 4 only: drop and rebuild the Chroma index")
    p.add_argument("--query", type=str, default=None,
                   help="Stage 4 only: one-shot query from the CLI")
    p.add_argument("--validate-only", action="store_true",
                   help="Only validate the input PDF, then exit")
    return p.parse_args()


def load_config(path):
    if not os.path.exists(path):
        print(f"[ERROR] Config not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Pipeline] Config loaded: {path}")
    return cfg


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — Extraction & Markdown Compilation
# ══════════════════════════════════════════════════════════════════

def run_stage1(config, test_pages=None):
    from stage1 import TextExtractor, ImageExtractor, TableExtractor, MarkdownCompiler
    from stage1.file_validator import FileValidator

    print("\n" + "=" * 60)
    print("  STAGE 1 — Extraction & Markdown Compilation")
    print("=" * 60)

    pdf_path   = config["input"]["pdf_path"]
    output_dir = config["stage1"]["output_dir"]

    # Validate input file
    if not FileValidator().report(pdf_path):
        print("[Stage 1] Aborting — invalid input.")
        sys.exit(1)

    # Determine page scope
    if test_pages:
        page_numbers = test_pages
        print(f"\n[Stage 1] Mode: TEST — pages {page_numbers}")
    elif config["input"].get("test_pages"):
        page_numbers = config["input"]["test_pages"]
        print(f"\n[Stage 1] Mode: TEST (config) — pages {page_numbers}")
    else:
        page_numbers = None
        print("\n[Stage 1] Mode: FULL DOCUMENT")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Step 1: Text
    print("\n[Step 1/4] Extracting text blocks...")
    text_elements = TextExtractor(config["stage1"]["text"]).extract(
        pdf_path, page_numbers
    )

    # Step 2: Images
    print("\n[Step 2/4] Extracting images...")
    image_elements = ImageExtractor(config["stage1"], output_dir).extract(
        pdf_path, page_numbers
    )

    # Step 3: Tables
    print("\n[Step 3/4] Extracting tables...")
    table_elements = TableExtractor(config["stage1"], output_dir).extract(
        pdf_path, page_numbers
    )

    # Step 4: Compile
    print("\n[Step 4/4] Compiling Markdown...")
    all_elements = text_elements + image_elements + table_elements
    pdf_base     = os.path.splitext(os.path.basename(pdf_path))[0]
    tag          = ("_pages_" + "_".join(str(p) for p in page_numbers)
                    if page_numbers else "")
    output_md    = os.path.join(output_dir, f"{pdf_base}{tag}.md")
    MarkdownCompiler().compile(all_elements, output_md)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  STAGE 1 COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  Text   : {len(text_elements)}")
    print(f"  Images : {len(image_elements)}")
    print(f"  Tables : {len(table_elements)}")
    print(f"  Output : {output_md}")
    return output_md


# ══════════════════════════════════════════════════════════════════
# STAGE 2 — Dataset Construction
# ══════════════════════════════════════════════════════════════════

def run_stage2(config, md_path, dry_run=False):
    stage2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2")
    if stage2_dir not in sys.path:
        sys.path.insert(0, stage2_dir)

    from stage2.stage2_pipeline import run_pipeline as run_stage2_pipeline

    if md_path and not os.path.exists(md_path):
        print(f"[Stage 2] ERROR: Markdown file not found: {md_path}")
        print("[Stage 2] Run Stage 1 first.")
        sys.exit(1)

    if md_path:
        if "stage2" not in config:
            config["stage2"] = {}
        config["stage2"]["input_md"] = md_path

    run_stage2_pipeline(config, dry_run=dry_run)


# ══════════════════════════════════════════════════════════════════
# STAGE 3 — Dataset Augmentation
# ══════════════════════════════════════════════════════════════════

def run_stage3(config, dry_run=False, limit=None):
    stage3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage3")
    if stage3_dir not in sys.path:
        sys.path.insert(0, stage3_dir)

    from stage3.stage3_pipeline import run_pipeline as run_stage3_pipeline

    s2_dir = config.get("stage2", {}).get("output_dir", "output/stage2")
    expected_input = config.get("stage3", {}).get("input_jsonl") \
                     or os.path.join(s2_dir, "qa_pairs.jsonl")
    if not os.path.exists(expected_input):
        print(f"[Stage 3] ERROR: Stage 2 output not found: {expected_input}")
        print("[Stage 3] Run Stage 2 first.")
        sys.exit(1)

    run_stage3_pipeline(config, dry_run=dry_run, limit=limit)


# ══════════════════════════════════════════════════════════════════
# STAGE 4 — RAG Chat Service (index build + optional CLI query)
# ══════════════════════════════════════════════════════════════════

def run_stage4(config, rebuild=False, query=None):
    stage4_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage4")
    if stage4_dir not in sys.path:
        sys.path.insert(0, stage4_dir)

    from stage4.stage4_pipeline import run_index_pipeline, answer_query

    if query is not None:
        # One-shot CLI query mode — skip indexing, just answer
        print("\n" + "=" * 60)
        print("  STAGE 4 — CLI QUERY")
        print("=" * 60)
        out = answer_query(query, config)
        print(f"\nQ: {out['query']}")
        print(f"\nA: {out['answer']}\n")
        print(f"   (model={out['model']}  ·  {out['elapsed_s']}s  ·  "
              f"{out['n_hits']} hits  ·  refused={out['refused']})")
        if out["sources"]:
            print("\nSOURCES:")
            for i, src in enumerate(out["sources"], 1):
                print(f"  [{i}] p.{src.get('page')}  ·  "
                      f"score={src.get('score'):.3f}  ·  "
                      f"{src.get('kind')}  ·  {src.get('section') or 'n/a'}")
        return

    # Index build mode
    run_index_pipeline(config, rebuild=rebuild)
    print("\nNext: launch the chat UI with:")
    print("  streamlit run stage4/streamlit_app.py")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    config = load_config(args.config)

    test_pages = None
    if args.test_pages:
        try:
            test_pages = [int(p.strip()) for p in args.test_pages.split(",")]
        except ValueError:
            print("[ERROR] --test-pages must be comma-separated integers, e.g. 14")
            sys.exit(1)

    if args.validate_only:
        from stage1.file_validator import FileValidator
        ok = FileValidator().report(config["input"]["pdf_path"])
        sys.exit(0 if ok else 1)

    md_path = None
    if args.stage in ("1", "both", "all", "all-rag"):
        md_path = run_stage1(config, test_pages)

    if args.stage in ("2", "both", "all", "all-rag"):
        if md_path is None:
            pdf_base = os.path.splitext(os.path.basename(
                config["input"]["pdf_path"]))[0]
            md_path = os.path.join(
                config["stage1"]["output_dir"], f"{pdf_base}.md"
            )
        run_stage2(config, md_path, dry_run=args.dry_run)

    if args.stage in ("3", "all", "all-rag"):
        run_stage3(config, dry_run=args.dry_run, limit=args.limit)

    if args.stage in ("4", "all-rag"):
        run_stage4(config, rebuild=args.rebuild, query=args.query)

    print("\n[Pipeline] All done.\n")


if __name__ == "__main__":
    main()
