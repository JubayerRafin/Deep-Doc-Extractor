"""
stage2_pipeline.py — Stage 2 Orchestrator: Markdown → Q&A Pairs (JSONL)

Usage:
    python stage2_pipeline.py                        # uses config.yaml
    python stage2_pipeline.py --config path.yaml     # custom config
    python stage2_pipeline.py --dry-run              # no LLM calls
"""
import os, sys, json, yaml, csv, argparse, time
from collections import Counter
from markdown_parser import parse_markdown
from semantic_classifier import classify_blocks
from chunker import chunk_all
from qa_generator import generate_qa_for_chunk, qa_pair_to_jsonl, QAPair
from quality_filters import run_all_filters


def _export_dli_kb(accepted, chunks, out_path):
    """
    Export DLI Knowledge Base snapshot as structured text.
    Groups content by entry type (procedure, spec, rule)
    with provenance metadata and extracted keywords.
    """
    from collections import defaultdict
    import re

    # Build chunk text lookup
    chunk_lookup = {}
    for c in chunks:
        key = f"{c.heading_path or c.heading} (chunk {c.chunk_index})"
        chunk_lookup[key] = c

    # Group Q&A pairs by chunk reference
    chunk_groups = defaultdict(list)
    for pair in accepted:
        chunk_groups[pair.chunk_ref].append(pair)

    # Map category to KB entry type
    entry_type_map = {
        "procedure": "PROCEDURE",
        "spec": "SPECIFICATION",
        "rule_error": "RULE",
        "figure": "PROCEDURE",
    }

    # Collect entries grouped by type
    entries_by_type = defaultdict(list)
    for chunk_ref, pairs in chunk_groups.items():
        chunk = chunk_lookup.get(chunk_ref)
        if not chunk:
            continue

        # Extract keywords
        words = re.findall(r'\b[A-Za-z][a-z]{3,}\b', chunk.text)
        stopwords = {"this", "that", "with", "from", "have", "will", "been",
                     "they", "their", "them", "your", "into", "when", "then",
                     "than", "what", "which", "where", "there", "these",
                     "those", "each", "other", "some", "such", "more",
                     "also", "about", "after", "before", "should", "does",
                     "make", "made", "like", "just", "only", "very", "most"}
        word_freq = defaultdict(int)
        for w in words:
            wl = w.lower()
            if wl not in stopwords and len(wl) > 3:
                word_freq[wl] += 1
        keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:8]

        category = pairs[0].category
        entry_type = entry_type_map.get(category, "PROCEDURE")

        entries_by_type[entry_type].append({
            "title": chunk.heading if chunk.heading else "Untitled",
            "content": chunk.text,
            "pairs": pairs,
            "keywords": keywords,
            "source_file": pairs[0].source_file,
            "chunk_ref": chunk_ref,
            "page": pairs[0].page_hint,
        })

    # Write structured text file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  DLI KNOWLEDGE BASE SNAPSHOT\n")
        f.write("  Deep Doc Extractor — Team DocuForge\n")
        f.write("=" * 70 + "\n\n")

        total_entries = sum(len(v) for v in entries_by_type.values())
        f.write(f"Total entries: {total_entries}\n")
        for etype in ["PROCEDURE", "SPECIFICATION", "RULE"]:
            count = len(entries_by_type.get(etype, []))
            if count:
                f.write(f"  {etype}: {count}\n")
        f.write("\n")

        entry_num = 0
        for etype in ["PROCEDURE", "SPECIFICATION", "RULE"]:
            entries = entries_by_type.get(etype, [])
            if not entries:
                continue

            f.write("=" * 70 + "\n")
            f.write(f"  {etype}S ({len(entries)} entries)\n")
            f.write("=" * 70 + "\n\n")

            for entry in entries:
                entry_num += 1
                f.write("-" * 50 + "\n")
                f.write(f"Entry {entry_num}: [{etype}] {entry['title']}\n")
                f.write(f"Source: {entry['source_file']} | Page: {entry['page'] or '?'}\n")
                f.write(f"Keywords: {', '.join(entry['keywords'])}\n")
                f.write("-" * 50 + "\n\n")

                f.write("CONTENT:\n")
                f.write(entry["content"].strip() + "\n\n")

                f.write("Q&A PAIRS:\n")
                for i, p in enumerate(entry["pairs"], 1):
                    f.write(f"  Q{i}: {p.question}\n")
                    f.write(f"  A{i}: {p.answer}\n\n")

                f.write("\n")

    # Print summary
    type_counts = {k: len(v) for k, v in entries_by_type.items() if v}
    print(f"      DLI KB: {total_entries} entries", end="")
    for t, n in sorted(type_counts.items()):
        print(f" | {t}: {n}", end="")
    print()


def _resolve_paths(config: dict):
    """Derive input/output paths from the unified config."""
    s1 = config.get("stage1", {})
    s2 = config.get("stage2", {})
    s1_dir = s1.get("output_dir", "output/stage1")
    s2_dir = s2.get("output_dir", "output/stage2")
    tables_dir = os.path.join(s1_dir, s1.get("tables", {}).get("output_subdir", "tables"))

    # Auto-detect the .md file from Stage 1 output
    input_md = s2.get("input_md")
    if not input_md:
        pdf_name = os.path.splitext(os.path.basename(config.get("input", {}).get("pdf_path", "doc")))[0]
        input_md = os.path.join(s1_dir, f"{pdf_name}.md")

    return input_md, tables_dir, s2_dir


def run_pipeline(config: dict, dry_run: bool = False):
    input_md, tables_dir, s2_dir = _resolve_paths(config)
    os.makedirs(s2_dir, exist_ok=True)

    out_jsonl    = os.path.join(s2_dir, "qa_pairs.jsonl")
    out_rejected = os.path.join(s2_dir, "qa_rejected.jsonl")
    out_csv      = os.path.join(s2_dir, "qa_pairs.csv")

    print("=" * 60)
    print("STAGE 2: Dataset Generation Pipeline")
    print("=" * 60)

    # --- 1. Parse ---
    print(f"\n[1/6] Parsing: {input_md}")
    blocks = parse_markdown(input_md, tables_dir)
    print(f"      → {len(blocks)} content blocks")

    # --- 2. Classify ---
    print(f"\n[2/6] Classifying...")
    classified = classify_blocks(blocks, config)
    for cat, n in Counter(c for _, c in classified).most_common():
        print(f"      {cat:12s}: {n}")

    # --- 3. Chunk ---
    print(f"\n[3/6] Chunking...")
    chunks = chunk_all(classified, config)
    print(f"      → {len(chunks)} chunks")

    if dry_run:
        print("\n[DRY RUN] Stopping before LLM calls.\n")
        for i, c in enumerate(chunks[:8]):
            print(f"  [{i}] {c.category:12s} | {c.heading[:30]:30s} | {c.text[:70].replace(chr(10),' ')}...")
        return

    # Build chunk text map for hallucination filter
    # Match the key format used by qa_generator.py line 151
    chunks_map = {f"{c.heading} (chunk {c.chunk_index})": c.text for c in chunks}
    # --- 4. Generate Q&A ---
    model = config.get("stage2", {}).get("llm", {}).get("model", "?")
    print(f"\n[4/6] Generating Q&A via Ollama ({model})...")
    all_pairs: list[QAPair] = []
    fails = 0
    t0 = time.time()

    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el else 0
            eta = (len(chunks) - i - 1) / rate if rate else 0
            print(f"      {i+1}/{len(chunks)}  ({len(all_pairs)} pairs, ~{eta:.0f}s left)")

        n = 1
        try:
            all_pairs.extend(generate_qa_for_chunk(chunk, config, n_pairs=n))
        except Exception as e:
            print(f"      [WARN] chunk {i} failed: {e}"); fails += 1

    elapsed = time.time() - t0
    print(f"      → {len(all_pairs)} raw pairs in {elapsed:.1f}s  ({fails} failed chunks)")

    # --- 5. Filter ---
    print(f"\n[5/6] Quality filtering...")
    accepted, rejected = run_all_filters(all_pairs, chunks_map, config)
    print(f"      Accepted: {len(accepted)}  |  Rejected: {len(rejected)}")
    for reason, n in Counter(r for _, r in rejected).most_common():
        print(f"        - {reason}: {n}")

    # --- 6. Export ---
    print(f"\n[6/6] Exporting...")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in accepted:
            f.write(json.dumps(qa_pair_to_jsonl(p), ensure_ascii=False) + "\n")
    print(f"      → {out_jsonl} ({len(accepted)} pairs)")

    with open(out_rejected, "w", encoding="utf-8") as f:
        for p, reason in rejected:
            rec = qa_pair_to_jsonl(p); rec["rejection_reason"] = reason
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"      → {out_rejected} ({len(rejected)} rejected)")

    # CSV export
    if config.get("stage2", {}).get("export", {}).get("csv", False) and accepted:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["system_role", "question", "answer", "category", "source_file", "chunk_ref", "page"])
            for p in accepted:
                w.writerow([p.system_role, p.question, p.answer, p.category, p.source_file, p.chunk_ref, p.page_hint])
        print(f"      → {out_csv}")

    # DLI Knowledge Base snapshot
    if config.get("stage2", {}).get("export", {}).get("dli_kb", False) and accepted:
        out_kb = os.path.join(s2_dir, "dli_kb_snapshot.txt")
        _export_dli_kb(accepted, chunks, out_kb)
        print(f"      → {out_kb}")

    print("\n" + "=" * 60)
    print(f"STAGE 2 COMPLETE  |  {len(accepted)} accepted  |  {elapsed:.1f}s")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Stage 2: Dataset Generation")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}"); sys.exit(1)
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_pipeline(config, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
