"""
test_stage3.py — Offline smoke test for Stage 3 components.

Verifies module structure and data flow WITHOUT making LLM calls.
Run this first to confirm the package imports cleanly and
augment_filter + final_exporter work with a synthetic dataset.

Usage:
    python test_stage3.py
"""
import os
import sys
import json
import tempfile
import shutil

# Make the stage3 package importable whether you run this from the
# repo root OR from inside the stage3/ directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


# ── Synthetic test data (mirrors your Stage 2 output schema) ─────────

SAMPLE_STAGE2_RECORDS = [
    {
        "messages": [
            {"role": "system",    "content": "You are a helpful HP printer technician."},
            {"role": "user",      "content": "How do I replace the toner cartridge?"},
            {"role": "assistant", "content": "Open the front door, remove the old cartridge, insert the new one, and close the door."},
        ],
        "provenance": {
            "source_file": "hp-e877-series-user-guide.md",
            "chunk": "Replace toner cartridge (chunk 0)",
            "category": "procedure",
            "page": 35,
        },
    },
    {
        "messages": [
            {"role": "system",    "content": "You are an HP technical support specialist."},
            {"role": "user",      "content": "What is the paper capacity of Tray 2?"},
            {"role": "assistant", "content": "Tray 2 holds up to 550 sheets of standard paper."},
        ],
        "provenance": {
            "source_file": "hp-e877-series-user-guide.md",
            "chunk": "Paper tray specifications (chunk 0)",
            "category": "spec",
            "page": 18,
        },
    },
]

# Synthetic variations (the kind of thing the LLM would return).
SAMPLE_VARIATIONS = [
    "What is the procedure for replacing the toner cartridge?",   # KEEP (paraphrase)
    "How do I replace the toner cartridge?",                      # DROP (exact)
    "What colour is the printer?",                                # DROP (drift)
    "Steps to swap out the toner cartridge?",                     # KEEP
    "How do I replace the toner cartridge please?",               # DROP (near-dup)
]


# ── Tests ────────────────────────────────────────────────────────────

def test_imports():
    """All modules import cleanly."""
    print("[1/5] Test imports...")
    from llm_rephraser import LLMRephraser, rephrase_question
    from augment_filter import AugmentFilter, filter_variations
    from final_exporter import FinalExporter, export_augmented_dataset, _question_id, _build_record
    from stage3_pipeline import run_pipeline, _load_stage2_records, _extract_question
    print("      ✓ all module imports ok")


def test_parser():
    """Variation-parser handles JSON array, code fences, numbered list, <think> tags."""
    print("[2/5] Test variation parser...")
    from llm_rephraser import LLMRephraser
    cases = [
        ('["a", "b", "c"]',                                   3, "plain JSON"),
        ('```json\n["a", "b"]\n```',                          2, "fenced JSON"),
        ('<think>thinking</think>\n["x", "y"]',               2, "with <think>"),
        ('1. First variation\n2. Second variation',           2, "numbered list"),
    ]
    for raw, expected_n, label in cases:
        out = LLMRephraser._parse_variations(raw, n=3)
        assert len(out) == expected_n, f"{label}: expected {expected_n}, got {len(out)}"
        print(f"      ✓ {label}: {out}")


def test_filter_offline():
    """Filter works without the LLM. Falls back to Jaccard if sbert missing."""
    print("[3/5] Test augment_filter...")
    from augment_filter import AugmentFilter

    cfg = {
        "stage3": {
            "filter": {
                "similarity_max": 0.97,
                "similarity_min": 0.40,  # relaxed for Jaccard fallback
                "min_variation_length": 5,
            }
        }
    }
    f = AugmentFilter(cfg)
    original = "How do I replace the toner cartridge?"
    kept, dropped = f.filter(original, SAMPLE_VARIATIONS)
    print(f"      original: {original}")
    print(f"      kept ({len(kept)}):")
    for v in kept:
        print(f"        ✓ {v}")
    print(f"      dropped ({len(dropped)}):")
    for v, reason in dropped:
        print(f"        ✗ [{reason}] {v}")
    assert len(kept) + len(dropped) == len(SAMPLE_VARIATIONS), \
        "filter should account for every input variation"
    assert any("drift" in r.lower() or "intra-batch" in r.lower() or "duplicate" in r.lower()
               for _, r in dropped), "at least one variation should be rejected"


def test_exporter():
    """Exporter writes the expected files with the expected schema."""
    print("[4/5] Test final_exporter...")
    from final_exporter import export_augmented_dataset, _build_record, _question_id

    tmp = tempfile.mkdtemp(prefix="stage3_test_")
    try:
        cfg = {
            "stage3": {
                "output_dir": tmp,
                "export": {"jsonl": True, "csv": True, "mentor_batch": True},
            }
        }
        # Build one synthetic augmentation of record[0]
        orig_q = SAMPLE_STAGE2_RECORDS[0]["messages"][1]["content"]
        aug_rec = _build_record(
            messages=[
                {"role": "system",    "content": "You are a helpful HP printer technician."},
                {"role": "user",      "content": "What is the procedure for replacing the toner cartridge?"},
                {"role": "assistant", "content": SAMPLE_STAGE2_RECORDS[0]["messages"][2]["content"]},
            ],
            provenance=SAMPLE_STAGE2_RECORDS[0]["provenance"],
            status="augmented",
            augmented_from=_question_id(orig_q),
            variation_index=1,
        )

        stats = {"test": True, "input_pairs": len(SAMPLE_STAGE2_RECORDS), "augmented": 1}
        written = export_augmented_dataset(
            accepted_stage2_records=SAMPLE_STAGE2_RECORDS,
            augmentations=[aug_rec],
            config=cfg,
            stats=stats,
        )

        # Verify files exist
        for name in ["jsonl", "csv", "mentor_batch", "stats"]:
            assert name in written, f"missing output: {name}"
            assert os.path.exists(written[name]), f"file not written: {written[name]}"
            print(f"      ✓ {name}: {os.path.basename(written[name])} "
                  f"({os.path.getsize(written[name])} bytes)")

        # Verify JSONL has 3 records (2 originals + 1 augmented)
        with open(written["jsonl"]) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 3, f"expected 3 records, got {len(lines)}"

        # Verify schema integrity of augmented record
        aug_in_file = [l for l in lines
                       if l["provenance"]["stage3"]["status"] == "augmented"]
        assert len(aug_in_file) == 1
        aug = aug_in_file[0]
        assert aug["provenance"]["stage3"]["augmented_from"] == _question_id(orig_q)
        assert aug["messages"][2]["content"] == SAMPLE_STAGE2_RECORDS[0]["messages"][2]["content"], \
            "augmented record must preserve the original answer unchanged"
        print(f"      ✓ schema: stage3.status + augmented_from + preserved answer")

        # Verify mentor_batch contains originals only
        with open(written["mentor_batch"]) as f:
            mb_lines = [json.loads(l) for l in f if l.strip()]
        assert len(mb_lines) == 2, f"mentor_batch should have 2 originals, got {len(mb_lines)}"
        assert all("original_question" in r for r in mb_lines)
        print(f"      ✓ mentor_batch: {len(mb_lines)} originals, no variations (as per FR021)")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pipeline_dry_run():
    """End-to-end dry run: writes originals-only export without calling the LLM."""
    print("[5/5] Test stage3_pipeline dry-run...")
    from stage3_pipeline import run_pipeline

    tmp = tempfile.mkdtemp(prefix="stage3_dry_")
    try:
        # Write a fake Stage 2 output
        s2_dir = os.path.join(tmp, "s2")
        os.makedirs(s2_dir)
        s2_jsonl = os.path.join(s2_dir, "qa_pairs.jsonl")
        with open(s2_jsonl, "w") as f:
            for rec in SAMPLE_STAGE2_RECORDS:
                f.write(json.dumps(rec) + "\n")

        cfg = {
            "stage2": {"output_dir": s2_dir},
            "stage3": {
                "output_dir": os.path.join(tmp, "s3"),
                "llm": {"model": "qwen2.5:3b"},
                "augmentation": {"variations_per_qa": 3},
                "filter": {"similarity_max": 0.97, "similarity_min": 0.70},
                "export": {"jsonl": True, "csv": False, "mentor_batch": False},
            },
        }
        stats = run_pipeline(cfg, dry_run=True)
        assert stats.get("dry_run") is True
        assert stats["accepted_input"] == 2
        out_jsonl = os.path.join(tmp, "s3", "augmented.jsonl")
        assert os.path.exists(out_jsonl)
        with open(out_jsonl) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2, f"dry-run should emit originals only (2), got {len(lines)}"
        assert all(l["provenance"]["stage3"]["status"] == "original" for l in lines)
        print(f"      ✓ dry run emitted {len(lines)} originals, no LLM calls")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Runner ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STAGE 3 SMOKE TEST")
    print("=" * 60)
    try:
        test_imports()
        test_parser()
        test_filter_offline()
        test_exporter()
        test_pipeline_dry_run()
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nNext step: run the real pipeline against Stage 2 output:")
    print("  python stage3_pipeline.py --limit 3    # 3-pair sanity check")
    print("  python stage3_pipeline.py              # full run")


if __name__ == "__main__":
    main()
