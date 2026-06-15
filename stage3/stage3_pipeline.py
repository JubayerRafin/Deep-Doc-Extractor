"""
stage3_pipeline.py — Stage 3 Orchestrator: Augment accepted Q&A dataset.

Reads: output/stage2/qa_pairs.jsonl  (from Stage 2)
Writes: output/stage3/augmented.jsonl
        output/stage3/augmented.csv           (if enabled)
        output/stage3/mentor_batch.jsonl      (if enabled — FR021)
        output/stage3/stats.json

Each accepted Stage 2 pair yields:
  - 1 "original" record (unchanged, marked status=original)
  - 0..N "augmented" records (status=augmented, same answer,
    paraphrased question, augmented_from = hash of original question)

Usage:
    python stage3_pipeline.py                    # uses config.yaml
    python stage3_pipeline.py --config path.yaml # custom config
    python stage3_pipeline.py --dry-run          # no LLM calls, skeleton export
    python stage3_pipeline.py --limit 5          # only process first 5 pairs
"""
import os
import sys
import json
import time
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import yaml

from llm_rephraser import LLMRephraser
from augment_filter import AugmentFilter
from final_exporter import (
    FinalExporter, _question_id, _build_record, _build_rejected_record,
)


# ── Path helpers ─────────────────────────────────────────────────────

def _resolve_paths(config: Dict) -> Tuple[str, str]:
    """Return (input_jsonl, stage3_output_dir)."""
    s2 = config.get("stage2", {})
    s3 = config.get("stage3", {})
    s2_dir = s2.get("output_dir", "output/stage2")
    s3_dir = s3.get("output_dir", "output/stage3")
    input_jsonl = s3.get("input_jsonl") or os.path.join(s2_dir, "qa_pairs.jsonl")
    return input_jsonl, s3_dir


def _load_stage2_records(path: str) -> List[Dict[str, Any]]:
    """Read Stage 2 qa_pairs.jsonl and return list of records."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 2 output not found: {path}\n"
            f"Run Stage 2 first (python stage2_pipeline.py)."
        )
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] line {i} in {path} is not valid JSON: {e}")
    return records


def _extract_question(rec: Dict[str, Any]) -> str:
    msgs = rec.get("messages", [])
    return next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")


# ── Pipeline ─────────────────────────────────────────────────────────

def run_pipeline(config: Dict, dry_run: bool = False,
                 limit: Optional[int] = None) -> Dict[str, Any]:
    """Run the Stage 3 pipeline end-to-end. Returns stats dict."""
    input_jsonl, s3_dir = _resolve_paths(config)
    os.makedirs(s3_dir, exist_ok=True)

    print("=" * 60)
    print("STAGE 3: Dataset Augmentation")
    print("=" * 60)

    # --- 1. Load accepted Stage 2 records ---
    print(f"\n[1/5] Loading: {input_jsonl}")
    records = _load_stage2_records(input_jsonl)
    if limit:
        records = records[:limit]
        print(f"      → --limit {limit} applied")
    print(f"      → {len(records)} accepted Q&A pairs")

    if not records:
        print("\n[Stage 3] No Stage 2 records to augment. Aborting.")
        return {"accepted_input": 0, "augmented_out": 0}

    # Category breakdown (useful for the presentation)
    cat_counter = Counter(
        r.get("provenance", {}).get("category", "unknown") for r in records
    )
    for cat, n in cat_counter.most_common():
        print(f"      {cat:12s}: {n}")

    # --- 2. Initialise rephraser and filter ---
    print(f"\n[2/5] Initialising components...")
    rephraser = LLMRephraser(config)
    aug_filter = AugmentFilter(config)
    print(f"      LLM       : {rephraser.model} @ {rephraser.base_url}")
    print(f"      n per Q&A : {rephraser.n_variations}")
    print(f"      sim bounds: [{aug_filter.sim_min}, {aug_filter.sim_max}]")

    if dry_run:
        print("\n[DRY RUN] Skipping LLM calls. Writing originals-only export.")
        stats = {"dry_run": True, "accepted_input": len(records),
                 "augmented_out": 0}
        exporter = FinalExporter(config)
        originals = []
        for r in records:
            rec_out = _build_record(r["messages"], r.get("provenance", {}),
                                    "original", None, 0)
            rec_out["chunk_text"] = r.get("chunk_text", "")
            originals.append(rec_out)
        written = exporter.export(originals, stats)
        for name, path in written.items():
            print(f"      → {name}: {path}")
        return stats

    # --- 3. Rephrase + filter loop ---
    print(f"\n[3/5] Generating paraphrased variations...")
    t0 = time.time()
    last_t = t0
    augmented_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    variations_kept = 0
    variations_dropped = 0
    dropped_reasons: Counter = Counter()
    llm_failures = 0

    for idx, rec in enumerate(records):
        q = _extract_question(rec).strip()
        if not q:
            continue

        # Progress line every 10 items (or first item)
        if idx % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(records) - idx - 1) / rate if rate > 0 else 0
            item_dt = (time.time() - last_t) / 10 if idx > 0 else 0
            last_t = time.time()
            print(f"      {idx+1}/{len(records)}  "
                  f"({variations_kept} kept, ~{eta:.0f}s left, "
                  f"{item_dt:.1f}s/item)")

        # Generate variations
        try:
            variations = rephraser.rephrase(q)
        except Exception as e:
            print(f"      [WARN] rephrase failed for pair {idx}: {e}")
            llm_failures += 1
            variations = []

        if not variations:
            llm_failures += 1
            continue

        # Filter variations; retain both kept and dropped for export
        kept, dropped = aug_filter.filter(q, variations)
        variations_dropped += len(dropped)
        orig_qid = _question_id(q)
        for variation, reason in dropped:
            dropped_reasons[reason.split("(")[0].strip()] += 1
            rej_msgs = [
                m if m.get("role") != "user" else {"role": "user", "content": variation}
                for m in rec["messages"]
            ]
            rejected_records.append(_build_rejected_record(
                messages=rej_msgs,
                provenance=rec.get("provenance", {}),
                augmented_from=orig_qid,
                original_question=q,
                rejection_reason=reason,
                chunk_text=rec.get("chunk_text", ""),
            ))

        # Emit one augmented record per kept variation
        for var_idx, variation in enumerate(kept, start=1):
            new_msgs = [
                m if m.get("role") != "user" else {"role": "user", "content": variation}
                for m in rec["messages"]
            ]
            aug_rec = _build_record(
                messages=new_msgs,
                provenance=rec.get("provenance", {}),
                status="augmented",
                augmented_from=orig_qid,
                variation_index=var_idx,
            )
            aug_rec["chunk_text"] = rec.get("chunk_text", "")
            augmented_records.append(aug_rec)
            variations_kept += 1

    elapsed = time.time() - t0
    print(f"      → {variations_kept} kept, {variations_dropped} dropped "
          f"in {elapsed:.1f}s  ({llm_failures} LLM failures)")
    for reason, n in dropped_reasons.most_common():
        print(f"        - {reason}: {n}")

    # --- 4. Assemble and export ---
    print(f"\n[4/5] Exporting...")
    exporter = FinalExporter(config)
    originals = []
    for r in records:
        rec_out = _build_record(r["messages"], r.get("provenance", {}),
                                "original", None, 0)
        rec_out["chunk_text"] = r.get("chunk_text", "")
        originals.append(rec_out)

    all_records = originals + augmented_records

    stats = {
        "input_stage2_pairs": len(records),
        "original_records_written": len(originals),
        "augmented_records_written": len(augmented_records),
        "total_records_written": len(all_records),
        "variations_kept": variations_kept,
        "variations_dropped": variations_dropped,
        "rejected_records_written": len(rejected_records),
        "llm_failures": llm_failures,
        "dropped_reasons": dict(dropped_reasons),
        "category_distribution": dict(cat_counter),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "model": rephraser.model,
            "variations_per_qa": rephraser.n_variations,
            "sim_max": aug_filter.sim_max,
            "sim_min": aug_filter.sim_min,
            "temperature": rephraser.temperature,
        },
    }
    if len(originals) > 0:
        stats["multiplier"] = round(len(all_records) / len(originals), 2)

    written = exporter.export(all_records, stats)
    rejected_path = exporter.export_rejected(rejected_records)
    if rejected_path:
        written["rejected"] = rejected_path
    for name, path in written.items():
        print(f"      → {name:14s}: {path}")

    # --- 5. Summary ---
    print(f"\n[5/5] Summary")
    print(f"      Accepted input pairs : {len(records)}")
    print(f"      Originals written    : {len(originals)}")
    print(f"      Augmented written    : {len(augmented_records)}")
    print(f"      Rejected written     : {len(rejected_records)}")
    print(f"      Total records        : {len(all_records)}")
    if len(originals) > 0:
        print(f"      Multiplier           : {stats['multiplier']}×")

    print("\n" + "=" * 60)
    print(f"STAGE 3 COMPLETE  |  {len(all_records)} records  |  {elapsed:.1f}s")
    print("=" * 60)

    return stats


# ── Entry point ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Stage 3: Dataset Augmentation")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip LLM calls; write originals-only export.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N accepted pairs (debug/demo).")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_pipeline(config, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
