"""
final_exporter.py — Write Stage 3 augmented dataset to disk.
Reads settings from config["stage3"]["output_dir"] and config["stage3"]["export"].

Outputs:
  output/stage3/augmented.jsonl          — one record per accepted variation (+ original)
  output/stage3/augmented.csv            — CSV mirror (optional)
  output/stage3/augmented_rejected.jsonl — paraphrased variations dropped by the filter
  output/stage3/mentor_batch.jsonl       — optional FR021 package for mentor GPU processing
  output/stage3/stats.json               — run statistics

Each accepted record preserves the Stage 2 schema plus a Stage 3 provenance extension:
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "<rephrased question>"},
    {"role": "assistant", "content": "<original answer, unchanged>"}
  ],
  "provenance": {
    "source_file": "...",
    "chunk": "...",
    "category": "procedure",
    "page": 35,
    "stage3": {
      "status": "original" | "augmented",
      "augmented_from": "<hash of original question>",  # null for originals
      "variation_index": 0  # 0 = original, 1..N = variations
    }
  },
  "chunk_text": "<source chunk text used by evaluator>"
}

Rejected records use the same schema with status "rejected" and two extra
fields for review (mirrors Stage 2's qa_rejected.jsonl):
  "rejection_reason":  "<filter reason, e.g. 'semantic drift (sim=0.62 < 0.70)'>"
  "original_question": "<the source question this variation was derived from>"
"""
import os
import csv
import json
import hashlib
import time
from typing import List, Dict, Any, Optional


def _question_id(question: str) -> str:
    """Stable id for cross-referencing augmented→original.
    Full SHA-1 hex (40 chars). Truncation removed — 12-char ids collided
    on byte-identical questions in the HP E877 dataset (2 collisions / 589 parents).
    """
    return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()


def _build_record(messages: List[Dict], provenance: Dict,
                  status: str, augmented_from: Optional[str],
                  variation_index: int,
                  chunk_text: str = "") -> Dict[str, Any]:
    """
    Build a final record. chunk_text is preserved at the top level so the
    evaluator (NLI faithfulness) can verify answers against the source
    without re-loading Stage 1 markdown.
    """
    prov = dict(provenance)
    prov["stage3"] = {
        "status": status,
        "augmented_from": augmented_from,
        "variation_index": variation_index,
    }
    rec: Dict[str, Any] = {"messages": messages, "provenance": prov}
    if chunk_text:
        rec["chunk_text"] = chunk_text
    return rec


def _build_rejected_record(messages: List[Dict], provenance: Dict,
                           augmented_from: Optional[str],
                           original_question: str,
                           rejection_reason: str,
                           chunk_text: str = "") -> Dict[str, Any]:
    """
    Build a rejected-variation record. Uses the same schema as an accepted
    record (status "rejected") with two review fields appended: the filter
    reason and the source question the variation was derived from.
    """
    rec = _build_record(messages, provenance, "rejected",
                        augmented_from, 0, chunk_text)
    rec["rejection_reason"] = rejection_reason
    rec["original_question"] = original_question
    return rec


class FinalExporter:
    """Writes augmented dataset and per-run statistics to stage3 output dir."""

    def __init__(self, config: Dict):
        s3 = config.get("stage3", {})
        self.out_dir = s3.get("output_dir", "output/stage3")
        export = s3.get("export", {})
        self.write_jsonl = export.get("jsonl", True)
        self.write_csv = export.get("csv", False)
        self.write_rejected = export.get("rejected", True)
        self.write_mentor_batch = export.get("mentor_batch", False)
        os.makedirs(self.out_dir, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────

    def export(self, augmented_records: List[Dict[str, Any]],
               stats: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Write all configured outputs. `augmented_records` should already be
        in the final schema (one record per line).

        Returns a dict of {name: path} for files actually written.
        """
        written: Dict[str, str] = {}

        if self.write_jsonl:
            p = os.path.join(self.out_dir, "augmented.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for rec in augmented_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written["jsonl"] = p

        if self.write_csv and augmented_records:
            p = os.path.join(self.out_dir, "augmented.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "status", "variation_index", "augmented_from",
                    "system_role", "question", "answer",
                    "category", "source_file", "chunk", "page"
                ])
                for rec in augmented_records:
                    msgs = rec["messages"]
                    prov = rec["provenance"]
                    s3p = prov.get("stage3", {})
                    sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
                    q_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    a_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    w.writerow([
                        s3p.get("status", ""),
                        s3p.get("variation_index", 0),
                        s3p.get("augmented_from", "") or "",
                        sys_msg, q_msg, a_msg,
                        prov.get("category", ""),
                        prov.get("source_file", ""),
                        prov.get("chunk", ""),
                        prov.get("page", ""),
                    ])
            written["csv"] = p

        if self.write_mentor_batch and augmented_records:
            p = self._write_mentor_batch(augmented_records)
            written["mentor_batch"] = p

        if stats is not None:
            p = os.path.join(self.out_dir, "stats.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            written["stats"] = p

        return written

    def export_rejected(self, rejected_records: List[Dict[str, Any]]) -> Optional[str]:
        """
        Write rejected paraphrase variations to augmented_rejected.jsonl for
        review and tuning of the filter thresholds. Mirrors Stage 2's
        qa_rejected.jsonl. Controlled by config stage3.export.rejected (default
        True). Returns the path written, or None if disabled.
        """
        if not self.write_rejected:
            return None
        p = os.path.join(self.out_dir, "augmented_rejected.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for rec in rejected_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return p

    # ── Helpers ──────────────────────────────────────────────────────

    def _write_mentor_batch(self, records: List[Dict[str, Any]]) -> str:
        """
        FR021 — Package originals + prompt templates so the mentor can
        re-run augmentation on a GPU if local compute is insufficient.
        One line per original question; variations are NOT included so
        the mentor's run produces fresh alternatives.
        """
        p = os.path.join(self.out_dir, "mentor_batch.jsonl")
        seen = set()
        with open(p, "w", encoding="utf-8") as f:
            for rec in records:
                s3p = rec["provenance"].get("stage3", {})
                if s3p.get("status") != "original":
                    continue
                msgs = rec["messages"]
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                if not q or q in seen:
                    continue
                seen.add(q)
                batch_rec = {
                    "qid": _question_id(q),
                    "original_question": q,
                    "original_answer": next((m["content"] for m in msgs
                                             if m["role"] == "assistant"), ""),
                    "provenance": rec["provenance"],
                    "chunk_text": rec.get("chunk_text", ""),
                }
                f.write(json.dumps(batch_rec, ensure_ascii=False) + "\n")
        return p


# ── Functional wrapper for Stage 3 pipeline use ──────────────────────

def export_augmented_dataset(
    accepted_stage2_records: List[Dict[str, Any]],
    augmentations: List[Dict[str, Any]],
    config: Dict,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Build the full records list (originals + augmentations) and delegate to
    FinalExporter.export. Both inputs are already in JSONL dict form.

    `augmentations` is expected to be a list of records already in the
    final schema (status='augmented', augmented_from set, chunk_text set, etc).

    Originals are finalised here: a stage3 block is added and chunk_text
    is passed through from the Stage 2 input record.
    """
    exporter = FinalExporter(config)
    finalised_originals = []
    for rec in accepted_stage2_records:
        msgs = rec["messages"]
        finalised = _build_record(
            msgs, rec.get("provenance", {}),
            status="original",
            augmented_from=None,
            variation_index=0,
            chunk_text=rec.get("chunk_text", ""),
        )
        finalised_originals.append(finalised)
    all_records = finalised_originals + augmentations
    return exporter.export(all_records, stats)


# Re-export helper for external use
__all__ = ["FinalExporter", "export_augmented_dataset", "_question_id", "_build_record"]


if __name__ == "__main__":
    # Minimal smoke test
    demo_rec = {
        "messages": [
            {"role": "system",    "content": "You are a helpful HP printer technician."},
            {"role": "user",      "content": "How do I replace the toner cartridge?"},
            {"role": "assistant", "content": "Open the front door, remove the old cartridge, insert the new one, and close the door."},
        ],
        "provenance": {
            "source_file": "hp-e877.md",
            "chunk": "Replace toner cartridge (chunk 0)",
            "category": "procedure",
            "page": 35,
        },
        "chunk_text": "To replace the toner cartridge, open the front door, grip the cartridge handle and pull straight out, unpack the new cartridge and remove protective tape, then align with guides and push firmly until it clicks.",
    }
    cfg = {"stage3": {"output_dir": "/tmp/stage3_smoke",
                      "export": {"jsonl": True, "csv": True, "mentor_batch": True}}}
    paths = export_augmented_dataset([demo_rec], [], cfg, stats={"demo": True})
    print("Wrote:", paths)
