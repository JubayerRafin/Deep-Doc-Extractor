#!/usr/bin/env python3
"""
switch_experiment.py — One-command config switcher for multi-model experiments.

Each team member runs a different LLM on the same code. This script updates
config.yaml in one shot so nobody has to manually edit 8 different fields.

Usage:
    # Jubayer (clean baseline on GPU)
    python3 scripts/switch_experiment.py --tag qwen35_9b_gpu --model qwen3.5:9b

    # Jubayer (then 27B run)
    python3 scripts/switch_experiment.py --tag qwen36_27b --model qwen3.6:27b

    # Arlen
    python3 scripts/switch_experiment.py --tag qwen35_35b --model qwen3.5:35b

    # Sangwook
    python3 scripts/switch_experiment.py --tag mistral_24b --model mistral-small3.2:24b

    # Inyoung / Sungho
    python3 scripts/switch_experiment.py --tag gemma4_26b --model gemma4:26b

After running, the following fields are updated atomically:
    - experiment.tag
    - stage2.llm.model
    - stage2.output_dir          → output/stage2_<tag>/
    - stage3.llm.model
    - stage3.output_dir          → output/stage3_<tag>/
    - stage4.llm.model
    - stage4.output_dir          → output/stage4_<tag>/
    - stage4.chroma.persist_dir  → output/stage4_<tag>/chroma/
"""
import argparse
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run:  pip install pyyaml")
    sys.exit(1)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str) -> None:
    """Write config preserving YAML key order. We use safe_dump (not dump)
    so we don't introduce Python-specific tags."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config, f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )


def backup_config(path: str) -> str:
    """Save a timestamped backup before mutating config.yaml."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak.{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def update_config(config: dict, tag: str, model: str) -> dict:
    """
    Update all model-related fields in one pass.
    Returns the updated config (modifies in place).
    """
    # 1. experiment.tag
    config.setdefault("experiment", {})["tag"] = tag

    # 2. stage2 — model + output dir
    s2 = config.setdefault("stage2", {})
    s2.setdefault("llm", {})["model"] = model
    s2["output_dir"] = f"output/stage2_{tag}"

    # 3. stage3 — model (and fallback) + output dir
    s3 = config.setdefault("stage3", {})
    s3.setdefault("llm", {})["model"] = model
    s3["llm"]["fallback_model"] = model
    s3["output_dir"] = f"output/stage3_{tag}"

    # 4. stage4 — model + output dir + chroma persist dir
    s4 = config.setdefault("stage4", {})
    s4.setdefault("llm", {})["model"] = model
    s4["output_dir"] = f"output/stage4_{tag}"
    s4.setdefault("chroma", {})["persist_dir"] = f"output/stage4_{tag}/chroma"
    s4.setdefault("offline", {})["log_path"] = f"output/stage4_{tag}/offline_guard.log"

    return config


def print_summary(config: dict) -> None:
    """Show the user exactly what changed so they can verify."""
    exp_tag = config.get("experiment", {}).get("tag", "?")
    print("=" * 60)
    print(f"Experiment configured: {exp_tag}")
    print("=" * 60)
    rows = [
        ("experiment.tag",            config["experiment"]["tag"]),
        ("stage2.llm.model",          config["stage2"]["llm"]["model"]),
        ("stage2.output_dir",         config["stage2"]["output_dir"]),
        ("stage3.llm.model",          config["stage3"]["llm"]["model"]),
        ("stage3.llm.fallback_model", config["stage3"]["llm"]["fallback_model"]),
        ("stage3.output_dir",         config["stage3"]["output_dir"]),
        ("stage4.llm.model",          config["stage4"]["llm"]["model"]),
        ("stage4.output_dir",         config["stage4"]["output_dir"]),
        ("stage4.chroma.persist_dir", config["stage4"]["chroma"]["persist_dir"]),
    ]
    for k, v in rows:
        print(f"  {k:32s} = {v}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(
        description="Update config.yaml for a multi-model experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--tag", required=True,
                    help="Experiment tag, e.g. 'qwen35_9b_gpu', 'mistral_24b'. "
                         "Used in output folder names.")
    ap.add_argument("--model", required=True,
                    help="Ollama model name, e.g. 'qwen3.5:9b', 'mistral-small3.2:24b'.")
    ap.add_argument("--config", default="config.yaml",
                    help="Path to config.yaml (default: config.yaml in cwd)")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip creating a timestamped backup of config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing the file")
    args = ap.parse_args()

    # Validate tag format (alphanumeric + underscore + dash only)
    bad_chars = set(args.tag) - set("abcdefghijklmnopqrstuvwxyz"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "0123456789_-")
    if bad_chars:
        print(f"ERROR: --tag contains invalid characters: {bad_chars}")
        print("       Use only letters, digits, underscore, and dash.")
        sys.exit(1)

    # Load
    config = load_config(args.config)

    # Update
    update_config(config, args.tag, args.model)

    # Print summary
    print_summary(config)

    if args.dry_run:
        print("\n[DRY RUN] No file written. Re-run without --dry-run to apply.")
        return

    # Backup
    if not args.no_backup:
        backup_path = backup_config(args.config)
        print(f"\nBackup saved: {backup_path}")

    # Save
    save_config(config, args.config)
    print(f"Updated:      {args.config}")

    # Hint
    print("\nNext step — verify your model is pulled:")
    print(f"  ollama list | grep {args.model.split(':')[0]}")
    print("\nThen run Stage 2:")
    print(f"  nohup python3 -u stage2/stage2_pipeline.py > ~/stage2_run.log 2>&1 &")


if __name__ == "__main__":
    main()
