# scripts/

Helper scripts that don't fit in any single stage.

## `switch_experiment.py`

One-command config switcher for multi-model experiments. Updates 9 fields in `config.yaml` atomically so teammates don't have to manually edit YAML.

**Usage:**
```bash
python3 scripts/switch_experiment.py --tag <tag> --model <ollama-model>
```

**Examples for each team member:**

| Member | Command |
|---|---|
| Jubayer (9B baseline) | `python3 scripts/switch_experiment.py --tag qwen35_9b_gpu --model qwen3.5:9b` |
| Jubayer (27B) | `python3 scripts/switch_experiment.py --tag qwen36_27b --model qwen3.6:27b` |
| Arlen | `python3 scripts/switch_experiment.py --tag qwen35_35b --model qwen3.5:35b` |
| Sangwook | `python3 scripts/switch_experiment.py --tag mistral_24b --model mistral-small3.2:24b` |
| Inyoung / Sungho | `python3 scripts/switch_experiment.py --tag gemma4_26b --model gemma4:26b` |

After running, your output files will be namespaced:
```
output/stage2_<tag>/qa_pairs.jsonl
output/stage3_<tag>/augmented.jsonl
output/stage4_<tag>/chroma/
```

This means **multiple team members can keep their results side-by-side** without overwriting each other.

**Flags:**
- `--dry-run` — Show what would change without writing
- `--no-backup` — Skip creating timestamped `.bak.YYYYMMDD_HHMMSS` backup
- `--config path/to/config.yaml` — Use a non-default config file
