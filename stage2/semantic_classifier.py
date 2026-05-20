"""
semantic_classifier.py — Classify ContentBlocks into schema categories.
Categories: procedure, spec, rule_error, figure
"""
import re
from typing import List, Dict
from markdown_parser import ContentBlock

CATEGORIES = ("procedure", "spec", "rule_error", "figure")

def classify_block(block: ContentBlock, config: Dict) -> str:
    s2 = config.get("stage2", {})
    clf = s2.get("classifier", {})
    proc_kw = [k.lower() for k in clf.get("procedure_keywords", [])]
    spec_kw = [k.lower() for k in clf.get("spec_keywords", [])]
    rule_kw = [k.lower() for k in clf.get("rule_error_keywords", [])]

    if block.block_type == "table":
        return "spec"

    text_len = len(block.body.replace("\n", " ").strip())
    if len(block.images) > 0 and text_len < 30:
        return "figure"

    searchable = (block.heading + " " + block.body).lower()

    def _score(kws):
        score = 0
        for k in kws:
            # Multi-word keywords: substring match (intentional)
            if " " in k:
                if k in searchable:
                    score += 1
            # Single word: word-boundary match
            else:
                if re.search(r'\b' + re.escape(k) + r'\b', searchable):
                    score += 1
        return score

    scores = {"procedure": _score(proc_kw), "rule_error": _score(rule_kw), "spec": _score(spec_kw)}
    scores["procedure"] += len(re.findall(r"^\s*\d+\.\s", block.body, re.M)) * 2
    if re.match(r"^\s*\*?\*?(CAUTION|WARNING|NOTE|IMPORTANT)", block.body, re.I):
        scores["rule_error"] += 5

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "procedure"

def classify_blocks(blocks: List[ContentBlock], config: Dict) -> List[tuple]:
    return [(b, classify_block(b, config)) for b in blocks]

if __name__ == "__main__":
    import yaml, sys
    from markdown_parser import parse_markdown
    from collections import Counter
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f: config = yaml.safe_load(f)
    s1 = config.get("stage1", {})
    md = config["stage2"].get("input_md") or f"{s1['output_dir']}/hp-e877-series-user-guide.md"
    blocks = parse_markdown(md, f"{s1['output_dir']}/{s1.get('tables',{}).get('output_subdir','tables')}")
    classified = classify_blocks(blocks, config)
    for cat, n in Counter(c for _, c in classified).most_common():
        print(f"  {cat:12s}: {n}")
