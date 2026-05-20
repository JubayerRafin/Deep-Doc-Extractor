"""
chunker.py — Split ContentBlocks into LLM-sized chunks.
Reads settings from config["stage2"]["chunking"].

CHANGES (vs original):
  - Fix #3: Carry `heading_path` from ContentBlock into Chunk.
"""
import re, json
from typing import List, Dict
from dataclasses import dataclass, field
from markdown_parser import ContentBlock

@dataclass
class Chunk:
    text: str
    heading: str
    category: str
    images: List[str] = field(default_factory=list)
    source_file: str = ""
    page_hint: int = 0
    chunk_index: int = 0
    heading_path: str = ""           # <-- FIX #3: full parent chain

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def _est_tokens(text: str) -> int:
    return len(text) // 4

def chunk_block(block: ContentBlock, category: str, config: Dict) -> List[Chunk]:
    cfg = config.get("stage2", {}).get("chunking", {})
    max_tok = cfg.get("max_tokens", 400)
    overlap = cfg.get("overlap_sentences", 1)
    min_len = cfg.get("min_chunk_length", 50)

    body = block.body.strip()
    if len(body) < min_len:
        return []

    # Use heading_path if it exists, else fall back to heading (FIX #3)
    hpath = block.heading_path or block.heading

    if _est_tokens(body) <= max_tok:
        return [Chunk(body, block.heading, category, block.images,
                       block.source_file, block.page_hint or 0, 0,
                       heading_path=hpath)]                    # <-- FIX #3

    sents = _split_sentences(body)
    if not sents: return []

    chunks, idx, i = [], 0, 0
    while i < len(sents):
        cur, tok = [], 0
        while i < len(sents):
            st = _est_tokens(sents[i])
            if tok + st > max_tok and cur: break
            cur.append(sents[i]); tok += st; i += 1
        chunks.append(Chunk(" ".join(cur), block.heading, category,
                            block.images,
                            block.source_file, block.page_hint or 0, idx,
                            heading_path=hpath))               # <-- FIX #3
        idx += 1
        if overlap > 0 and i < len(sents):
            i = max(i - overlap, i - len(cur) + 1)
    return chunks

def chunk_all(classified_blocks: List[tuple], config: Dict) -> List[Chunk]:
    out = []
    for block, cat in classified_blocks:
        if block.block_type == "table" and block.table_data:
            tbl_text = f"Section: {block.heading}\nTable data:\n{json.dumps(block.table_data, indent=2, ensure_ascii=False)}"
            # Preserve heading_path on the synthetic table block (FIX #3)
            b2 = ContentBlock(block.heading, tbl_text, "table", [], block.table_data,
                              block.page_hint, block.source_file,
                              heading_path=block.heading_path)
            out.extend(chunk_block(b2, cat, config))
        else:
            out.extend(chunk_block(block, cat, config))
    return out

if __name__ == "__main__":
    import yaml, sys
    from markdown_parser import parse_markdown
    from semantic_classifier import classify_blocks
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f: config = yaml.safe_load(f)
    s1 = config.get("stage1", {})
    md = config["stage2"].get("input_md") or f"{s1['output_dir']}/hp-e877-series-user-guide.md"
    blocks = parse_markdown(md, f"{s1['output_dir']}/{s1.get('tables',{}).get('output_subdir','tables')}")
    chunks = chunk_all(classify_blocks(blocks, config), config)
    print(f"{len(chunks)} chunks from {len(blocks)} blocks")
    for c in chunks[:5]:
        # Show heading_path now
        print(f"  [{c.chunk_index}] {c.category:12s} | {c.heading_path[:50]:50s} | {c.text[:50].replace(chr(10),' ')}...")
