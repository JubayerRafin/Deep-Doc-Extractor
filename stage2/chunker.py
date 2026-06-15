"""
chunker.py — Split ContentBlocks into LLM-sized chunks.
Reads settings from config["stage2"]["chunking"].
 
CHANGES (vs original):
  - Fix #3: Carry `heading_path` from ContentBlock into Chunk.
  - Fix #4: TOC-aware splitting. Dot-leaders (e.g. "1 Printer overview...1")
    have no whitespace, so the old regex treated the whole TOC as one
    sentence and bypassed the token cap. Now splits on dot-leaders AND
    drops lines that are mostly dots.
  - Fix #5: Hard character cap. Force-split any chunk over CHAR_HARD_CAP
    at character boundary — safety net so we never emit a 69 KB chunk.
"""
import re, json
from typing import List, Dict
from dataclasses import dataclass, field
from markdown_parser import ContentBlock
 
 
# Lines like "Printer views...........................2"
TOC_DOT_LEADER = re.compile(r'\.{4,}')
 
# Safety net: no chunk may exceed this many characters under any circumstances
CHAR_HARD_CAP = 2000
 
 
@dataclass
class Chunk:
    text: str
    heading: str
    category: str
    images: List[str] = field(default_factory=list)
    source_file: str = ""
    page_hint: int = 0
    chunk_index: int = 0
    heading_path: str = ""
 
 
def _is_toc_line(line: str) -> bool:
    """A line is a TOC entry if dots are >25% of its characters."""
    line = line.strip()
    if len(line) < 10:
        return False
    return line.count('.') / len(line) > 0.25
 
 
def _strip_toc_lines(text: str) -> str:
    """Drop TOC dot-leader lines while keeping real prose."""
    return "\n".join(ln for ln in text.split("\n") if not _is_toc_line(ln))
 
 
def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
 
    Splits on:
      - `.`/`!`/`?` followed by whitespace (normal prose)
      - Dot-leaders of 4+ consecutive dots (TOC, even without trailing space)
      - Hard newlines (paragraph breaks)
    """
    # Replace dot-leaders with a single period + space so the next split works
    text = TOC_DOT_LEADER.sub('. ', text)
    # Normalise paragraph breaks into sentence boundaries
    text = re.sub(r'\n+', '. ', text)
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]
 
 
def _est_tokens(text: str) -> int:
    return len(text) // 4
 
 
def _force_split(text: str, max_chars: int = CHAR_HARD_CAP) -> List[str]:
    """Fallback: split at char boundary when a sentence has no breakpoints."""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    i = 0
    while i < len(text):
        pieces.append(text[i:i + max_chars])
        i += max_chars
    return pieces
 
 
def chunk_block(block: ContentBlock, category: str, config: Dict) -> List[Chunk]:
    cfg = config.get("stage2", {}).get("chunking", {})
    max_tok = cfg.get("max_tokens", 400)
    overlap = cfg.get("overlap_sentences", 1)
    min_len = cfg.get("min_chunk_length", 50)
 
    # Fix #4: drop TOC dot-leader lines before doing anything
    body = _strip_toc_lines(block.body.strip())
    if len(body) < min_len:
        return []
 
    # Use heading_path if it exists, else fall back to heading
    hpath = block.heading_path or block.heading
 
    # Single-chunk fast path
    if _est_tokens(body) <= max_tok and len(body) <= CHAR_HARD_CAP:
        return [Chunk(body, block.heading, category, block.images,
                      block.source_file, block.page_hint or 0, 0,
                      heading_path=hpath)]
 
    sents = _split_sentences(body)
    if not sents:
        return []
 
    # Fix #5: any monster sentence (e.g. table dump with no period) gets
    # force-split at character boundary so it never poisons a chunk
    safe_sents: List[str] = []
    for s in sents:
        if len(s) > CHAR_HARD_CAP:
            safe_sents.extend(_force_split(s, CHAR_HARD_CAP))
        else:
            safe_sents.append(s)
    sents = safe_sents
 
    chunks, idx, i = [], 0, 0
    while i < len(sents):
        cur, tok, char_len = [], 0, 0
        while i < len(sents):
            st = _est_tokens(sents[i])
            sl = len(sents[i])
            # Stop if either token budget OR char hard-cap would be exceeded
            if cur and (tok + st > max_tok or char_len + sl > CHAR_HARD_CAP):
                break
            cur.append(sents[i])
            tok += st
            char_len += sl + 1   # +1 for space
            i += 1
        if not cur:
            # Shouldn't happen after force-split, but guard anyway
            break
        chunks.append(Chunk(" ".join(cur), block.heading, category,
                            block.images, block.source_file,
                            block.page_hint or 0, idx, heading_path=hpath))
        idx += 1
        # Sentence overlap (NOT page rewind)
        if overlap > 0 and i < len(sents):
            i = max(i - overlap, i - len(cur) + 1)
    return chunks
 
 
def chunk_all(classified_blocks: List[tuple], config: Dict) -> List[Chunk]:
    out = []
    for block, cat in classified_blocks:
        if block.block_type == "table" and block.table_data:
            tbl_text = (f"Section: {block.heading}\nTable data:\n"
                        f"{json.dumps(block.table_data, indent=2, ensure_ascii=False)}")
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
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    s1 = config.get("stage1", {})
    md = config["stage2"].get("input_md") or f"{s1['output_dir']}/hp-e877-series-user-guide.md"
    blocks = parse_markdown(md, f"{s1['output_dir']}/{s1.get('tables', {}).get('output_subdir', 'tables')}")
    chunks = chunk_all(classify_blocks(blocks, config), config)
    print(f"{len(chunks)} chunks from {len(blocks)} blocks")
    sizes = sorted([len(c.text) for c in chunks])
    n = len(sizes)
    print(f"Chunk size distribution: min={sizes[0]}, p25={sizes[n//4]}, median={sizes[n//2]}, p75={sizes[3*n//4]}, max={sizes[-1]}")
    for c in chunks[:5]:
        print(f"  [{c.chunk_index}] {c.category:12s} | {c.heading_path[:50]:50s} | {c.text[:50].replace(chr(10), ' ')}...")