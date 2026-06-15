"""
markdown_parser.py — Parse Stage 1 Markdown into structured content blocks.

CHANGES (vs original):
  - Fix #3: Track heading hierarchy. Each ContentBlock now has a
            `heading_path` field showing the parent chain
            (e.g., "Scan to SharePoint > Before you begin").
            This disambiguates sections that share the same leaf title.
"""
import re, json, os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ContentBlock:
    heading: str
    body: str
    block_type: str = "text"              # text | image | table
    images: List[str] = field(default_factory=list)
    table_data: Optional[dict] = None
    page_hint: Optional[int] = None
    source_file: str = ""
    heading_path: str = ""                # <-- FIX #3: parent > self chain


def _table_heading(td: dict, fallback: str) -> str:
    """
    Derive a meaningful heading for a table block.
    Priority: table headers > fallback heading (if not Untitled).
    """
    if isinstance(td, dict):
        headers = td.get("headers", [])
        if headers and len(headers) >= 2:
            return " / ".join(str(h) for h in headers[:3])
        rows = td.get("rows", [])
        if rows and isinstance(rows[0], dict):
            return " / ".join(list(rows[0].keys())[:3])
    if fallback and fallback != "Untitled":
        return fallback
    return "Untitled"


def parse_markdown(md_path: str, tables_dir: str = "") -> List[ContentBlock]:
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks: List[ContentBlock] = []
    cur_heading = "Untitled"
    cur_body: List[str] = []
    cur_images: List[str] = []
    cur_page: Optional[int] = None
    src = os.path.basename(md_path)

    # === FIX #3: track full heading stack by depth (1..6) ===
    # heading_stack[i] = the most recent heading at level i (1-based: 1..6)
    heading_stack: List[Optional[str]] = [None] * 7   # index 0 unused
    cur_level: int = 0

    def _heading_path() -> str:
        """Build 'Top > Mid > Leaf' path from the current stack."""
        parts = [h for h in heading_stack[1:] if h]
        return " > ".join(parts)

    heading_re    = re.compile(r"^(#{1,4})\s+(.+)")
    image_re      = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    page_re       = re.compile(r"<!--\s*page[:\s]*(\d+)\s*-->", re.I)
    img_page_re   = re.compile(r"image_p(\d+)")
    json_start_re = re.compile(r"^```json\s*$", re.I)
    json_end_re   = re.compile(r"^```\s*$")
    tbl_ref_re    = re.compile(r"\[Table[:\s]*([^\]]*)\]\(([^)]+\.json)\)")
    json_obj_re   = re.compile(r"^\{")

    in_json = False
    json_buf: List[str] = []
    in_inline_json = False
    inline_json_buf: List[str] = []
    brace_depth = 0

    def _flush():
        body = "\n".join(cur_body).strip()
        if body or cur_images:
            blocks.append(ContentBlock(
                cur_heading, body, "text",
                list(cur_images), None, cur_page, src,
                heading_path=_heading_path(),     # <-- FIX #3
            ))

    def _make_table_block(td):
        heading = _table_heading(td, cur_heading)
        # For tables, extend the path with the table heading if it differs
        path = _heading_path()
        if heading != cur_heading and path:
            path = f"{path} > {heading}"
        elif heading != cur_heading:
            path = heading
        blocks.append(ContentBlock(
            heading, f"[Table under: {heading}]",
            "table", [], td, cur_page, src,
            heading_path=path,                    # <-- FIX #3
        ))

    for raw_line in lines:
        raw = raw_line.rstrip("\n").rstrip("\r")

        # Page comment markers
        pm = page_re.search(raw)
        if pm:
            cur_page = int(pm.group(1))
            continue

        # Extract page number from image references
        im_page = img_page_re.search(raw)
        if im_page:
            cur_page = int(im_page.group(1))

        # Fenced JSON block (```json ... ```)
        if json_start_re.match(raw.strip()):
            in_json = True
            json_buf = []
            continue
        if in_json:
            if json_end_re.match(raw.strip()):
                in_json = False
                try:
                    td = json.loads("\n".join(json_buf))
                except Exception:
                    td = {"raw": "\n".join(json_buf)}
                _make_table_block(td)
            else:
                json_buf.append(raw)
            continue

        # Inline JSON object
        if not in_inline_json and json_obj_re.match(raw.strip()):
            in_inline_json = True
            inline_json_buf = [raw]
            brace_depth = raw.count("{") - raw.count("}")
            if brace_depth <= 0:
                in_inline_json = False
                try:
                    td = json.loads("\n".join(inline_json_buf))
                    if isinstance(td, dict) and ("table_index" in td or "headers" in td or "rows" in td):
                        _make_table_block(td)
                        continue
                except Exception:
                    pass
                cur_body.append(raw)
            continue
        if in_inline_json:
            inline_json_buf.append(raw)
            brace_depth += raw.count("{") - raw.count("}")
            if brace_depth <= 0:
                in_inline_json = False
                try:
                    td = json.loads("\n".join(inline_json_buf))
                    if isinstance(td, dict) and ("table_index" in td or "headers" in td or "rows" in td):
                        _make_table_block(td)
                    else:
                        cur_body.extend(inline_json_buf)
                except Exception:
                    cur_body.extend(inline_json_buf)
            continue

        # Heading
        hm = heading_re.match(raw)
        if hm:
            _flush()
            cur_body = []
            cur_images = []
            cur_heading = hm.group(2).strip()

            # === FIX #3: update the heading stack at this depth ===
            level = len(hm.group(1))                  # 1..4 for # to ####
            cur_level = level
            heading_stack[level] = cur_heading
            # Clear any deeper levels (they belong to the previous parent)
            for i in range(level + 1, 7):
                heading_stack[i] = None
            continue

        # Image reference
        im = image_re.search(raw)
        if im:
            cur_images.append(im.group(2))
            cur_body.append(raw)
            continue

        # Table file reference
        tm = tbl_ref_re.search(raw)
        if tm and tables_dir:
            tp = os.path.join(tables_dir, os.path.basename(tm.group(2)))
            if os.path.exists(tp):
                try:
                    with open(tp, "r", encoding="utf-8") as tf:
                        td = json.load(tf)
                        _make_table_block(td)
                    continue
                except Exception:
                    pass
            cur_body.append(raw)
            continue

        cur_body.append(raw)

    _flush()

    # ── Backfill remaining "Untitled" headings ──
    for i in range(len(blocks)):
        if blocks[i].heading != "Untitled":
            continue

        # Try backward: nearest preceding block with a real heading
        for j in range(i - 1, -1, -1):
            if blocks[j].heading != "Untitled":
                blocks[i].heading = blocks[j].heading
                # FIX #3: also inherit the heading path
                if not blocks[i].heading_path:
                    blocks[i].heading_path = blocks[j].heading_path
                break

        # Still untitled? Try forward
        if blocks[i].heading == "Untitled":
            for j in range(i + 1, len(blocks)):
                if blocks[j].heading != "Untitled":
                    blocks[i].heading = blocks[j].heading
                    if not blocks[i].heading_path:
                        blocks[i].heading_path = blocks[j].heading_path
                    break

        # Last resort: extract from body (skip markup)
        if blocks[i].heading == "Untitled" and blocks[i].body:
            first_line = blocks[i].body.strip().split("\n")[0].strip()
            first_line = re.sub(r"^\*\*(.+?)\*\*", r"\1", first_line)
            first_line = re.sub(r"!\[.*?\]\(.*?\)", "", first_line).strip()
            first_line = re.sub(r"\[Table.*?\]", "", first_line).strip()
            if len(first_line) > 5:
                blocks[i].heading = first_line[:80]
                if not blocks[i].heading_path:
                    blocks[i].heading_path = blocks[i].heading

    return blocks

if __name__ == "__main__":
    import sys
    md = sys.argv[1] if len(sys.argv) > 1 else "output/stage1/hp-e877-series-user-guide.md"
    bs = parse_markdown(md)
    print(f"Parsed {len(bs)} blocks")
    for i, b in enumerate(bs[:12]):
        pg = f"p{b.page_hint}" if b.page_hint else "p?"
        # Show heading_path now
        print(f"  [{i}] {b.block_type:6s} | {pg:>5} | {b.heading_path[:60]:60s} | {b.body[:40].replace(chr(10),' ')}...")
