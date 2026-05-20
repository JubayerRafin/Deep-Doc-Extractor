"""
markdown_chunker.py — Re-chunk Stage 1 markdown into retrieval-friendly blocks.

Why re-chunk? Stage 2's chunks are tuned for Q&A generation (400 tokens, aggressive
splitting on subheadings). For retrieval we want slightly longer, overlapping
chunks so context around the match is preserved.

Produces RetrievalChunk dataclass objects that carry:
  - text:       the chunk body (~512 tokens target)
  - source_file
  - page:       inferred from the nearest `<!-- page: N -->` marker
  - section:    the nearest markdown heading above the chunk
  - content_type: heuristic guess (procedure | spec | rule | figure | text)
  - chunk_id:   stable sha1-derived id

Reads settings from config["stage4"]["chunking"].
"""
import os
import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)
HEADING_RE     = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
JSON_FENCE_RE  = re.compile(r"^```json\s*$")
FENCE_RE       = re.compile(r"^```")

_PROCEDURE_KWS = ["step", "follow these", "how to", "install", "replace", "remove",
                  "press", "select", "open the", "close the", "insert"]
_SPEC_KWS      = ["dimensions", "weight", "capacity", "specification", "supported",
                  "compatible", "resolution", "speed"]
_RULE_KWS      = ["caution", "warning", "note:", "do not", "important", "error",
                  "troubleshoot"]


@dataclass
class RetrievalChunk:
    text: str
    source_file: str
    page: int = 0
    section: str = ""
    content_type: str = "text"   # procedure | spec | rule | figure | text
    chunk_id: str = ""
    kind: str = "chunk"          # "chunk" | "qa_pair"  — distinguishes hybrid index rows

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Helpers ──────────────────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _classify(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in _RULE_KWS):
        return "rule"
    if any(kw in t for kw in _PROCEDURE_KWS):
        return "procedure"
    if any(kw in t for kw in _SPEC_KWS):
        return "spec"
    if text.strip().startswith("!["):
        return "figure"
    return "text"


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _chunk_id(source_file: str, page: int, section: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(source_file.encode("utf-8"))
    h.update(str(page).encode("utf-8"))
    h.update(section.encode("utf-8"))
    h.update(text[:200].encode("utf-8"))
    return h.hexdigest()[:16]


# ── Main chunker ─────────────────────────────────────────────────────

class MarkdownChunker:
    """Re-chunks a Stage 1 markdown file into RetrievalChunk objects."""

    def __init__(self, config: Dict):
        s4 = config.get("stage4", {})
        ch = s4.get("chunking", {})
        self.max_tokens  = ch.get("max_tokens", 512)
        self.overlap_sents = ch.get("overlap_sentences", 1)
        self.min_chars   = ch.get("min_chunk_chars", 80)

    # ── Public API ───────────────────────────────────────────────────

    def chunk_file(self, md_path: str) -> List[RetrievalChunk]:
        """Read a markdown file and return a list of RetrievalChunks."""
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown not found: {md_path}")
        with open(md_path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        source_file = os.path.basename(md_path)
        blocks = self._split_into_blocks(lines)
        chunks: List[RetrievalChunk] = []
        for block in blocks:
            chunks.extend(self._chunk_block(block, source_file))
        return chunks

    # ── Internals ────────────────────────────────────────────────────

    def _split_into_blocks(self, lines: List[str]) -> List[Dict]:
        """Walk the file; emit one 'block' per heading section.
        Each block dict: { page, section, body (str), has_image (bool) }.
        """
        blocks: List[Dict] = []
        current_page = 0
        current_section = ""
        current_body: List[str] = []
        current_block_page = 0   # page at the moment the current block started
        has_image = False
        in_json_fence = False

        def flush():
            nonlocal current_body, has_image
            body = "\n".join(current_body).strip()
            if body:
                blocks.append({
                    "page": current_block_page or current_page,
                    "section": current_section,
                    "body": body,
                    "has_image": has_image,
                })
            current_body = []
            has_image = False

        for line in lines:
            # Page marker -> update state, don't emit
            m_pg = PAGE_MARKER_RE.match(line.strip())
            if m_pg:
                current_page = int(m_pg.group(1))
                # If the current block has no content yet, stamp this page as its start
                if not current_body:
                    current_block_page = current_page
                continue

            # Toggle JSON fence (tables embedded as JSON — keep them as one chunk block)
            if JSON_FENCE_RE.match(line.strip()):
                in_json_fence = not in_json_fence
                current_body.append(line)
                continue
            if in_json_fence:
                current_body.append(line)
                continue

            # Heading -> flush previous and start new section at CURRENT page
            m_hd = HEADING_RE.match(line)
            if m_hd:
                flush()
                current_section = m_hd.group(2).strip()
                current_block_page = current_page
                continue

            # Image link
            if line.strip().startswith("!["):
                has_image = True

            # First content line of a fresh block — stamp the page now
            if not current_body:
                current_block_page = current_page

            current_body.append(line)

        flush()
        return blocks

    def _chunk_block(self, block: Dict, source_file: str) -> List[RetrievalChunk]:
        body = block["body"].strip()
        if len(body) < self.min_chars:
            return []

        category = _classify(body)

        # Fits in one chunk
        if _est_tokens(body) <= self.max_tokens:
            cid = _chunk_id(source_file, block["page"], block["section"], body)
            return [RetrievalChunk(
                text=body, source_file=source_file,
                page=block["page"], section=block["section"],
                content_type=category, chunk_id=cid, kind="chunk",
            )]

        # Split on sentences with overlap
        sents = _split_sentences(body)
        chunks: List[RetrievalChunk] = []
        buf: List[str] = []
        buf_tok = 0
        idx = 0
        while idx < len(sents):
            s = sents[idx]
            t = _est_tokens(s)
            if buf_tok + t > self.max_tokens and buf:
                text = " ".join(buf).strip()
                cid = _chunk_id(source_file, block["page"], block["section"], text)
                chunks.append(RetrievalChunk(
                    text=text, source_file=source_file,
                    page=block["page"], section=block["section"],
                    content_type=_classify(text), chunk_id=cid, kind="chunk",
                ))
                # Overlap
                if self.overlap_sents > 0:
                    buf = buf[-self.overlap_sents:]
                    buf_tok = sum(_est_tokens(x) for x in buf)
                else:
                    buf, buf_tok = [], 0
            buf.append(s)
            buf_tok += t
            idx += 1
        if buf:
            text = " ".join(buf).strip()
            cid = _chunk_id(source_file, block["page"], block["section"], text)
            chunks.append(RetrievalChunk(
                text=text, source_file=source_file,
                page=block["page"], section=block["section"],
                content_type=_classify(text), chunk_id=cid, kind="chunk",
            ))
        return chunks


# ── QA pair loader (for hybrid index) ────────────────────────────────

def load_qa_pairs_as_retrieval_chunks(jsonl_path: str) -> List[RetrievalChunk]:
    """Load a Stage 2 (or Stage 3) JSONL and convert each record to a
    RetrievalChunk whose `text` is 'Q: ... A: ...' for direct FAQ match.
    kind = 'qa_pair'.
    """
    if not os.path.exists(jsonl_path):
        return []
    chunks: List[RetrievalChunk] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages", [])
            q = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            a = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            if not q or not a:
                continue
            prov = rec.get("provenance", {})
            combined = f"Q: {q}\nA: {a}"
            cid = _chunk_id(
                prov.get("source_file", "qa"),
                prov.get("page", 0),
                prov.get("chunk", ""),
                combined,
            )
            chunks.append(RetrievalChunk(
                text=combined,
                source_file=prov.get("source_file", "qa"),
                page=prov.get("page", 0),
                section=prov.get("chunk", "") or prov.get("section", ""),
                content_type=prov.get("category", "text"),
                chunk_id=cid,
                kind="qa_pair",
            ))
    return chunks
