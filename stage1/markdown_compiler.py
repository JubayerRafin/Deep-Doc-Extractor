"""
markdown_compiler.py
--------------------
Merges all ExtractedElements into a single Markdown file, sorted by
page number then Y-coordinate (top-to-bottom reading order).

Features:
  - Tables rendered as fenced JSON code blocks
  - Images get descriptive alt text derived from the preceding step/heading
  - NOTE/CAUTION icons labeled appropriately
  - Smart step summarization with pattern matching for common HP procedures
  - Content deduplication (exact match on first 100 chars)

Team DocuForge | Woosong University Capstone 2026
"""

import json
import os
import re
from .extracted_element import ExtractedElement


class MarkdownCompiler:

    def compile(self, elements: list, output_path: str) -> str:
        """
        Sort elements by page/Y-coordinate, deduplicate, and write to .md file.
        """
        sorted_elements = sorted(
            elements,
            key=lambda el: (el.page_num, el.y_coordinate)
        )
        deduped = self._deduplicate(sorted_elements)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        lines          = []
        prev_page      = None
        prev_step_text = ""   # last numbered step (sticky — used for image alt)
        prev_text      = ""   # last text block (fallback for alt text)

        for el in deduped:
            # Page break separator with page marker
            if el.page_num != prev_page:
                if prev_page is not None:
                    lines.append("")
                lines.append(f"<!-- page: {el.page_num} -->")
                prev_page = el.page_num

            if el.element_type == "text":
                lines.append(el.content)
                lines.append("")
                prev_text = el.content
                stripped  = el.content.strip()
                if re.match(r"^\d+\.", stripped) or stripped.startswith("#"):
                    prev_step_text = el.content

            elif el.element_type == "image":
                alt_source     = prev_step_text if prev_step_text else prev_text
                alt            = self._make_alt_text(alt_source, el.content)
                prev_step_text = ""   # consumed
                img_path       = el.content.replace("\\", "/")
                lines.append(f"![{alt}]({img_path})")
                lines.append("")

            elif el.element_type == "table":
                try:
                    table_data = json.loads(el.content)
                    pretty     = json.dumps(table_data, ensure_ascii=False, indent=2)
                except Exception:
                    pretty = el.content
                lines.append("```json")
                lines.append(pretty)
                lines.append("```")
                lines.append("")

        # Remove trailing blank lines
        while lines and lines[-1] == "":
            lines.pop()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        text_count  = sum(1 for e in deduped if e.element_type == "text")
        image_count = sum(1 for e in deduped if e.element_type == "image")
        table_count = sum(1 for e in deduped if e.element_type == "table")

        print(f"  [MarkdownCompiler] Written: {output_path}")
        print(f"  [MarkdownCompiler] Total elements: {len(deduped)} "
              f"(text: {text_count}, images: {image_count}, tables: {table_count})")
        return output_path

    # ── Alt text generation ─────────────────────────────────────────

    def _make_alt_text(self, prev_text: str, img_path: str) -> str:
        """Generate descriptive alt text from the preceding text context."""
        text = prev_text.strip()

        # Strip markdown formatting
        text_clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text_clean = re.sub(r"^#{1,4}\s+", "", text_clean)

        # NOTE / CAUTION icon (small inline image after alert label)
        for kw in ["NOTE:", "CAUTION:", "WARNING:", "IMPORTANT:", "TIP:"]:
            if text_clean.strip().startswith(kw):
                label = kw.rstrip(":")
                return f"{label} icon"

        # Numbered step → "Step N: summary"
        m = re.match(r"^(\d+)\.\s+(.+)", text_clean.strip(), re.DOTALL)
        if m:
            step_num  = m.group(1)
            full_text = m.group(2).split("\n")[0].strip()
            summary   = self._summarize_step(full_text)
            return f"Step {step_num}: {summary}"

        # Heading
        if text.startswith("#"):
            heading = text_clean.split("\n")[0].strip()
            return self._summarize_step(heading)

        # Plain paragraph fallback
        first_line = text_clean.split("\n")[0].rstrip(".").strip()
        if len(first_line) > 5:
            return self._summarize_step(first_line)

        # Last resort: use filename
        return os.path.splitext(os.path.basename(img_path))[0]

    def _summarize_step(self, text: str) -> str:
        """
        Match known HP manual step patterns first, then fall back to
        intelligent verb-phrase extraction.
        """
        # ── Pattern table (most specific first) ────────────────────
        PATTERNS = [
            # Toner cartridge
            (r"open the front door",                        "Open front door"),
            (r"close the front door",                       "Close front door"),
            (r"select the eject button.+eject the cartridge.+pull",
                                                            "Eject and pull cartridge"),
            (r"eject.+cartridge",                           "Eject cartridge"),
            (r"pull.+cartridge.+straight out",              "Pull out cartridge"),
            (r"remove.+new toner cartridge from its package",
                                                            "Unpack new cartridge"),
            (r"hold both ends.+rock it",                    "Rock cartridge and remove seal"),
            (r"rock.+toner",                                "Rock cartridge"),
            (r"remove the seal",                            "Remove seal"),
            (r"align.+cartridge.+insert",                   "Insert cartridge"),
            (r"align the toner cartridge",                  "Align and insert cartridge"),
            (r"pack the used toner cartridge",              "Pack used cartridge"),
            (r"remove and replace the toner cartridge",     "Replace toner cartridge"),
            # Paper trays
            (r"open tray\s*1",                              "Open Tray 1"),
            (r"open the tray",                              "Open the tray"),
            (r"pull out the tray extension",                "Pull out tray extension"),
            (r"use the adjustment latch to spread",         "Spread paper guides"),
            (r"load paper in(to)? the tray",                "Load paper in tray"),
            (r"adjust the side guides.+lightly touch",      "Adjust side guides"),
            (r"using the adjustment latch.+adjust",         "Adjust paper guides"),
            (r"adjust the paper.length guide",              "Adjust paper-length guide"),
            (r"adjust the paper.width guide",               "Adjust paper-width guide"),
            (r"close the tray",                             "Close the tray"),
            (r"slide the tray",                             "Slide tray into printer"),
            (r"before loading paper",                       "Adjust guides before loading"),
            # General UI steps
            (r"click\s+next\b",                             "Click Next"),
            (r"click\s+ok\b",                               "Click OK"),
            (r"click\s+save\b",                             "Click Save"),
            (r"click\s+apply\b",                            "Click Apply"),
            (r"click\s+finish\b",                           "Click Finish"),
            (r"select\s+apply\b",                           "Select Apply"),
            (r"click\s+add\b",                              "Click Add"),
            (r"open the hp embedded web server",            "Open HP Embedded Web Server"),
            (r"sign in.+administrator",                     "Sign in as administrator"),
            (r"on the control panel",                       "Use control panel"),
        ]

        lower = text.lower()
        for pattern, label in PATTERNS:
            if re.search(pattern, lower):
                return label

        # ── Smart fallback: verb + object ──────────────────────────
        clean = text
        clean = re.split(r",\s+(?:and then|so that|in order to)", clean, flags=re.I)[0]
        clean = re.sub(r"\s+(?:by using|from its|so that|in order|to support)\b.*",
                       "", clean, flags=re.I)
        clean = clean.rstrip(".,").strip()

        words = clean.split()
        if len(words) <= 5:
            return clean

        # Take up to 6 words, stop at preposition if it makes a clean cut
        STOP_PREPS = {"to", "from", "into", "onto", "until", "before", "after",
                      "by", "with", "so", "and"}
        result_words = []
        for i, w in enumerate(words):
            if i >= 4 and w.lower() in STOP_PREPS:
                break
            result_words.append(w)
            if i >= 6:
                break
        return " ".join(result_words)

    # ── Deduplication ───────────────────────────────────────────────

    def _deduplicate(self, elements: list) -> list:
        """Remove exact duplicate elements (same type + first 100 chars)."""
        deduped      = []
        seen_content = set()
        for el in elements:
            key = (el.element_type, el.content.strip()[:100])
            if key in seen_content:
                continue
            seen_content.add(key)
            deduped.append(el)
        return deduped
