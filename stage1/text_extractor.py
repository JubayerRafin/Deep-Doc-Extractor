"""
text_extractor.py
-----------------
Extracts text blocks from PDF pages using pdfplumber.

Key design decisions:
  - Heading detection uses char-level font size via max() (not average),
    because pdfplumber's extract_words() can return duplicate words that
    would dilute the average.
  - Bold detection checks fontname for 'bold'/'semibold'/'black'/'heavy'.
  - NOTE/CAUTION/WARNING keywords are force-marked as bold.
  - Footer/header lines (page numbers, chapter labels) are filtered out.
  - Words overlapping table bounding boxes are excluded to avoid
    duplicating content that TableExtractor handles.

Team DocuForge | Woosong University Capstone 2026
"""

import re
import pdfplumber
from .extracted_element import ExtractedElement


class TextExtractor:

    def __init__(self, config: dict):
        self.min_chars         = config.get("min_block_chars", 5)
        self.heading_threshold = config.get("heading_font_threshold", 1.2)
        self.bold_keywords     = config.get("bold_font_keywords", [
            "NOTE:", "CAUTION:", "WARNING:", "IMPORTANT:", "TIP:"
        ])

    # ── Public API ──────────────────────────────────────────────────

    def extract(self, pdf_path: str, page_numbers: list = None) -> list:
        """
        Extract text elements from the PDF.

        Parameters
        ----------
        pdf_path     : Path to the PDF file.
        page_numbers : 1-based page numbers to process (None = all pages).

        Returns
        -------
        List of ExtractedElement with element_type='text'.
        """
        elements = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages      = len(pdf.pages)
            pages_to_process = self._resolve_pages(page_numbers, total_pages)

            # Collect all char sizes across selected pages for median baseline
            all_sizes = []
            for page_idx in pages_to_process:
                for char in (pdf.pages[page_idx].chars or []):
                    s = char.get("size", 0)
                    if s and s > 0:
                        all_sizes.append(s)

            median_font = self._median(all_sizes) if all_sizes else 10.0
            heading_min = median_font * self.heading_threshold
            print(f"  [TextExtractor] Median font size: {median_font}pt  "
                  f"Heading threshold: {heading_min:.1f}pt")

            for page_idx in pages_to_process:
                page         = pdf.pages[page_idx]
                page_num     = page_idx + 1
                table_bboxes = self._get_table_bboxes(page)
                elements.extend(
                    self._extract_page(page, page_num, median_font, table_bboxes)
                )

        print(f"  [TextExtractor] Extracted {len(elements)} text elements "
              f"from {len(pages_to_process)} pages.")
        return elements

    # ── Page-level extraction ───────────────────────────────────────

    def _extract_page(self, page, page_num: int,
                      median_font: float, table_bboxes: list) -> list:
        elements = []

        # extra_attrs gives each word dict 'size' and 'fontname' fields
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["size", "fontname"]
        ) or []

        if not words:
            return elements

        # Exclude words that fall inside table bounding boxes
        if table_bboxes:
            words = [w for w in words
                     if not self._word_in_table(w, table_bboxes)]

        lines  = self._group_words_into_lines(words)
        blocks = self._group_lines_into_blocks(lines)

        for block in blocks:
            text, y_top, block_font, is_bold = self._summarize_block(block)

            if len(text.strip()) < self.min_chars:
                continue
            if self._is_footer_or_header(text, y_top, page.height):
                continue

            formatted = self._format_text(text, block_font, median_font, is_bold)

            elements.append(ExtractedElement(
                page_num=page_num,
                y_coordinate=y_top,
                element_type="text",
                content=formatted,
                font_size=block_font,
                is_bold=is_bold,
            ))

        return elements

    # ── Word → Line → Block grouping ───────────────────────────────

    def _group_words_into_lines(self, words: list) -> list:
        """Group words into lines based on Y-position proximity (≤3pt)."""
        lines        = []
        current_line = []
        prev_top     = None

        for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
            top = word["top"]
            if prev_top is None or abs(top - prev_top) <= 3:
                current_line.append(word)
                prev_top = top
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                prev_top     = top

        if current_line:
            lines.append(current_line)
        return lines

    def _group_lines_into_blocks(self, lines: list) -> list:
        """Group consecutive lines into blocks (gap > 8pt = new block)."""
        if not lines:
            return []

        blocks        = []
        current_block = [lines[0]]

        for i in range(1, len(lines)):
            prev_bottom = max(w["bottom"] for w in lines[i - 1])
            curr_top    = min(w["top"]    for w in lines[i])
            if curr_top - prev_bottom > 8:
                blocks.append(current_block)
                current_block = [lines[i]]
            else:
                current_block.append(lines[i])

        if current_block:
            blocks.append(current_block)
        return blocks

    def _summarize_block(self, block: list) -> tuple:
        """
        Compute text, position, font size, and bold status for a block.

        Font size uses max() across all words — not average — because
        pdfplumber sometimes returns duplicate words that would dilute
        the average and cause headings (14pt) to appear as body (10pt).
        """
        all_words  = [w for line in block for w in line]
        y_top      = min(w["top"] for w in all_words)
        lines_text = [" ".join(w["text"] for w in line) for line in block]
        full_text  = "\n".join(lines_text)

        # Font size: use MAX to ensure heading detection works
        sizes = []
        for w in all_words:
            sz = w.get("size")
            if sz and sz > 0:
                sizes.append(float(sz))
        block_font = max(sizes) if sizes else 10.0

        # Bold: check fontname keywords
        fontnames = [w.get("fontname", "") for w in all_words]
        is_bold   = any(self._is_bold_fontname(fn) for fn in fontnames)

        # Force-bold if block starts with a bold keyword
        stripped = full_text.strip()
        if any(stripped.startswith(kw) for kw in self.bold_keywords):
            is_bold = True

        return full_text.strip(), y_top, block_font, is_bold

    # ── Markdown formatting ─────────────────────────────────────────

    def _format_text(self, text: str, font_size: float,
                     median_font: float, is_bold: bool) -> str:
        stripped = text.strip()

        # 1. Font-size-based heading detection
        #    HP E877: body=10pt, headings=14pt → ratio 1.4 → triggers ##
        if median_font > 0 and font_size >= self.heading_threshold * median_font:
            ratio = font_size / median_font
            if ratio >= 1.8:
                return f"# {stripped}"
            elif ratio >= 1.4:
                return f"## {stripped}"
            else:
                return f"### {stripped}"

        # 2. NOTE / CAUTION / WARNING → bold keyword + plain rest
        for keyword in self.bold_keywords:
            if stripped.startswith(keyword):
                rest = stripped[len(keyword):].strip()
                return f"**{keyword}** {rest}" if rest else f"**{keyword}**"

        # 3. General bold text (short blocks only)
        if is_bold and len(stripped) < 200:
            return f"**{stripped}**"

        # 4. Numbered list items (pass through as-is)
        if re.match(r"^\d+\.", stripped):
            return stripped

        # 5. Plain paragraph
        return stripped

    # ── Filters ─────────────────────────────────────────────────────

    def _is_footer_or_header(self, text: str, y_top: float,
                              page_height: float) -> bool:
        """Filter out page numbers, chapter headers, and footer lines."""
        s = text.strip()

        # Standalone page number
        if re.match(r"^\d+$", s):
            return True
        # "42 Chapter 3 ..." style header
        if re.match(r"^\d+\s+Chapter\s+\d+", s):
            return True
        # Line ending with a page number (but not a numbered step)
        if (re.search(r"\s+\d+$", s)
                and len(s) < 80
                and not re.match(r"^\d+[\.\)]\s+", s)):
            return True
        # Near top/bottom margin and short
        if (y_top < 36 or y_top > page_height - 45) and len(s) < 60:
            return True

        return False

    def _word_in_table(self, word: dict, table_bboxes: list) -> bool:
        """Check if a word overlaps any detected table region."""
        wx0  = word.get("x0",     0)
        wtop = word.get("top",    0)
        wx1  = word.get("x1",     0)
        wbot = word.get("bottom", 0)
        for (tx0, ttop, tx1, tbot) in table_bboxes:
            if wx0 < tx1 and wx1 > tx0 and wtop < tbot and wbot > ttop + 10:
                return True
        return False

    def _get_table_bboxes(self, page) -> list:
        """Get bounding boxes of all tables detected by pdfplumber."""
        try:
            return [tbl.bbox for tbl in page.find_tables()]
        except Exception:
            return []

    def _is_bold_fontname(self, fontname: str) -> bool:
        fn = fontname.lower()
        return any(k in fn for k in ["bold", "semibold", "black", "heavy"])

    # ── Utilities ───────────────────────────────────────────────────

    def _resolve_pages(self, page_numbers, total_pages):
        if page_numbers is None:
            return list(range(total_pages))
        return [p - 1 for p in page_numbers if 1 <= p <= total_pages]

    @staticmethod
    def _median(values: list) -> float:
        if not values:
            return 10.0
        sv  = sorted(values)
        mid = len(sv) // 2
        return (sv[mid - 1] + sv[mid]) / 2.0 if len(sv) % 2 == 0 else float(sv[mid])
