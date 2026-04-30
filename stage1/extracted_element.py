"""
extracted_element.py
--------------------
Shared data structure for all Stage 1 extractors.
Every text block, image reference, and table is wrapped in an
ExtractedElement so the MarkdownCompiler can sort them uniformly.

Team DocuForge | Woosong University Capstone 2026
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ExtractedElement:
    """
    One extracted piece of content from a PDF page.

    Attributes
    ----------
    page_num      : 1-based page number (matches PDF page numbering).
    y_coordinate  : Distance from the TOP of the page in points.
                    pdfplumber uses y=0 at the top, so smaller y = higher.
    element_type  : One of 'text', 'image', or 'table'.
    content       : text  → formatted Markdown string
                    image → relative path to saved PNG
                    table → JSON string of table data
    font_size     : (text only) largest font size in the block.
    is_bold       : (text only) True if the block contains bold text.
    """

    page_num: int
    y_coordinate: float
    element_type: Literal["text", "image", "table"]
    content: str
    font_size: float = 0.0
    is_bold: bool = False

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return (
            f"ExtractedElement("
            f"page={self.page_num}, y={self.y_coordinate:.1f}, "
            f"type={self.element_type!r}, content={preview!r}...)"
        )
