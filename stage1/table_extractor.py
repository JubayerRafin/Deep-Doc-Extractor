"""
table_extractor.py
------------------
Extracts tables from PDF pages using Camelot-py (primary)
with pdfplumber as fallback for borderless/thin-line tables.

Includes validation to reject false positives common in HP manuals:
  1. CAUTION/NOTE boxes — single-column bordered text boxes
  2. Image + text side-by-side layouts — two columns with one mostly empty
  3. Page footer fragments — small tables with only page numbers

Output format is JSON (not Markdown pipe tables).

Team DocuForge | Woosong University Capstone 2026
"""

import os
import json
import camelot
import pdfplumber
from .extracted_element import ExtractedElement


FAKE_TABLE_KEYWORDS = [
    "CAUTION:", "NOTE:", "WARNING:", "IMPORTANT:", "TIP:",
    "caution:", "note:", "warning:"
]

# Fill rate threshold — tables below this are likely layout grids, not data tables
TABLE_FILL_THRESH = 0.40


class TableExtractor:

    def __init__(self, config: dict, output_dir: str):
        tables_cfg           = config.get("tables", {})
        subdir               = tables_cfg.get("output_subdir", "tables")
        self.tables_dir      = os.path.join(output_dir, subdir)
        self.flavor          = tables_cfg.get("flavor", "lattice")
        self.fallback_flavor = tables_cfg.get("fallback_flavor", "stream")
        os.makedirs(self.tables_dir, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────

    def extract(self, pdf_path: str, page_numbers: list = None) -> list:
        elements = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        pages_list = self._resolve_pages(page_numbers, total_pages)
        if not pages_list:
            return elements

        # Track which pages Camelot successfully finds tables on
        camelot_pages = set()
        page_str = ",".join(str(p) for p in pages_list)

        # Primary: Camelot lattice
        tables = self._run_camelot(pdf_path, page_str, self.flavor)

        # Fallback flavor if nothing found
        if not tables and self.fallback_flavor != self.flavor:
            print(f"  [TableExtractor] {self.flavor} found 0 tables — "
                  f"trying fallback: {self.fallback_flavor}")
            tables = self._run_camelot(pdf_path, page_str, self.fallback_flavor)

        table_index_global = 0
        rejected = 0

        # Track per-page table counts from Camelot
        page_table_count = {}
        for table in tables:
            pg = int(table.page)
            page_table_count[pg] = page_table_count.get(pg, 0) + 1

        for table in tables:
            if not self._is_real_table(table):
                rejected += 1
                continue
            el = self._camelot_table_to_element(
                table, table_index_global, pdf_path
            )
            if el is not None:
                elements.append(el)
                camelot_pages.add(int(table.page))
                table_index_global += 1

        # ── pdfplumber fallback for pages Camelot missed ────────────
        missed_pages = [p for p in pages_list if p not in camelot_pages]
        if missed_pages:
            print(f"  [TableExtractor] pdfplumber fallback for "
                  f"{len(missed_pages)} pages: {missed_pages}")
            fallback_els = self._pdfplumber_fallback(
                pdf_path, missed_pages, table_index_global
            )
            elements.extend(fallback_els)
            table_index_global += len(fallback_els)

        print(f"  [TableExtractor] Extracted {len(elements)} real tables "
              f"(rejected {rejected} false positives).")
        return elements

    # ── pdfplumber fallback ─────────────────────────────────────────

    def _pdfplumber_fallback(self, pdf_path: str, page_nums: list,
                              start_index: int) -> list:
        """
        Use pdfplumber find_tables() for pages where Camelot found nothing.
        Applies fill-rate filter to avoid detecting layout grids as tables.
        """
        elements = []
        idx = start_index

        with pdfplumber.open(pdf_path) as pdf:
            for page_num in page_nums:
                page = pdf.pages[page_num - 1]
                try:
                    tables = page.find_tables()
                except Exception:
                    continue

                for tbl in tables:
                    try:
                        extracted = tbl.extract()
                        if not extracted:
                            continue

                        # Fill rate check — reject layout grids
                        total  = sum(len(r) for r in extracted)
                        filled = sum(
                            1 for r in extracted
                            for c in r if c and str(c).strip()
                        )
                        fill_rate = filled / max(total, 1)
                        if fill_rate < TABLE_FILL_THRESH:
                            continue

                        # Must have at least 2 columns and 2 rows
                        n_rows = len(extracted)
                        n_cols = max((len(r) for r in extracted), default=0)
                        if n_rows < 2:
                            continue

                        # Build JSON
                        table_json = self._pdfplumber_to_json(
                            extracted, page_num, idx
                        )

                        # Save file
                        json_filename = f"table_p{page_num}_{idx}.json"
                        json_path = os.path.join(
                            self.tables_dir, json_filename
                        )
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(table_json, f,
                                      ensure_ascii=False, indent=2)

                        # bbox — pdfplumber uses top-left origin
                        bbox = tbl.bbox  # (x0, top, x1, bottom)
                        y_coord = bbox[1]

                        elements.append(ExtractedElement(
                            page_num=page_num,
                            y_coordinate=y_coord,
                            element_type="table",
                            content=json.dumps(table_json, ensure_ascii=False),
                        ))
                        idx += 1
                        print(f"    [pdfplumber] Page {page_num}: "
                              f"table {idx-1} saved "
                              f"({n_rows}x{n_cols}, fill={fill_rate:.2f})")

                    except Exception as e:
                        print(f"    [pdfplumber] Warning page {page_num}: {e}")

        return elements

    def _pdfplumber_to_json(self, extracted: list, page_num: int,
                             table_index: int) -> dict:
        """Convert pdfplumber extracted table to JSON dict."""
        # Clean cells
        cleaned = [
            [str(c).strip() if c else "" for c in row]
            for row in extracted
        ]

        # Detect header row
        first_row = cleaned[0]
        is_header = (
            len(set(first_row)) == len(first_row)
            and all(len(v) > 0 for v in first_row)
        )

        if is_header:
            headers   = first_row
            data_rows = cleaned[1:]
        else:
            headers   = [f"Column_{i}" for i in range(len(first_row))]
            data_rows = cleaned

        rows = [
            {h: v for h, v in zip(headers, row)}
            for row in data_rows
        ]

        return {
            "page":        page_num,
            "table_index": table_index,
            "headers":     headers,
            "rows":        rows,
            "source":      "pdfplumber_fallback",
        }

    # ── False-positive validation (Camelot) ────────────────────────

    def _is_real_table(self, table) -> bool:
        try:
            df     = table.df
            n_cols = df.shape[1]
            n_rows = df.shape[0]

            if n_cols < 2:
                return False

            first_row = [str(v).strip() for v in df.iloc[0]]
            for cell in first_row:
                for kw in FAKE_TABLE_KEYWORDS:
                    if cell.startswith(kw):
                        return False

            for col_idx in range(n_cols):
                col_vals    = [str(v).strip() for v in df.iloc[:, col_idx]]
                empty_count = sum(1 for v in col_vals if v == "" or v == "nan")
                empty_ratio = empty_count / len(col_vals) if col_vals else 1.0
                if empty_ratio > 0.7:
                    return False

            if n_rows <= 2:
                first_col_vals = [str(v).strip() for v in df.iloc[:, 0]]
                if all(v.isdigit() or v == "" for v in first_col_vals):
                    return False

            return True

        except Exception:
            return False

    # ── Camelot helpers ─────────────────────────────────────────────

    def _run_camelot(self, pdf_path: str, page_str: str, flavor: str):
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=page_str,
                flavor=flavor,
                suppress_stdout=True
            )
            return list(tables)
        except Exception as e:
            print(f"  [TableExtractor] Warning: Camelot {flavor} error: {e}")
            return []

    def _camelot_table_to_element(self, table, table_index: int,
                                   pdf_path: str):
        try:
            df       = table.df
            page_num = int(table.page)

            with pdfplumber.open(pdf_path) as pdf:
                page_height = pdf.pages[page_num - 1].height

            bbox           = table._bbox
            y_top_of_table = page_height - bbox[3]
            table_json     = self._dataframe_to_json(df, page_num, table_index)

            json_filename = f"table_p{page_num}_{table_index}.json"
            json_path     = os.path.join(self.tables_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(table_json, f, ensure_ascii=False, indent=2)

            return ExtractedElement(
                page_num=page_num,
                y_coordinate=y_top_of_table,
                element_type="table",
                content=json.dumps(table_json, ensure_ascii=False),
            )

        except Exception as e:
            print(f"  [TableExtractor] Warning: could not convert table: {e}")
            return None

    def _dataframe_to_json(self, df, page_num: int, table_index: int) -> dict:
        df = df.map(lambda x: str(x).strip() if x else "")

        first_row = list(df.iloc[0])
        is_header = (
            len(set(first_row)) == len(first_row)
            and all(len(v) > 0 for v in first_row)
        )

        if is_header:
            headers   = first_row
            data_rows = df.iloc[1:]
        else:
            headers   = [f"Column_{i}" for i in range(df.shape[1])]
            data_rows = df

        rows = []
        for _, row in data_rows.iterrows():
            row_dict = {h: v for h, v in zip(headers, row)}
            rows.append(row_dict)

        return {
            "page":        page_num,
            "table_index": table_index,
            "headers":     headers,
            "rows":        rows,
        }

    def _resolve_pages(self, page_numbers, total_pages):
        if page_numbers is None:
            return list(range(1, total_pages + 1))
        return sorted([p for p in page_numbers if 1 <= p <= total_pages])
