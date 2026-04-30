"""
file_validator.py
-----------------
Validates the input PDF before extractors run.
Checks: file existence, extension, PDF header, page count.

Team DocuForge | Woosong University Capstone 2026
"""

import os


class FileValidator:

    def validate(self, pdf_path: str) -> dict:
        """Run all validation checks and return a result dict."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "page_count": None,
            "file_size_mb": 0.0,
        }

        # File existence
        if not os.path.exists(pdf_path):
            result["errors"].append(f"File not found: {pdf_path}")
            result["valid"] = False
            return result

        # File extension
        ext = os.path.splitext(pdf_path)[1].lower()
        if ext not in (".pdf", ".docx", ".xlsx"):
            result["errors"].append(f"Unsupported file type: '{ext}'")
            result["valid"] = False
            return result

        # File size
        size_bytes = os.path.getsize(pdf_path)
        result["file_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        if size_bytes < 1024:
            result["warnings"].append(
                f"File very small ({size_bytes} bytes) — may be corrupt."
            )

        # PDF header check
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            if not header.startswith(b"%PDF-"):
                result["errors"].append(
                    f"Not a valid PDF (header: {header!r}). "
                    "Check the file is not corrupted."
                )
                result["valid"] = False
                return result
        except Exception as e:
            result["errors"].append(f"Cannot read file: {e}")
            result["valid"] = False
            return result

        # Page count via pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                result["page_count"] = len(pdf.pages)
        except Exception as e:
            result["errors"].append(f"Cannot open PDF: {e}")
            result["valid"] = False

        return result

    def report(self, pdf_path: str) -> bool:
        """Validate and print a human-readable report. Returns True if valid."""
        print(f"\n[FileValidator] Validating: {pdf_path}")
        result = self.validate(pdf_path)

        print(f"  File size : {result['file_size_mb']} MB")
        if result["page_count"]:
            print(f"  Pages     : {result['page_count']}")
        for warn in result["warnings"]:
            print(f"  WARNING   : {warn}")

        if result["valid"]:
            print(f"  Status    : VALID [OK]")
        else:
            print(f"  Status    : INVALID [FAIL]")
            for err in result["errors"]:
                print(f"  ERROR     : {err}")

        return result["valid"]
