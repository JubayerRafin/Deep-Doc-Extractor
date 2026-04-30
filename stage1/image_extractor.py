"""
image_extractor.py
------------------
Extracts images from PDF pages using a render-and-mask approach.

Strategy:
  1. Render the full page as a pixel image (150 DPI)
  2. Build a text mask: white out every character bounding box
  3. White out thin lines and rectangle borders (table borders, rules)
  4. Scan the masked image for remaining non-white content regions
  5. Filter regions by minimum size and content density
  6. Crop each surviving region from the ORIGINAL (unmasked) render

This approach naturally handles:
  - Vector graphics (Form XObjects) rendered to pixels
  - Icons inside table cells (text is masked, icon pixels remain)
  - Full-page illustrations and diagrams
  - Mixed layouts (text beside images)

Team DocuForge | Woosong University Capstone 2026
"""

import os
import pdfplumber
from .extracted_element import ExtractedElement


class ImageExtractor:

    DPI   = 150
    SCALE = DPI / 72.0       # PDF points → pixels

    MIN_REGION_W  = 40       # min region width  (pixels)
    MIN_REGION_H  = 40       # min region height (pixels)
    MIN_DENSITY   = 0.02     # min non-white ratio inside region bbox
    MERGE_GAP     = 15       # merge regions within this pixel gap
    TEXT_PAD       = 3       # expand text mask by this many pixels
    LINE_PAD       = 2       # expand line mask by this many pixels
    MARGIN_PT      = 40      # page margin to ignore (PDF points)
    DOWNSAMPLE     = 3       # scan at 1/N resolution for speed

    def __init__(self, config: dict, output_dir: str):
        images_cfg      = config.get("images", {})
        subdir          = images_cfg.get("output_subdir", "images")
        self.images_dir = os.path.join(output_dir, subdir)
        self.min_width  = images_cfg.get("min_width",  50)
        self.min_height = images_cfg.get("min_height", 50)
        os.makedirs(self.images_dir, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────

    def extract(self, pdf_path: str, page_numbers: list = None) -> list:
        elements = []
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            pages = self._resolve_pages(page_numbers, total)
            for page_idx in pages:
                try:
                    elems = self._extract_page(pdf.pages[page_idx],
                                               page_idx + 1)
                    elements.extend(elems)
                except Exception as e:
                    print(f"  [ImageExtractor] Page {page_idx+1} failed: {e}")
        print(f"  [ImageExtractor] Extracted {len(elements)} images.")
        return elements

    # ── Per-page pipeline ────────────────────────────────────────────

    def _extract_page(self, page, page_num: int) -> list:
        from PIL import Image, ImageDraw

        elements = []

        # ── 1. Render full page ──
        try:
            full_img = page.to_image(resolution=self.DPI).original.convert("RGB")
        except Exception as e:
            print(f"    [ImageExtractor] p{page_num} render failed: {e}")
            return elements

        img_w, img_h = full_img.size

        # ── 2. Build masked copy ──
        masked = full_img.copy()
        draw   = ImageDraw.Draw(masked)
        S      = self.SCALE
        TP     = self.TEXT_PAD
        LP     = self.LINE_PAD

        # White out all text characters
        for ch in (page.chars or []):
            draw.rectangle([
                int(ch["x0"] * S) - TP,  int(ch["top"] * S) - TP,
                int(ch["x1"] * S) + TP,  int(ch["bottom"] * S) + TP
            ], fill=(255, 255, 255))

        # White out lines (table borders, horizontal rules)
        for ln in (page.lines or []):
            draw.rectangle([
                int(ln["x0"] * S) - LP,  int(ln["top"] * S) - LP,
                int(ln["x1"] * S) + LP,  int(ln["bottom"] * S) + LP
            ], fill=(255, 255, 255))

        # White out rectangle borders (table cell outlines)
        for rect in (page.rects or []):
            rx0 = int(rect["x0"] * S)
            ry0 = int(rect["top"] * S)
            rx1 = int(rect["x1"] * S)
            ry1 = int(rect["bottom"] * S)
            bw = LP + 2
            draw.rectangle([rx0-bw, ry0-bw, rx1+bw, ry0+bw], fill=(255,255,255))
            draw.rectangle([rx0-bw, ry1-bw, rx1+bw, ry1+bw], fill=(255,255,255))
            draw.rectangle([rx0-bw, ry0-bw, rx0+bw, ry1+bw], fill=(255,255,255))
            draw.rectangle([rx1-bw, ry0-bw, rx1+bw, ry1+bw], fill=(255,255,255))

        # White out page margins
        m = int(self.MARGIN_PT * S)
        draw.rectangle([0, 0, img_w, m], fill=(255, 255, 255))
        draw.rectangle([0, img_h - m, img_w, img_h], fill=(255, 255, 255))

        # ── 3. Detect content regions (downsampled for speed) ──
        ds = self.DOWNSAMPLE
        small = masked.resize((img_w // ds, img_h // ds), Image.NEAREST)
        raw_regions = self._find_content_regions(small)

        # Scale back to full resolution
        regions = [(x0*ds, y0*ds, x1*ds, y1*ds)
                   for (x0, y0, x1, y1) in raw_regions]

        # Merge nearby regions
        regions = self._merge_regions(regions)

        # ── 4. Filter, crop, save ──
        img_index = 0
        for (rx0, ry0, rx1, ry1) in regions:
            rw, rh = rx1 - rx0, ry1 - ry0

            if rw < self.MIN_REGION_W or rh < self.MIN_REGION_H:
                continue

            # Density check on masked image
            if self._region_density(masked, rx0, ry0, rx1, ry1) < self.MIN_DENSITY:
                continue

            # Crop from ORIGINAL render (with small padding)
            pad = 5
            cx0 = max(0, rx0 - pad)
            cy0 = max(0, ry0 - pad)
            cx1 = min(img_w, rx1 + pad)
            cy1 = min(img_h, ry1 + pad)
            cropped = full_img.crop((cx0, cy0, cx1, cy1))

            if cropped.size[0] < self.min_width or cropped.size[1] < self.min_height:
                continue

            if self._is_blank(cropped):
                continue

            # Save
            filename = f"image_p{page_num}_{img_index:02d}.png"
            filepath = os.path.join(self.images_dir, filename)
            cropped.save(filepath, format="PNG")

            rel_path = os.path.join(os.path.basename(self.images_dir), filename)
            elements.append(ExtractedElement(
                page_num=page_num,
                y_coordinate=ry0 / self.SCALE,   # back to PDF points
                element_type="image",
                content=rel_path,
            ))
            img_index += 1

        return elements

    # ── Region scanning ──────────────────────────────────────────────

    def _find_content_regions(self, img) -> list:
        """
        Scan image for clusters of non-white pixels using row extents.
        Returns list of (x0, y0, x1, y1) bounding boxes.
        """
        width, height = img.size
        pix = img.load()
        THRESH = 240

        # Per-row: find leftmost/rightmost non-white pixel
        bands = []          # (y_start, y_end, x_min, x_max)
        band_start = None
        bx_min, bx_max = width, 0

        for y in range(height):
            row_min, row_max = width, 0
            has = False
            for x in range(width):
                r, g, b = pix[x, y]
                if r < THRESH or g < THRESH or b < THRESH:
                    row_min = min(row_min, x)
                    row_max = max(row_max, x)
                    has = True

            if has:
                if band_start is None:
                    band_start = y
                    bx_min, bx_max = row_min, row_max
                else:
                    bx_min = min(bx_min, row_min)
                    bx_max = max(bx_max, row_max)
            else:
                if band_start is not None:
                    bands.append((bx_min, band_start, bx_max + 1, y))
                    band_start = None
                    bx_min, bx_max = width, 0

        if band_start is not None:
            bands.append((bx_min, band_start, bx_max + 1, height))

        return bands

    def _merge_regions(self, regions: list) -> list:
        """Merge vertically close regions."""
        if not regions:
            return regions
        regions = sorted(regions, key=lambda r: r[1])
        merged = [list(regions[0])]
        for (x0, y0, x1, y1) in regions[1:]:
            p = merged[-1]
            if y0 - p[3] <= self.MERGE_GAP:
                p[0] = min(p[0], x0)
                p[1] = min(p[1], y0)
                p[2] = max(p[2], x1)
                p[3] = max(p[3], y1)
            else:
                merged.append([x0, y0, x1, y1])
        return [tuple(r) for r in merged]

    # ── Pixel helpers ────────────────────────────────────────────────

    def _region_density(self, img, x0, y0, x1, y1) -> float:
        cropped = img.crop((x0, y0, x1, y1))
        pixels  = list(cropped.getdata())
        total   = len(pixels)
        if total == 0:
            return 0.0
        nw = sum(1 for r, g, b in pixels if r < 240 or g < 240 or b < 240)
        return nw / total

    def _is_blank(self, img, threshold=0.98) -> bool:
        pixels = list(img.getdata())
        total  = len(pixels)
        if total == 0:
            return True
        w = sum(1 for r, g, b in pixels if r > 245 and g > 245 and b > 245)
        return (w / total) >= threshold

    def _resolve_pages(self, page_numbers, total_pages):
        if page_numbers is None:
            return list(range(total_pages))
        return [p - 1 for p in page_numbers if 1 <= p <= total_pages]
