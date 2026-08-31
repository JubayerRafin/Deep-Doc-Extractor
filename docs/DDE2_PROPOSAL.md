# Deep Doc Extractor 2.0 — Document Transformation

**Proposal · Woosong University Capstone 2026-2 · Industry Partner: HP Printing Korea (Mintae Kim, DST)**

Extending Deep Doc Extractor from document *extraction* to document *transformation*:
translate and de-identify a document while preserving its original layout and formatting.

---

## 1. What the project asks for

From the HP project brief:

| Column | Requirement |
|---|---|
| **Objectives** | Document structure analysis (text / table / image region identification); multilingual translation + automatic PII detection & de-identification; layout- and format-preserving reconstruction; verification of conversion accuracy and layout retention rate |
| **Domain knowledge** | Document parsing, OCR, layout analysis; LLM-based translation and PII detection; on-prem LLM service; DDE code and sample documents (provided by HP) |
| **Output** | AI document-transformation prototype; web demo for upload + result inspection; conversion accuracy and layout-retention evaluation results |
| **Expected value** | Automating in-house document translation and PII protection; advancing DDE; foundation for a future Document AI platform |

---

## 2. Where DDE 1.0 stands, and the one thing that has to change

DDE 1.0 is a validated four-stage extraction pipeline (Stage 1 extraction → Stage 2 Q&A
generation → Stage 3 augmentation → Stage 4 offline RAG), with Stage 1 scoring **table F1 0.94**
and **image F1 0.90** on the 241-page HP E877 guide — beating DocLayout-YOLO (0.72) and
gap-based detection (0.62) without a GPU.

Stage 1 already does most of the *analysis* half of Objective 1. What it does **not** do is
keep enough information to put a document back together.

**The core architectural gap.** `stage1/extracted_element.py` carries exactly this:

```python
page_num, y_coordinate, element_type, content, font_size, is_bold
```

That is a *reading-order* model. It is deliberately lossy: `markdown_compiler.py` sorts by
`(page_num, y_coordinate)` and flattens everything into Markdown. Along the way DDE 1.0 discards
**x-coordinates, box width/height, font family, colour, character spacing, rotation, and
z-order** — none of which matter for building a Q&A dataset, and all of which are mandatory for
rendering a translated page that still looks like the original.

Extraction is a one-way projection. Transformation is a **round trip**:

```
DDE 1.0   PDF ──parse──> Markdown ──> dataset          (lossy is fine)
DDE 2.0   PDF ──parse──> Document IR ──transform──> Document IR' ──render──> PDF/DOCX
                              └────────── must be reconstruction-complete ──────────┘
```

So the first and most important engineering task of DDE 2.0 is **not** translation or PII — it is
replacing `ExtractedElement` with a reconstruction-complete document intermediate representation
(IR). Everything else depends on it.

---

## 3. Proposed architecture

Five stages, keeping DDE 1.0's file-based hand-off so any stage can be re-run independently.

```
   Source document (PDF / DOCX / scanned image)
        │
   [T1] Layout Analysis        → document.ir.json   (regions + geometry + style)
        │
   [T2] Semantic Enrichment    → reading order, block roles, PII spans, translate/no-translate flags
        │
   [T3] Transformation         → translated text, redacted spans   (on-prem LLM)
        │
   [T4] Reconstruction         → output PDF / DOCX, layout preserved
        │
   [T5] Verification           → accuracy + layout-retention report
        │
   Web demo (upload → side-by-side original / translated / redacted)
```

### T1 — Layout Analysis
Extend Stage 1's proven extractors to emit full geometry instead of a y-coordinate.

```python
@dataclass
class Region:
    page_num: int
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 in PDF points
    kind: Literal["text", "table", "image", "header", "footer", "caption", "list"]
    reading_order: int
    z_index: int
    rotation: float
    content: TextContent | TableContent | ImageContent

@dataclass
class TextSpan:                 # style-uniform run inside a text region
    text: str
    bbox: tuple[float, float, float, float]
    font_family: str
    font_size: float
    bold: bool
    italic: bool
    color: tuple[int, int, int]
    direction: Literal["ltr", "rtl"]
```

- **Text**: `text_extractor.py` already reads `size` and `fontname` per word via
  `extract_words(extra_attrs=...)` — the geometry is there, it is simply thrown away at block
  summarisation. Recovering it is a change of a few dozen lines, plus span-level grouping so a
  bold run inside a sentence survives.
- **Tables**: use `docling_table_extractor.py` from the research branch as primary (it recovers
  cell structure for borderless tables), keeping Camelot/pdfplumber as fallback. Store **per-cell
  bboxes**, not just values, because each cell is translated independently.
- **Images**: `image_extractor.py`'s render-and-mask approach already produces pixel regions;
  keep the crop and its bbox, and pass the image through untouched.
- **Scanned input**: DDE 1.0 lists OCR as out of scope. DDE 2.0 needs it (the brief names OCR
  explicitly). Add a detector — a page with no extractable text layer routes to Tesseract or
  PaddleOCR, which returns word boxes in the same `TextSpan` shape, so downstream stages do not
  care whether text came from the PDF layer or from OCR.

### T2 — Semantic Enrichment
- Reading-order recovery for multi-column pages (XY-cut or a column-detection pass); DDE 1.0's
  pure y-sort breaks on two-column layouts.
- Block role classification, reusing `semantic_classifier.py`'s keyword approach as a baseline.
- **PII detection**, two layers:
  - **Deterministic**: regex + checksum for structured PII — emails, phone numbers, Korean RRN
    (주민등록번호, with the check-digit test), credit cards (Luhn), IPs, MACs, serial numbers, URLs.
    High precision, auditable, zero-cost, and it never depends on an LLM being in a good mood.
  - **Contextual**: a local NER / LLM pass for names, addresses, organisations, and free-text
    identifiers that regex cannot catch.
  - Union the two, then apply a confidence threshold that is **recall-biased** — a missed PII item
    is a data leak, a false positive is a redacted word. These costs are not symmetric.
- **Do-not-translate** flags: part numbers, model codes (`E877`), UI button labels, code blocks,
  units. Driven by a glossary file so HP can maintain it without touching code.

### T3 — Transformation (on-prem LLM)
Reuses the existing Ollama client pattern and `offline_guard.py` — the whole point of an on-prem
service is that a document containing PII never leaves the network, and DDE 1.0 already enforces
that at runtime by blocking non-localhost calls.

- **Translation** is done **per region with document context**, not per page and not per sentence:
  sentence-at-a-time loses pronoun/terminology consistency, page-at-a-time makes it impossible to
  map output back onto the right box. Each request carries the region's text, its role, the
  surrounding headings, and the glossary.
- **Length control**: prompt for a target length band and re-request when a segment overshoots
  badly. Translated text is routinely longer than the source (EN→KO shrinks, KO→EN and EN→AR
  grow); overflow is the single biggest cause of layout breakage, so it is fought at generation
  time first and at render time second.
- **Redaction** replaces PII spans with a masking token that carries the original span width, so
  T4 can decide between a black box and a `[REDACTED]` label.

### T4 — Reconstruction
Two output paths, chosen by document type:

1. **Overlay path (PDF in → PDF out)** — highest fidelity. Cover the original text region and
   draw the translated spans in the same box with the same style, keeping images, vector graphics,
   table rules, and page geometry pixel-identical. Overflow ladder, applied in order:
   *(a)* shrink tracking, *(b)* reduce font size within a floor (e.g. ≥ 85% of original),
   *(c)* re-wrap into more lines if vertical slack exists, *(d)* expand the box into adjacent
   whitespace, *(e)* flag the region as **overflowed** in the report rather than silently
   producing an unreadable page. An honest overflow flag is worth more than a hidden failure.
2. **Reflow path (DOCX out)** — for documents that must stay editable; layout retention is lower
   but the artifact is usable downstream.

Two things that will bite and need budgeting for:
- **Fonts.** Target scripts need embedded fonts that actually cover the glyphs — CJK and Arabic
  will not render in a Latin subset. Ship a licensed font set mapped by script.
- **RTL.** Arabic (named in the brief's mockup) needs bidi reordering and right-anchored boxes.
  This is a real amount of work; it should be planned, not discovered in week 12.

**Redaction must be irreversible.** Drawing a black rectangle over text in a PDF hides it visually
while leaving it fully extractable with `Ctrl-A` — the classic redaction failure that has embarrassed
governments and law firms. DDE 2.0 must **delete the underlying text objects**, and T5 must
verify by re-extracting text from the output and asserting the PII strings are absent.

### T5 — Verification
See §5.

---

## 4. What carries over

DDE 2.0 is an extension, not a rewrite. Roughly two-thirds of the foundation already exists.

| Asset | From | Use in 2.0 |
|---|---|---|
| `stage1/text_extractor.py` | DDE 1.0 | Extend to emit span geometry + style |
| `stage1/image_extractor.py` (F1 0.90) | DDE 1.0 | Reuse; keep crops **and** bboxes |
| `stage1/table_extractor.py` | DDE 1.0 | Fallback table path |
| `stage1/docling_table_extractor.py` | research branch | Primary table path (cell structure) |
| Ollama client + `config.yaml` control plane | DDE 1.0 | On-prem LLM service layer |
| `stage4/offline_guard.py` | DDE 1.0 | Non-negotiable for PII work |
| Streamlit app pattern | DDE 1.0 | Base for the web demo |
| `scorers/`, `baseline/`, `analysis/significance.py` | research branch | Evaluation harness + paired-bootstrap testing |
| Multi-vendor corpus (HP, Canon, Samsung, Cisco) | research branch | Cross-vendor generalisation set |

The evaluation discipline from the research work — independent scorers, real baselines, paired
bootstrap significance testing — is arguably the most transferable asset here, and it is what will
make the 2.0 results defensible rather than anecdotal.

---

## 5. Evaluation plan

The brief asks specifically for *conversion accuracy* and *layout retention rate*, so these need
real definitions, not impressions.

### 5.1 Layout retention
Re-parse the **output** document with T1 and compare region geometry against the input IR.

| Metric | Definition | Target |
|---|---|---|
| **Region IoU** | Mean IoU between matched input/output region bboxes | ≥ 0.90 |
| **Layout retention rate** | % of regions with IoU ≥ 0.90 | ≥ 90% |
| **Overflow rate** | % of text regions that exceeded their box or hit the font floor | ≤ 5% |
| **Collision rate** | % of regions overlapping a neighbour post-render | ≈ 0% |
| **Page-count delta** | Output pages − input pages | 0 (overlay path) |
| **Asset preservation** | % of images/tables present at the same position | 100% |

### 5.2 Translation accuracy
No reference translations exist for these manuals, so combine reference-free signals and anchor
them with a small human sample:

- **COMET-QE** or equivalent reference-free QE on all segments.
- **Back-translation similarity** (embedding cosine between source and round-tripped source),
  reusing the sentence-transformers setup already in the repo.
- **LLM-as-judge** with a fixed rubric, run on a model *independent* of the translation model —
  the same independence principle as the existing NLI faithfulness evaluator.
- **Glossary/term consistency**: % of glossary terms rendered per the HP-approved translation.
- **Human-scored sample** of ~100 segments per language pair, to calibrate the automatic metrics.
  This is what makes the automatic numbers trustworthy.

### 5.3 PII detection & redaction
The annotation-cost problem has a cheap solution: **inject synthetic PII into real manual pages**.
Generating the PII means the ground truth is exact and free, and the surrounding document is
genuine. Supplement with a modest hand-labelled real set to check the synthetic set is not too easy.

| Metric | Why | Target |
|---|---|---|
| **Recall** | Missed PII = leak; the metric that matters | ≥ 0.98 |
| **Precision** | Over-redaction destroys usability | ≥ 0.90 |
| **F1** | Headline | ≥ 0.94 |
| **Leakage check** | Re-extract text from output; PII strings **must** be absent | Pass/fail, 100% |

### 5.4 Baselines
Ours vs. published/commercial reference points on the same pages and the same metrics, with
paired-bootstrap significance testing over pages (`analysis/significance.py` already implements
this): a layout-naive baseline (extract → translate → re-emit, no geometry), an off-the-shelf PDF
translator, and a regex-only PII baseline. A result without a baseline is not a result.

### 5.5 Test corpus
HP E877 as primary, plus the Canon / Samsung / Cisco manuals already collected — cross-vendor
evidence that the method generalises rather than being tuned to one document's quirks. Add a
scanned-document subset to exercise the OCR path.

---

## 6. Web demo scope

Streamlit, following the Stage 4 app pattern, running fully on-prem:

1. Upload a document, pick target language and redaction mode (off / mask / label).
2. Progress by stage.
3. **Side-by-side page viewer** — original | translated | redacted — the money shot from the brief's mockup.
4. Per-page layout-retention score with overflowed regions highlighted, so a reviewer can see
   exactly where the system struggled instead of having to hunt for it.
5. Download the output document + a JSON transformation report.
6. PII review panel: every detected span, its type, confidence, and an accept/reject toggle —
   a human-in-the-loop checkpoint, mirroring DDE 1.0's stance that automated filtering plus a
   sampled human review beats either alone.

---

## 7. Milestones

Sixteen weeks, front-loading the IR because everything else sits on top of it.

| Weeks | Milestone | Exit criterion |
|---|---|---|
| 1–2 | Requirements freeze with HP; sample documents; language pairs; glossary v0 | Signed-off scope |
| 2–4 | **T1: Document IR + geometry-preserving extractors** | Round-trip test: PDF → IR → PDF with *no* transformation is visually identical |
| 4–5 | OCR path for scanned input | Scanned page produces the same IR shape |
| 5–6 | T2: reading order, roles, glossary flags | Multi-column pages ordered correctly |
| 6–8 | T2: PII detection (regex + contextual), labelled eval set | Recall ≥ 0.98 on the injected set |
| 8–10 | T3: translation service, length control, terminology | First end-to-end translated document |
| 10–12 | T4: reconstruction, overflow ladder, fonts, RTL | Layout retention ≥ 90% on HP E877 |
| 11–13 | Web demo | Upload → side-by-side → download works |
| 13–15 | T5: full evaluation, baselines, significance testing | Complete results table |
| 15–16 | Report, documentation, HP handover | Deliverables submitted |

The week 2–4 exit criterion is the one to defend hardest: **an untransformed round trip must be
pixel-identical**. If the pipeline cannot rebuild a document it has not changed, no amount of
translation quality will save the output — and this is exactly the check that catches a lossy IR
early, while it is still cheap to fix.

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Text expansion breaks layout | High | Length-controlled generation; overflow ladder; honest overflow flags |
| RTL/CJK font and shaping work underestimated | High | Scope Arabic explicitly in week 1; licensed font set early; prototype one RTL page by week 10 |
| Reconstruction fidelity on complex vector pages | High | Overlay path keeps original graphics untouched; reflow path only where editability is required |
| Redaction leaves recoverable text | Critical | Delete text objects, never overlay; automated leakage check in T5 as a hard gate |
| PII recall short of target | High | Recall-biased thresholds; regex layer as a floor; human review panel in the demo |
| Local LLM translation quality below commercial | Medium | Benchmark several local models (the DDE 1.0 five-model comparison methodology transfers directly); report the gap honestly |
| Scope creep across 3 hard problems (layout, MT, PII) | Medium | IR first; one language pair end-to-end before adding more |
| GPU availability for on-prem LLM | Medium | Confirm hardware with HP in week 1; DDE 1.0's CPU-first design is the fallback |

---

## 9. Open questions for HP

1. **Language pairs** — the mockup shows Arabic; is KO↔EN the primary pair, and which others are in scope for the prototype?
2. **Document types** — PDF only, or DOCX/PPTX/scanned images too? Which dominate the real internal workload?
3. **Output format** — is a non-editable layout-faithful PDF acceptable, or is an editable DOCX required?
4. **PII taxonomy** — which categories are in scope (Korean RRN, employee IDs, customer names, addresses, device serial numbers)? Is there an existing internal definition to conform to?
5. **Redaction semantics** — irreversible black-box, or reversible pseudonymisation with a key held internally? These are very different systems.
6. **Hardware** — what on-prem GPU/CPU budget is available for the LLM service?
7. **Glossary** — does HP have an approved terminology base for the target languages?
8. **Sample documents** — how soon can representative documents (ideally including scanned ones) be provided? The evaluation set gates everything in §5.

---

## 10. Summary

DDE 2.0 keeps DDE 1.0's validated extraction core and evaluation discipline, and adds the three
things transformation requires: a **reconstruction-complete document IR**, an **on-prem
translate-and-redact stage**, and a **layout-preserving renderer** — measured by layout retention,
translation accuracy, and PII recall against real baselines with significance testing.

The single highest-risk item is not translation quality; it is whether the IR is lossless enough to
rebuild a document. That is why it comes first, and why it has a pixel-identical round-trip test
attached to it.
