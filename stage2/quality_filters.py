"""
quality_filters.py — Filter Q&A pairs for quality.
Reads settings from config["stage2"]["filters"].

Filters: Length → Weak Answer → Language → Hallucination → Semantic Dedup

Design notes (v3):
  - Hallucination check uses embedding cosine similarity (per HLD §11.2 / LLD §3.3).
  - SPEC chunks bypass the hallucination check entirely. Spec chunks are typically
    table fragments ("Item / Specification") which embed poorly with MiniLM —
    answers like "6 GB" or "70 ppm" are correct but score < 0.20 against fragmented
    table source text. The bypass prevents over-rejection of valid spec data.
  - Word-overlap remains as a fallback for environments without sentence-transformers,
    at a lenient 0.30 threshold.
  - Embedding model is loaded ONCE and shared across hallucination + dedup filters.
  - Language filter is OPT-IN (the project supports Korean per FR030 config).
  - Markdown normalization strips a wider range of syntax to reduce false rejections.
  - run_all_filters() prints per-filter rejection counts so tuning is observable.
"""
import re
from typing import List, Tuple, Dict
from qa_generator import QAPair


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding model (loaded once, reused across filters)
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDER_CACHE: Dict[str, object] = {}

def _get_embedder(model_name: str):
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")
        _EMBEDDER_CACHE[model_name] = model
        return model
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Markdown-aware text normalization
# ─────────────────────────────────────────────────────────────────────────────

_MD_PATTERNS = [
    (re.compile(r'```[\s\S]*?```'), ' '),                  # fenced code blocks
    (re.compile(r'`[^`]*`'), ' '),                         # inline code
    (re.compile(r'!\[[^\]]*\]\([^)]*\)'), ' '),            # image refs
    (re.compile(r'\[([^\]]*)\]\([^)]*\)'), r'\1'),         # links → keep visible text
    (re.compile(r'^\s*#{1,6}\s*', re.MULTILINE), ''),      # ATX headers
    (re.compile(r'^\s*[-*+]\s+', re.MULTILINE), ''),       # bullet markers
    (re.compile(r'^\s*\d+\.\s+', re.MULTILINE), ''),       # numbered list markers
    (re.compile(r'\*\*|__|\*|_'), ''),                     # bold / italic
    (re.compile(r'\|'), ' '),                              # table pipes
]

def _normalize(text: str) -> str:
    for pat, repl in _MD_PATTERNS:
        text = pat.sub(repl, text)
    return re.sub(r'\s+', ' ', text.lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# Filter 1 — Length
# ─────────────────────────────────────────────────────────────────────────────

def filter_length(pairs: List[QAPair], config: Dict) -> Tuple[List[QAPair], List[Tuple[QAPair, str]]]:
    """
    Length gate. Spec answers can legitimately be very short ("6 GB", "70 ppm"),
    so min_answer_length should be set low (5) when the corpus contains many specs.
    """
    f = config.get("stage2", {}).get("filters", {})
    min_q = f.get("min_question_length", 10)
    min_a = f.get("min_answer_length", 5)              # was 20 — too strict for spec answers
    max_a = f.get("max_answer_length", 2000)
    passed, rejected = [], []
    for p in pairs:
        if   len(p.question) < min_q: rejected.append((p, f"question too short ({len(p.question)}<{min_q})"))
        elif len(p.answer)   < min_a: rejected.append((p, f"answer too short ({len(p.answer)}<{min_a})"))
        elif len(p.answer)   > max_a: rejected.append((p, f"answer too long ({len(p.answer)}>{max_a})"))
        else: passed.append(p)
    return passed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Filter 2 — Weak answers (hedging phrases)
# ─────────────────────────────────────────────────────────────────────────────

def filter_weak_answers(pairs: List[QAPair], config: Dict) -> Tuple[List[QAPair], List[Tuple[QAPair, str]]]:
    pats = [p.lower() for p in config.get("stage2", {}).get("filters", {}).get("weak_answer_patterns", [])]
    if not pats:
        return pairs, []
    passed, rejected = [], []
    for p in pairs:
        al = p.answer.lower()
        hit = next((pat for pat in pats if pat in al), None)
        if hit: rejected.append((p, f"weak answer: '{hit}'"))
        else:   passed.append(p)
    return passed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Filter 3 — Language (OPT-IN)
# ─────────────────────────────────────────────────────────────────────────────

def _is_cjk(char):
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF) or      # CJK Unified Ideographs
        (0x3400 <= cp <= 0x4DBF) or      # CJK Extension A
        (0x3040 <= cp <= 0x309F) or      # Hiragana
        (0x30A0 <= cp <= 0x30FF) or      # Katakana
        (0xAC00 <= cp <= 0xD7AF) or      # Hangul Syllables
        (0xF900 <= cp <= 0xFAFF) or      # CJK Compatibility
        (0x20000 <= cp <= 0x2A6DF)       # CJK Extension B
    )

def filter_language(pairs: List[QAPair], config: Dict) -> Tuple[List[QAPair], List[Tuple[QAPair, str]]]:
    """
    Opt-in script enforcement. Off by default — the pipeline supports Korean (FR030).
    Set config["stage2"]["filters"]["enforce_language"] to "en", "ko", or "any" (=off).
    """
    f = config.get("stage2", {}).get("filters", {})
    mode = f.get("enforce_language")          # None / "en" / "ko" / "any"
    threshold = f.get("language_cjk_threshold", 0.30)
    if not mode or mode == "any":
        return pairs, []

    passed, rejected = [], []
    for p in pairs:
        text = p.question + " " + p.answer
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            passed.append(p); continue
        cjk_ratio = sum(1 for c in alpha if _is_cjk(c)) / len(alpha)
        if mode == "en" and cjk_ratio > threshold:
            rejected.append((p, f"non-English ({cjk_ratio:.0%} CJK)"))
        elif mode == "ko" and cjk_ratio < (1 - threshold):
            rejected.append((p, f"non-Korean ({cjk_ratio:.0%} CJK)"))
        else:
            passed.append(p)
    return passed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Filter 4 — Hallucination (embedding cosine, with overlap fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _word_overlap_ratio(answer: str, source: str, min_word_len: int = 4) -> float:
    """Lenient fallback when embeddings are unavailable. Substring search permits
    subword matches (e.g., 'print' → 'printer'), which is intentionally permissive."""
    src_n, ans_n = _normalize(source), _normalize(answer)
    words = set(w for w in ans_n.split() if len(w) >= min_word_len)
    if not words:
        return 1.0
    return sum(1 for w in words if w in src_n) / len(words)

def filter_hallucination(pairs: List[QAPair], chunks_map: Dict[str, str], config: Dict) -> Tuple[List[QAPair], List[Tuple[QAPair, str]]]:
    """
    Reject answers not grounded in their source chunk.

    PRIMARY: cosine(embed(answer), embed(source)) ≥ similarity_threshold (default 0.40).
             Default lowered from 0.55 → 0.40 because MiniLM under-scores technical
             printer-domain text; empirical threshold for HP E877 corpus.
    BYPASS:  spec-category chunks skip this check (table fragments embed poorly).
    FALLBACK (no sentence-transformers): word-overlap ratio ≥ overlap_threshold (0.30).
    """
    f = config.get("stage2", {}).get("filters", {})
    sim_threshold     = f.get("hallucination_similarity_threshold", 0.40)
    overlap_threshold = f.get("hallucination_overlap_threshold", 0.30)
    bypass_categories = set(f.get("hallucination_bypass_categories", ["spec"]))
    model_name        = f.get("embedding_model", "all-MiniLM-L6-v2")

    embedder = _get_embedder(model_name)
    passed, rejected = [], []

    # ── Fallback path (no embeddings available) ──────────────────────────────
    if embedder is None:
        for p in pairs:
            if p.category in bypass_categories:
                passed.append(p); continue
            src = chunks_map.get(p.chunk_ref, "")
            if not src:
                passed.append(p); continue
            ratio = _word_overlap_ratio(p.answer, src)
            if ratio < overlap_threshold:
                rejected.append((p, f"hallucination ({ratio:.0%} overlap, fallback)"))
            else:
                passed.append(p)
        return passed, rejected

    # ── Primary path: batched embedding cosine ───────────────────────────────
    import numpy as np

    # Separate pairs that bypass from pairs that need the check
    to_check_idx = []
    for i, p in enumerate(pairs):
        if p.category in bypass_categories:
            continue                                   # bypass → auto-pass below
        if not chunks_map.get(p.chunk_ref, ""):
            continue                                   # no source → auto-pass below
        to_check_idx.append(i)

    sim_lookup = {}
    if to_check_idx:
        sources = [chunks_map[pairs[i].chunk_ref] for i in to_check_idx]
        answers = [pairs[i].answer for i in to_check_idx]
        src_emb = embedder.encode(sources, normalize_embeddings=True, show_progress_bar=False)
        ans_emb = embedder.encode(answers, normalize_embeddings=True, show_progress_bar=False)
        sims = np.sum(src_emb * ans_emb, axis=1)
        sim_lookup = {to_check_idx[k]: float(sims[k]) for k in range(len(to_check_idx))}

    for i, p in enumerate(pairs):
        if i not in sim_lookup:
            passed.append(p); continue                # bypassed or no source — pass through
        s = sim_lookup[i]
        if s < sim_threshold:
            rejected.append((p, f"hallucination (sim={s:.2f}<{sim_threshold})"))
        else:
            passed.append(p)
    return passed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Filter 5 — Semantic deduplication
# ─────────────────────────────────────────────────────────────────────────────

def filter_dedup(pairs: List[QAPair], config: Dict) -> Tuple[List[QAPair], List[Tuple[QAPair, str]]]:
    f = config.get("stage2", {}).get("filters", {})
    thresh     = f.get("dedup_similarity_threshold", 0.92)
    model_name = f.get("embedding_model", "all-MiniLM-L6-v2")
    if len(pairs) <= 1:
        return pairs, []

    embedder = _get_embedder(model_name)
    if embedder is None:
        print("  [WARN] sentence-transformers not installed — skipping dedup.")
        return pairs, []

    import numpy as np
    embs = embedder.encode([p.question for p in pairs],
                           normalize_embeddings=True, show_progress_bar=False)
    sim = np.dot(embs, embs.T)
    dup = [False] * len(pairs)
    for i in range(len(pairs)):
        if dup[i]:
            continue
        for j in range(i + 1, len(pairs)):
            if not dup[j] and sim[i][j] >= thresh:
                dup[j] = True
    passed   = [p for i, p in enumerate(pairs) if not dup[i]]
    rejected = [(p, f"duplicate (sim>={thresh})") for i, p in enumerate(pairs) if dup[i]]
    return passed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator with per-filter visibility
# ─────────────────────────────────────────────────────────────────────────────

def run_all_filters(pairs: List[QAPair], chunks_map: Dict[str, str], config: Dict,
                    verbose: bool = True) -> Tuple[List[QAPair], List[Tuple[QAPair, str]]]:
    """Run all five gates in order; print per-filter rejection counts when verbose."""
    stages = [
        ("length",        lambda ps: filter_length(ps, config)),
        ("weak_answer",   lambda ps: filter_weak_answers(ps, config)),
        ("language",      lambda ps: filter_language(ps, config)),
        ("hallucination", lambda ps: filter_hallucination(ps, chunks_map, config)),
        ("dedup",         lambda ps: filter_dedup(ps, config)),
    ]
    initial = len(pairs)
    all_rej = []
    for name, fn in stages:
        before = len(pairs)
        pairs, rej = fn(pairs)
        all_rej.extend(rej)
        if verbose:
            print(f"  [{name:13s}] in={before:5d}  out={len(pairs):5d}  rejected={len(rej):4d}")
    if verbose:
        keep_rate = (len(pairs) / initial * 100) if initial else 0.0
        print(f"  ──────────────  total kept={len(pairs)}/{initial} ({keep_rate:.1f}%)")
    return pairs, all_rej
