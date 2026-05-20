"""
augment_filter.py — Filter paraphrased question variations.
Reads settings from config["stage3"]["filter"].

Drops variations that are:
  - Too similar to the original question (no diversity gain)
  - Too dissimilar from the original question (semantic drift)
  - Identical to other variations in the same batch
  - Empty or trivially short

Uses sentence-transformers for embedding (same model as Stage 2 dedup).
If sentence-transformers is not installed, falls back to a token-overlap
heuristic so the pipeline still runs.
"""
import re
from typing import List, Tuple, Dict, Optional


class AugmentFilter:
    """Filters paraphrased question variations by cosine similarity to the original."""

    def __init__(self, config: Dict):
        s3 = config.get("stage3", {})
        flt = s3.get("filter", {})
        self.sim_max = flt.get("similarity_max", 0.97)  # drop near-duplicates
        self.sim_min = flt.get("similarity_min", 0.70)  # drop drift
        self.min_length = flt.get("min_variation_length", 5)
        self.model_name = flt.get("embedding_model",
                                  config.get("stage2", {}).get("filters", {})
                                       .get("embedding_model", "all-MiniLM-L6-v2"))
        self._model = None  # lazy
        self._fallback_mode = False

    # ── Public API ───────────────────────────────────────────────────

    def filter(self, original: str, variations: List[str]
               ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Return (kept, dropped) where dropped is a list of (variation, reason).
        Runs trivial-length check first, then similarity bounds, then dedup.
        """
        dropped: List[Tuple[str, str]] = []
        kept: List[str] = []

        # 1. Length and trivial filters
        candidates: List[str] = []
        seen_exact = {original.strip().lower()}
        for v in variations:
            v = (v or "").strip()
            if not v:
                dropped.append((v, "empty"))
                continue
            if len(v) < self.min_length:
                dropped.append((v, f"too short (<{self.min_length})"))
                continue
            key = v.lower()
            if key in seen_exact:
                dropped.append((v, "exact duplicate"))
                continue
            seen_exact.add(key)
            candidates.append(v)

        if not candidates:
            return [], dropped

        # 2. Similarity bounds (vs original)
        scores = self._similarities(original, candidates)
        for v, sim in zip(candidates, scores):
            if sim > self.sim_max:
                dropped.append((v, f"near-duplicate (sim={sim:.2f} > {self.sim_max})"))
            elif sim < self.sim_min:
                dropped.append((v, f"semantic drift (sim={sim:.2f} < {self.sim_min})"))
            else:
                kept.append(v)

        # 3. Dedup within kept set (avoid 3 near-identical variations)
        if len(kept) > 1:
            kept, dedup_dropped = self._dedup_within(kept)
            dropped.extend(dedup_dropped)

        return kept, dropped

    # ── Similarity backends ──────────────────────────────────────────

    def _get_model(self):
        """Lazy-load sentence-transformers; fall back to token overlap if missing."""
        if self._model is not None or self._fallback_mode:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            print("  [WARN] sentence-transformers not installed — "
                  "using token-overlap heuristic for AugmentFilter.")
            self._fallback_mode = True
        return self._model

    def _similarities(self, anchor: str, candidates: List[str]) -> List[float]:
        """Cosine similarities between anchor and each candidate."""
        model = self._get_model()
        if model is None:
            return [self._jaccard(anchor, c) for c in candidates]
        try:
            import numpy as np
            embs = model.encode([anchor] + candidates, normalize_embeddings=True)
            anchor_emb = embs[0]
            cand_embs = embs[1:]
            return [float(np.dot(anchor_emb, c)) for c in cand_embs]
        except Exception as e:
            print(f"  [WARN] Embedding failed ({e}) — falling back to Jaccard.")
            return [self._jaccard(anchor, c) for c in candidates]

    def _dedup_within(self, variations: List[str]
                      ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Drop variations that are >sim_max similar to an earlier kept variation."""
        model = self._get_model()
        kept: List[str] = []
        dropped: List[Tuple[str, str]] = []
        if model is None:
            # Fallback: Jaccard greedy
            for v in variations:
                if any(self._jaccard(v, k) > self.sim_max for k in kept):
                    dropped.append((v, f"intra-batch near-duplicate"))
                else:
                    kept.append(v)
            return kept, dropped
        try:
            import numpy as np
            embs = model.encode(variations, normalize_embeddings=True)
            kept_idxs: List[int] = []
            for i in range(len(variations)):
                is_dup = False
                for j in kept_idxs:
                    if float(np.dot(embs[i], embs[j])) > self.sim_max:
                        is_dup = True
                        break
                if is_dup:
                    dropped.append((variations[i], "intra-batch near-duplicate"))
                else:
                    kept_idxs.append(i)
            kept = [variations[i] for i in kept_idxs]
        except Exception:
            for v in variations:
                if any(self._jaccard(v, k) > self.sim_max for k in kept):
                    dropped.append((v, "intra-batch near-duplicate"))
                else:
                    kept.append(v)
        return kept, dropped

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Token-overlap fallback when embeddings are unavailable."""
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


# ── Functional wrapper ───────────────────────────────────────────────

def filter_variations(original: str, variations: List[str], config: Dict
                      ) -> Tuple[List[str], List[Tuple[str, str]]]:
    return AugmentFilter(config).filter(original, variations)


if __name__ == "__main__":
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    original = "How do I replace the toner cartridge?"
    samples = [
        "What is the procedure for replacing the toner cartridge?",  # should keep
        "How do I replace the toner cartridge?",                     # exact dup
        "What colour is the printer?",                               # drift
        "Steps to swap out the toner cartridge?",                    # should keep
        "How do I replace the toner cartridge please?",              # near-dup
    ]
    kept, dropped = filter_variations(original, samples, cfg)
    print(f"Original: {original}")
    print(f"Kept ({len(kept)}):")
    for v in kept:
        print(f"  ✓ {v}")
    print(f"Dropped ({len(dropped)}):")
    for v, r in dropped:
        print(f"  ✗ [{r}] {v}")
