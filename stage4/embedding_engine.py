"""
embedding_engine.py — Thin wrapper around sentence-transformers.
Reads settings from config["stage4"]["embedding"].

Default model: all-MiniLM-L6-v2 (384-d, matches Stage 2 dedup filter).
Lazy-loaded on first encode call so that modules importing this don't pay
the ~90 MB download cost until they actually need embeddings.
"""
from typing import List, Dict, Iterable
import numpy as np


class EmbeddingEngine:
    """Sentence-transformer wrapper producing L2-normalised 384-d vectors."""

    def __init__(self, config: Dict):
        s4 = config.get("stage4", {})
        emb = s4.get("embedding", {})
        self.model_name = emb.get("model", "all-MiniLM-L6-v2")
        self.batch_size = emb.get("batch_size", 32)
        self.device     = emb.get("device", "cpu")
        self._model     = None

    # ── Public API ───────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Single-string embedding. Shape: (dim,)."""
        vec = self._encode([text])
        return vec[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding. Shape: (n, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    @property
    def dim(self) -> int:
        return 384  # all-MiniLM-L6-v2

    # ── Internals ────────────────────────────────────────────────────

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # so dot-product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


if __name__ == "__main__":
    cfg = {"stage4": {"embedding": {"model": "all-MiniLM-L6-v2"}}}
    e = EmbeddingEngine(cfg)
    v = e.embed_text("How do I replace the toner cartridge?")
    print(f"embed_text shape: {v.shape}, norm: {np.linalg.norm(v):.4f}")
    vs = e.embed_batch(["question one", "question two", "question three"])
    print(f"embed_batch shape: {vs.shape}")
    # Self-similarity ≈ 1.0
    sim = float(np.dot(v, v))
    print(f"self-similarity: {sim:.4f} (expect ≈ 1.0)")
