"""
rag_retriever.py — Orchestrates embedding + Chroma query + score filtering.

Strategy for the HYBRID index:
  1. Run ONE top-k query across the whole collection (chunks + qa_pairs mixed).
  2. Apply min_score threshold.
  3. Optionally promote a perfectly-matching QA-pair hit (score > qa_fast_path)
     to the top of the list — this implements the "FAQ direct match" behaviour.

Reads settings from config["stage4"]["retrieval"].
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

from chroma_indexer import ChromaIndexer


@dataclass
class Hit:
    text: str
    score: float
    page: int
    section: str
    source_file: str
    content_type: str
    kind: str            # "chunk" | "qa_pair"
    id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGRetriever:
    """Embeds the query and fetches top-k hits from Chroma, score-filtered."""

    def __init__(self, config: Dict, indexer: Optional[ChromaIndexer] = None):
        s4 = config.get("stage4", {})
        r  = s4.get("retrieval", {})
        self.top_k          = r.get("top_k", 5)
        self.min_score      = r.get("min_score", 0.3)
        self.qa_fast_path   = r.get("qa_fast_path_score", 0.85)
        self.max_final_hits = r.get("max_final_hits", 5)
        self.indexer = indexer if indexer is not None else ChromaIndexer(config)

    # ── Public API ───────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Hit]:
        """Return score-filtered hits, ranked by cosine similarity.
        Empty list means 'nothing confident enough' — the AnswerGenerator
        will refuse rather than hallucinate.
        """
        raw = self.indexer.query(query, k=self.top_k * 2)  # over-fetch, then trim
        hits: List[Hit] = []
        for r in raw:
            if r["score"] < self.min_score:
                continue
            hits.append(Hit(
                text=r["text"],
                score=float(r["score"]),
                page=int(r.get("page", 0)),
                section=r.get("section", ""),
                source_file=r.get("source_file", ""),
                content_type=r.get("content_type", "text"),
                kind=r.get("kind", "chunk"),
                id=r.get("id", ""),
            ))
            if len(hits) >= self.max_final_hits:
                break

        # Promote a confident QA-pair match (if any) to the top
        for i, h in enumerate(hits):
            if h.kind == "qa_pair" and h.score >= self.qa_fast_path and i > 0:
                hits.insert(0, hits.pop(i))
                break

        return hits
