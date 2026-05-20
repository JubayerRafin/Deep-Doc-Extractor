"""
chroma_indexer.py — Persistent ChromaDB client for the hybrid retrieval index.

Hybrid index: a single collection holds BOTH
  - markdown chunks (kind='chunk'), and
  - Q&A pairs      (kind='qa_pair')
Each row is distinguished by its metadata['kind']. Retrieval can either query
the whole collection (default) or filter by kind for debugging.

Reads settings from config["stage4"]["chroma"] and config["stage4"]["embedding"].
"""
import os
from typing import List, Dict, Optional, Any
import numpy as np

from markdown_chunker import RetrievalChunk
from embedding_engine import EmbeddingEngine


class ChromaIndexer:
    """Wraps a persistent Chroma collection for hybrid retrieval."""

    def __init__(self, config: Dict):
        s4 = config.get("stage4", {})
        ch = s4.get("chroma", {})
        self.persist_dir     = ch.get("persist_dir", "output/stage4/chroma")
        self.collection_name = ch.get("collection_name", "hp_e877_manual")
        self.embedder        = EmbeddingEngine(config)
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = None
        self._collection = None

    # ── Client / collection handling ─────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            # Use cosine distance (matches normalised embeddings)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Public API ───────────────────────────────────────────────────

    def index(self, chunks: List[RetrievalChunk], batch: int = 64) -> int:
        """Embed and upsert a list of chunks. Returns the number indexed."""
        if not chunks:
            return 0
        col = self._get_collection()
        total = 0
        for i in range(0, len(chunks), batch):
            sub = chunks[i:i + batch]
            texts = [c.text for c in sub]
            embs  = self.embedder.embed_batch(texts)
            metas = [{
                "source_file":  c.source_file,
                "page":         int(c.page or 0),
                "section":      c.section or "",
                "content_type": c.content_type or "text",
                "kind":         c.kind or "chunk",
            } for c in sub]
            ids = [c.chunk_id for c in sub]
            col.upsert(
                ids=ids,
                embeddings=embs.tolist(),
                documents=texts,
                metadatas=metas,
            )
            total += len(sub)
        return total

    def query(self, query_text: str, k: int = 5,
              kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """Embed the query, run ANN search, return top-k hits as dicts with
        keys: text, score, source_file, page, section, content_type, kind.

        kind: if set, filter to 'chunk' or 'qa_pair' only.
        """
        col = self._get_collection()
        q_emb = self.embedder.embed_text(query_text).tolist()
        where = {"kind": kind} if kind else None
        # Some Chroma versions require `where` be omitted rather than None
        kwargs = {"query_embeddings": [q_emb], "n_results": k,
                  "include": ["documents", "metadatas", "distances"]}
        if where is not None:
            kwargs["where"] = where
        res = col.query(**kwargs)

        hits: List[Dict[str, Any]] = []
        if not res.get("ids") or not res["ids"][0]:
            return hits
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i] if res.get("metadatas") else {}
            dist = float(res["distances"][0][i]) if res.get("distances") else 1.0
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            hits.append({
                "id":            doc_id,
                "text":          res["documents"][0][i],
                "score":         score,
                "source_file":   meta.get("source_file", ""),
                "page":          meta.get("page", 0),
                "section":       meta.get("section", ""),
                "content_type":  meta.get("content_type", "text"),
                "kind":          meta.get("kind", "chunk"),
            })
        return hits

    def count(self) -> int:
        return self._get_collection().count()

    def reset(self) -> None:
        """Drop and re-create the collection. Use when re-indexing from scratch."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._get_collection()

    def stats(self) -> Dict[str, int]:
        """Return kind-level counts — useful for Streamlit status bar."""
        col = self._get_collection()
        # Chroma doesn't have aggregation, so we do a coarse scan
        try:
            all_rows = col.get(include=["metadatas"])
            metas = all_rows.get("metadatas") or []
            n_chunks = sum(1 for m in metas if m.get("kind") == "chunk")
            n_qa     = sum(1 for m in metas if m.get("kind") == "qa_pair")
            return {"total": len(metas), "chunks": n_chunks, "qa_pairs": n_qa}
        except Exception:
            return {"total": col.count(), "chunks": -1, "qa_pairs": -1}
