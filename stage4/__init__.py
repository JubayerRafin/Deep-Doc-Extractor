"""
Stage 4 — RAG Chat Service

Exports the main classes and orchestrators.

Team DocuForge | Woosong University Capstone 2026
Mentor: Mintae Kim (HP Korea)

Modules
-------
offline_guard       — enforces FR028 (no external network)
embedding_engine    — sentence-transformer wrapper (all-MiniLM-L6-v2, 384-d)
chroma_indexer      — ChromaDB persistent client for HYBRID index (chunks + QA pairs)
markdown_chunker    — re-chunks Stage 1 markdown for retrieval
rag_retriever       — orchestrates embed + query + score-filter
answer_generator    — grounded Qwen3.5:9B prompt with citations
streamlit_app       — UI (run with: streamlit run stage4/streamlit_app.py)

Pipelines
---------
index_pipeline      — one-time build: re-chunks Stage 1 MD, loads Stage 2 JSONL,
                      embeds both, upserts into Chroma
query_pipeline      — per-query: retrieve → generate → return Answer
"""
from .offline_guard      import OfflineGuard, assert_offline
from .embedding_engine   import EmbeddingEngine
from .chroma_indexer     import ChromaIndexer
from .markdown_chunker   import MarkdownChunker, RetrievalChunk
from .rag_retriever      import RAGRetriever, Hit
from .answer_generator   import AnswerGenerator, Answer
from .stage4_pipeline    import run_index_pipeline, answer_query

__all__ = [
    "OfflineGuard", "assert_offline",
    "EmbeddingEngine",
    "ChromaIndexer",
    "MarkdownChunker", "RetrievalChunk",
    "RAGRetriever", "Hit",
    "AnswerGenerator", "Answer",
    "run_index_pipeline", "answer_query",
]
