"""
streamlit_app.py — Deep Doc Extractor RAG Chat (Stage 4).

Run with:
    streamlit run stage4/streamlit_app.py

What it does
------------
  · Enforces FR028 (offline) on startup via offline_guard.
  · Shows a chat-style UI where the user types a question about the HP E877 manual.
  · Retrieves hits from the hybrid Chroma index (chunks + QA pairs).
  · Generates a grounded answer via Qwen3.5:9B (Ollama).
  · Displays the answer AND a full grounding panel with:
        - source text (expandable)
        - page number
        - section heading
        - similarity score
        - kind (chunk or qa_pair)
  · Sidebar shows index status, session history, config knobs.
"""
# ── Enforce offline FIRST, before any HTTP library loads ─────────────
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from offline_guard import assert_offline
assert_offline(strict=True, log_path="output/stage4/offline_guard.log")

import os
import sys
import yaml
import streamlit as st

# Make sibling modules importable when run via `streamlit run`
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from chroma_indexer   import ChromaIndexer
from rag_retriever    import RAGRetriever
from answer_generator import AnswerGenerator


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deep Doc Extractor — HP E877 RAG Chat",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Config loading ───────────────────────────────────────────────────

@st.cache_data
def load_config():
    candidates = [
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.path.dirname(_this_dir), "config.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, encoding="utf-8") as f:
                return yaml.safe_load(f), c
    return None, None


cfg, cfg_path = load_config()
if cfg is None:
    st.error("Could not find config.yaml. Run the app from the repo root.")
    st.stop()


# ── Module-level singletons (loaded ONCE per process, never reloaded) ──

_INDEXER = None
_RETRIEVER = None
_GENERATOR = None
_EMBEDDER_WARMED = False


def get_indexer():
    global _INDEXER
    if _INDEXER is None:
        _INDEXER = ChromaIndexer(cfg)
    return _INDEXER


def get_retriever():
    global _RETRIEVER, _EMBEDDER_WARMED
    if _RETRIEVER is None:
        _RETRIEVER = RAGRetriever(cfg, indexer=get_indexer())
        # Force embedder to load NOW so the first query doesn't pay the cost
        if not _EMBEDDER_WARMED:
            _RETRIEVER.indexer.embedder.embed_text("warmup")
            _EMBEDDER_WARMED = True
    return _RETRIEVER


def get_generator():
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = AnswerGenerator(cfg)
    return _GENERATOR

# ── Session state ────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts {q, answer, sources, elapsed}


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📖 Deep Doc Extractor")
    st.caption("HP E877 Manual RAG · Stage 4")

    try:
        stats = get_indexer().stats()
        st.markdown("#### Index status")
        c1, c2 = st.columns(2)
        c1.metric("Chunks",   stats.get("chunks", "?"))
        c2.metric("Q&A pairs", stats.get("qa_pairs", "?"))
        st.caption(f"Collection: `{get_indexer().collection_name}`")
    except Exception as e:
        st.warning(f"Index not ready: {e}")
        st.caption("Run: `python -m stage4.stage4_pipeline index`")

    st.markdown("---")
    st.markdown("#### Runtime")
    llm = cfg.get("stage4", {}).get("llm", {})
    st.markdown(
        f"- Model: `{llm.get('model', 'qwen3.5:9b')}`\n"
        f"- Temperature: `{llm.get('temperature', 0.2)}`\n"
        f"- Offline guard: **on** (FR028)"
    )

    st.markdown("---")
    if st.session_state.history:
        st.markdown(f"#### History ({len(st.session_state.history)})")
        for i, turn in enumerate(reversed(st.session_state.history[-8:])):
            st.caption(f"**Q{len(st.session_state.history) - i}.** {turn['q'][:60]}…"
                       if len(turn['q']) > 60 else
                       f"**Q{len(st.session_state.history) - i}.** {turn['q']}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


# ── Main area ────────────────────────────────────────────────────────

st.markdown("# HP LaserJet E877  ·  Grounded Q&A")
st.caption("Ask a question about the manual. All answers are grounded in "
           "retrieved source text and cite page numbers.")

# Question input
query = st.chat_input("Ask anything about the HP E877 manual...")

# Example prompts
if not query and not st.session_state.history:
    st.markdown("##### Try asking:")
    ex_cols = st.columns(3)
    examples = [
        "What is the purpose of item 7?",
        "How do I replace the toner cartridge?",
        "What are the dimensions of Tray 2?",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, use_container_width=True):
            query = ex


# ── Answer + render ──────────────────────────────────────────────────

def render_turn(turn: dict):
    """Render one Q/A turn with the full grounding panel."""
    # Question bubble
    with st.chat_message("user"):
        st.markdown(turn["q"])

    # Answer bubble
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        meta = (f"🕐 {turn['elapsed']}s · "
                f"🔎 {len(turn['sources'])} sources · "
                f"🤖 {turn['model']}")
        st.caption(meta)

        # Grounding panel
        if turn["sources"]:
            with st.expander(f"📑 Grounding — {len(turn['sources'])} source(s)", expanded=False):
                for i, src in enumerate(turn["sources"], 1):
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                    c1.markdown(f"**[{i}]**")
                    c2.metric("Page", src.get("page", "?"))
                    c3.metric("Score", f"{src.get('score', 0):.3f}")
                    kind_label = "📄 chunk" if src.get("kind") == "chunk" else "💬 Q&A pair"
                    c4.markdown(
                        f"**{kind_label}**  ·  _{src.get('section') or 'no section'}_"
                    )
                    body = src.get("text", "")
                    if len(body) > 400:
                        body = body[:400] + "…"
                    st.text_area(
                        label=f"Source {i} text",
                        value=body,
                        height=110,
                        key=f"src_{id(turn)}_{i}",
                        label_visibility="collapsed",
                    )
                    if i < len(turn["sources"]):
                        st.markdown("---")
        else:
            st.info("No source above similarity threshold. Answer refused by design (FR028 grounding).")


# Replay history
for turn in st.session_state.history:
    render_turn(turn)


# Handle new query
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Retrieving from index and generating grounded answer..."):
        retriever = get_retriever()
        generator = get_generator()
        hits = retriever.retrieve(query)
        ans = generator.answer(query, hits)

    new_turn = {
        "q":        query,
        "answer":   ans.text,
        "sources":  ans.sources,
        "elapsed":  ans.elapsed_s,
        "model":    ans.model,
    }
    st.session_state.history.append(new_turn)

    with st.chat_message("assistant"):
        st.markdown(ans.text)
        meta = (f"🕐 {ans.elapsed_s}s · "
                f"🔎 {len(ans.sources)} sources · "
                f"🤖 {ans.model}")
        st.caption(meta)
        if ans.sources:
            with st.expander(f"📑 Grounding — {len(ans.sources)} source(s)", expanded=True):
                for i, src in enumerate(ans.sources, 1):
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                    c1.markdown(f"**[{i}]**")
                    c2.metric("Page", src.get("page", "?"))
                    c3.metric("Score", f"{src.get('score', 0):.3f}")
                    kind_label = "📄 chunk" if src.get("kind") == "chunk" else "💬 Q&A pair"
                    c4.markdown(
                        f"**{kind_label}**  ·  _{src.get('section') or 'no section'}_"
                    )
                    body = src.get("text", "")
                    if len(body) > 400:
                        body = body[:400] + "…"
                    st.text_area(
                        label=f"Source {i} text",
                        value=body,
                        height=110,
                        key=f"live_src_{i}",
                        label_visibility="collapsed",
                    )
                    if i < len(ans.sources):
                        st.markdown("---")
        else:
            st.info("No source above similarity threshold. Answer refused by design (FR028 grounding).")
