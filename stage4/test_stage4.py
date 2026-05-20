"""
test_stage4.py — Offline smoke test for Stage 4 modules.

Verifies module structure and data flow WITHOUT making LLM calls.
Run this before the full index build.

Usage:
    cd stage4
    python test_stage4.py
"""
import os
import sys
import json
import tempfile
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


# ── Test 1 — imports ─────────────────────────────────────────────────

def test_imports():
    print("[1/6] Test imports...")
    from offline_guard    import OfflineGuard, assert_offline
    from embedding_engine import EmbeddingEngine
    from markdown_chunker import MarkdownChunker, RetrievalChunk, load_qa_pairs_as_retrieval_chunks
    from chroma_indexer   import ChromaIndexer
    from rag_retriever    import RAGRetriever, Hit
    from answer_generator import AnswerGenerator, Answer, REFUSAL_TEXT
    from stage4_pipeline  import run_index_pipeline, answer_query
    print("      ✓ all module imports ok")


# ── Test 2 — offline guard ───────────────────────────────────────────

def test_offline_guard():
    print("[2/6] Test offline_guard...")
    from offline_guard import OfflineGuard, _is_loopback

    # loopback detection
    assert _is_loopback("localhost")
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("127.5.5.5")
    assert _is_loopback("::1")
    assert not _is_loopback("google.com")
    assert not _is_loopback("8.8.8.8")
    print("      ✓ _is_loopback classifier correct")

    # install/uninstall
    g = OfflineGuard(strict=True)
    g.install()
    import socket
    try:
        try:
            socket.create_connection(("93.184.216.34", 80), timeout=0.1)
            assert False, "external connection should have been blocked"
        except RuntimeError as e:
            assert "OfflineGuard" in str(e)
            print(f"      ✓ blocked external: {str(e)[:60]}...")
    finally:
        g.uninstall()


# ── Test 3 — markdown chunker ────────────────────────────────────────

def test_chunker():
    print("[3/6] Test markdown_chunker...")
    from markdown_chunker import MarkdownChunker, RetrievalChunk

    tmp = tempfile.mkdtemp()
    try:
        md_path = os.path.join(tmp, "test.md")
        with open(md_path, "w") as f:
            f.write(
                "<!-- page: 14 -->\n"
                "### Printer views\n"
                "The HP LaserJet E877 has a front panel with 15 labeled parts. "
                "Item 7 is the easy-access USB port which allows firmware updates.\n"
                "\n"
                "<!-- page: 35 -->\n"
                "### Replace the toner cartridge\n"
                "1. Open the front door.\n"
                "2. Press the release button.\n"
                "3. Remove the old cartridge.\n"
                "4. Insert the new cartridge.\n"
                "5. Close the front door.\n"
            )

        cfg = {"stage4": {"chunking": {"max_tokens": 512, "min_chunk_chars": 20}}}
        ck = MarkdownChunker(cfg)
        chunks = ck.chunk_file(md_path)
        assert len(chunks) == 2, f"expected 2 sections → 2 chunks, got {len(chunks)}"
        print(f"      ✓ chunked 2 sections into {len(chunks)} RetrievalChunks")

        # Verify page/section propagation
        by_page = {c.page: c for c in chunks}
        assert 14 in by_page and 35 in by_page
        assert "Printer views" in by_page[14].section
        assert "toner cartridge" in by_page[35].section.lower()
        print(f"      ✓ page numbers propagated (14, 35)")
        print(f"      ✓ sections inferred from headings")

        # Verify content_type classification
        assert by_page[35].content_type == "procedure", \
            f"expected procedure, got {by_page[35].content_type}"
        print(f"      ✓ content_type classification (p.35 → procedure)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 4 — QA pair loader ──────────────────────────────────────────

def test_qa_loader():
    print("[4/6] Test QA-pair loader...")
    from markdown_chunker import load_qa_pairs_as_retrieval_chunks

    tmp = tempfile.mkdtemp()
    try:
        jsonl_path = os.path.join(tmp, "qa.jsonl")
        with open(jsonl_path, "w") as f:
            rec = {
                "messages": [
                    {"role": "system",    "content": "You are a technician."},
                    {"role": "user",      "content": "What is Item 7?"},
                    {"role": "assistant", "content": "Item 7 is the USB port."},
                ],
                "provenance": {
                    "source_file": "hp-e877.md",
                    "chunk": "Printer views (chunk 0)",
                    "category": "spec",
                    "page": 14,
                },
            }
            f.write(json.dumps(rec) + "\n")

        chunks = load_qa_pairs_as_retrieval_chunks(jsonl_path)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.kind == "qa_pair"
        assert c.page == 14
        assert "Q:" in c.text and "A:" in c.text
        print(f"      ✓ QA pair loaded as RetrievalChunk(kind='qa_pair', page=14)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 5 — answer generator with no hits (refusal) ────────────────

def test_answer_refusal():
    print("[5/6] Test AnswerGenerator refusal on empty hits...")
    from answer_generator import AnswerGenerator, Answer, REFUSAL_TEXT

    cfg = {"stage4": {"llm": {"model": "qwen3.5:9b"}}}
    gen = AnswerGenerator(cfg)
    ans = gen.answer("completely off-topic question", hits=[])
    assert ans.refused is True
    assert REFUSAL_TEXT in ans.text
    assert ans.sources == []
    print(f"      ✓ empty hits → refusal, no LLM call made")


# ── Test 6 — prompt builder ──────────────────────────────────────────

def test_prompt_builder():
    print("[6/6] Test prompt builder...")
    from rag_retriever    import Hit
    from answer_generator import AnswerGenerator

    cfg = {"stage4": {"llm": {"model": "qwen3.5:9b", "max_context_chars": 10000}}}
    gen = AnswerGenerator(cfg)
    hits = [
        Hit(text="Item 7 is the easy-access USB port.",
            score=0.87, page=14, section="Printer views",
            source_file="hp-e877.md", content_type="spec", kind="chunk"),
        Hit(text="Q: What is Item 7?\nA: Item 7 is the USB port.",
            score=0.81, page=14, section="Printer views (chunk 0)",
            source_file="hp-e877.md", content_type="spec", kind="qa_pair"),
    ]
    prompt = gen._build_prompt("What is item 7?", hits)
    assert "CONTEXT:" in prompt
    assert "QUESTION:" in prompt
    assert "p.14" in prompt
    assert "[source 1" in prompt
    assert "[source 2" in prompt
    assert "Item 7 is the easy-access USB port" in prompt
    print(f"      ✓ prompt includes both sources with page + section")
    print(f"      ✓ prompt length: {len(prompt)} chars")


# ── Runner ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STAGE 4 SMOKE TEST")
    print("=" * 60)
    try:
        test_imports()
        test_offline_guard()
        test_chunker()
        test_qa_loader()
        test_answer_refusal()
        test_prompt_builder()
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Install deps     : pip install -r stage4/requirements.txt")
    print("  2. Add stage4 block to config.yaml (see config_stage4_block.yaml)")
    print("  3. Build the index  : python -m stage4.stage4_pipeline index")
    print("  4. CLI smoke query  : python -m stage4.stage4_pipeline query \"What is Item 7?\"")
    print("  5. Launch UI        : streamlit run stage4/streamlit_app.py")


if __name__ == "__main__":
    main()
