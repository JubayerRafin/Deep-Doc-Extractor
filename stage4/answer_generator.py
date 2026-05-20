"""
answer_generator.py — Grounded Qwen3.5:9B answer with citations.

Reads settings from config["stage4"]["llm"].

Call pattern mirrors Stage 2's qa_generator.py:
  - Ollama /api/generate
  - think: false (Qwen thinking mode off, lower latency)
  - <think>...</think> tag stripping defensively
  - Graceful handling of ConnectionError / Timeout

The prompt is designed to minimise hallucination:
  - "Use ONLY the context."
  - "If the context does not contain the answer, say 'I don't have that
     information in the manual.'"
  - "Cite page numbers inline as [p.N]."
"""
import re
import requests
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any

from rag_retriever import Hit


REFUSAL_TEXT = ("I don't have that information in the HP E877 manual. "
                "Please consult a different source or rephrase the question.")

SYSTEM_PROMPT = (
    "You are an HP LaserJet E877 technical support assistant. "
    "Answer the user's question using ONLY the information in the CONTEXT below. "
    "If the CONTEXT does not contain the answer, reply exactly: "
    f"\"{REFUSAL_TEXT}\" "
    "Do not use outside knowledge. "
    "Cite page numbers inline in the form [p.N] where N is the page from the context. "
    "Be concise — 1-3 sentences for factual questions, a short numbered list for procedures."
)


@dataclass
class Answer:
    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    elapsed_s: float = 0.0
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnswerGenerator:
    """Grounded answer via Ollama; returns Answer with citations."""

    def __init__(self, config: Dict):
        s4 = config.get("stage4", {})
        llm = s4.get("llm", {})
        self.base_url    = llm.get("base_url", "http://localhost:11434")
        self.model       = llm.get("model", "qwen3.5:9b")
        self.temperature = llm.get("temperature", 0.2)
        self.max_tokens  = llm.get("max_tokens", 1024)
        self.timeout     = llm.get("timeout", 600)
        self.max_context_chars = llm.get("max_context_chars", 6000)

    # ── Public API ───────────────────────────────────────────────────

    def answer(self, query: str, hits: List[Hit]) -> Answer:
        import time
        t0 = time.time()

        # Refuse early if nothing confident retrieved
        if not hits:
            return Answer(
                text=REFUSAL_TEXT,
                sources=[],
                refused=True,
                elapsed_s=round(time.time() - t0, 2),
                model=self.model,
            )

        prompt = self._build_prompt(query, hits)
        raw = self._call_ollama(prompt)
        elapsed = round(time.time() - t0, 2)

        if raw is None:
            return Answer(
                text="(LLM call failed — check that Ollama is running.)",
                sources=[h.to_dict() for h in hits],
                refused=False,
                elapsed_s=elapsed,
                model=self.model,
            )

        text = self._clean(raw)
        refused = REFUSAL_TEXT in text or len(text.strip()) < 5
        return Answer(
            text=text,
            sources=[h.to_dict() for h in hits],
            refused=refused,
            elapsed_s=elapsed,
            model=self.model,
        )

    # ── Internals ────────────────────────────────────────────────────

    def _build_prompt(self, query: str, hits: List[Hit]) -> str:
        # Cap total context length so we don't blow past Qwen's window
        ctx_lines: List[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            head = f"[source {i} · p.{h.page} · {h.section or 'n/a'}]"
            body = h.text.strip()
            block = f"{head}\n{body}\n"
            if used + len(block) > self.max_context_chars:
                break
            ctx_lines.append(block)
            used += len(block)
        context = "\n".join(ctx_lines)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n"
            f"QUESTION: {query.strip()}\n\n"
            f"ANSWER:"
        )

    def _call_ollama(self, prompt: str) -> Optional[str]:
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": 2048,
                },
            }, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            print(f"  [AnswerGenerator] Cannot connect to Ollama at {self.base_url}.")
        except requests.exceptions.Timeout:
            print(f"  [AnswerGenerator] Ollama request timed out.")
        except Exception as e:
            print(f"  [AnswerGenerator] Ollama error: {e}")
        return None

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip <think> blocks and code fences that Qwen sometimes emits."""
        t = raw or ""
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
        t = re.sub(r"^```[a-z]*\s*", "", t.strip())
        t = re.sub(r"\s*```$", "", t)
        return t.strip()
