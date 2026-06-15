"""
llm_rephraser.py — Generate paraphrased question variations using Ollama.
Reads settings from config["stage3"]["llm"] and config["stage3"]["augmentation"].

For each accepted Q&A pair, asks the LLM to produce N paraphrased versions
of the question that preserve the original intent. The answer text is
preserved unchanged — only the question is varied.

Mirrors the Ollama call pattern used in stage2/qa_generator.py:
  - /api/generate endpoint
  - think: false (Qwen thinking mode off)
  - <think>...</think> tag stripping from responses
  - Graceful handling of ConnectionError / Timeout
"""
import json
import re
import requests
from typing import List, Dict, Optional


REPHRASE_PROMPT = (
    "You are paraphrasing a technical question from an HP printer manual dataset. "
    "Rewrite the question below in {n} alternative ways that PRESERVE its exact meaning. "
    "Vary the phrasing, word order, and syntactic form. Do NOT change what is being asked. "
    "Do NOT add new information. Do NOT answer the question.\n\n"
    "ORIGINAL QUESTION:\n{question}\n\n"
    "Respond ONLY with a JSON array of {n} strings: [\"variation 1\", \"variation 2\", ...]"
)


class LLMRephraser:
    """Wraps Ollama for question paraphrasing."""

    def __init__(self, config: Dict):
        s3 = config.get("stage3", {})
        llm = s3.get("llm", {})
        self.base_url = llm.get("base_url", "http://localhost:11434")
        self.model = llm.get("model", "qwen3.5:9b")
        self.fallback_model = llm.get("fallback_model", self.model)
        self.temperature = llm.get("temperature", 0.7)
        self.max_tokens = llm.get("max_tokens", 512)
        self.timeout = llm.get("timeout", 600)

        aug = s3.get("augmentation", {})
        self.n_variations = aug.get("variations_per_qa", 3)

    # ── Public API ───────────────────────────────────────────────────

    def rephrase(self, question: str, n: Optional[int] = None) -> List[str]:
        """Return up to n paraphrased variations of the input question."""
        n = n if n is not None else self.n_variations
        prompt = REPHRASE_PROMPT.format(n=n, question=question.strip())
        raw = self._call_ollama(prompt, self.model)
        if not raw:
            # Retry once with fallback model if different
            if self.fallback_model and self.fallback_model != self.model:
                raw = self._call_ollama(prompt, self.fallback_model)
        return self._parse_variations(raw, n)

    # ── Internals ────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, model: str) -> Optional[str]:
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json={  # ← /api/chat
                "model": model,
                "messages": [
                    {"role": "system", "content": "/no_think\nYou paraphrase technical questions. Output ONLY a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "think": False,                # ← honored by /api/chat
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # /api/chat returns {"message": {"role": "assistant", "content": "..."}, ...}
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            print(f"  [ERROR] Cannot connect to Ollama at {self.base_url}. Is it running?")
        except requests.exceptions.Timeout:
            print(f"  [ERROR] Ollama request timed out.")
        except Exception as e:
            print(f"  [ERROR] Ollama call failed: {e}")
        return None

    @staticmethod
    def _parse_variations(raw: Optional[str], n: int) -> List[str]:
        if not raw:
            return []
        text = raw.strip()
        # Strip markdown code fences and Qwen <think> blocks
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Find the JSON array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            # Fallback: look for numbered list "1. ...", "2. ..."
            lines = [re.sub(r"^\s*\d+\.\s*", "", ln).strip()
                     for ln in text.split("\n")]
            return [ln.strip('"').strip() for ln in lines if ln][:n]
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if isinstance(x, str) and x.strip()][:n]
        except json.JSONDecodeError:
            return []


# ── Functional wrapper (for convenience / matches Stage 2 style) ─────

def rephrase_question(question: str, config: Dict, n: Optional[int] = None) -> List[str]:
    """One-shot wrapper: build rephraser, rephrase, discard."""
    return LLMRephraser(config).rephrase(question, n)


if __name__ == "__main__":
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rr = LLMRephraser(cfg)
    test_q = "How do I replace the toner cartridge?"
    variations = rr.rephrase(test_q)
    print(f"Original: {test_q}")
    for i, v in enumerate(variations, 1):
        print(f"  [{i}] {v}")
