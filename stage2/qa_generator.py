"""
qa_generator.py — Generate Q&A pairs from chunks using Ollama.
Reads settings from config["stage2"]["llm"].

IMPORTANT: Uses /api/chat endpoint (not /api/generate).
The /api/generate endpoint has a documented bug where it IGNORES think:false
for Qwen models — see https://github.com/ollama/ollama/issues/14793
This caused massive slowdowns on Qwen 3.5/3.6 (thinking mode runs anyway,
burning the entire num_predict budget on hidden reasoning).
The /api/chat endpoint correctly disables thinking mode.

For extra safety we also inject "/no_think" into the system prompt as a
belt-and-suspenders fallback — some Qwen builds need the prompt-level marker.

FIX (Fix #1): The QAPair record and JSONL export both include `chunk_text`
so the downstream evaluation pipeline can verify faithfulness without
re-loading and re-chunking the source markdown.
"""
import json, random, requests, re
from typing import List, Dict, Optional
from dataclasses import dataclass
from chunker import Chunk

@dataclass
class QAPair:
    question: str
    answer: str
    system_role: str
    category: str
    source_file: str
    chunk_ref: str
    page_hint: int = 0
    chunk_text: str = ""        # Fix #1 — needed by evaluation pipeline

PROMPTS = {
    "procedure": (
        "You are reading a technical printer manual. Based ONLY on the text below, "
        "generate {n} question-answer pairs about the procedure described.\n"
        "Each question should ask HOW to perform a step or WHAT to do.\n"
        "Each answer must be grounded in the provided text — do NOT add information not present.\n\n"
        "TEXT:\n{chunk}\n\n"
        "Respond ONLY with a JSON array: [{{\"question\": \"...\", \"answer\": \"...\"}}]"
    ),
    "spec": (
        "You are reading a technical printer manual. Based ONLY on the text below, "
        "generate {n} question-answer pairs about specifications or technical details.\n"
        "Questions should ask WHAT a specification is, or what values/limits apply.\n\n"
        "TEXT:\n{chunk}\n\n"
        "Respond ONLY with a JSON array: [{{\"question\": \"...\", \"answer\": \"...\"}}]"
    ),
    "rule_error": (
        "You are reading a technical printer manual. Based ONLY on the text below, "
        "generate {n} question-answer pairs about warnings, cautions, errors, or troubleshooting.\n"
        "Questions should ask WHAT to avoid, WHAT causes an issue, or HOW to fix it.\n\n"
        "TEXT:\n{chunk}\n\n"
        "Respond ONLY with a JSON array: [{{\"question\": \"...\", \"answer\": \"...\"}}]"
    ),
    "figure": (
        "You are reading a technical printer manual. The text below accompanies a figure/image. "
        "Based ONLY on the text, generate {n} question-answer pairs.\n\n"
        "TEXT:\n{chunk}\n\n"
        "Respond ONLY with a JSON array: [{{\"question\": \"...\", \"answer\": \"...\"}}]"
    ),
}

# Inject /no_think at top of system prompt — second-line defence in case the
# Ollama think:false flag is ignored for the active model build.
_NO_THINK_SYSTEM = (
    "/no_think\n"
    "You generate factual Q&A pairs from technical text. "
    "Write questions that someone unfamiliar with the document would ask. "
    "Vary question phrasing — use 'how', 'what', 'why', 'when' as appropriate. "
    "Output ONLY a JSON array starting with '['. Do not add reasoning."
)


def _call_ollama(prompt: str, config: Dict) -> Optional[str]:
    """
    POST to /api/chat (not /api/generate) so think:false is honoured.
    Returns the assistant's content string, or None on failure.
    """
    llm = config.get("stage2", {}).get("llm", {})
    base_url = llm.get("base_url", "http://localhost:11434")
    model = llm.get("model", "qwen3.5:9b")
    temp = llm.get("temperature", 0.3)
    max_tok = llm.get("max_tokens", 4096)
    timeout = llm.get("timeout", 600)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _NO_THINK_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,                # honoured on /api/chat
        "options": {
            "temperature": temp,
            "num_predict": max_tok,
        },
    }

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # /api/chat returns {"message": {"role": "assistant", "content": "..."}, ...}
        return data.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Cannot connect to Ollama at {base_url}. Is it running?")
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Ollama request timed out.")
    except Exception as e:
        print(f"  [ERROR] Ollama call failed: {e}")
    return None


def _parse_qa_json(raw: str) -> List[dict]:
    if not raw:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    # Strip any leaked <think> blocks just in case
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [x for x in arr if isinstance(x, dict) and "question" in x and "answer" in x]
    except json.JSONDecodeError:
        return []


def generate_qa_for_chunk(chunk: Chunk, config: Dict, n_pairs: int = 2) -> List[QAPair]:
    llm = config.get("stage2", {}).get("llm", {})
    roles = llm.get("system_roles", ["You are a helpful assistant."])

    template = PROMPTS.get(chunk.category, PROMPTS["procedure"])
    prompt = template.format(n=n_pairs, chunk=chunk.text)
    raw = _call_ollama(prompt, config)
    print(f"  [DEBUG] Raw response ({len(raw) if raw else 0} chars): {(raw or '')[:300]}")
    parsed = _parse_qa_json(raw)
    print(f"  [DEBUG] Parsed {len(parsed)} pairs")

    return [QAPair(
        question=item["question"].strip(),
        answer=item["answer"].strip(),
        system_role=random.choice(roles),
        category=chunk.category,
        source_file=chunk.source_file,
        chunk_ref=f"{chunk.heading} (chunk {chunk.chunk_index})",
        page_hint=chunk.page_hint,
        chunk_text=chunk.text,                  # Fix #1 — pass through for eval
    ) for item in parsed]


def qa_pair_to_jsonl(pair: QAPair) -> dict:
    return {
        "messages": [
            {"role": "system", "content": pair.system_role},
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.answer},
        ],
        "provenance": {
            "source_file": pair.source_file,
            "chunk": pair.chunk_ref,
            "category": pair.category,
            "page": pair.page_hint,
        },
        "chunk_text": pair.chunk_text,          # Fix #1 — exported for eval
    }
