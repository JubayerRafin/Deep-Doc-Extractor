"""
qa_generator.py — Generate Q&A pairs from chunks using Ollama.
Reads settings from config["stage2"]["llm"].
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

def _call_ollama(prompt: str, config: Dict) -> Optional[str]:
    llm = config.get("stage2", {}).get("llm", {})
    base_url = llm.get("base_url", "http://localhost:11434")
    model = llm.get("model", "qwen3.5:9b")
    temp = llm.get("temperature", 0.7)
    max_tok = llm.get("max_tokens", 4096)

    try:
        resp = requests.post(f"{base_url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temp, "num_predict": max_tok},
        }, timeout=600)
        resp.raise_for_status()
        return resp.json().get("response", "")
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
    # Handle Qwen's think tags if present
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
    }
