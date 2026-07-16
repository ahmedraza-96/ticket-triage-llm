"""Shared completion-record schema + JSONL IO.

Every arm (base vLLM, tuned vLLM, Groq 70B few-shot) writes the SAME record
shape, so score/judge/report never care where a completion came from:

{
  "case_id": "t0042", "arm": "tuned", "model": "...",
  "prompt_sha256": "...",                 # sha256 of the rendered system prompt
  "ticket": "...", "gold_category": "REFUND", "gold_reply": "...",
  "completion_text": "...",               # raw model output, untouched
  "parsed": {"category": "...", "reply": "..."} | null,
  "parse_error": "..." | null,
  "usage": {"prompt_tokens": n, "completion_tokens": n},
  "latency_ms": n,
  "gen_config": {"temperature": 0.0, "max_tokens": 400}
}
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ARMS = ("base", "tuned", "groq70b")
GEN_CONFIG = {"temperature": 0.0, "max_tokens": 400}

REPO_ROOT = Path(__file__).parents[1]
COMPLETIONS_DIR = REPO_ROOT / "artifacts" / "completions"


def prompt_sha256(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def parse_completion(text: str, categories: list[str]) -> tuple[dict | None, str | None]:
    """Strict-ish parse: accept the raw text or one fenced/embedded JSON object.

    Deliberately unforgiving beyond that — JSON validity is a headline metric,
    so we must not paper over format failures.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json-decode: {exc}"
    if not isinstance(obj, dict):
        return None, "not-an-object"
    if set(obj.keys()) != {"category", "reply"}:
        return None, f"wrong-keys: {sorted(obj.keys())}"
    if not isinstance(obj["category"], str) or not isinstance(obj["reply"], str):
        return None, "wrong-value-types"
    if obj["category"] not in categories:
        return None, f"unknown-category: {obj['category']!r}"
    return obj, None


def make_record(case_id: str, arm: str, model: str, system_prompt: str,
                ticket: str, gold_category: str, gold_reply: str,
                completion_text: str, usage: dict, latency_ms: float,
                categories: list[str]) -> dict:
    parsed, parse_error = parse_completion(completion_text, categories)
    return {
        "case_id": case_id, "arm": arm, "model": model,
        "prompt_sha256": prompt_sha256(system_prompt),
        "ticket": ticket, "gold_category": gold_category, "gold_reply": gold_reply,
        "completion_text": completion_text,
        "parsed": parsed, "parse_error": parse_error,
        "usage": usage, "latency_ms": round(latency_ms, 1),
        "gen_config": dict(GEN_CONFIG),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def completions_path(arm: str, subset: str) -> Path:
    return COMPLETIONS_DIR / f"{arm}__{subset}.jsonl"
