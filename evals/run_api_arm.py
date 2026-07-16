"""The Groq Llama-3.3-70B few-shot reference arm — "the expensive API model
this fine-tune replaces". Produces the same completion-record JSONL as the
vLLM arms so downstream scoring is arm-agnostic.

Free-tier friendly: paced call starts (TPM budget), sha256-keyed SQLite cache
(interrupt + resume for free), exponential backoff on 429s.

Usage:
    python evals/run_api_arm.py                 # full groq_subset (550 rows)
    python evals/run_api_arm.py --limit 5       # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.cache import ResponseCache, make_key                      # noqa: E402
from evals.common import GEN_CONFIG, completions_path, make_record, write_jsonl  # noqa: E402
from prompts.triage_prompt import CATEGORIES, SYSTEM_PROMPT, build_messages, load_few_shot  # noqa: E402

MODEL = "llama-3.3-70b-versatile"
SUBSET = "groq_subset"
CONCURRENCY = 2
SPACING_S = 4.5   # Groq free tier ~6k TPM; few-shot prompt ~1.3k tokens/call
MAX_RETRIES = 5


class Pacer:
    """Enforces a minimum interval between call starts (token/min budgets)."""

    def __init__(self, spacing_s: float):
        self.spacing_s = spacing_s
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._next_start - now
            self._next_start = max(now, self._next_start) + self.spacing_s
        if wait > 0:
            await asyncio.sleep(wait)


def call_groq(client, messages: list[dict]) -> tuple[str, dict, float]:
    delay = 6.0
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            result = client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=GEN_CONFIG["temperature"],
                max_tokens=GEN_CONFIG["max_tokens"],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            usage = {"prompt_tokens": result.usage.prompt_tokens,
                     "completion_tokens": result.usage.completion_tokens}
            return result.choices[0].message.content, usage, latency_ms
        except Exception as exc:  # 429s and transient 5xx
            last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} attempts: {last}")


async def run_case(row, client, few_shot, fs_fingerprint, cache, sem, pacer) -> dict:
    key = make_key(SYSTEM_PROMPT, MODEL, GEN_CONFIG, fs_fingerprint, row["text"])
    cached = cache.get(key)
    if cached is None:
        async with sem:
            await pacer.wait()
            messages = build_messages(row["text"], few_shot=few_shot)
            text, usage, latency_ms = await asyncio.to_thread(call_groq, client, messages)
        cached = {"text": text, "usage": usage, "latency_ms": latency_ms}
        cache.put(key, cached)
        tag = ""
    else:
        tag = " (cached)"
    rec = make_record(
        case_id=row["case_id"], arm="groq70b", model=MODEL,
        system_prompt=SYSTEM_PROMPT, ticket=row["text"],
        gold_category=row["label"], gold_reply=row["reply"],
        completion_text=cached["text"], usage=cached["usage"],
        latency_ms=cached["latency_ms"], categories=CATEGORIES,
    )
    ok = "ok" if rec["parsed"] else f"PARSE-FAIL ({rec['parse_error']})"
    print(f"  [{ok}] {row['case_id']}{tag}")
    return rec


async def main() -> None:
    import pandas as pd
    from dotenv import load_dotenv
    from groq import Groq

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(Path(__file__).parents[1] / ".env")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    df = pd.read_csv(Path(__file__).parents[1] / "data" / "eval" / f"{SUBSET}.csv")
    if args.limit:
        df = df.head(args.limit)

    few_shot = load_few_shot()
    fs_fingerprint = hashlib.sha256(
        json.dumps(few_shot, sort_keys=True).encode()).hexdigest()

    cache = ResponseCache()
    sem = asyncio.Semaphore(CONCURRENCY)
    pacer = Pacer(SPACING_S)
    t0 = time.perf_counter()
    records = await asyncio.gather(*[
        run_case(row, client, few_shot, fs_fingerprint, cache, sem, pacer)
        for _, row in df.iterrows()
    ])
    cache.close()

    out = completions_path("groq70b", "subset")
    write_jsonl(out, sorted(records, key=lambda r: r["case_id"]))
    valid = sum(1 for r in records if r["parsed"])
    print(f"\nWrote {out} — {len(records)} records, {valid} valid JSON, "
          f"{(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
