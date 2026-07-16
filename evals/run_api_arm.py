"""The Groq Llama-3.3-70B few-shot reference arm — "the expensive API model
this fine-tune replaces". Produces the same completion-record JSONL as the
vLLM arms so downstream scoring is arm-agnostic.

Free-tier friendly: the binding constraint is Groq's 100k tokens-per-DAY cap
on this model, so the arm runs on the 176-row judged subset with a 3-shot
prompt (~140k tokens total) and waits out 429 windows instead of failing —
combined with the sha256-keyed SQLite cache the run survives any interruption
and simply resumes where it left off, even across the daily reset.

Usage:
    python evals/run_api_arm.py                 # judged subset (176 rows)
    python evals/run_api_arm.py --limit 5       # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.cache import ResponseCache, make_key                      # noqa: E402
from evals.common import GEN_CONFIG, completions_path, make_record, write_jsonl  # noqa: E402
from prompts.triage_prompt import CATEGORIES, SYSTEM_PROMPT, build_messages, load_few_shot  # noqa: E402

MODEL = "llama-3.3-70b-versatile"
SUBSET = "judged_subset"   # 176 rows — sized for Groq's 100k tokens/day cap
FEW_SHOT_N = 3             # 3-shot: the 3 most common categories, format anchors
FEW_SHOT_LABELS = ["ORDER", "REFUND", "SHIPPING"]
CONCURRENCY = 2
SPACING_S = 4.5
MAX_RETRIES = 8


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


def retry_after_s(exc: Exception) -> float | None:
    """Extract Groq's 'Please try again in 9m12.96s' hint from a 429 message."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


MAX_RATE_LIMIT_WAIT_S = 6 * 3600  # give up on a single call after 6h of 429 waits


def call_groq(client, messages: list[dict]) -> tuple[str, dict, float]:
    delay = 6.0
    errors = 0
    waited = 0.0
    last = None
    while True:
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
        except Exception as exc:
            last = exc
            hint = retry_after_s(exc)
            if hint is not None:
                # Tokens-per-day 429: Groq tells us exactly how long to wait.
                # These don't count as errors — waiting out the daily window is
                # the expected behavior on the free tier (cache keeps progress).
                wait = hint + 10
                waited += wait
                if waited > MAX_RATE_LIMIT_WAIT_S:
                    raise RuntimeError(f"Rate-limited beyond {MAX_RATE_LIMIT_WAIT_S}s: {exc}")
                print(f"    rate-limited: waiting {wait:.0f}s "
                      f"({waited/60:.0f} min total so far)")
                time.sleep(wait)
                continue
            errors += 1
            if errors >= MAX_RETRIES:
                break
            time.sleep(delay)
            delay = min(delay * 2, 120)
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

    all_shots = {ex["category"]: ex for ex in load_few_shot()}
    few_shot = [all_shots[lab] for lab in FEW_SHOT_LABELS][:FEW_SHOT_N]
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
