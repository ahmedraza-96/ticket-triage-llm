"""Generate raw completions for one arm against a vLLM OpenAI-compatible
endpoint (runs inside the Kaggle GPU session; artifacts are downloaded and
committed so scoring/judging re-run on CPU forever).

Usage (inside the vLLM session):
    python infer/generate_completions.py --arm tuned --base-url http://localhost:8000/v1 \
        --model ahmedraza-96/Qwen3-4B-Instruct-2507-ticket-triage
    python infer/generate_completions.py --arm base --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 --limit 20   # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.common import GEN_CONFIG, completions_path, make_record, write_jsonl  # noqa: E402
from prompts.triage_prompt import CATEGORIES, SYSTEM_PROMPT, build_messages      # noqa: E402

CONCURRENCY = 16  # vLLM batches continuously; higher concurrency = better T4 utilization


async def run_case(row, client, model: str, arm: str, sem) -> dict:
    async with sem:
        t0 = time.perf_counter()
        result = await client.chat.completions.create(
            model=model,
            messages=build_messages(row["text"]),
            temperature=GEN_CONFIG["temperature"],
            max_tokens=GEN_CONFIG["max_tokens"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
    return make_record(
        case_id=row["case_id"], arm=arm, model=model,
        system_prompt=SYSTEM_PROMPT, ticket=row["text"],
        gold_category=row["label"], gold_reply=row["reply"],
        completion_text=result.choices[0].message.content or "",
        usage={"prompt_tokens": result.usage.prompt_tokens,
               "completion_tokens": result.usage.completion_tokens},
        latency_ms=latency_ms, categories=CATEGORIES,
    )


async def main() -> None:
    import pandas as pd
    from openai import AsyncOpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "tuned"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--subset", default="test_full")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(Path(__file__).parents[1] / "data" / "eval" / f"{args.subset}.csv")
    if args.limit:
        df = df.head(args.limit)

    client = AsyncOpenAI(base_url=args.base_url, api_key="local")
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.perf_counter()
    records = await asyncio.gather(*[
        run_case(row, client, args.model, args.arm, sem) for _, row in df.iterrows()
    ])
    records.sort(key=lambda r: r["case_id"])

    out = completions_path(args.arm, args.subset)
    write_jsonl(out, records)
    valid = sum(1 for r in records if r["parsed"])
    mins = (time.perf_counter() - t0) / 60
    print(f"Wrote {out} — {len(records)} records, {valid} valid JSON "
          f"({valid/len(records):.1%}), {mins:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
