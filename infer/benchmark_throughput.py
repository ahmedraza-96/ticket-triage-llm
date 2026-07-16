"""Measure vLLM serving throughput on the T4 for the cost table.

Two modes against a running vLLM endpoint:
  single — one request at a time (interactive latency, TTFT via streaming)
  batched — CONCURRENCY simultaneous requests (server throughput)

Usage (inside the vLLM session):
    python infer/benchmark_throughput.py --arm tuned --model <served-model-name>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.common import GEN_CONFIG                       # noqa: E402
from prompts.triage_prompt import build_messages          # noqa: E402

SINGLE_N = 10
BATCHED_N = 64
CONCURRENCY = 16
OUT_DIR = Path(__file__).parents[1] / "artifacts" / "benchmarks"


async def timed_call(client, model, ticket, stream=False):
    t0 = time.perf_counter()
    ttft = None
    if stream:
        out_tokens = 0
        async with client.chat.completions.stream(
            model=model, messages=build_messages(ticket),
            temperature=GEN_CONFIG["temperature"], max_tokens=GEN_CONFIG["max_tokens"],
        ) as s:
            async for event in s:
                if event.type == "content.delta" and ttft is None:
                    ttft = time.perf_counter() - t0
            final = await s.get_final_completion()
            out_tokens = final.usage.completion_tokens
    else:
        result = await client.chat.completions.create(
            model=model, messages=build_messages(ticket),
            temperature=GEN_CONFIG["temperature"], max_tokens=GEN_CONFIG["max_tokens"],
        )
        out_tokens = result.usage.completion_tokens
    return time.perf_counter() - t0, ttft, out_tokens


async def main() -> None:
    import pandas as pd
    from openai import AsyncOpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "tuned"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    df = pd.read_csv(Path(__file__).parents[1] / "data" / "eval" / "test_full.csv")
    tickets = df["text"].head(BATCHED_N).tolist()
    client = AsyncOpenAI(base_url=args.base_url, api_key="local")

    # Warmup
    await timed_call(client, args.model, tickets[0])

    single = []
    for t in tickets[:SINGLE_N]:
        dur, ttft, toks = await timed_call(client, args.model, t, stream=True)
        single.append({"s": dur, "ttft_s": ttft, "tokens": toks})

    sem = asyncio.Semaphore(CONCURRENCY)

    async def batched_one(t):
        async with sem:
            return await timed_call(client, args.model, t)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[batched_one(t) for t in tickets])
    wall = time.perf_counter() - t0
    total_tokens = sum(r[2] for r in results)

    report = {
        "arm": args.arm, "model": args.model, "gpu": "T4 (Kaggle)",
        "single": {
            "n": SINGLE_N,
            "avg_ttft_s": round(sum(x["ttft_s"] for x in single) / SINGLE_N, 3),
            "output_tokens_per_s": round(
                sum(x["tokens"] for x in single) / sum(x["s"] for x in single), 1),
        },
        "batched": {
            "n": BATCHED_N, "concurrency": CONCURRENCY,
            "wall_s": round(wall, 1),
            "output_tokens_per_s": round(total_tokens / wall, 1),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.arm}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
