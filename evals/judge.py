"""LLM-as-judge for reply quality — grades the drafted customer reply of every
arm on the shared judged_subset (identical cases across arms).

Criteria (1-5 each):
  correctness   — the reply addresses the ticket's actual issue and is
                  consistent with the gold reference reply's intent.
  tone          — professional, empathetic, customer-appropriate.
  policy_safety — invents NO specifics (order numbers, links, phone numbers,
                  commitments). Using {{placeholder}} slots for unknown
                  specifics is CORRECT behavior and must score well.

Deterministic gate: records that failed JSON parsing are never judged (they
already scored zero on validity); saves judge quota. Judge = Groq
openai/gpt-oss-120b (separate per-model quota from the 70B arm), cached.

Usage:
    python evals/judge.py                # all arms, judged_subset cases
    python evals/judge.py --limit 5      # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.cache import ResponseCache, make_key            # noqa: E402
from evals.common import completions_path, read_jsonl      # noqa: E402
from evals.run_api_arm import Pacer, retry_after_s         # noqa: E402

JUDGE_MODEL = "openai/gpt-oss-120b"
CONCURRENCY = 2
SPACING_S = 4.0
MAX_RETRIES = 5
OUT_PATH = Path(__file__).parent / "runs" / "judgments.json"

JUDGE_PROMPT = """\
You are a strict quality judge for customer-support reply drafts. Grade the
CANDIDATE REPLY below against the customer's ticket and the REFERENCE REPLY
(written by the vendor's own support team; treat its intent as ground truth,
but do not punish different wording).

CUSTOMER TICKET:
{ticket}

REFERENCE REPLY (ground-truth intent):
{gold_reply}

CANDIDATE REPLY to grade:
{reply}

Score each criterion from 1 (terrible) to 5 (perfect):
1. correctness — the candidate addresses the ticket's actual issue and its
   guidance is consistent with the reference reply's intent.
2. tone — professional, empathetic, appropriate for a customer.
3. policy_safety — the candidate invents NO concrete specifics (order numbers,
   URLs, phone numbers, refund amounts, dates, promises). Placeholder slots
   like {{{{Order Number}}}} are the CORRECT way to reference unknown specifics
   and must NOT be penalized. Any invented specific caps this at 2.

Return ONLY a JSON object, no other text:
{{"correctness": n, "tone": n, "policy_safety": n, "reasoning": "1-3 sentences citing specifics"}}
"""


def call_judge(client, prompt: str) -> dict:
    delay = 6.0
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            result = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            scores = json.loads(result.choices[0].message.content)
            return {
                "correctness": int(scores["correctness"]),
                "tone": int(scores["tone"]),
                "policy_safety": int(scores["policy_safety"]),
                "reasoning": scores.get("reasoning", ""),
                "judge_tokens": {"prompt": result.usage.prompt_tokens,
                                 "completion": result.usage.completion_tokens},
            }
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES - 1:
                hint = retry_after_s(exc)  # daily-cap 429s say how long to wait
                wait = hint + 10 if hint is not None else delay
                time.sleep(wait)
                delay = min(delay * 2, 120)
    raise RuntimeError(f"Judge failed after {MAX_RETRIES} attempts: {last}")


async def judge_record(rec: dict, client, cache, sem, pacer) -> dict:
    prompt = JUDGE_PROMPT.format(ticket=rec["ticket"], gold_reply=rec["gold_reply"],
                                 reply=rec["parsed"]["reply"])
    key = make_key(prompt, JUDGE_MODEL, {"temperature": 0.0}, "judge-v1", rec["case_id"])
    cached = cache.get(key)
    if cached is None:
        async with sem:
            await pacer.wait()
            cached = await asyncio.to_thread(call_judge, client, prompt)
        cache.put(key, cached)
        tag = ""
    else:
        tag = " (cached)"
    print(f"  [{rec['arm']}] {rec['case_id']}{tag} "
          f"c={cached['correctness']} t={cached['tone']} p={cached['policy_safety']}")
    return {"case_id": rec["case_id"], "arm": rec["arm"], **cached}


async def main() -> None:
    import pandas as pd
    from dotenv import load_dotenv
    from groq import Groq

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(Path(__file__).parents[1] / ".env")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    judged_ids = set(pd.read_csv(
        Path(__file__).parents[1] / "data" / "eval" / "judged_subset.csv")["case_id"])
    if args.limit:
        judged_ids = set(sorted(judged_ids)[:args.limit])

    arm_files = {"base": completions_path("base", "test_full"),
                 "tuned": completions_path("tuned", "test_full"),
                 "groq70b": completions_path("groq70b", "subset")}

    todo, skipped = [], 0
    for arm, path in arm_files.items():
        if not path.exists():
            print(f"NOTE: {path.name} missing — skipping arm '{arm}'")
            continue
        for rec in read_jsonl(path):
            if rec["case_id"] not in judged_ids:
                continue
            if rec["parsed"] is None:
                skipped += 1        # deterministic gate: unparseable ⇒ not judged
                continue
            todo.append(rec)

    print(f"Judging {len(todo)} replies ({skipped} skipped at the parse gate)")
    cache = ResponseCache()
    sem = asyncio.Semaphore(CONCURRENCY)
    pacer = Pacer(SPACING_S)
    results = await asyncio.gather(*[
        judge_record(rec, client, cache, sem, pacer) for rec in todo
    ])
    cache.close()

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"judge_model": JUDGE_MODEL, "skipped_at_gate": skipped,
                   "judgments": sorted(results, key=lambda r: (r["arm"], r["case_id"]))},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_PATH} ({len(results)} judgments)")


if __name__ == "__main__":
    asyncio.run(main())
