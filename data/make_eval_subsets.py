"""Freeze the eval subsets (committed to the repo, never regenerated casually).

- test_full.csv    : all 2,464 held-out test rows — deterministic metrics for
                     the base and fine-tuned arms.
- groq_subset.csv  : stratified 550 rows (50/class) — the Groq 70B few-shot
                     arm (free-tier TPM budget makes the full set impractical).
- judged_subset.csv: 176 rows (16/class), a SUBSET of groq_subset — all three
                     arms are LLM-judged on identical cases.
- prompts/few_shot_examples.json: 11 examples (1/class) drawn from TRAIN
                     (never test) for the Groq reference arm.

Usage:
    python data/make_eval_subsets.py
"""

from __future__ import annotations

import json
from pathlib import Path

SEED = 42
GROQ_PER_CLASS = 50
JUDGED_PER_CLASS = 16


def main() -> None:
    import pandas as pd

    data_dir = Path(__file__).parent
    out_dir = data_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    test = pd.read_csv(data_dir / "test.csv")
    test = test.reset_index(drop=True)
    test.insert(0, "case_id", [f"t{i:04d}" for i in range(len(test))])

    test.to_csv(out_dir / "test_full.csv", index=False)
    print(f"Wrote test_full.csv ({len(test)} rows)")

    groq = (
        test.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(GROQ_PER_CLASS, random_state=SEED))
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    groq.to_csv(out_dir / "groq_subset.csv", index=False)
    print(f"Wrote groq_subset.csv ({len(groq)} rows, {GROQ_PER_CLASS}/class)")

    judged = (
        groq.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(JUDGED_PER_CLASS, random_state=SEED))
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    judged.to_csv(out_dir / "judged_subset.csv", index=False)
    print(f"Wrote judged_subset.csv ({len(judged)} rows, {JUDGED_PER_CLASS}/class)")

    # Few-shot examples come from TRAIN — using test rows here would leak.
    train = pd.read_csv(data_dir / "train.csv")
    shots = (
        train.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(1, random_state=SEED))
        .sort_values("label")
    )
    few_shot = [
        {"ticket": r["text"], "category": r["label"], "reply": r["reply"]}
        for _, r in shots.iterrows()
    ]
    fs_path = data_dir.parent / "prompts" / "few_shot_examples.json"
    with open(fs_path, "w", encoding="utf-8") as f:
        json.dump(few_shot, f, indent=2, ensure_ascii=False)
    print(f"Wrote {fs_path} ({len(few_shot)} examples, from train split only)")


if __name__ == "__main__":
    main()
