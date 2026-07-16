"""Build the SFT chat dataset from the prepared split.

- Stratified, seeded 8,000-row subsample of train.csv (~727/class): Bitext is
  heavily templated, so one epoch over 8k rows teaches the format and routing
  without memorizing phrasing (and fits a free T4 session).
- Targets are compact JSON via prompts.triage_prompt.target_json — 100% of
  training targets are machine-valid JSON, which is how the model learns the
  format without guided decoding.
- Asserts p99 rendered-token length < MAX_SEQ_LEN so nothing is truncated.

Usage:
    python data/build_sft_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from prompts.triage_prompt import SYSTEM_PROMPT, target_json  # noqa: E402

SEED = 42
TRAIN_ROWS = 8000
VAL_ROWS = 500
MAX_SEQ_LEN = 1024
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"  # only for the length check


def sample(df, n: int, seed: int):
    """Stratified proportional sample without replacement."""
    frac = n / len(df)
    out = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed))
        .sample(frac=1.0, random_state=seed)  # shuffle
        .reset_index(drop=True)
    )
    return out.head(n)


def to_chat(row) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": target_json(row["label"], row["reply"])},
        ]
    }


def main() -> None:
    import pandas as pd

    data_dir = Path(__file__).parent
    out_dir = data_dir / "sft"
    out_dir.mkdir(exist_ok=True)

    for split, n in (("train", TRAIN_ROWS), ("val", VAL_ROWS)):
        df = pd.read_csv(data_dir / f"{split}.csv")
        part = sample(df, n, SEED)
        out = out_dir / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for _, row in part.iterrows():
                f.write(json.dumps(to_chat(row), ensure_ascii=False) + "\n")
        print(f"Wrote {out} ({len(part)} rows, {part['label'].nunique()} classes)")

    check_lengths(out_dir / "train.jsonl")


def check_lengths(path: Path) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed locally — length check will run in the "
              "Kaggle notebook instead (train_qlora.py re-asserts it).")
        return
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    lengths = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            ids = tok.apply_chat_template(msgs, tokenize=True)
            lengths.append(len(ids))
    lengths.sort()
    p99 = lengths[int(len(lengths) * 0.99)]
    print(f"Token lengths: max={lengths[-1]} p99={p99} median={lengths[len(lengths)//2]}")
    assert p99 < MAX_SEQ_LEN, f"p99 token length {p99} >= {MAX_SEQ_LEN} — raise MAX_SEQ_LEN"
    print(f"Length check OK (p99 < {MAX_SEQ_LEN}).")


if __name__ == "__main__":
    main()
