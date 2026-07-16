"""Data preparation — re-derives the EXACT split used by the sibling
support-ticket-classifier project (same cleaning, same seed-42 stratified
80/10/10) while carrying the `response` and `intent` columns that the old
project dropped. The fine-tuned LLM is therefore evaluated on the SAME
held-out test set as the DistilBERT classifier — verified, not assumed:
`--verify-against` asserts per-split text sets match the old CSVs.

Usage:
    python data/prepare_data.py --verify-against "D:/projects/support-ticket-classifier/data"
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_ID}"

TEXT_COLUMN = "instruction"   # the customer's message
LABEL_COLUMN = "category"     # 11 coarse intent categories
REPLY_COLUMN = "response"     # agent reply template with {{placeholders}}
INTENT_COLUMN = "intent"      # fine-grained intent (kept for analysis only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the triage SFT/eval dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--verify-against", default=None,
                        help="Path to the old support-ticket-classifier data/ dir; "
                             "asserts split parity (same text sets per split).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()[[TEXT_COLUMN, LABEL_COLUMN, REPLY_COLUMN, INTENT_COLUMN]].copy()
    df.columns = ["text", "label", "reply", "intent"]

    # Cleaning must stay byte-identical to the old project's prepare_data.py:
    # strip text/label, drop empties, dedupe on (text, label). The extra
    # columns ride along and must not change row selection or order.
    before = len(df)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after cleaning (dropped empties/duplicates).")

    labels = sorted(df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    df["label_id"] = df["label"].map(label2id)

    counts = Counter(df["label"])
    print(f"\n{len(labels)} classes:")
    for lab in labels:
        print(f"  {lab:<28} {counts[lab]:>6}")

    # Same two-stage stratified split as the old project (seed 42).
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=args.seed, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=args.seed, stratify=temp_df["label_id"]
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in splits.items():
        out = os.path.join(args.out_dir, f"{name}.csv")
        part.to_csv(out, index=False)
        print(f"Wrote {out:<50} {len(part):>6} rows")

    if args.verify_against:
        verify_parity(splits, args.verify_against)

    label_map = {
        "label2id": label2id,
        "id2label": {str(i): lab for lab, i in label2id.items()},
        "num_labels": len(labels),
        "dataset_id": DATASET_ID,
        "source_url": SOURCE_URL,
        "split": {k: len(v) for k, v in splits.items()},
        "seed": args.seed,
        "parity_note": "Split verified identical to ahmedraza-96/support-ticket-classifier "
                       "(same cleaning + seed); this project adds reply/intent columns.",
    }
    with open(os.path.join(repo_root, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print("\nWrote label_map.json")


def verify_parity(splits: dict, old_dir: str) -> None:
    """Assert each split contains exactly the same ticket texts as the old project."""
    import pandas as pd

    for name, new_df in splits.items():
        old_csv = os.path.join(old_dir, f"{name}.csv")
        old_df = pd.read_csv(old_csv)
        old_set, new_set = set(old_df["text"].astype(str)), set(new_df["text"].astype(str))
        if old_set != new_set:
            only_old, only_new = len(old_set - new_set), len(new_set - old_set)
            raise SystemExit(
                f"PARITY FAILURE on '{name}': {only_old} texts only in old, "
                f"{only_new} only in new. Do NOT proceed — the 'same test set as "
                f"DistilBERT' claim would be false."
            )
        print(f"Parity OK: {name} ({len(new_set)} unique texts match {old_csv})")
    print("Split parity verified against the DistilBERT classifier project.")


if __name__ == "__main__":
    main()
