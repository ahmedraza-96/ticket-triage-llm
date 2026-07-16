"""Merge the LoRA adapter into fp16 weights and push both artifacts to HF Hub.

Pushes:
  <user>/Qwen3-4B-Instruct-2507-ticket-triage-LoRA   (adapter, ~100 MB)
  <user>/Qwen3-4B-Instruct-2507-ticket-triage        (merged fp16, ~8 GB)

Model cards are rendered from train/model_card_template.md; eval numbers are
injected later by re-running with --card-only once evals/runs/latest.json
exists.

Usage (inside the training session, HF_TOKEN in env):
    python train/merge_and_push.py --adapter outputs/qlora/adapter --hf-user ahmedraza-96
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
UPSTREAM_BASE = "Qwen/Qwen3-4B-Instruct-2507"


def render_card(hf_user: str, kind: str) -> str:
    template = (REPO_ROOT / "train" / "model_card_template.md").read_text(encoding="utf-8")
    run_config_path = Path("outputs/qlora/run_config.json")
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    latest = REPO_ROOT / "evals" / "runs" / "latest.json"
    results_md = "_Evaluation in progress — results land here after the full run._"
    if latest.exists():
        tables = REPO_ROOT / "evals" / "runs" / "README_tables.md"
        if tables.exists():
            results_md = tables.read_text(encoding="utf-8")
    return (template
            .replace("{{HF_USER}}", hf_user)
            .replace("{{KIND}}", kind)
            .replace("{{RUN_CONFIG}}", json.dumps(run_config, indent=2))
            .replace("{{RESULTS}}", results_md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/qlora/adapter")
    parser.add_argument("--hf-user", default="ahmedraza-96")
    parser.add_argument("--card-only", action="store_true",
                        help="only re-render + push the model cards (CPU-safe)")
    args = parser.parse_args()

    adapter_repo = f"{args.hf_user}/Qwen3-4B-Instruct-2507-ticket-triage-LoRA"
    merged_repo = f"{args.hf_user}/Qwen3-4B-Instruct-2507-ticket-triage"

    if args.card_only:
        from huggingface_hub import HfApi
        api = HfApi()
        for repo, kind in ((adapter_repo, "adapter"), (merged_repo, "merged")):
            card = render_card(args.hf_user, kind)
            api.upload_file(path_or_fileobj=card.encode("utf-8"),
                            path_in_repo="README.md", repo_id=repo)
            print(f"Updated model card: {repo}")
        return

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=1024, load_in_4bit=True, dtype=None,
    )

    print(f"Pushing adapter -> {adapter_repo}")
    model.push_to_hub(adapter_repo, tokenizer=tokenizer)

    print(f"Merging fp16 + pushing -> {merged_repo}")
    model.push_to_hub_merged(merged_repo, tokenizer, save_method="merged_16bit")

    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind in ((adapter_repo, "adapter"), (merged_repo, "merged")):
        card = render_card(args.hf_user, kind)
        api.upload_file(path_or_fileobj=card.encode("utf-8"),
                        path_in_repo="README.md", repo_id=repo)
    print("Done: adapter + merged + model cards pushed.")


if __name__ == "__main__":
    main()
