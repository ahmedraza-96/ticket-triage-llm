"""QLoRA fine-tune of Qwen3-4B-Instruct-2507 on the triage SFT dataset.

Runs on a single free T4 (Kaggle/Colab) in ~45-60 min via Unsloth. All logic
lives here — the notebook only pip-installs, clones the repo, and calls this.

T4 notes: compute capability 7.5 ⇒ fp16 only (no bf16). Unsloth detects this;
we still assert to fail fast on misconfigured runtimes.

Fallback if Unsloth breaks (version churn): the same dataset/config trains
with plain TRL SFTTrainer + peft.LoraConfig — swap FastLanguageModel for
AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True) and
get_peft_model; expect ~2x the wall-clock.

Usage:
    python train/train_qlora.py --output-dir outputs/qlora
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prompts.triage_prompt import CATEGORIES  # noqa: E402  (smoke check uses it)

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"  # Unsloth mirror of Qwen/Qwen3-4B-Instruct-2507
MAX_SEQ_LEN = 1024
CONFIG = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
    "num_train_epochs": 1, "learning_rate": 2e-4, "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05, "weight_decay": 0.01, "seed": 42,
    "logging_steps": 25, "eval_steps": 200, "save_steps": 200,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/qlora")
    parser.add_argument("--max-steps", type=int, default=0, help="override for smoke runs")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    assert torch.cuda.is_available(), "This script needs a GPU (run on Kaggle/Colab T4)."
    cc = torch.cuda.get_device_capability()
    use_fp16 = cc[0] < 8  # T4 = (7,5): fp16; A100+ would allow bf16
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={cc} fp16={use_fp16}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True, dtype=None,  # Unsloth picks fp16 on T4
    )
    model = FastLanguageModel.get_peft_model(
        model, r=CONFIG["r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"], bias=CONFIG["bias"],
        target_modules=CONFIG["target_modules"],
        use_gradient_checkpointing="unsloth", random_state=CONFIG["seed"],
    )

    data_files = {"train": str(REPO_ROOT / "data" / "sft" / "train.jsonl"),
                  "eval": str(REPO_ROOT / "data" / "sft" / "val.jsonl")}
    ds = load_dataset("json", data_files=data_files)

    def render(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(render, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds["train"], eval_dataset=ds["eval"],
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            num_train_epochs=CONFIG["num_train_epochs"],
            max_steps=args.max_steps if args.max_steps else -1,
            learning_rate=CONFIG["learning_rate"],
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
            warmup_ratio=CONFIG["warmup_ratio"],
            weight_decay=CONFIG["weight_decay"],
            seed=CONFIG["seed"],
            fp16=use_fp16, bf16=not use_fp16,
            logging_steps=CONFIG["logging_steps"],
            eval_strategy="steps", eval_steps=CONFIG["eval_steps"],
            save_strategy="steps", save_steps=CONFIG["save_steps"],
            report_to="none",
        ),
    )
    # Loss only on the assistant JSON, not the system/user text.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    stats = trainer.train()
    print(stats)

    out = Path(args.output_dir)
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    run_config = {"base_model": BASE_MODEL, "max_seq_len": MAX_SEQ_LEN, **CONFIG,
                  "train_rows": len(ds["train"]), "eval_rows": len(ds["eval"]),
                  "train_loss": stats.training_loss}
    with open(out / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (out / "pip_freeze.txt").write_text(freeze)

    smoke(model, tokenizer)


def smoke(model, tokenizer) -> None:
    """10 greedy generations must all parse as valid JSON, use a known
    category, and stop before max_tokens. Abort the run (and the push) if not."""
    import pandas as pd
    from unsloth import FastLanguageModel

    sys.path.insert(0, str(REPO_ROOT))
    from evals.common import parse_completion
    from prompts.triage_prompt import build_messages

    FastLanguageModel.for_inference(model)
    df = pd.read_csv(REPO_ROOT / "data" / "eval" / "test_full.csv").head(10)
    failures = []
    for _, row in df.iterrows():
        ids = tokenizer.apply_chat_template(
            build_messages(row["text"]), tokenize=True,
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(input_ids=ids, max_new_tokens=400, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen = out[0][ids.shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        parsed, err = parse_completion(text, CATEGORIES)
        stopped = len(gen) < 400
        status = "OK" if parsed and stopped else f"FAIL ({err or 'no-eos'})"
        print(f"  [{status}] {row['case_id']}: {text[:90]}...")
        if not (parsed and stopped):
            failures.append(row["case_id"])
    if failures:
        raise SystemExit(f"SMOKE FAILED on {failures} — do NOT merge/push this adapter.")
    print("Smoke passed: 10/10 valid JSON with EOS stop.")


if __name__ == "__main__":
    main()
