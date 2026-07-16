"""Single source of truth for the triage prompt.

Training, eval (all three arms), and the demo UI import from here so the
fine-tuned model, the base-model arm, and the Groq reference arm are always
graded on exactly the same task definition. Any edit here invalidates the
right cache entries (the prompt text is part of every cache key).
"""

from __future__ import annotations

import json

CATEGORIES = [
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]

SYSTEM_PROMPT = f"""\
You are a support-ticket triage assistant for an e-commerce company.
Given a customer's ticket, respond with ONLY a JSON object, no other text:
{{"category": "<one of: {', '.join(CATEGORIES)}>", "reply": "<a helpful, professional customer-facing reply>"}}

Rules:
- "category" must be exactly one of the {len(CATEGORIES)} categories listed, uppercase.
- "reply" must address the customer's issue with a warm, professional tone.
- Never invent specifics you don't know (order numbers, names, dates, links).
  Use double-brace placeholder slots instead, e.g. {{{{Order Number}}}} or
  {{{{Customer Support Phone Number}}}}, exactly like a reply template.
- Output compact JSON on a single line. No markdown, no explanations."""


def target_json(category: str, reply: str) -> str:
    """The assistant target string used for SFT — compact, deterministic key order."""
    return json.dumps({"category": category, "reply": reply}, ensure_ascii=False)


def build_messages(ticket: str, few_shot: list[dict] | None = None) -> list[dict]:
    """Chat messages for one ticket. `few_shot` items: {"ticket", "category", "reply"}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot or []:
        messages.append({"role": "user", "content": ex["ticket"]})
        messages.append({"role": "assistant", "content": target_json(ex["category"], ex["reply"])})
    messages.append({"role": "user", "content": ticket})
    return messages


# Curated 11-shot block (one per category) for the Groq 70B reference arm.
# Built by data/make_eval_subsets.py from TRAIN rows only (never test) and
# frozen here as a committed JSON file so the arm is reproducible.
FEW_SHOT_PATH = "few_shot_examples.json"


def load_few_shot() -> list[dict]:
    from pathlib import Path
    path = Path(__file__).parent / FEW_SHOT_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)
