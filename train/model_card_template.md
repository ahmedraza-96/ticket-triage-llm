---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
datasets:
  - bitext/Bitext-customer-support-llm-chatbot-training-dataset
language:
  - en
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - unsloth
  - customer-support
  - ticket-triage
  - vllm
---

# Qwen3-4B Ticket Triage ({{KIND}})

QLoRA fine-tune of [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
that turns a raw customer-support ticket into strict single-line JSON:

```json
{"category": "REFUND", "reply": "I'm sorry to hear that... use {{Order Number}} slots for unknown specifics."}
```

- **category** — one of 11 routing intents: ACCOUNT, CANCEL, CONTACT, DELIVERY,
  FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
- **reply** — a drafted, customer-facing response. Unknown specifics (order
  numbers, links, phone numbers) are emitted as `{{Placeholder}}` slots rather
  than hallucinated — trained-in policy safety.

Trained and evaluated end-to-end on **free T4 GPUs**; the JSON format is
**learned, not enforced** — validity is measured without guided decoding.

Code, eval harness, and raw completion artifacts:
**https://github.com/ahmedraza-96/ticket-triage-llm**
Sibling project (same held-out test set, classification-only DistilBERT):
[support-ticket-classifier](https://github.com/ahmedraza-96/support-ticket-classifier)

## Results

{{RESULTS}}

## Serving (vLLM, fits a 16 GB T4)

```bash
vllm serve {{HF_USER}}/Qwen3-4B-Instruct-2507-ticket-triage \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.85
```

Prompting: use the exact system prompt in
[`prompts/triage_prompt.py`](https://github.com/ahmedraza-96/ticket-triage-llm/blob/main/prompts/triage_prompt.py)
with `temperature=0`.

## Training

```json
{{RUN_CONFIG}}
```

## Limitations

- Trained on Bitext's templated single-vendor e-commerce dataset — replies are
  fluent but stylistically templated; category vocabulary is fixed at 11.
- English only.
- `{{Placeholder}}` slots are left for the ticketing system to fill; the model
  does not do slot-filling.
- Reply-quality scores come from an LLM judge (openai/gpt-oss-120b) and share
  the usual limits of LLM-as-judge evaluation.
