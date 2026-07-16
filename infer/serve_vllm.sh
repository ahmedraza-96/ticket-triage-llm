#!/usr/bin/env bash
# Serve the model with vLLM on a 16 GB T4 (Kaggle/Colab).
#
# T4 = compute capability 7.5:
#   --dtype float16 is REQUIRED (Qwen3 config declares bf16, unsupported on T4)
#   FlashAttention-2 is unavailable on sm75 — vLLM falls back to xformers;
#   if auto-detection misbehaves: export VLLM_ATTENTION_BACKEND=XFORMERS
#
# Usage:
#   bash infer/serve_vllm.sh ahmedraza-96/Qwen3-4B-Instruct-2507-ticket-triage
#   bash infer/serve_vllm.sh Qwen/Qwen3-4B-Instruct-2507          # base arm

MODEL="${1:?usage: serve_vllm.sh <model-id>}"

vllm serve "$MODEL" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
