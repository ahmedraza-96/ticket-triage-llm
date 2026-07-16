"""Deterministic scoring — CPU-only, no API keys, re-runnable in CI.

Per arm: JSON-validity rate, category accuracy, macro-F1 (+ per-class F1),
Wilson 95% CIs on rates, latency p50/p95. Reads committed completion JSONL
artifacts; writes evals/runs/scores.json consumed by report.py.

Usage:
    python evals/score.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.common import COMPLETIONS_DIR, read_jsonl  # noqa: E402
from prompts.triage_prompt import CATEGORIES          # noqa: E402

OUT_PATH = Path(__file__).parent / "runs" / "scores.json"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(center - margin, 4), round(center + margin, 4))


def macro_f1(records: list[dict]) -> tuple[float, dict]:
    """F1 per gold class. A parse failure counts as a wrong prediction —
    an unusable answer must hurt the headline metric, not be excluded."""
    tp: dict = defaultdict(int); fp: dict = defaultdict(int); fn: dict = defaultdict(int)
    for r in records:
        gold = r["gold_category"]
        pred = r["parsed"]["category"] if r["parsed"] else None
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred is not None:
                fp[pred] += 1
    per_class = {}
    for cat in CATEGORIES:
        prec = tp[cat] / (tp[cat] + fp[cat]) if tp[cat] + fp[cat] else 0.0
        rec = tp[cat] / (tp[cat] + fn[cat]) if tp[cat] + fn[cat] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[cat] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4), "support": tp[cat] + fn[cat]}
    macro = sum(v["f1"] for v in per_class.values()) / len(CATEGORIES)
    return round(macro, 4), per_class


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return round(values[min(len(values) - 1, int(len(values) * q))], 1)


def score_arm(records: list[dict]) -> dict:
    n = len(records)
    valid = sum(1 for r in records if r["parsed"])
    correct = sum(1 for r in records if r["parsed"] and
                  r["parsed"]["category"] == r["gold_category"])
    macro, per_class = macro_f1(records)
    latencies = [r["latency_ms"] for r in records]
    return {
        "n": n,
        "model": records[0]["model"] if records else None,
        "json_validity": {"rate": round(valid / n, 4), "n_valid": valid,
                          "ci95": wilson_ci(valid, n)},
        "category_accuracy": {"rate": round(correct / n, 4),
                              "ci95": wilson_ci(correct, n)},
        "macro_f1": macro,
        "per_class": per_class,
        "latency_ms": {"p50": percentile(latencies, 0.50),
                       "p95": percentile(latencies, 0.95)},
        "parse_errors": sorted({r["parse_error"] for r in records if r["parse_error"]}),
    }


def main() -> None:
    scores = {}
    for path in sorted(COMPLETIONS_DIR.glob("*__*.jsonl")):
        arm, subset = path.stem.split("__", 1)
        records = read_jsonl(path)
        scores[f"{arm}__{subset}"] = score_arm(records)
        s = scores[f"{arm}__{subset}"]
        print(f"{path.stem:<24} n={s['n']:<5} validity={s['json_validity']['rate']:.3f} "
              f"acc={s['category_accuracy']['rate']:.3f} macroF1={s['macro_f1']:.3f}")
    if not scores:
        raise SystemExit(f"No completion artifacts in {COMPLETIONS_DIR}")
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
