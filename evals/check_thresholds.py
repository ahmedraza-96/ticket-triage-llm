"""CI quality gate — exits non-zero if the fine-tune stops beating its bars.

Runs on the committed completion artifacts (no GPU, no secrets), so any PR
that changes prompts/parsing/scoring is re-checked against real outputs.

Gates:
  tuned JSON validity            >= 0.98
  tuned macro-F1                 >= base macro-F1 + 0.15
  tuned judge avg (all criteria) >= base judge avg   (when judgments exist)

Usage:
    python evals/check_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"

MIN_TUNED_VALIDITY = 0.98
MIN_F1_DELTA = 0.15


def main() -> int:
    with open(RUNS_DIR / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    base = scores.get("base__test_full")
    tuned = scores.get("tuned__test_full")
    if not base or not tuned:
        print("Missing base/tuned scores — nothing to gate yet (pass).")
        return 0

    failures = []
    validity = tuned["json_validity"]["rate"]
    if validity < MIN_TUNED_VALIDITY:
        failures.append(f"tuned JSON validity {validity:.3f} < {MIN_TUNED_VALIDITY}")

    delta = tuned["macro_f1"] - base["macro_f1"]
    if delta < MIN_F1_DELTA:
        failures.append(f"tuned macro-F1 delta {delta:+.3f} < +{MIN_F1_DELTA} "
                        f"(base {base['macro_f1']:.3f}, tuned {tuned['macro_f1']:.3f})")

    latest = RUNS_DIR / "latest.json"
    if latest.exists():
        with open(latest, encoding="utf-8") as f:
            js = json.load(f).get("judge_summary", {})
        if "base" in js and "tuned" in js:
            crits = ("avg_correctness", "avg_tone", "avg_policy_safety")
            b = sum(js["base"][c] for c in crits) / 3
            t = sum(js["tuned"][c] for c in crits) / 3
            if t < b:
                failures.append(f"tuned judge avg {t:.2f} < base {b:.2f}")

    if failures:
        print("QUALITY GATE FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print(f"Quality gate passed: validity={validity:.3f}, macro-F1 delta {delta:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
