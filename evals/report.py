"""Merge scores + judgments + benchmarks into one run report, README-ready
markdown tables, and charts.

Outputs:
  evals/runs/<run-id>.json  — full merged report (also copied to latest.json
                              and demo/latest.json for results.html)
  evals/runs/README_tables.md — paste-ready markdown for the README
  evals/runs/charts/*.png  — comparison bar charts (matplotlib)

Usage:
    python evals/report.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVALS_DIR = Path(__file__).parent
REPO_ROOT = EVALS_DIR.parent
RUNS_DIR = EVALS_DIR / "runs"
BENCH_DIR = REPO_ROOT / "artifacts" / "benchmarks"

ARM_LABELS = {
    "base__test_full": ("base", "Base Qwen3-4B (zero-shot)"),
    "tuned__test_full": ("tuned", "Fine-tuned Qwen3-4B (QLoRA)"),
    "groq70b__subset": ("groq70b", "Llama-3.3-70B API (3-shot)"),
}

# API pricing for the cost comparison table ($/1M tokens, input/output).
# Groq on-demand pricing for llama-3.3-70b-versatile; T4 cost assumes a
# typical cloud spot price (~$0.20/h) at the measured batched throughput.
GROQ_70B_PRICE = {"input": 0.59, "output": 0.79}
T4_HOURLY_USD = 0.20


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=REPO_ROOT).stdout.strip()
    except OSError:
        return "unknown"


def judge_summary(judgments: dict | None) -> dict:
    out: dict = {}
    if not judgments:
        return out
    by_arm: dict = {}
    for j in judgments["judgments"]:
        by_arm.setdefault(j["arm"], []).append(j)
    for arm, items in by_arm.items():
        out[arm] = {
            "n_judged": len(items),
            "avg_correctness": round(mean(j["correctness"] for j in items), 2),
            "avg_tone": round(mean(j["tone"] for j in items), 2),
            "avg_policy_safety": round(mean(j["policy_safety"] for j in items), 2),
        }
    return out


def cost_rows(scores: dict, benchmarks: dict) -> dict:
    """Estimated $/1M output tokens: API list price vs measured T4 throughput."""
    rows = {"groq70b": {"basis": "Groq list price",
                        "usd_per_1m_output_tokens": GROQ_70B_PRICE["output"]}}
    tuned_bench = benchmarks.get("tuned")
    if tuned_bench:
        tps = tuned_bench.get("batched", {}).get("output_tokens_per_s")
        if tps:
            usd = T4_HOURLY_USD / (tps * 3600 / 1e6)
            rows["tuned"] = {"basis": f"T4 @ ${T4_HOURLY_USD}/h, "
                                      f"{tps:.0f} tok/s batched (measured)",
                             "usd_per_1m_output_tokens": round(usd, 3)}
    return rows


def build_tables(report: dict) -> str:
    lines = ["### Results", "",
             "| Arm | n | JSON validity | Category accuracy | Macro-F1 |",
             "|---|---|---|---|---|"]
    for key, (arm, label) in ARM_LABELS.items():
        s = report["scores"].get(key)
        if not s:
            continue
        v, a = s["json_validity"], s["category_accuracy"]
        note = "" if s["n"] >= 2000 else f" (95% CI {a['ci95'][0]:.3f}–{a['ci95'][1]:.3f})"
        lines.append(f"| {label} | {s['n']} | {v['rate']:.1%} | "
                     f"{a['rate']:.1%}{note} | {s['macro_f1']:.3f} |")
    js = report["judge_summary"]
    if js:
        lines += ["", f"### Reply quality (LLM-as-judge, {report.get('judge_model', '')}, "
                      "1–5, identical cases across arms)", "",
                  "| Arm | judged | Correctness | Tone | Policy safety |",
                  "|---|---|---|---|---|"]
        for key, (arm, label) in ARM_LABELS.items():
            j = js.get(arm)
            if j:
                lines.append(f"| {label} | {j['n_judged']} | {j['avg_correctness']} | "
                             f"{j['avg_tone']} | {j['avg_policy_safety']} |")
    costs = report["cost_estimates"]
    if costs:
        lines += ["", "### Serving cost (est.)", "",
                  "| Arm | Basis | $/1M output tokens |", "|---|---|---|"]
        for key, (arm, label) in ARM_LABELS.items():
            c = costs.get(arm)
            if c:
                lines.append(f"| {label} | {c['basis']} | ${c['usd_per_1m_output_tokens']} |")
    return "\n".join(lines) + "\n"


def make_charts(report: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping charts")
        return
    charts_dir = RUNS_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)

    arms = [(k, lbl) for k, (a, lbl) in ARM_LABELS.items() if k in report["scores"]]
    labels = [lbl.replace(" (", "\n(") for _, lbl in arms]
    colors = ["#94a3b8", "#f97316", "#38bdf8"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in (
        (axes[0], "macro_f1", "Category routing (macro-F1)"),
        (axes[1], None, "JSON validity"),
    ):
        vals = [report["scores"][k]["macro_f1"] if metric else
                report["scores"][k]["json_validity"]["rate"] for k, _ in arms]
        bars = ax.bar(labels, vals, color=colors[:len(arms)])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.3f" if metric else "%.1%%".replace("%%", "%"))
    fig.tight_layout()
    fig.savefig(charts_dir / "deterministic.png", dpi=150)

    js = report["judge_summary"]
    if js:
        crits = ["avg_correctness", "avg_tone", "avg_policy_safety"]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        width = 0.25
        for i, (k, lbl) in enumerate(arms):
            arm = ARM_LABELS[k][0]
            if arm not in js:
                continue
            vals = [js[arm][c] for c in crits]
            ax.bar([x + i * width for x in range(len(crits))], vals, width,
                   label=lbl, color=colors[i])
        ax.set_xticks([x + width for x in range(len(crits))])
        ax.set_xticklabels(["Correctness", "Tone", "Policy safety"])
        ax.set_ylim(0, 5.3)
        ax.set_title("Reply quality (LLM-as-judge, 1–5)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(charts_dir / "judge.png", dpi=150)
    print(f"Charts -> {charts_dir}")


def main() -> None:
    scores = load_json(RUNS_DIR / "scores.json")
    if not scores:
        raise SystemExit("Run evals/score.py first")
    judgments = load_json(RUNS_DIR / "judgments.json")
    benchmarks = {p.stem: load_json(p) for p in BENCH_DIR.glob("*.json")}

    report = {
        "run_id": None,  # stamped below
        "git_sha": git_sha(),
        "judge_model": judgments["judge_model"] if judgments else None,
        "scores": scores,
        "judge_summary": judge_summary(judgments),
        "benchmarks": benchmarks,
        "cost_estimates": cost_rows(scores, benchmarks),
    }
    report["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{report['run_id']}.json"
    for path in (out, RUNS_DIR / "latest.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    demo_copy = REPO_ROOT / "demo" / "latest.json"
    shutil.copyfile(RUNS_DIR / "latest.json", demo_copy)

    tables = build_tables(report)
    with open(RUNS_DIR / "README_tables.md", "w", encoding="utf-8") as f:
        f.write(tables)
    print(tables)
    make_charts(report)
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
