"""Plot Layer 1-A four-way response category breakdown.

Three panels (one per benchmark). Each panel shows two vertical stacked
columns (Apertus, GPT-5). Percentages annotated on every segment that
is at least 3% tall.

Usage:
    python scripts/plot_1a_categories.py \
        --apertus results/1a-jailbreak/apertus \
        --gpt5    results/1a-jailbreak/gpt5 \
        --out     results/1a-jailbreak/plots/jailbreak_categories.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = ("harmbench", "jailbreakbench", "strongreject")
LABELS = {
    "harmbench":      "HarmBench",
    "jailbreakbench": "JailbreakBench",
    "strongreject":   "StrongREJECT",
}
NS = {"harmbench": 320, "jailbreakbench": 100, "strongreject": 313}
MODELS = (("Apertus", "apertus"), ("GPT-5", "gpt5"))
CATS = ("Jailbroken", "Refused", "Complied", "Ambiguous")
COLORS = {
    "Jailbroken": "#b91c1c",
    "Refused":    "#15803d",
    "Complied":   "#a8b3c4",
    "Ambiguous":  "#f59e0b",
}


def recategorize(run_dir: Path, bench: str):
    scored = {r["id"]: r for r in (json.loads(l) for l in open(run_dir / f"scores_{bench}.jsonl"))}
    cat = [json.loads(l) for l in open(run_dir / f"categorized_{bench}.jsonl")]
    counts = Counter()
    for cr in cat:
        sr = scored.get(cr["prompt_id"], {})
        if sr.get("jailbroken"):
            counts["Jailbroken"] += 1
        else:
            label = cr.get("category", "Unknown")
            if label == "Jailbroken":
                label = "Complied"
            elif label in ("Complied (safe)", "Complied_safe"):
                label = "Complied"
            counts[label] += 1
    n = sum(counts.values())
    return n, counts


def wilson_ci(x: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus", required=True, type=Path)
    ap.add_argument("--gpt5", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.0), sharey=True)

    for ax, bench in zip(axes, BENCHMARKS):
        x = np.array([0, 1])
        width = 0.55
        bar_data = []
        for model_label, model_dir_name in MODELS:
            run_dir = args.apertus if model_dir_name == "apertus" else args.gpt5
            n, c = recategorize(run_dir, bench)
            bar_data.append((model_label, n, c))

        bottom = np.zeros(2)
        for cat in CATS:
            vals = np.array([100 * bd[2].get(cat, 0) / bd[1] if bd[1] else 0 for bd in bar_data])
            ax.bar(x, vals, width, bottom=bottom, color=COLORS[cat],
                   edgecolor="white", linewidth=1.2)
            for xi, vi, bi in zip(x, vals, bottom):
                if vi >= 3.0:
                    txt_color = "white" if cat in ("Jailbroken", "Refused") else "#1f2937"
                    ax.text(xi, bi + vi / 2, f"{vi:.1f}%",
                            ha="center", va="center",
                            fontsize=8.5, color=txt_color, fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([bd[0] for bd in bar_data], fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title(f"{LABELS[bench]}  (n={NS[bench]})", fontsize=11, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([0, 25, 50, 75, 100])
        if ax is axes[0]:
            ax.set_yticklabels(["0", "25", "50", "75", "100%"])
            ax.set_ylabel("Share of responses", fontsize=10)
        else:
            ax.set_yticklabels(["", "", "", "", ""])
        ax.tick_params(axis="x", bottom=False, pad=4)
        ax.set_axisbelow(True)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[c], ec="white") for c in CATS]
    fig.legend(legend_handles, CATS, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10,
               handlelength=1.4, handleheight=1.0, columnspacing=2.0)

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
