"""Plot Layer 1-A canonical-scorer results across models.

Reads per-benchmark summaries from each run dir, makes a grouped bar
chart of ASR by benchmark and a separate panel for StrongREJECT's mean
rubric score.

Usage:
    python scripts/plot_1a_asr.py \\
        --apertus results/1a-jailbreak/apertus \\
        --gpt5    results/1a-jailbreak/gpt5 \\
        --out     results/1a-jailbreak/plots/jailbreak_eval.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = ("harmbench", "jailbreakbench", "strongreject")
LABELS = {"harmbench": "HarmBench", "jailbreakbench": "JailbreakBench", "strongreject": "StrongREJECT"}


def load(run_dir: Path) -> dict:
    out = {}
    for b in BENCHMARKS:
        path = run_dir / f"summary_{b}.json"
        if path.exists():
            out[b] = json.load(open(path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus", required=True, type=Path)
    ap.add_argument("--gpt5", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    apertus = load(args.apertus)
    gpt5 = load(args.gpt5)

    def asr(s):
        # harmbench/jbb: asr ; strongreject: asr_at_zero
        return s.get("asr", s.get("asr_at_zero", 0.0)) * 100

    def sr_mean(s):
        return (s.get("mean_score") or 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    # ── Panel 1: per-benchmark ASR (grouped bars) ──
    x = np.arange(len(BENCHMARKS))
    w = 0.35
    apertus_asrs = [asr(apertus.get(b, {})) for b in BENCHMARKS]
    gpt5_asrs    = [asr(gpt5.get(b, {})) for b in BENCHMARKS]

    b1 = ax1.bar(x - w/2, apertus_asrs, w, label="Apertus", color="#ef4444")
    b2 = ax1.bar(x + w/2, gpt5_asrs, w, label="GPT-5", color="#3b82f6")
    ax1.set_xticks(x); ax1.set_xticklabels([LABELS[b] for b in BENCHMARKS])
    ax1.set_ylabel("Attack Success Rate (%)")
    ax1.set_title("Attack Success Rate by benchmark", fontsize=11)
    ax1.set_ylim(0, max(15, max(apertus_asrs + gpt5_asrs) * 1.2))
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)
    ax1.text(0.02, -0.18,
             "ASR is the paper-headline binary metric for HarmBench (Mazeika 2024) "
             "and JailbreakBench (Chao 2024).\nFor StrongREJECT (Souly 2024) ASR is "
             "secondary — the paper-headline number is the mean rubric score (right panel).",
             transform=ax1.transAxes, fontsize=8, color="gray")

    # ── Panel 2: StrongREJECT mean rubric score (its paper-headline metric) ──
    sr_a = sr_mean(apertus.get("strongreject", {}))
    sr_g = sr_mean(gpt5.get("strongreject", {}))
    bars = ax2.bar(["Apertus", "GPT-5"], [sr_a, sr_g], color=["#ef4444", "#3b82f6"])
    ax2.set_ylabel("Mean rubric score (0–1)")
    ax2.set_title("StrongREJECT — mean rubric score", fontsize=11)
    ax2.set_ylim(0, max(0.1, max(sr_a, sr_g) * 1.4))
    ax2.grid(axis="y", alpha=0.3)
    for r in bars:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    plt.suptitle("Jailbreak Benchmark Evaluation — Apertus vs GPT-5", fontsize=13)
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
