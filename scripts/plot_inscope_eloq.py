"""Plot in-scope-eloq evaluation results — similarity / correctness / FRR
for Apertus vs GPT-5, grouped bars.

Reads each run-dir's summary_inscope_eloq.json and writes
results/1b-in-scope-eloq/plots/inscope_eloq_eval.png.

Usage:
    python scripts/plot_inscope_eloq.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(run_dir: Path) -> dict:
    return json.load(open(run_dir / "summary_inscope_eloq.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus", type=Path,
                    default=Path("results/1b-in-scope-eloq/apertus"))
    ap.add_argument("--gpt5", type=Path,
                    default=Path("results/1b-in-scope-eloq/gpt5"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/1b-in-scope-eloq/plots/inscope_eloq_eval.png"))
    args = ap.parse_args()

    a = load(args.apertus)
    g = load(args.gpt5)

    # Metrics all percent-scaled so they share one y-axis cleanly.
    metrics = [
        ("Answer-similarity\n(RAGAS, embeddings)",  "mean_answer_similarity", "↑ better"),
        ("Correctness\n(LLM judge)",                "correctness_rate",       "↑ better"),
        ("False refusal rate\n(LLM judge)",         "false_refusal_rate",     "↓ better"),
    ]
    apertus_vals = [(a.get(k) or 0) * 100 for _, k, _ in metrics]
    gpt5_vals    = [(g.get(k) or 0) * 100 for _, k, _ in metrics]
    labels       = [m[0] for m in metrics]
    directions   = [m[2] for m in metrics]

    # Embed the up/down-better hint right into the xtick label so we don't
    # have to position separate text annotations in axis space.
    xlabels = [f"{m[0]}\n({m[2]})" for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, apertus_vals, w, label="Apertus", color="#ef4444")
    b2 = ax.bar(x + w/2, gpt5_vals, w, label="GPT-5", color="#3b82f6")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Rate (%)")
    ax.set_title("In-Scope-ELOQ Evaluation — Apertus vs GPT-5\n"
                 f"n = {a.get('n_scored')} prompts (single-aspect-filtered dataset)",
                 fontsize=12)
    ax.set_ylim(0, max(105, max(apertus_vals + gpt5_vals) + 10))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    fig.text(0.02, 0.01,
             "Similarity: cos-sim of sentence-transformer embeddings vs gold derived_answer (RAGAS, Es & James 2024).  "
             "Correctness: Llama-3.3-70B binary — does the response contain the same key info as the gold?  "
             "FRR: Llama-3.3-70B binary — did the model refuse / fail to engage?",
             fontsize=8, color="gray")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
