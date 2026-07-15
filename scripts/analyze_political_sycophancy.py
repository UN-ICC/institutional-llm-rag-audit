"""Layer 4 political-sycophancy benchmark — statistical analysis.

A pre-specified analysis plan with primary, secondary, and exploratory
tiers. Reports effect estimates with Wilson and Newcombe 95% confidence
intervals, exact McNemar p-values, and Holm-Bonferroni adjustment within
each hypothesis tier.

Setup
-----
N = 110 paired items, each tested in four variants
  (clean, plain, load_a, load_b).
Two systems: Apertus RAG, GPT-5 RAG, sharing the same retrieval index.
Outcome per (item, variant, system): binary `corrects_premise` from a
GPT-4o LLM-as-judge adapted from ELEPHANT (Cheng et al. 2025).

Pre-specified hypotheses
------------------------
PRIMARY tier (alpha = 0.05, Holm-Bonferroni across the three tests):
  H1: PCR_plain differs between Apertus and GPT-5 RAG.
      (Operational headline: does the deployed system correct false
       premises as well as a stronger LLM with the same retrieval?)
  H2a: Apertus DA = PCR_load_a - PCR_load_b is non-zero.
       (Political-bias asymmetry, system 1.)
  H2b: GPT-5 RAG DA is non-zero.
       (Political-bias asymmetry, system 2.)

SECONDARY tier (alpha = 0.05, Holm-Bonferroni across the six tests):
  S1, S2: System comparison on load_a and load_b (paired McNemar).
  S3, S4: Political loading effect, plain vs load_a / plain vs load_b,
          within Apertus.
  S5, S6: Same within GPT-5 RAG.

EXPLORATORY (no multiplicity correction; flagged as exploratory):
  DA stratified by perturbation pole alignment (n = 6, 18 subgroups).
  Robustness to the calibration filter (item subset where
  answer_correct=True on clean).

Effect sizes and confidence intervals
-------------------------------------
  - Wilson 95% CI for each marginal correction rate.
  - Newcombe's method 10 for the paired risk-difference 95% CI.
  - McNemar's odds ratio (b / c) where both > 0.

Outputs
-------
  results/2c-political-sycophancy/analysis/results.json
  results/2c-political-sycophancy/analysis/report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

from scipy.stats import binomtest, norm


# ── Confidence-interval helpers ────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired_ci(a: int, b: int, c: int, d: int,
                       alpha: float = 0.05) -> tuple[float, float, float]:
    """Newcombe's method 10 for the paired proportion difference 95% CI.

    Returns (point_estimate, lower, upper) for p1 - p2 where p1 is the
    marginal proportion of system-1 successes and p2 the marginal
    proportion of system-2 successes.

    Inputs are the 2x2 cell counts in the paired layout
      a = both success, b = sys1 only, c = sys2 only, d = both fail.

    Reference: Newcombe (1998), Statistics in Medicine, 17, 2635-2650.
    """
    n = a + b + c + d
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p1 = (a + b) / n
    p2 = (a + c) / n
    diff = p1 - p2
    l1, u1 = wilson_ci(a + b, n, alpha)
    l2, u2 = wilson_ci(a + c, n, alpha)
    if a + d == 0:
        phi = 0.0
    else:
        denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        if denom == 0:
            phi = 0.0
        else:
            phi = (a * d - b * c) / denom
    lower = diff - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return diff, max(-1.0, lower), min(1.0, upper)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for the McNemar null b == c."""
    n_disc = b + c
    if n_disc == 0:
        return float("nan")
    return binomtest(b, n_disc, p=0.5, alternative="two-sided").pvalue


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values
    (each value compared to alpha, identical to the standard convention)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [None] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        raw = pvalues[idx] * (m - rank)
        prev = max(prev, min(raw, 1.0))
        adjusted[idx] = prev
    return adjusted


# ── Paired counts builders ─────────────────────────────────────────────

def paired_two_systems(items: list[str], A: dict, G: dict,
                       variant: str) -> tuple[int, int, int, int]:
    """Return (a, b, c, d) for the paired 2x2 between Apertus and GPT-5 on `variant`.
       a = both correct, b = Apertus only, c = GPT-5 only, d = both wrong.
    """
    a = b = c = d = 0
    for iid in items:
        ax = A[iid]["corrects_premise"].get(variant)
        gx = G[iid]["corrects_premise"].get(variant)
        if ax is None or gx is None:
            continue
        if ax and gx:        a += 1
        elif ax and not gx:  b += 1
        elif gx and not ax:  c += 1
        else:                d += 1
    return a, b, c, d


def paired_within_system(items: list[str], S: dict,
                          v1: str, v2: str) -> tuple[int, int, int, int]:
    """Same shape, but within-system between two variants v1 vs v2.
       a = both correct, b = v1 only, c = v2 only, d = both wrong.
    """
    a = b = c = d = 0
    for iid in items:
        x1 = S[iid]["corrects_premise"].get(v1)
        x2 = S[iid]["corrects_premise"].get(v2)
        if x1 is None or x2 is None:
            continue
        if x1 and x2:        a += 1
        elif x1 and not x2:  b += 1
        elif x2 and not x1:  c += 1
        else:                d += 1
    return a, b, c, d


# ── Result wrapper ─────────────────────────────────────────────────────

def make_paired_result(label: str, a: int, b: int, c: int, d: int,
                       direction: str = "diff") -> OrderedDict:
    """Build the standard result block for a paired binary contrast."""
    n = a + b + c + d
    p_value = mcnemar_exact(b, c)
    diff, lo, hi = newcombe_paired_ci(a, b, c, d)
    odds_ratio = (b / c) if c > 0 else (float("inf") if b > 0 else float("nan"))
    return OrderedDict([
        ("label", label),
        ("direction", direction),
        ("n_pairs", n),
        ("counts", OrderedDict([("a_both", a), ("b_only_1", b), ("c_only_2", c), ("d_both_wrong", d)])),
        ("p1_marginal", (a + b) / n if n else float("nan")),
        ("p2_marginal", (a + c) / n if n else float("nan")),
        ("diff", diff),
        ("diff_ci_95", [lo, hi]),
        ("mcnemar_odds_ratio_b_over_c", odds_ratio),
        ("n_discordant", b + c),
        ("p_exact_two_sided", p_value),
    ])


def make_rate(label: str, k: int, n: int) -> OrderedDict:
    lo, hi = wilson_ci(k, n)
    return OrderedDict([
        ("label", label), ("k", k), ("n", n),
        ("rate", k / n if n else float("nan")),
        ("wilson_ci_95", [lo, hi]),
    ])


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus-per-item",
        default="results/2c-political-sycophancy/apertus/scores_political_sycophancy_per_item.jsonl")
    ap.add_argument("--gpt5-per-item",
        default="results/2c-political-sycophancy/gpt5_rag/scores_political_sycophancy_per_item.jsonl")
    ap.add_argument("--out-dir",
        default="results/2c-political-sycophancy/analysis")
    args = ap.parse_args()

    A = {r["item_id"]: r for r in (json.loads(l) for l in open(args.apertus_per_item))}
    G = {r["item_id"]: r for r in (json.loads(l) for l in open(args.gpt5_per_item))}
    common = sorted(set(A) & set(G))
    n_items = len(common)
    variants = ["clean", "plain", "load_a", "load_b"]

    out = OrderedDict([
        ("benchmark", "political_sycophancy_layer4"),
        ("n_items_paired", n_items),
        ("variants", variants),
        ("alpha", 0.05),
        ("multiple_comparisons", "Holm-Bonferroni within tier"),
        ("confidence_intervals", "Wilson 95% (rates), Newcombe method 10 (paired diff)"),
        ("test", "McNemar exact (binomial on discordant pairs)"),
    ])

    # ── Marginal rates with Wilson CI ──
    rates = OrderedDict()
    for system_name, S in [("apertus", A), ("gpt5_rag", G)]:
        block = OrderedDict()
        for v in variants:
            k = sum(int(bool(S[i]["corrects_premise"].get(v))) for i in common
                    if S[i]["corrects_premise"].get(v) is not None)
            n = sum(1 for i in common if S[i]["corrects_premise"].get(v) is not None)
            block[v] = make_rate(f"{system_name}.{v}", k, n)
        n_cal = sum(1 for i in common if S[i]["answer_correct"].get("clean"))
        block["calibration_baseline"] = make_rate(f"{system_name}.calibrated", n_cal, len(common))
        rates[system_name] = block
    out["marginal_rates"] = rates

    # ── PRIMARY tier ──
    primary = OrderedDict()
    # H1: system comparison on plain
    a, b, c, d = paired_two_systems(common, A, G, "plain")
    primary["H1_system_on_plain"] = make_paired_result(
        "Apertus vs GPT-5 RAG on PCR_plain", a, b, c, d, "p1_minus_p2"
    )
    # H2a / H2b: pole asymmetry within each system
    for system_name, S in [("apertus", A), ("gpt5_rag", G)]:
        a, b, c, d = paired_within_system(common, S, "load_a", "load_b")
        primary[f"H2_DA_{system_name}"] = make_paired_result(
            f"{system_name} pole asymmetry (load_a vs load_b)", a, b, c, d, "load_a_minus_load_b"
        )

    primary_pvalues = [primary[k]["p_exact_two_sided"] for k in primary]
    primary_adj = holm_bonferroni(primary_pvalues, 0.05)
    for (key, _), adj in zip(primary.items(), primary_adj):
        primary[key]["p_holm_adjusted"] = adj
        primary[key]["primary_significant_at_0.05"] = bool(adj < 0.05)
    out["primary_tests"] = primary

    # ── SECONDARY tier ──
    secondary = OrderedDict()
    for v in ("load_a", "load_b"):
        a, b, c, d = paired_two_systems(common, A, G, v)
        secondary[f"S_system_on_{v}"] = make_paired_result(
            f"Apertus vs GPT-5 RAG on PCR_{v}", a, b, c, d, "p1_minus_p2"
        )
    for system_name, S in [("apertus", A), ("gpt5_rag", G)]:
        for v in ("load_a", "load_b"):
            a, b, c, d = paired_within_system(common, S, "plain", v)
            secondary[f"S_PLE_{system_name}_plain_vs_{v}"] = make_paired_result(
                f"{system_name} PLE (plain vs {v})", a, b, c, d, "plain_minus_load"
            )
    secondary_pvalues = [secondary[k]["p_exact_two_sided"] for k in secondary]
    secondary_adj = holm_bonferroni(secondary_pvalues, 0.05)
    for (key, _), adj in zip(secondary.items(), secondary_adj):
        secondary[key]["p_holm_adjusted"] = adj
        secondary[key]["secondary_significant_at_0.05"] = bool(adj < 0.05)
    out["secondary_tests"] = secondary

    # ── EXPLORATORY tier ──
    exploratory = OrderedDict()
    # DA subgroups by alignment, per system (no multiplicity correction)
    for system_name, S in [("apertus", A), ("gpt5_rag", G)]:
        for pole in ("pole_A", "pole_B", "ambiguous"):
            subset = [i for i in common if S[i].get("false_claim_aligns_with") == pole]
            if not subset:
                continue
            a, b, c, d = paired_within_system(subset, S, "load_a", "load_b")
            exploratory[f"E_DA_{system_name}_{pole}_aligned"] = make_paired_result(
                f"{system_name} DA | false aligns {pole} (n={len(subset)})",
                a, b, c, d, "load_a_minus_load_b"
            )
    # Calibration sensitivity: between-system on PLAIN restricted to items where
    # BOTH systems were calibrated.
    cal_both = [i for i in common
                if A[i]["answer_correct"].get("clean") and G[i]["answer_correct"].get("clean")]
    a, b, c, d = paired_two_systems(cal_both, A, G, "plain")
    exploratory["E_calibration_sensitivity_H1_intersection"] = make_paired_result(
        f"H1 restricted to common-calibrated items (n={len(cal_both)})", a, b, c, d, "p1_minus_p2"
    )
    out["exploratory_tests"] = exploratory

    # ── Post-hoc power note ──
    out["sample_size_note"] = (
        "At n_paired = 110 with the observed concordance, McNemar exact has "
        "approx. 0.80 power to detect a true risk difference of about 0.15 "
        "and approx. 0.50 power for a 0.10 difference. Subgroup tests at "
        "n in [6, 18] are effectively descriptive."
    )

    # ── Write JSON ──
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # ── Write text report ──
    def fmt_pct(x):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

    def fmt_p(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        if x < 1e-4:
            return f"{x:.2e}"
        return f"{x:.4f}"

    lines = []
    lines.append("LAYER 4 POLITICAL-SYCOPHANCY: STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"N paired items:            {n_items}")
    lines.append(f"Alpha:                     0.05")
    lines.append(f"Multiplicity within tier:  Holm-Bonferroni")
    lines.append(f"Tests:                     McNemar exact (two-sided binomial on discordant pairs)")
    lines.append(f"Rate CIs:                  Wilson 95%")
    lines.append(f"Paired diff CIs:           Newcombe method 10 95%")
    lines.append("")

    lines.append("MARGINAL CORRECTION RATES (PCR) — Wilson 95% CI")
    lines.append("-" * 60)
    for system_name in ("apertus", "gpt5_rag"):
        lines.append(f"  {system_name}:")
        for v in variants:
            r = rates[system_name][v]
            lo, hi = r["wilson_ci_95"]
            lines.append(f"    PCR_{v:<6s} = {fmt_pct(r['rate'])}  ({r['k']}/{r['n']})  Wilson 95% CI [{fmt_pct(lo)}, {fmt_pct(hi)}]")
        cal = rates[system_name]["calibration_baseline"]
        lines.append(f"    calibrated     = {fmt_pct(cal['rate'])}  ({cal['k']}/{cal['n']})")
        lines.append("")

    def render_tier(title, tier):
        lines.append(title)
        lines.append("-" * 60)
        for key, r in tier.items():
            cnts = r["counts"]
            ci_lo, ci_hi = r["diff_ci_95"]
            lines.append(f"  {key}: {r['label']}")
            lines.append(f"    n = {r['n_pairs']}  discordant = {r['n_discordant']}  (b={cnts['b_only_1']}, c={cnts['c_only_2']})")
            lines.append(f"    diff = {fmt_pct(r['diff'])}  Newcombe 95% CI [{fmt_pct(ci_lo)}, {fmt_pct(ci_hi)}]")
            lines.append(f"    p_exact = {fmt_p(r['p_exact_two_sided'])}    p_holm = {fmt_p(r.get('p_holm_adjusted', float('nan')))}")
            lines.append("")
        lines.append("")

    render_tier("PRIMARY TESTS", primary)
    render_tier("SECONDARY TESTS", secondary)
    render_tier("EXPLORATORY TESTS (no multiplicity correction)", exploratory)

    lines.append("POWER NOTE")
    lines.append("-" * 60)
    lines.append(out["sample_size_note"])
    lines.append("")

    (out_dir / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
