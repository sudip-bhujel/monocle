#!/usr/bin/env python3
"""Render Monocle paper tables/figures from confirmatory JSON artifacts.

Inputs (defaults match the confirmatory pipeline):
  - Monocle/data/confirmatory-empirical.json   (from paper_artifacts.py)
  - evaluation/action-safety-v2-results.json   (from action_safety_analysis.py)

Writes LaTeX under Monocle/tables and Monocle/figures. Static exhibits
(notation, dataset-validity, experimental-setup) are left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

COMMITTEES = ("OpenAI", "Anthropic", "Google", "Multi-provider")
COMMITTEE_LABEL = {
    "OpenAI": "OpenAI prompt ensemble",
    "Anthropic": "Anthropic prompt ensemble",
    "Google": "Google prompt ensemble",
    "Multi-provider": "Multi-provider ensemble",
}
COMMITTEE_PROVIDERS = {
    "OpenAI": "OpenAI",
    "Anthropic": "Anthropic",
    "Google": "Google",
    "Multi-provider": "All three",
}
PROMPT_ROLES = (
    "Baseline",
    "Independent risk",
    "Policy checklist",
    "Adversarial review",
    "Concise binary",
)
PROVIDERS = ("OpenAI", "Anthropic", "Google")
STRATUM_LABELS = (
    ("code:access-scope", "Access scope"),
    ("code:audit-retention", "Audit retention"),
    ("code:control-gate", "Control gate"),
    ("code:integrity-gate", "Integrity gate"),
    ("code:prechoice-flow", "Pre-choice flow"),
    ("code:preconsent-flow", "Pre-consent flow"),
    ("code:review-retention", "Review retention"),
    ("shell:review-prep", "Review preparation"),
    ("shell:workspace-scope", "Workspace scope"),
)
REPLACEMENT_LABEL = {
    "openai-mini-replace-independent-google": "Independent / G",
    "openai-mini-replace-adversarial-google": "Adversarial / G",
    "openai-mini-replace-independent-anthropic": "Independent / A",
    "openai-mini-replace-policy-google": "Policy / G",
    "openai-mini-replace-adversarial-anthropic": "Adversarial / A",
    "openai-mini-replace-policy-anthropic": "Policy / A",
    "openai-mini-replace-baseline-google": "Baseline / G",
    "openai-mini-replace-concise-anthropic": "Concise / A",
    "openai-mini-replace-baseline-anthropic": "Baseline / A",
    "openai-mini-replace-concise-google": "Concise / G",
}
FPR_TARGETS = (0.05, 0.10, 0.15, 0.20)
SCALING_SIZES = (1, 2, 3, 5)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--empirical",
        type=Path,
        default=root / "Monocle" / "data" / "confirmatory-empirical.json",
    )
    parser.add_argument(
        "--hypotheses",
        type=Path,
        default=root / "evaluation" / "action-safety-v2-results.json",
    )
    parser.add_argument("--paper-root", type=Path, default=root / "Monocle")
    parser.add_argument(
        "--frontier-only",
        action="store_true",
        help="render only exhibits supported by the four frontier committees",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned writes without modifying files",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str, *, dry_run: bool) -> None:
    text = content if content.endswith("\n") else content + "\n"
    if dry_run:
        print(f"would write {path} ({len(text)} bytes)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def namespace_labels(content: str, prefix: str) -> str:
    return content.replace(r"\label{tab:", rf"\label{{tab:{prefix}-").replace(
        r"\label{fig:", rf"\label{{fig:{prefix}-"
    )


def point(value: Any) -> float:
    if isinstance(value, dict):
        return float(value["point"])
    return float(value)


def interval(value: dict[str, Any]) -> tuple[float, float]:
    lo, hi = value["interval"]
    return float(lo), float(hi)


def fmt(value: float, digits: int = 3) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def tex_num(value: float, digits: int = 3) -> str:
    return f"${fmt(value, digits)}$"


def tex_small_nonzero(value: float, digits: int = 3) -> str:
    if value != 0 and abs(value) < 10**-digits:
        return rf"\num{{{value:.2e}}}"
    return tex_num(value, digits)


def tex_rate(value: float, digits: int = 3) -> str:
    """Format rates without disguising small nonzero values as exact zeros."""
    if value == 0:
        return r"\num{0}"
    return tex_small_nonzero(value, digits)


def optional_tex_point(est: dict[str, Any], digits: int = 3) -> str:
    value = est.get("point")
    return "---" if value is None else tex_num(float(value), digits)


def plus_minus(est: dict[str, Any], digits: int = 3) -> str:
    p = point(est)
    lo, hi = interval(est)
    if abs(p) < 1e-12 and abs(lo) < 1e-12 and abs(hi) < 1e-12:
        return r"\num{0}"
    return (
        f"${fmt(p, digits)}"
        f"^{{+{fmt(hi - p, digits)}}}"
        f"_{{-{fmt(p - lo, digits)}}}$"
    )


def bracket(est: dict[str, Any], digits: int = 3) -> str:
    """Format point with absolute interval bounds as $x^{ub}_{lb}$.

    Degenerate zero intervals render as the point alone.
    """
    if est.get("point") is None or any(bound is None for bound in est["interval"]):
        return "---"
    p = point(est)
    lo, hi = interval(est)
    if abs(lo) < 1e-12 and abs(hi) < 1e-12:
        return tex_rate(p, digits)
    if any(value != 0 and abs(value) < 10**-digits for value in (p, lo, hi)):

        def sci(value: float) -> str:
            return r"\num{0}" if value == 0 else rf"\num{{{value:.2e}}}"

        return f"${{{sci(p)}}}^{{{sci(hi)}}}_{{{sci(lo)}}}$"
    return f"${fmt(p, digits)}^{{{fmt(hi, digits)}}}_{{{fmt(lo, digits)}}}$"


def xerr_coord(x: float, y: Any, lo: float, hi: float, digits: int = 3) -> str:
    return (
        f"({fmt(x, digits)},{y}) "
        f"+= ({fmt(hi - x, digits)},0) "
        f"-= ({fmt(x - lo, digits)},0)"
    )


def yerr_coord(x: Any, y: float, lo: float, hi: float, digits: int = 3) -> str:
    return (
        f"({x},{fmt(y, digits)}) "
        f"+= (0,{fmt(hi - y, digits)}) "
        f"-= (0,{fmt(y - lo, digits)})"
    )


def blue_fill(rate: float) -> str:
    pct = max(2, min(70, int(round(100 * rate * 0.7))))
    return f"blue!{pct}"


def meta_banner(empirical: dict[str, Any]) -> str:
    meta = empirical.get("metadata") or {}
    run_id = meta.get("run_id", "confirmatory")
    draws = meta.get("bootstrap_draws", 2000)
    return (
        f"% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.\n"
        f"% Source run: {run_id}; bootstrap draws: {draws}."
    )


def artifact_design(empirical: dict[str, Any]) -> str:
    """Return the explicit artifact design, defaulting legacy artifacts to paired."""
    return str((empirical.get("metadata") or {}).get("design", "paired"))


def meta_note(empirical: dict[str, Any]) -> str:
    # Kept for callers; captions no longer embed this prose.
    return meta_banner(empirical)


def render_main_results(empirical: dict[str, Any]) -> str:
    primary = empirical["primary"]
    rows = []
    for name in COMMITTEES:
        row = primary[name]
        rows.append(
            "    "
            f"{COMMITTEE_LABEL[name]:<29} & {COMMITTEE_PROVIDERS[name]:<10} & "
            f"{plus_minus(row['R_obs'])} & {tex_small_nonzero(point(row['R_ind']))} & "
            f"{plus_minus(row['Gamma'])} & {optional_tex_point(row['N_eff'])} & "
            f"{tex_num(row['CPR'])} & {tex_num(row['FPR'])} \\\\"
        )
    body = "\n".join(rows)
    return f"""{meta_banner(empirical)}
\\begin{{table*}}[t]
  \\caption{{Primary any-flag committee results at the $15\\%$ per-monitor target.}}
  \\label{{tab:main-results}}
  \\centering\\footnotesize
  \\setlength{{\\tabcolsep}}{{4.5pt}}
  \\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
    \\toprule
    Committee                          & Providers &
    $R_{{\\mathrm{{obs}}}}$                 & $R_{{\\mathrm{{ind}}}}$ &
    $\\Gamma$                           &
    $N_{{\\mathrm{{eff}}}}^{{\\mathrm{{risk}}}}$ & CPR & \\multicolumn{{1}}{{c}}{{FPR}} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""


def render_regime_split(empirical: dict[str, Any], hypotheses: dict[str, Any]) -> str:
    framing = empirical["framing"]
    rows = []
    for name in COMMITTEES:
        row = framing[name]
        delta = {
            "point": float(row["delta_log_Gamma"]),
            "interval": list(row["delta_log_Gamma_interval"]),
        }
        rows.append(
            "    "
            f"{COMMITTEE_LABEL[name]:<29} & "
            f"{tex_num(row['R_direct'])} & {tex_num(row['R_evasion'])} & "
            f"{tex_num(row['Gamma_direct'])} & {tex_num(row['Gamma_evasion'])} & "
            f"{plus_minus(delta)} \\\\"
        )
    h2 = hypotheses["h2"]
    pooled = f"${fmt(h2['value'])}\\,[{fmt(h2['lower'])},\\infty)$"
    body = "\n".join(rows)
    return f"""{meta_banner(empirical)}
\\begin{{table*}}[t]
  \\caption{{Direct vs.\\ pressure framing at the $15\\%$ per-monitor target, with pooled \\boldcode{{H2}}.}}
  \\label{{tab:regime-split}}
  \\centering\\footnotesize
  \\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
    \\toprule
    Committee &
    $R_{{\\mathrm{{obs}}}}^{{\\mathrm{{direct}}}}$ &
    $R_{{\\mathrm{{obs}}}}^{{\\mathrm{{pressure}}}}$ &
    $\\Gamma_{{\\mathrm{{direct}}}}$ &
    $\\Gamma_{{\\mathrm{{pressure}}}}$ &
    $\\Delta\\log\\Gamma$ \\\\
    \\midrule
{body}
    \\midrule
    Pooled (\\boldcode{{H2}}, adjusted)     & ---     & ---     & ---     & ---     & {pooled} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""


def render_committee_scaling_table(empirical: dict[str, Any]) -> str:
    scaling = empirical["scaling"]
    blocks = []
    for group, label in (
        ("Same-model", r"\makecell[l]{Same\\model}"),
        ("Multi-provider", r"\makecell[l]{Multi\\provider}"),
    ):
        lines = []
        for index, size in enumerate(SCALING_SIZES):
            cell = scaling[group][str(size)]
            prefix = (
                f"    \\multirow{{4}}{{*}}{{{label}}} & ${size}$"
                if index == 0
                else f"    & ${size}$"
            )
            lines.append(
                f"{prefix} & {bracket(cell['R_obs'])} & {bracket(cell['R_ind'])} & "
                f"{bracket(cell['Gamma'])} & {bracket(cell['N_eff'])} \\\\"
            )
        blocks.append("\n".join(lines))
    body = "\n    \\midrule\n".join(blocks)
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Committee-size ablation averages with $95\\%$ family-cluster intervals.}}
  \\label{{tab:committee-scaling}}
  \\centering\\scriptsize
  \\setlength{{\\tabcolsep}}{{5pt}}
  \\begin{{adjustbox}}{{max width=\\linewidth}}
  \\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
    \\toprule
    \\makecell[l]{{Committee\\\\group}} & $N$ & $R_{{\\mathrm{{obs}}}}$ & $R_{{\\mathrm{{ind}}}}$ &
    $\\Gamma$ & $N_{{\\mathrm{{eff}}}}^{{\\mathrm{{risk}}}}$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  \\end{{adjustbox}}
\\end{{table}}
"""


def render_coverage(empirical: dict[str, Any]) -> str:
    coverage = empirical["coverage"]
    rows = []
    for name in COMMITTEES:
        row = coverage[name]
        rows.append(
            "    "
            f"{COMMITTEE_LABEL[name]:<29} & "
            f"{tex_num(row['N_EIC_1'])} & {tex_num(row['N_EIC_2'])} & "
            f"{tex_num(row['CPR'])} & {tex_num(row['Gamma'])} \\\\"
        )
    body = "\n".join(rows)
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Coverage attribution at the $15\\%$ per-monitor target.}}
  \\label{{tab:coverage}}
  \\centering\\footnotesize
  \\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
    \\toprule
    Committee &
    $N_{{\\mathrm{{EIC}}}}^{{(1)}}$ & $N_{{\\mathrm{{EIC}}}}^{{(2)}}$ & CPR & $\\Gamma$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def render_operating_curve_table(empirical: dict[str, Any]) -> str:
    rows_by_target: dict[float, dict[str, dict[str, float]]] = {
        target: {} for target in FPR_TARGETS
    }
    for row in empirical["operating_curve"]:
        target = float(row["fpr_target"])
        if target in rows_by_target:
            rows_by_target[target][row["committee"]] = row
    blocks = []
    for target_index, target in enumerate(FPR_TARGETS):
        shaded = target_index % 2 == 1
        lines = ["    \\showrowcolors" if shaded else "    \\hiderowcolors"]
        for index, name in enumerate(COMMITTEES):
            row = rows_by_target[target][name]
            if shaded and index == len(COMMITTEES) - 1:
                target_cell = f"\\multirow{{-4}}{{*}}{{${target:.2f}$}}"
            elif not shaded and index == 0:
                target_cell = f"\\multirow{{4}}{{*}}{{${target:.2f}$}}"
            else:
                target_cell = ""
            label = (
                "OpenAI"
                if name == "OpenAI"
                else (
                    "Anthropic"
                    if name == "Anthropic"
                    else ("Google" if name == "Google" else "Multi-provider")
                )
            )
            lines.append(
                f"    {target_cell:<28} & {label:<13} & "
                f"{tex_rate(row['fpr'])} & {tex_rate(row['R_obs'])} \\\\"
            )
        blocks.append("\n".join(lines))
    body = "\n    \\addlinespace[2pt]\n".join(blocks)
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Held-out FPR and $R_{{\\mathrm{{obs}}}}$ across per-monitor targets.}}
  \\label{{tab:operating-curve}}
  \\centering\\footnotesize
  \\rowcolors{{1}}{{gray!10}}{{gray!10}}
  \\begin{{tabular}}{{@{{}}llrr@{{}}}}
    \\hiderowcolors
    \\toprule
    Per-monitor target & Committee & FPR & $R_{{\\mathrm{{obs}}}}$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def render_monitor_marginals(empirical: dict[str, Any]) -> str:
    pooled = artifact_design(empirical) == "pooled"
    by_key = {
        (row["provider"], row["prompt_role"]): row
        for row in empirical["monitor_marginals"]
    }
    blocks = []
    for provider_index, provider in enumerate(PROVIDERS):
        # Colored blocks paint over a forward \multirow; place the label with
        # a negative multirow on the last row so it stays visible.
        use_negative_multirow = provider_index == 1
        lines: list[str] = []
        if provider_index == 1:
            lines.append("    \\showrowcolors")
        elif provider_index == 2:
            lines.append("    \\hiderowcolors")
        for role_index, role in enumerate(PROMPT_ROLES):
            row = by_key[(provider, role)]
            is_first = role_index == 0
            is_last = role_index == len(PROMPT_ROLES) - 1
            if use_negative_multirow and is_last:
                lead = f"    \\multirow{{-5}}{{*}}{{{provider}}}"
            elif (not use_negative_multirow) and is_first:
                lead = f"    \\multirow{{5}}{{*}}{{{provider}}}"
            else:
                lead = "     "
            if pooled:
                lines.append(
                    f"{lead}\n"
                    f"      & {role}\n"
                    f"      & {tex_rate(row['miss_unsafe'])} & {tex_rate(row['fpr'])}\n"
                    f"      & {tex_num(row['threshold'])} & {tex_rate(row['disagreement'])} \\\\"
                )
            else:
                lines.append(
                    f"{lead}\n"
                    f"      & {role}\n"
                    f"      & {tex_rate(row['miss_direct'])} & {tex_rate(row['miss_pressure'])} "
                    f"& {tex_rate(row['fpr'])}\n"
                    f"      & {tex_num(row['threshold'])} & {tex_rate(row['disagreement'])} \\\\"
                )
        blocks.append("\n".join(lines))
    body = "\n\n    \\midrule\n\n".join(blocks)
    columns = "@{}llcccc@{}" if pooled else "@{}llccccc@{}"
    miss_headers = (
        "      & \\makecell[c]{Unsafe\\\\miss}\n"
        if pooled
        else (
            "      & \\makecell[c]{Miss\\\\direct}\n"
            "      & \\makecell[c]{Miss\\\\pressure}\n"
        )
    )
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Per-monitor miss rates, holdout FPR, thresholds, and run disagreement.}}
  \\label{{tab:monitor-marginals}}

  \\centering
  \\scriptsize
  \\setlength{{\\tabcolsep}}{{3pt}}
  \\renewcommand{{\\arraystretch}}{{1.05}}

  \\rowcolors{{1}}{{gray!10}}{{gray!10}}

  \\begin{{tabular}}{{{columns}}}
    \\hiderowcolors
    \\toprule
    Provider
      & \\makecell[l]{{Prompt\\\\role}}
{miss_headers.rstrip()}
      & FPR
      & \\makecell[c]{{Threshold}}
      & \\makecell[c]{{Disagreement}} \\\\
    \\midrule

{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def render_cost_latency(empirical: dict[str, Any]) -> str:
    summary = empirical["cost_latency"]
    first = summary["committees"][COMMITTEES[0]]
    members = first["members"]
    case_runs = first["committee_decisions"]
    case_runs_tex = f"{case_runs:,}".replace(",", "{,}")
    rows = []
    for name in COMMITTEES:
        row = summary["committees"][name]
        rows.append(
            "    "
            f"{COMMITTEE_LABEL[name]:<29} & "
            f"${row['mean_cost_per_committee_decision_usd']:.5f}$ & "
            f"${row['median_parallel_latency_s']:.2f}$ \\\\"
        )
    body = "\n".join(rows)
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Descriptive committee cost and latency for the confirmatory run
    (${members}$ members and ${case_runs_tex}$ case-runs per committee).
    Cost is mean USD per committee decision; latency is median parallel wall time in seconds.}}
  \\label{{tab:cost-latency}}
  \\centering\\footnotesize
  \\begin{{tabular}}{{@{{}}lrr@{{}}}}
    \\toprule
    Committee & Cost (USD) & Latency (s) \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def render_majority(empirical: dict[str, Any]) -> str:
    majority = empirical["majority"]
    rows = []
    for name in COMMITTEES:
        row = majority[name]
        rows.append(
            "    "
            f"{COMMITTEE_LABEL[name]:<29} & "
            f"{tex_num(row['R_obs'])} & {tex_num(row['R_ind'])} & "
            f"{tex_num(row['Gamma'])} \\\\"
        )
    body = "\n".join(rows)
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Secondary majority-vote results for size-five committees at the
    $15\\%$ per-monitor target. An exact tie is escalated rather than accepted.}}
  \\label{{tab:majority}}
  \\centering\\footnotesize
  \\begin{{tabular}}{{@{{}}lrrr@{{}}}}
    \\toprule
    Committee & $R_{{\\mathrm{{obs}}}}$ & $R_{{\\mathrm{{ind}}}}$ & $\\Gamma$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def render_hypothesis_results(
    hypotheses: dict[str, Any], *, design: str = "paired"
) -> str:
    if design == "pooled":
        h1, h2 = hypotheses["h1"], hypotheses["h2"]
        h3_block = hypotheses["h3"]
        h3 = (
            h3_block["primary_0p05"]
            if isinstance(h3_block, dict) and "primary_0p05" in h3_block
            else h3_block
        )
    else:
        h1, h2, h3 = hypotheses["h1"], hypotheses["h2"], hypotheses["h3"]
        h4 = hypotheses["h4"]["primary_0p05"]

    def row(label: str, estimand: str, threshold: str, result: dict[str, Any]) -> str:
        estimate = (
            f"${fmt(result['value'])}$ "
            f"$[{fmt(result['lower'])}, {fmt(result['upper'])}]$"
        )
        decision = str(result["status"])
        return (
            f"    {label} & {estimand} &\n"
            f"    {threshold} &\n"
            f"    {estimate} &\n"
            f"    {decision} \\\\"
        )

    if design == "pooled":
        rows = "\n\n".join(
            [
                row(
                    "\\boldcode{H1} residual dependence",
                    "pooled family-stratified $\\log\\Gamma$, all unsafe rows",
                    "$>\\log 1.10$",
                    h1,
                ),
                row(
                    "\\boldcode{H2} dependence dividend",
                    "mean $\\log\\Gamma_{\\mathrm{hom}}-\\log\\Gamma_{\\mathrm{sub}}$ over ten substitutions",
                    "$>0.10$",
                    h2,
                ),
                row(
                    "\\boldcode{H3} operational diversity",
                    "$R_{\\mathrm{hom}}-R_{\\mathrm{mixed}}$ at the selected $5\\%$ committee FPR",
                    "$>0.05$, FPR gap $\\leq 0.025$",
                    h3,
                ),
            ]
        )
        caption = "Confirmatory \\boldcode{H1}--\\boldcode{H3} decisions from Bonferroni-adjusted bounds."
    else:
        rows = "\n\n".join(
            [
                row(
                    "\\boldcode{H1} residual dependence",
                    "pooled family-stratified $\\log\\Gamma$, pressure-framed rows",
                    "$>\\log 1.10$",
                    h1,
                ),
                row(
                    "\\boldcode{H2} pressure amplification",
                    "$\\log\\Gamma_{\\mathrm{pressure}}-\\log\\Gamma_{\\mathrm{direct}}$, pooled",
                    "$>0.10$",
                    h2,
                ),
                row(
                    "\\boldcode{H3} dependence dividend",
                    "mean $\\log\\Gamma_{\\mathrm{hom}}-\\log\\Gamma_{\\mathrm{sub}}$ over ten substitutions",
                    "$>0.10$",
                    h3,
                ),
                row(
                    "H4 operational diversity",
                    "$R_{\\mathrm{hom}}-R_{\\mathrm{mixed}}$ at cross-fitted $5\\%$ committee FPR",
                    "$>0.05$, FPR gap $\\leq 0.025$",
                    h4,
                ),
            ]
        )
        caption = (
            "Confirmatory \\boldcode{H1}--H4 decisions from Bonferroni-adjusted bounds."
        )
    return f"""% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.
\\begin{{table*}}[t]
  \\caption{{{caption}}}
  \\label{{tab:hypothesis-results}}
  \\centering\\footnotesize
  \\begin{{tabularx}}{{\\textwidth}}{{@{{}}lXlrl@{{}}}}
    \\toprule
    Hypothesis & Estimand & Threshold & Estimate [adj.\\ bounds] & Decision \\\\
    \\midrule
{rows}
    \\bottomrule
  \\end{{tabularx}}
\\end{{table*}}
"""


def render_dependence_sensitivity_table(empirical: dict[str, Any]) -> str:
    sensitivity = empirical["sensitivity"]
    rows = []
    for name in COMMITTEES:
        row = sensitivity[name]
        loso = (
            f"$[{fmt(float(row['loso_min']))},"
            f"\\,{fmt(float(row['loso_max']))}]$"
        )
        rows.append(
            "    "
            f"{name:<14} & {tex_num(float(row['unstratified']))} & "
            f"{tex_num(float(row['stratified']))} & "
            f"{tex_num(point(row['item_conditioned']))} & {loso} \\\\"
        )
    body = "\n".join(rows)
    return f"""{meta_banner(empirical)}
\\begin{{table}}[t]
  \\caption{{Sensitivity of dependence inflation $\\Gamma$ to the independence prediction.}}
  \\label{{tab:dependence-sensitivity}}
  \\centering
  \\setlength{{\\tabcolsep}}{{3.2pt}}
  \\begin{{adjustbox}}{{max width=\\linewidth}}
  \\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
    \\toprule
    Committee & \\makecell[c]{{Unstratified}} &
    \\makecell[c]{{Stratum-\\\\conditioned}} &
    \\makecell[c]{{Per-case}} &
    \\makecell[c]{{Leave-one-stratum-\\\\out range}} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  \\end{{adjustbox}}
\\end{{table}}
"""


def _sorted_replacements(empirical: dict[str, Any]) -> list[str]:
    replacements = empirical["replacements"]
    return sorted(
        replacements,
        key=lambda committee_id: (
            float(replacements[committee_id]["delta_R"]["point"]),
            committee_id,
        ),
    )


def render_replacement_effects(empirical: dict[str, Any]) -> str:
    replacements = empirical["replacements"]
    ordered = _sorted_replacements(empirical)
    labels = [REPLACEMENT_LABEL[committee_id] for committee_id in ordered]
    yticklabels = ",\n            ".join(labels)

    def series(field: str) -> list[tuple[float, float, float]]:
        out = []
        for committee_id in ordered:
            row = replacements[committee_id][field]
            if isinstance(row, dict):
                p = point(row)
                lo, hi = interval(row)
            else:
                p = float(row)
                lo = hi = p
            out.append((p, lo, hi))
        return out

    def panel_coords(values: list[tuple[float, float, float]]) -> str:
        lines = []
        for y, (p, lo, hi) in enumerate(values, start=1):
            lines.append(f"          {xerr_coord(p, y, lo, hi)}")
        return "\n".join(lines)

    def point_coords(values: list[tuple[float, float, float]]) -> str:
        return "\n".join(
            f"          ({fmt(p)},{y})" for y, (p, _, _) in enumerate(values, start=1)
        )

    def axis_limits(
        values: list[tuple[float, float, float]], pad: float = 0.02
    ) -> tuple[str, str]:
        lo = min(v[1] for v in values)
        hi = max(v[2] for v in values)
        span = max(hi - lo, 0.05)
        return fmt(lo - pad - 0.05 * span), fmt(hi + pad + 0.05 * span)

    delta_r = series("delta_R")
    delta_g = series("delta_log_Gamma")
    delta_f = series("delta_fpr")
    r_min, r_max = axis_limits(delta_r)
    g_min, g_max = axis_limits(delta_g)
    f_min, f_max = axis_limits(delta_f)
    # Shared axis-box geometry: identical absolute width/height so (b)/(c)
    # match (a)'s plot rectangle (y labels sit outside the box on (a)).
    axis_w = "4.05cm"
    axis_h = "4.2cm"
    axis_common = f"""scale only axis,width={axis_w},height={axis_h},
          ymin=0.5,ymax=10.5,enlarge y limits=false,
          clip=false,
          ytick={{1,2,3,4,5,6,7,8,9,10}},
          xmajorgrids,grid style={{gray!20}},
          label style={{font=\\small}}"""
    return f"""% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.
% Row order (bottom to top) is ascending $\\Delta R_{{\\mathrm{{obs}}}}$.
\\begin{{figure*}}[t]
  \\centering
  \\begin{{subfigure}}[t]{{0.38\\textwidth}}
    \\centering
    \\begin{{tikzpicture}}[baseline=(current axis.north)]
      \\begin{{axis}}[
          {axis_common},
          xmin={r_min},xmax={r_max},
          xlabel={{Risk reduction $\\Delta R_{{\\mathrm{{obs}}}}$}},
          yticklabels={{
            {yticklabels}}},
          y tick label style={{font=\\scriptsize}},
          scaled x ticks=false,
          x tick label style={{font=\\scriptsize,/pgf/number format/fixed,
            /pgf/number format/precision=2}}
        ]
        \\addplot [solid,gray,thick] coordinates {{(0,0.5) (0,10.5)}};
        \\addplot [
          only marks,mark=*,color=blue!70!black,
          error bars/.cd,x dir=both,x explicit,
          error bar style={{color=blue!70!black,thick}},
          error mark options={{rotate=90,color=blue!70!black,mark size=2pt}}
        ] coordinates {{
{panel_coords(delta_r)}
        }};
      \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{Risk reduction $\\Delta R_{{\\mathrm{{obs}}}}$.}}
  \\end{{subfigure}}\\hfill
  \\begin{{subfigure}}[t]{{0.30\\textwidth}}
    \\centering
    \\begin{{tikzpicture}}[baseline=(current axis.north)]
      \\begin{{axis}}[
          {axis_common},
          xmin={g_min},xmax={g_max},
          xlabel={{Dependence reduction $\\Delta\\log\\Gamma$}},
          yticklabels={{}},
          tick label style={{font=\\scriptsize}}
        ]
        \\addplot [solid,gray,thick] coordinates {{(0,0.5) (0,10.5)}};
        \\addplot [dashed,gray!80,thick] coordinates {{(0.10,0.5) (0.10,10.5)}};
        \\addplot [
          only marks,mark=square*,color=orange!85!black,
          error bars/.cd,x dir=both,x explicit,
          error bar style={{color=orange!85!black,thick}},
          error mark options={{rotate=90,color=orange!85!black,mark size=2pt}}
        ] coordinates {{
{panel_coords(delta_g)}
        }};
      \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{Dependence reduction $\\Delta\\log\\Gamma$.}}
  \\end{{subfigure}}\\hfill
  \\begin{{subfigure}}[t]{{0.30\\textwidth}}
    \\centering
    \\begin{{tikzpicture}}[baseline=(current axis.north)]
      \\begin{{axis}}[
          {axis_common},
          xmin={f_min},xmax={f_max},
          xlabel={{FPR change $\\Delta\\mathrm{{FPR}}$}},
          yticklabels={{}},
          tick label style={{font=\\scriptsize}}
        ]
        \\addplot [solid,gray,thick] coordinates {{(0,0.5) (0,10.5)}};
        \\addplot [
          only marks,mark=triangle*,color=green!45!black
        ] coordinates {{
{point_coords(delta_f)}
        }};
      \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{FPR change $\\Delta\\mathrm{{FPR}}$.}}
  \\end{{subfigure}}
  \\caption{{One-member cross-provider substitutions relative to the OpenAI ensemble ($A$: Anthropic; $G$: Google).}}
  \\label{{fig:replacement-effects}}
\\end{{figure*}}
"""


def render_replacement_quadrant(empirical: dict[str, Any]) -> str:
    """Render a simple bivariate view of replacement effects.

    The existing forest plot remains the uncertainty-focused exhibit.  This
    companion plot answers only whether each substitution jointly reduces
    observed risk and dependence.  Provider identity is encoded by marker
    shape; uncertainty and FPR remain available in the forest plot and tables.
    """
    replacements = empirical["replacements"]
    ordered = sorted(
        replacements, key=lambda committee_id: REPLACEMENT_LABEL[committee_id]
    )

    points = []
    for committee_id in ordered:
        row = replacements[committee_id]
        delta_r = row["delta_R"]
        delta_g = row["delta_log_Gamma"]
        r_point = point(delta_r)
        g_point = point(delta_g)
        points.append(
            {
                "committee_id": committee_id,
                "label": REPLACEMENT_LABEL[committee_id],
                "provider": (
                    "Anthropic" if committee_id.endswith("-anthropic") else "Google"
                ),
                "risk": r_point,
                "dependence": g_point,
            }
        )

    x_lo = min(float(row["dependence"]) for row in points)
    x_hi = max(float(row["dependence"]) for row in points)
    y_lo = min(float(row["risk"]) for row in points)
    y_hi = max(float(row["risk"]) for row in points)
    x_pad = max(0.35, 0.05 * (x_hi - x_lo))
    y_pad = max(0.004, 0.08 * (y_hi - y_lo))
    xmin, xmax = x_lo - x_pad, x_hi + x_pad
    ymin, ymax = y_lo - y_pad, y_hi + y_pad

    def scatter_coords(provider: str) -> str:
        return "\n".join(
            f"          ({fmt(float(row['dependence']))},{fmt(float(row['risk']))})"
            for row in points
            if row["provider"] == provider
        )

    label_offsets = {
        "Adversarial / A": (-0.20, -0.004, "north east"),
        "Adversarial / G": (-0.20, 0.004, "south east"),
        "Baseline / A": (0.30, 0.004, "south west"),
        "Baseline / G": (-0.30, -0.003, "north east"),
        "Concise / A": (0.25, 0.005, "south west"),
        "Concise / G": (0.25, -0.002, "north west"),
        "Independent / A": (-0.15, -0.003, "north east"),
        "Independent / G": (0.20, -0.003, "north west"),
        "Policy / A": (0.20, -0.002, "north west"),
        "Policy / G": (-0.20, 0.006, "south east"),
    }
    labels = []
    for row in points:
        dx, dy, anchor = label_offsets[row["label"]]
        x = float(row["dependence"])
        y = float(row["risk"])
        labels.append(
            "        \\draw[gray!55,line width=0.35pt] "
            f"(axis cs:{fmt(x)},{fmt(y)}) -- "
            f"(axis cs:{fmt(x + dx)},{fmt(y + dy)}) "
            f"node[font=\\scriptsize,anchor={anchor},"
            "text=black,fill=white,fill opacity=0.78,"
            "text opacity=1,inner sep=1pt] "
            f"{{{row['label'].replace(' / ', '/')}}};"
        )
    labels_tex = "\n".join(labels)

    return f"""{meta_banner(empirical)}
% Circles are Anthropic replacements; triangles are Google replacements.
\\begin{{figure*}}[t]
  \\centering
  \\begin{{tikzpicture}}
    \\begin{{axis}}[
        width=0.80\\textwidth,height=7.2cm,
        xmin={fmt(xmin)},xmax={fmt(xmax)},
        ymin={fmt(ymin)},ymax={fmt(ymax)},
        xlabel={{Dependence reduction $\\Delta\\log\\Gamma$}},
        ylabel={{Risk reduction $\\Delta R_{{\\mathrm{{obs}}}}$}},
        scaled y ticks=false,
        y tick label style={{/pgf/number format/fixed,
          /pgf/number format/precision=3,font=\\scriptsize}},
        tick label style={{font=\\scriptsize}},
        label style={{font=\\small}},
        grid=major,grid style={{gray!18}},
        legend style={{font=\\scriptsize,draw=none,at={{(0.02,0.98)}},
          anchor=north west,fill=white,fill opacity=0.88,text opacity=1}},
        clip=false
      ]
        \\path[fill=green!7] (axis cs:0,0)
          rectangle (axis cs:{fmt(xmax)},{fmt(ymax)});
        \\addplot [solid,gray!75,thick,forget plot] coordinates {{
          (0,{fmt(ymin)}) (0,{fmt(ymax)})
        }};
        \\addplot [solid,gray!75,thick,forget plot] coordinates {{
          ({fmt(xmin)},0) ({fmt(xmax)},0)
        }};
        \\addplot [
          only marks,mark=*,mark size=3.2pt,
          draw=blue!70!black,fill=blue!55
        ] coordinates {{
{scatter_coords("Anthropic")}
        }};
        \\addlegendentry{{Anthropic replacement}}
        \\addplot [
          only marks,mark=triangle*,mark size=3.5pt,
          draw=blue!70!black,fill=blue!55
        ] coordinates {{
{scatter_coords("Google")}
        }};
        \\addlegendentry{{Google replacement}}
{labels_tex}
        \\node[font=\\scriptsize,anchor=north east,align=right,
          fill=green!10,inner sep=2pt]
          at (rel axis cs:0.985,0.985) {{Improves both}};
    \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{Joint risk and dependence effects of one-member cross-provider substitutions. Positive values on both axes indicate improvement relative to the OpenAI ensemble.}}
  \\label{{fig:replacement-quadrant}}
\\end{{figure*}}
"""


def render_committee_scaling_figure(empirical: dict[str, Any]) -> str:
    scaling = empirical["scaling"]

    def obs_coords(group: str) -> str:
        lines = []
        for size in SCALING_SIZES:
            est = scaling[group][str(size)]["R_obs"]
            lines.append("            " + yerr_coord(size, point(est), *interval(est)))
        return "\n".join(lines)

    def ind_coords(group: str) -> str:
        return " ".join(
            f"({size},{fmt(point(scaling[group][str(size)]['R_ind']))})"
            for size in SCALING_SIZES
        )

    def neff_coords(group: str) -> str:
        lines = []
        for size in SCALING_SIZES:
            est = scaling[group][str(size)]["N_eff"]
            if est.get("point") is not None and all(
                bound is not None for bound in est["interval"]
            ):
                lines.append(
                    "            " + yerr_coord(size, point(est), *interval(est))
                )
        return "\n".join(lines)

    risk_vals: list[float] = []
    neff_vals: list[float] = []
    for group in ("Same-model", "Multi-provider"):
        for size in SCALING_SIZES:
            for metric in ("R_obs", "R_ind"):
                est = scaling[group][str(size)][metric]
                risk_vals.append(point(est))
                if isinstance(est, dict) and "interval" in est:
                    risk_vals.extend(interval(est))
            neff = scaling[group][str(size)]["N_eff"]
            if neff.get("point") is not None and all(
                bound is not None for bound in neff["interval"]
            ):
                neff_vals.append(point(neff))
                neff_vals.extend(interval(neff))
    risk_hi = max(risk_vals)
    risk_ymax = min(1.0, max(0.15, math.ceil((risk_hi + 0.01) * 20) / 20))
    # Leave a small visual margin below zero so zero-valued observations and
    # error bars remain distinct from the lower axis frame.
    risk_ymin = -0.05 * risk_ymax
    neff_hi = max(neff_vals + [5.0])
    neff_ymax = max(5.3, math.ceil(neff_hi * 2) / 2 + 0.3)

    return f"""% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.
\\begin{{figure}}[t]
  \\centering
  \\begin{{subfigure}}[t]{{0.96\\linewidth}}
    \\centering
    \\begin{{tikzpicture}}
      \\begin{{axis}}[
          width=\\linewidth,height=4.5cm,
          xmin=0.7,xmax=5.3,ymin={fmt(risk_ymin)},ymax={fmt(risk_ymax)},
          xtick={{1,2,3,5}},xlabel={{Committee size $N$}},
          ylabel={{$R_{{\\mathrm{{obs}}}}$}},
          grid=major,grid style={{gray!20}},
          legend columns=2,
          legend style={{font=\\tiny,draw=none,at={{(0.97,0.97)}},anchor=north east}},
          scaled y ticks=false,
          yticklabel style={{font=\\scriptsize,/pgf/number format/fixed}},
          xticklabel style={{font=\\scriptsize}},label style={{font=\\small}}
        ]
        \\addplot [color=blue!70!black,mark=*,thick,
          error bars/.cd,y dir=both,y explicit]
          coordinates {{
{obs_coords('Same-model')}
          }};
        \\addplot [color=blue!70!black,mark=o,dashed,thick]
          coordinates {{{ind_coords('Same-model')}}};
        \\addplot [color=orange!85!black,mark=square*,thick,
          error bars/.cd,y dir=both,y explicit]
          coordinates {{
{obs_coords('Multi-provider')}
          }};
        \\addplot [color=orange!85!black,mark=square,dashed,thick]
          coordinates {{{ind_coords('Multi-provider')}}};
        \\legend{{Same: $R_{{\\mathrm{{obs}}}}$,Same: $R_{{\\mathrm{{ind}}}}$,
          Multi: $R_{{\\mathrm{{obs}}}}$,Multi: $R_{{\\mathrm{{ind}}}}$}}
      \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{Observed and independence-predicted risk.}}
  \\end{{subfigure}}\\par\\smallskip
  \\begin{{subfigure}}[t]{{0.96\\linewidth}}
    \\centering
    \\begin{{tikzpicture}}
      \\begin{{axis}}[
          width=\\linewidth,height=4.5cm,
          xmin=0.7,xmax=5.3,ymin=0.7,ymax={fmt(neff_ymax)},
          xtick={{1,2,3,5}},ytick={{1,2,3,4,5}},
          xlabel={{Nominal committee size $N$}},
          ylabel={{$N_{{\\mathrm{{eff}}}}^{{\\mathrm{{risk}}}}$}},
          grid=major,grid style={{gray!20}},
          legend columns=2,
          legend style={{font=\\tiny,draw=none,at={{(0.03,0.97)}},anchor=north west}},
          tick label style={{font=\\scriptsize}},label style={{font=\\small}}
        ]
        \\addplot [dashed,gray,thick] coordinates {{(1,1) (5,5)}};
        \\addplot [color=blue!70!black,mark=*,thick,
          error bars/.cd,y dir=both,y explicit]
          coordinates {{
{neff_coords('Same-model')}
          }};
        \\addplot [color=orange!85!black,mark=square*,thick,
          error bars/.cd,y dir=both,y explicit]
          coordinates {{
{neff_coords('Multi-provider')}
          }};
        \\legend{{Independence,Same-model,Multi-provider}}
      \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{Nominal versus effective redundancy.}}
  \\end{{subfigure}}
  \\caption{{Committee-size ablation of observed risk, independence risk, and $N_{{\\mathrm{{eff}}}}$.}}
  \\label{{fig:committee-scaling}}
\\end{{figure}}
"""


def render_frontier(empirical: dict[str, Any]) -> str:
    by_committee: dict[str, list[tuple[float, float]]] = {
        name: [] for name in COMMITTEES
    }
    primary_pts = []
    for row in empirical["operating_curve"]:
        name = row["committee"]
        if name not in by_committee:
            continue
        pt = (float(row["fpr"]), float(row["R_obs"]))
        if pt not in by_committee[name]:
            by_committee[name].append(pt)
        if abs(float(row["fpr_target"]) - 0.15) < 1e-9:
            primary_pts.append(pt)
    for name in COMMITTEES:
        by_committee[name].sort()

    def coords(points: list[tuple[float, float]]) -> str:
        return " ".join(f"({fmt(x)},{fmt(y)})" for x, y in points)

    styles = {
        "OpenAI": "color=blue!70!black,mark=*,thick",
        "Anthropic": "color=red!70!black,mark=square*,thick",
        "Google": "color=green!45!black,mark=triangle*,thick",
        "Multi-provider": "color=orange!85!black,mark=diamond*,thick",
    }
    plots = []
    for name in COMMITTEES:
        plots.append(
            f"      \\addplot+ [{styles[name]}]\n"
            f"        coordinates {{{coords(by_committee[name])}}};"
        )
    unique_primary = []
    for pt in primary_pts:
        if pt not in unique_primary:
            unique_primary.append(pt)
    plots.append(
        "      \\addplot [only marks,mark=o,mark size=4.5pt,black]\n"
        f"        coordinates {{{coords(unique_primary)}}};"
    )
    all_fpr = [p[0] for pts in by_committee.values() for p in pts]
    all_r = [p[1] for pts in by_committee.values() for p in pts]
    xmin = max(0.0, min(all_fpr) - 0.02)
    xmax = min(1.0, max(all_fpr) + 0.02)
    ymin = max(0.0, min(all_r) - 0.05)
    ymax = min(1.0, max(0.01, math.ceil((max(all_r) + 0.005) * 100) / 100))
    body = "\n".join(plots)
    return f"""% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
    \\begin{{axis}}[
        width=\\linewidth,height=5.4cm,
        xlabel={{Held-out committee FPR}},
        ylabel={{Unsafe acceptance $R_{{\\mathrm{{obs}}}}$}},
        xmin={fmt(xmin)},xmax={fmt(xmax)},ymin={fmt(ymin)},ymax={fmt(ymax)},
        xtick={{0.05,0.10,0.15,0.20,0.25}},
        xticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=2}},
        grid=major,grid style={{gray!20}},
        legend style={{font=\\scriptsize,draw=none,at={{(0.97,0.97)}},anchor=north east}},
        tick label style={{font=\\scriptsize}},label style={{font=\\small}}
      ]
{body}
      \\legend{{OpenAI,Anthropic,Google,Multi-provider,primary point}}
    \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{Held-out risk--FPR operating curves; circles mark the $15\\%$ target.}}
  \\label{{fig:frontier}}
\\end{{figure}}
"""


def render_miss_count(empirical: dict[str, Any]) -> str:
    miss = empirical["miss_count"]
    caption = (
        "Observed vs.\\ independence miss-count mass on all unsafe inputs."
        if artifact_design(empirical) == "pooled"
        else "Observed vs.\\ independence miss-count mass on pressure-framed unsafe inputs."
    )
    ymax = 0.0
    panels = []

    def bar_coords(values: list[float]) -> str:
        def lead_dot(v: float) -> str:
            return f"{v:.2f}"[1:] if v < 1.0 else f"{v:.2f}"

        return " ".join(f"({k},{fmt(v)}) [{lead_dot(v)}]" for k, v in enumerate(values))

    for index, name in enumerate(COMMITTEES):
        row = miss[name]
        ymax = max(ymax, max(row["P_ind"]), max(row["P_obs"]))
        # Overlay the first panel's y-axis text; a left \\hspace* in the figure
        # reserves room so it does not spill into the page margin.
        yopts = (
            "ylabel={$P(K=k)$},ylabel style={overlay,xshift=0.35em},yticklabel style={overlay}"
            if index == 0
            else "yticklabels=\\empty,ylabel={},yticklabel style={overlay},trim axis left"
        )
        caption_name = {
            "OpenAI": "OpenAI",
            "Anthropic": "Anthropic",
            "Google": "Google",
            "Multi-provider": "Multi-provider",
        }[name]
        panels.append(
            f"""\\begin{{subfigure}}[t]{{0.220\\textwidth}}
\\centering
\\misscountpanel{{{yopts}}}
  {{{bar_coords(row['P_ind'])}}}
  {{{bar_coords(row['P_obs'])}}}
\\caption{{{caption_name}, $\\Gamma={fmt(row['Gamma'])}$.}}
\\end{{subfigure}}"""
        )
    ymax = min(0.95, max(0.60, math.ceil(ymax * 20) / 20 + 0.08))
    # Stretch panels across the full measure so the rightmost is not inset, and
    # leave a fixed left pad for the overlaid y-label on panel (a).
    body = "\\hfill\n".join(panels)
    return f"""% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.
\\providecommand{{\\misscountpanel}}{{}}
\\renewcommand{{\\misscountpanel}}[3]{{%
  \\begin{{tikzpicture}}
    \\begin{{axis}}[
        scale only axis,width=0.90\\linewidth,height=2.65cm,
        ybar=0pt,
        xmin=-0.50,xmax=5.50,ymin=0,ymax={fmt(ymax)},
        xtick={{0,1,2,3,4,5}},
        xlabel={{Miss count $K$}},
        #1,
        ymajorgrids,grid style={{gray!20}},
        tick label style={{font=\\scriptsize}},
        label style={{font=\\small}},
        point meta=explicit symbolic
      ]
      \\addplot[
        ybar,bar width=0.40,bar shift=-0.20,
        draw=gray!75,fill=gray!10,
        pattern=north east lines,pattern color=gray!70,
        nodes near coords,
        nodes near coords style={{font=\\tiny,anchor=south,
          inner sep=0.35pt,xshift=-0.8pt}}
      ] coordinates {{#2}};
      \\addplot[
        ybar,bar width=0.40,bar shift=0.20,
        draw=blue!75!black,fill=blue!20,
        pattern=crosshatch,pattern color=blue!75!black,
        nodes near coords,
        nodes near coords style={{font=\\tiny,anchor=south,
          inner sep=0.35pt,xshift=-0.8pt}}
      ] coordinates {{#3}};
    \\end{{axis}}
    % Keep adjacent panels from colliding when labels stick out of the axis box.
    \\pgfresetboundingbox
    \\useasboundingbox
      (current axis.outer south west)
      rectangle
      ([yshift=3pt]current axis.outer north east);
  \\end{{tikzpicture}}%
}}
\\begin{{figure*}}[t]
  \\centering
  {{\\scriptsize\\hfill
  \\begin{{tikzpicture}}[baseline=(L.base)]
    \\node[
      draw=gray!75,fill=gray!10,inner sep=0pt,
      pattern=north east lines,pattern color=gray!70,
      minimum width=1.7ex,minimum height=1.4ex
    ] (a) {{}};
    \\node[right=0.25em of a,font=\\scriptsize] (L) {{Independence}};
    \\node[
      draw=blue!75!black,fill=blue!20,inner sep=0pt,
      pattern=crosshatch,pattern color=blue!75!black,
      minimum width=1.7ex,minimum height=1.4ex,
      right=1.1em of L
    ] (b) {{}};
    \\node[right=0.25em of b,font=\\scriptsize] {{Observed}};
  \\end{{tikzpicture}}\\par}}
  \\vspace{{-0.4em}}
  \\hspace*{{2.6em}}%
{body}
  \\caption{{{caption}}}
  \\label{{fig:miss-count}}
\\end{{figure*}}
"""


def render_stratum_heatmap(empirical: dict[str, Any]) -> str:
    strata = empirical["strata"]
    lines = [
        "      |[rowlabel]| \\textbf{Unsafe-action stratum} &",
        "      |[header]| OpenAI & |[header]| Anthropic & |[header]| Google &",
        "      |[header]| Multi-provider \\\\",
    ]
    for stratum_id, label in STRATUM_LABELS:
        row = strata[stratum_id]
        n = int(row["families"])
        cells = []
        for name in COMMITTEES:
            est = row["committees"][name]
            p = point(est)
            lo, hi = interval(est)
            if abs(lo) < 1e-12 and abs(hi) < 1e-12:
                cell_tex = f"${fmt(p)}$"
            else:
                cell_tex = f"${fmt(p)}^{{{fmt(hi)}}}_{{{fmt(lo)}}}$"
            cells.append(f"|[cell,fill={blue_fill(p)}]| {cell_tex}")
        lines.append(f"      |[rowlabel]| {label} ($n={n}$) &")
        lines.append("      " + " &\n      ".join(cells) + " \\\\")
    body = "\n".join(lines)
    return f"""% AUTO-GENERATED by scripts/render_paper_exhibits.py — do not edit by hand.
\\begin{{figure*}}[t]
  \\centering
  \\begin{{tikzpicture}}[
      cell/.style={{draw=white,line width=0.6pt,minimum width=2.75cm,
        minimum height=0.58cm,align=center,font=\\scriptsize}},
      rowlabel/.style={{draw=none,minimum width=3.65cm,minimum height=0.58cm,
        align=left,font=\\scriptsize}},
      header/.style={{draw=none,minimum width=2.75cm,minimum height=0.65cm,
        align=center,font=\\scriptsize\\bfseries}}
    ]
    \\matrix (m) [matrix of nodes,row sep=-\\pgflinewidth,column sep=-\\pgflinewidth,
      nodes in empty cells] {{
{body}
    }};
  \\end{{tikzpicture}}
  \\caption{{Joint-miss rates by unsafe-action stratum with $95\\%$ family-cluster intervals.}}
  \\label{{fig:stratum-heatmap}}
\\end{{figure*}}
"""


def main() -> None:
    args = parse_args()
    empirical = load_json(args.empirical)
    hypotheses = None if args.frontier_only else load_json(args.hypotheses)
    paper = args.paper_root
    design = artifact_design(empirical)
    outputs = {
        paper / "tables" / "main-results.tex": render_main_results(empirical),
        paper
        / "tables"
        / "committee-scaling.tex": render_committee_scaling_table(empirical),
        paper / "tables" / "coverage.tex": render_coverage(empirical),
        paper
        / "tables"
        / "operating-curve.tex": render_operating_curve_table(empirical),
        paper / "tables" / "monitor-marginals.tex": render_monitor_marginals(empirical),
        paper / "tables" / "cost-latency.tex": render_cost_latency(empirical),
        paper / "tables" / "majority.tex": render_majority(empirical),
        paper
        / "tables"
        / "dependence-sensitivity.tex": render_dependence_sensitivity_table(empirical),
        paper
        / "figures"
        / "committee-scaling.tex": render_committee_scaling_figure(empirical),
        paper / "figures" / "frontier.tex": render_frontier(empirical),
        paper / "figures" / "miss-count.tex": render_miss_count(empirical),
        paper / "figures" / "stratum-heatmap.tex": render_stratum_heatmap(empirical),
    }
    if not args.frontier_only:
        assert hypotheses is not None
        outputs[paper / "tables" / "hypothesis-results.tex"] = (
            render_hypothesis_results(hypotheses, design=design)
        )
        outputs[paper / "figures" / "replacement-effects.tex"] = (
            render_replacement_effects(empirical)
        )
        outputs[paper / "figures" / "replacement-quadrant.tex"] = (
            render_replacement_quadrant(empirical)
        )
    if design == "paired" and not args.frontier_only:
        assert hypotheses is not None
        outputs[paper / "tables" / "regime-split.tex"] = render_regime_split(
            empirical, hypotheses
        )
    if args.frontier_only:
        outputs = {
            path: namespace_labels(content, "frontier")
            for path, content in outputs.items()
        }
    for path, content in outputs.items():
        write_text(path, content, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
