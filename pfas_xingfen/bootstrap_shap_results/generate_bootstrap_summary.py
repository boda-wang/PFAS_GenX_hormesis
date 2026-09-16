"""Create a compact, GitHub-friendly summary of Bootstrap SHAP stability.

The figure displays all 18 retained RDKit descriptors. Points are mean absolute
SHAP values across 500 bootstrap refits; horizontal bars are empirical 95%
bootstrap intervals. The title reports ranking stability separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "rdkit_descriptor_stability"
TABLE_FILE = RESULTS_DIR / "descriptor_stability.csv"
SPEARMAN_FILE = RESULTS_DIR / "spearman_summary.json"
PNG_FILE = HERE / "bootstrap_shap_stability_summary.png"
PDF_FILE = HERE / "bootstrap_shap_stability_summary.pdf"
SVG_FILE = HERE / "bootstrap_shap_stability_summary.svg"
JSON_FILE = HERE / "bootstrap_shap_stability_summary.json"


def main() -> None:
    table = pd.read_csv(TABLE_FILE).sort_values("bootstrap_mean_abs_shap", ascending=True).reset_index(drop=True)
    summary = json.loads(SPEARMAN_FILE.read_text(encoding="utf-8"))

    result_summary = {
        "n_bootstrap_replicates": summary["n_replicates"],
        "spearman_rank_correlation": {
            "mean": summary["mean_rho"],
            "median": summary["median_rho"],
            "ci95_low": summary["ci95_low"],
            "ci95_high": summary["ci95_high"],
        },
        "rank_scope": summary["rank_scope"],
        "top_descriptors_by_bootstrap_mean_abs_shap": table.sort_values(
            "bootstrap_mean_abs_shap", ascending=False
        ).head(5)[[
            "descriptor", "bootstrap_mean_abs_shap", "shap_ci95_low", "shap_ci95_high",
            "primary_rank", "bootstrap_median_rank", "rank_ci95_low", "rank_ci95_high",
        ]].to_dict(orient="records"),
    }
    JSON_FILE.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
    y = range(len(table))
    mean = table["bootstrap_mean_abs_shap"]
    lower = mean - table["shap_ci95_low"]
    upper = table["shap_ci95_high"] - mean

    ax.errorbar(
        mean,
        y,
        xerr=[lower, upper],
        fmt="o",
        markersize=4.5,
        color="#1F5A85",
        ecolor="#91B4C8",
        elinewidth=1.3,
        capsize=2.2,
        capthick=1.0,
        zorder=3,
    )
    ax.set_yticks(list(y), table["descriptor"])
    ax.set_xlabel("Mean absolute SHAP value (95% bootstrap interval)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D5DDE2", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        "Bootstrap SHAP stability for 18 RDKit descriptors\n"
        f"500 refits; median Spearman ρ = {summary['median_rho']:.3f} "
        f"(95% interval {summary['ci95_low']:.3f}–{summary['ci95_high']:.3f})",
        fontsize=10,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(PNG_FILE, dpi=600, facecolor="white")
    fig.savefig(PDF_FILE, facecolor="white")
    fig.savefig(SVG_FILE, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
