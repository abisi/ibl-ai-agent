"""Final report: methodology, distributions, statistics matrix, post-hoc
where triggered, motion/licking control. See ../question.md.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PROJECT_DIR = Path(__file__).resolve().parents[1]
METRICS = ["continuous_burstiness", "burst_index_whisker", "burst_index_auditory"]
METRIC_LABELS = {"continuous_burstiness": "Continuous burstiness",
                  "burst_index_whisker": "Burst index (whisker)",
                  "burst_index_auditory": "Burst index (auditory)"}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fig_distributions(df: pd.DataFrame, metric: str, day_stage: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    valid_col = f"{metric}_valid"
    for ax, scope, scope_filter in [
        (axes[0], "all mice", df["reward_group"].isin(["R+", "R-"])),
        (axes[1], "learners only", df["learning_category"].isin(["good", "moderate"])),
    ]:
        sub = df[scope_filter & df[valid_col] & df[metric].notna()]
        for rg, color in [("R+", "#4C72B0"), ("R-", "#DD8452")]:
            vals = sub.loc[sub.reward_group == rg, metric]
            if len(vals):
                ax.hist(vals, bins=40, alpha=0.5, color=color, label=f"{rg} (n={len(vals)})", density=True)
        ax.set_title(scope)
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.legend(fontsize=7)
    fig.suptitle(f"{METRIC_LABELS[metric]} -- {day_stage}, good+MUA")
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_motion_control(motion_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for i, day_stage in enumerate(["learning", "expert"]):
        for j, var in enumerate(["orofacial_motion_energy", "lick_rate_hz"]):
            ax = axes[i, j]
            sub = motion_df[(motion_df.day_stage == day_stage) & (motion_df.tier == "good_mua")]
            ax.scatter(sub[var], sub["median_continuous_burstiness"], s=15, alpha=0.6)
            ax.set_xlabel(var)
            ax.set_ylabel("median continuous burstiness")
            ax.set_title(f"{day_stage}")
    fig.tight_layout()
    return fig_to_b64(fig)


def main() -> None:
    summary = pd.read_csv(ARTIFACTS_DIR / "statistics_summary.csv")
    motion_df = pd.read_parquet(ARTIFACTS_DIR / "motion_control_per_session.parquet")
    corr_df = pd.read_csv(ARTIFACTS_DIR / "motion_control_correlations.csv")

    figs = []
    for day_stage in ["learning", "expert"]:
        df = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics.parquet")
        df_good_mua = df[df["quality_label"].isin(["good", "mua"])]
        for metric in METRICS:
            figs.append((f"{day_stage}: {METRIC_LABELS[metric]} distributions", fig_distributions(df_good_mua, metric, day_stage)))

    motion_fig = fig_motion_control(motion_df)

    html = [
        "<title>Burstiness x Cohort Report</title>",
        "<meta charset='utf-8'>",
        "<body style='font-family:sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5'>",
        "<h1>Burstiness x cohort -- preliminary (exploratory) analysis</h1>",
        "<p><b>Exploratory pass, no confirmation split</b> (per explicit request) -- treat all results below as "
        "hypothesis-generating, not confirmed. See <code>question.md</code> for the full method spec.</p>",
        "<h2>Statistics summary (24 combinations)</h2>",
        summary.to_html(index=False, float_format=lambda x: f"{x:.4g}"),
        "<h2>Distributions</h2>",
    ]
    for title, src in figs:
        html.append(f"<h3>{title}</h3><img src='{src}' style='max-width:100%'>")

    html.append("<h2>Motion/licking control</h2>")
    html.append(corr_df.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
    html.append(f"<img src='{motion_fig}' style='max-width:100%'>")

    html.append("</body>")
    out_path = PROJECT_DIR / "report.html"
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
