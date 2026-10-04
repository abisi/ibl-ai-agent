"""Single combined grid across ALL 'regular decoding' area_group tags,
across-condition (half: first vs second) x cohort (R+, R-) -- user
request 2026-09-21: "position all figures in a single grid for
comparison", following up on `034`'s per-tag `_condition_cohorts.png`
figures (one file per tag). Also carries `034`'s desaturation retrofit
("Make the first half condition a less saturated than the second half"):
imports `034`'s `lighten()` and reuses it for the 'first' marker.

One row per (tag, metric-window) pair actually present for that tag
(row counts differ: hitmiss_stim/hitmiss_stim_expert have baseline+sensory
rows, modality_stim(_expert) has sensory only, modality_lick(_expert) has
pre-lick only -- read from each tag's own `load()` output, not assumed
fixed). Columns = R+, R- (the only condition pair currently usable for
every tag is 'half: first vs second' -- `CONDITION_PAIRS` has just the one
entry). All rows share ONE global ylim and ONE canonical area x-axis
(taken from the first tag with data -- all tags cover the same 13 common
areas here, `area_means` NaN-fills gracefully if a later tag were missing
one).

`perfstate_stim_area_group` has no computed decoding results yet (see
`034`'s own run output) so it's excluded here -- nothing to plot for it.

Usage: python 034b_area_condition_megagrid.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("m034", str(OUT_DIR / "034_area_window_quant_grid.py"))
m034 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m034)

TAGS = [
    "hitmiss_stim_area_group",
    "hitmiss_stim_expert_area_group",
    "modality_stim_area_group",
    "modality_stim_expert_area_group",
    "modality_lick_area_group",
    "modality_lick_expert_area_group",
]


def main():
    grid_rows = []  # (tag, metric_prefix, row_label, df, ylim_lo, ylim_hi)
    canonical_areas = None
    colors = None

    for tag in TAGS:
        df, rows, has_shift_null, _null_kind = m034.load(tag)  # 034.load returns 4 values since 2026-09-24 (null_kind)
        if df is None:
            print(f"{tag}: no data -- skipped")
            continue
        area_col = df["area_col"].iloc[0] if "area_col" in df.columns else "area_group"
        areas = m034.common_areas(df, area_col)
        if canonical_areas is None:
            canonical_areas = areas
            colors = m034.get_area_color_map(canonical_areas)
        columns = m034.present_columns(df)
        ylim = m034.compute_shared_ylim(df, areas, columns, rows, has_shift_null)
        for metric_prefix, _, row_label in rows:
            grid_rows.append((tag, metric_prefix, row_label, df, ylim[0], ylim[1]))
        print(f"{tag}: {len(rows)} row(s) -- {[r[2] for r in rows]}")

    if canonical_areas is None or not grid_rows:
        print("nothing to plot")
        return

    ylim = (min(r[4] for r in grid_rows), max(r[5] for r in grid_rows))
    print(f"{len(grid_rows)} total rows, global ylim={ylim}, {len(canonical_areas)} canonical areas")

    val_a, val_b = "first", "second"
    ctype = "half"
    n_cols = 2  # R+, R-
    fig, axes = plt.subplots(len(grid_rows), n_cols, figsize=(3.8 * n_cols, 3.6 * len(grid_rows)), squeeze=False)

    for r, (tag, metric_prefix, row_label, df, _, _) in enumerate(grid_rows):
        for ci, cohort in enumerate(("R+", "R-")):
            ax = axes[r][ci]
            sub = df[(df.reward_group == cohort) & (df.condition_type == ctype)]
            n_parts = []
            top_y = {area: -np.inf for area in canonical_areas}
            for val, off in ((val_a, -0.12), (val_b, 0.12)):
                vsub = sub[sub.condition_value == val]
                mean, sem = m034.area_means(vsub, canonical_areas, metric_prefix)
                diff = mean - 0.5
                for i, area in enumerate(canonical_areas):
                    marker_color = colors[area] if val == val_b else m034.lighten(colors[area])
                    ax.errorbar(i + off, diff[i], yerr=sem[i], fmt="o", color=marker_color,
                                markersize=6, capsize=2, markeredgecolor="black", markeredgewidth=0.5)
                    if not np.isnan(diff[i]):
                        top_y[area] = max(top_y[area], diff[i] + sem[i])
                n_parts.append(f"{val} n={vsub['session_id'].nunique()}")

            f_val, p_val, posthoc_q = m034.factor_anova_and_posthoc(sub, canonical_areas, metric_prefix, "condition_value", val_a, val_b)
            anova_note = f"ANOVA condition: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else "ANOVA condition: n/a"
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(canonical_areas):
                q = posthoc_q.get(area, float("nan"))
                if not np.isnan(q) and q < 0.05 and np.isfinite(top_y[area]):
                    ax.text(i, min(top_y[area] + 0.03 * y_span, ylim[1] - 0.02 * y_span), "*",
                            ha="center", va="bottom", fontsize=11, color="black")

            m034.style_ax(ax, canonical_areas, colors, f"{tag}\n{cohort} | {row_label}\n({', '.join(n_parts)})\n{anova_note}", ylim)

    fig.suptitle("All regular decoding analyses -- across condition (half: first vs second), cohorts side by side\n"
                  "(light marker = first half, full-saturation marker = second half)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # area_group (not whole_brain/area_acronym_custom -- every tag here uses area_group),
    # but at the TOP of that folder rather than learning/ or expert/, since this one figure
    # spans both stages (same "figures/<area level>/..." convention as `034`'s own fig_dir(),
    # just without the stage split that doesn't apply to a cross-stage figure).
    fig_dir = OUT_DIR / "figures" / "area_group"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "034b_area_condition_megagrid.png"
    m034.savefig_retry(fig, out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
