"""Initial style check (per user request 2026-09-11) for the new
comprehensive figure spec, using CURRENT data only (no rerun): hit/miss
(lick_flag) decoding, start_time-aligned, one area (Motor and frontal
areas -- best session coverage in `002_pilot_results_partial.parquet`),
session-half comparison ("condition" factor).

New style elements being validated here, ahead of the full pipeline
rebuild:
  - square subplots
  - 3-subplot layout for a "compare conditions" figure: R+, R-, and R+/R-
    aggregated (pooled across cohort)
  - n mice / n sessions annotated per subplot
  - cohort colors (#00B400 / #C800C8) kept; condition (half) encoded by
    linestyle, as in the existing 014/018 scripts
  - comparable y-axis (0.4-1.0) across figures

Does NOT yet include: cross-condition generalization, sensory/pre-lick
window quantification+stats, behavioral correlation, or the rebuilt
allen_utils area-group scheme (unit_area_labels.parquet still has the old
groups -- 'Somatosensory areas' not yet split into whisker/orofacial/body).
Those are part of the full rebuild, held pending user sign-off on this
style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
PARTIAL_PATH = OUT_DIR / "002_pilot_results_partial.parquet"
BIN_EDGES_PATH = OUT_DIR / "002_bin_edges.json"
AREA_VALUE = "Motor and frontal areas"

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
AGGREGATE_COLOR = "#2c5f5b"


def summarize(sub: pd.DataFrame) -> str:
    return f"n={sub['session_id'].nunique()} sessions, {sub['subject_id'].nunique()} mice"


def main():
    df = pd.read_parquet(PARTIAL_PATH)
    df = df[df["skipped_reason"].isna() & (df["area_col"] == "area_group") & (df["area_value"] == AREA_VALUE)].copy()

    bin_edges = json.loads(BIN_EDGES_PATH.read_text())
    bin_centers_ms = np.array([(b[0] + b[1]) / 2 * 1000 for b in bin_edges])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [("R+", df[df.reward_group == "R+"]), ("R-", df[df.reward_group == "R-"]), ("R+ & R- aggregated", df)]

    for ax, (label, sub) in zip(axes, panels):
        color = COHORT_COLOR.get(label, AGGREGATE_COLOR)
        for half, ls in (("first", "-"), ("second", "--")):
            half_sub = sub[sub["half"] == half]
            curves = [np.array(c) for c in half_sub["real_curve"]]
            if len(curves) == 0:
                continue
            stacked = np.stack(curves)
            mean_curve = np.nanmean(stacked, axis=0)
            sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
            ax.plot(bin_centers_ms, mean_curve, color=color, lw=2.2, linestyle=ls,
                     label=f"{'first' if half == 'first' else 'second'} half")
            ax.fill_between(bin_centers_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18, lw=0)

        ax.axhline(0.5, color="#888888", lw=1.2, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_title(f"{label}\n({summarize(sub)})", fontsize=13)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(bin_centers_ms.min(), bin_centers_ms.max())
        ax.set_box_aspect(1)
        ax.set_xlabel("time from start_time (ms)", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, frameon=False, loc="upper left")

    fig.suptitle(
        f"Hit vs miss (lick_flag) decoding, start_time-aligned, {AREA_VALUE}\n"
        "session-half comparison -- INITIAL STYLE CHECK (current data, old area scheme, no cross-gen/stats yet)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = OUT_DIR / "023_initial_check_hitmiss_halves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
