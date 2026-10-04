"""Compare `033_pseudopopulation_hitmiss_wholebrain.py`'s pooled
cross-session pseudo-population decoding curves against the average of the
per-session `hitmiss_stim_whole_brain` single-session decode curves (user
request 2026-09-13). Same target/alignment/area-scheme (hitmiss, stim,
whole_brain), different unit of analysis: one big pooled population vs. the
mean of many independent per-session populations.

Uses the pre-shift-null `hitmiss_stim_whole_brain` backup (complete, 89
sessions) for the single-session average, for the same reason as `035`/
`037`: it's already complete and real_curve itself doesn't depend on the
shift-null addition.

Caveat surfaced, not hidden: every single pseudopop draw in `033`'s own log
hit `peak_acc=1.000` at `C=1e5` (the top of `C_GRID`, i.e. essentially no
regularization) -- a strong overfitting signature given `033`'s own design
(total_units pooled across ~50-65 sessions, likely in the thousands, against
only `2*N_PSEUDO_PER_CLASS=60` pseudo-trials per draw -- a severe n<<p
regime) and its own documented lack of a label-shuffle null for this first
pass. Read the pseudopop curve's absolute level with that in mind; this
script plots what `033` produced, it doesn't correct for it.

Usage: python 038_pseudopop_vs_single_session_comparison.py
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
FIG_DIR = OUT_DIR / "figures" / "behavior"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_SESSION_TAG = "hitmiss_stim_whole_brain"
BACKUP_SUFFIX = ".bak-preshiftnull-20260913"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def load_pseudopop(cohort: str):
    tag = cohort.replace("+", "plus").replace("-", "minus")
    path = OUT_DIR / f"033_pseudopop_hitmiss_whole_wholebrain_{tag}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    curves = np.stack([np.array(c) for c in df["curve"]])
    bin_edges = json.loads((OUT_DIR / "033_bin_edges.json").read_text())
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
    return dict(bin_labels_ms=bin_labels_ms, mean=np.nanmean(curves, axis=0),
                sem=np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0]),
                n_draws=curves.shape[0], n_sessions=int(df["n_sessions"].iloc[0]),
                n_units=int(df["n_units"].iloc[0]))


def load_single_session_avg(cohort: str):
    df = pd.read_parquet(OUT_DIR / f"024_master_results_{SINGLE_SESSION_TAG}.parquet{BACKUP_SUFFIX}")
    df = df[(df["skipped_reason"].isna()) & (df["condition_type"] == "whole") & (df["reward_group"] == cohort)]
    bin_edges = json.loads((OUT_DIR / f"024_bin_edges_{SINGLE_SESSION_TAG}.json{BACKUP_SUFFIX}").read_text())
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
    curves = np.stack([np.array(c) for c in df["real_curve"]])
    return dict(bin_labels_ms=bin_labels_ms, mean=np.nanmean(curves, axis=0),
                sem=np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0]),
                n_sessions=df["session_id"].nunique())


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, cohort in zip(axes, ("R+", "R-")):
        color = COHORT_COLOR[cohort]
        pseudo = load_pseudopop(cohort)
        single = load_single_session_avg(cohort)

        if pseudo is not None:
            ax.plot(pseudo["bin_labels_ms"], pseudo["mean"], color=color, lw=2.2, linestyle="-",
                     label=f"pseudo-pop (n={pseudo['n_sessions']} sessions pooled, "
                           f"{pseudo['n_units']} units, {pseudo['n_draws']} draws)")
            ax.fill_between(pseudo["bin_labels_ms"], pseudo["mean"] - pseudo["sem"], pseudo["mean"] + pseudo["sem"],
                             color=color, alpha=0.18, lw=0)
        else:
            ax.text(0.5, 0.6, "pseudo-pop: not finished yet", transform=ax.transAxes, ha="center", fontsize=9, color="#888888")

        ax.plot(single["bin_labels_ms"], single["mean"], color=color, lw=2.0, linestyle="--",
                 label=f"mean of single-session curves (n={single['n_sessions']} sessions)")
        ax.fill_between(single["bin_labels_ms"], single["mean"] - single["sem"], single["mean"] + single["sem"],
                         color=color, alpha=0.12, lw=0)

        ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_ylim(0.4, 1.05)
        ax.set_box_aspect(1)
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("balanced accuracy")
        ax.set_title(cohort, fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("hitmiss_stim_whole_brain -- pseudo-population vs. mean single-session decoding\n"
                  "(caveat: every pseudopop draw hit peak_acc=1.000 at C=1e5 -- likely n<<p overfitting, no null control yet)",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = FIG_DIR / "038_pseudopop_vs_single_session_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
