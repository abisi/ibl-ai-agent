"""Test figure only (user request 2026-09-13): "just to see" whether a
reaction-time histogram (corrected reaction_time since 2026-09-27, hit trials) sharing the
x-axis with a stim-aligned decode curve is a useful way to show when mice
typically respond relative to the decoding time course. Not wired into any
pipeline -- one-off preview for one tag/condition.

Uses the pre-shift-null `hitmiss_stim_whole_brain` backup (complete, 89
sessions) rather than the live in-progress rerun, purely because it's
already fully computed and real_curve itself is unaffected by the
shift-null addition -- this is just a layout preview, not a result.

Usage: python 035_test_rt_histogram_shared_axis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import add_first_lick_time, hitmiss_session_list, prep_hitmiss_trials  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TAG = "hitmiss_stim_whole_brain"
BACKUP_SUFFIX = ".bak-preshiftnull-20260913"
STIM_WINDOW = (-0.2, 0.6)


def main():
    partial_path = OUT_DIR / f"024_master_results_{TAG}.parquet{BACKUP_SUFFIX}"
    edges_path = OUT_DIR / f"024_bin_edges_{TAG}.json{BACKUP_SUFFIX}"
    df = pd.read_parquet(partial_path)
    df = df[(df["skipped_reason"].isna()) & (df["condition_type"] == "whole")]
    bin_edges = json.loads(edges_path.read_text())
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])

    curves = np.stack([np.array(c) for c in df["real_curve"]])
    mean_curve = np.nanmean(curves, axis=0)
    sem_curve = np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0])
    session_ids = df["session_id"].unique().tolist()
    print(f"decode curve: {len(session_ids)} sessions, {curves.shape[1]} bins")

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    rt_ms = []
    for session_id in session_ids:
        trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        if trials is None:
            continue
        hits = add_first_lick_time(trials[trials["lick_flag"] == 1])  # corrected first lick (2026-09-27)
        rt = hits["reaction_time"].to_numpy() * 1000
        rt_ms.append(rt)
    rt_ms = np.concatenate(rt_ms)
    rt_ms = rt_ms[(rt_ms >= STIM_WINDOW[0] * 1000) & (rt_ms <= STIM_WINDOW[1] * 1000)]
    print(f"RT histogram: {len(rt_ms)} hit-trial reaction times pooled across {len(session_ids)} sessions")

    ylim = (0.4, 1.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax_hist = ax.twinx()

    # Histogram drawn first, on its own independent y-axis, scaled so its
    # tallest bar sits just under the 0.5 chance line on the curve's axis
    # (not sharing that axis's scale -- purely a visual floor) -- user
    # request 2026-09-13. `ax` stays on top (transparent patch) so the curve
    # draws over the histogram rather than being occluded by it.
    counts, bin_edges_h = np.histogram(rt_ms, bins=40, range=(STIM_WINDOW[0] * 1000, STIM_WINDOW[1] * 1000))
    frac_at_chance = (0.5 - ylim[0]) / (ylim[1] - ylim[0])
    ax_hist.set_ylim(0, counts.max() / (frac_at_chance * 0.85))
    ax_hist.bar(bin_edges_h[:-1], counts, width=np.diff(bin_edges_h), align="edge",
                color="#888888", alpha=0.35, edgecolor="#333333", linewidth=1.5)
    ax_hist.set_yticks([])
    ax_hist.set_ylabel("hit trials (count)", color="#555555")
    ax_hist.spines[["top", "left"]].set_visible(False)

    ax.set_zorder(ax_hist.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(bin_labels_ms, mean_curve, color="#2c5f5b", lw=2.0)
    ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color="#2c5f5b", alpha=0.18, lw=0)
    ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("time from start_time (ms)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"{TAG} -- whole session, R+ & R- aggregated (n={len(session_ids)} sessions)\nhit-trial reaction-time distribution overlaid below chance", fontsize=10)
    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "035_test_rt_histogram_shared_axis.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
