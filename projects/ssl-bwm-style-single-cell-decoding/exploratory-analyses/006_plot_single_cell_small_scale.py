"""Diagnostic figures for the 3 single-cell tests' small-scale checkpoint
(20-unit subsets, 2 sessions), requested by the user before the full-cohort
confirmatory run: (1) a p-value summary panel across tests x sessions, and
(2) a didactic single-unit illustration of the Modality test's most
significant unit, showing the actual per-trial firing rates the test acted on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssl_bwm_trial_prep import prep_session
from ssl_bwm_windows import unit_rates_for_trials
from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard

HERE = Path(__file__).resolve().parent
DATASET_ROOT = resolve_dataset_dir("ssl_ephys")
EVOKED_WINDOW = (0.005, 0.035)

TESTS = ["modality", "response", "prior_outcome"]
TEST_LABELS = {"modality": "Modality\n(whisker vs auditory)", "response": "Response\n(lick vs no-lick)",
               "prior_outcome": "Prior-outcome\n(t-1 rewarded vs not)"}
SESSION_LABELS = {"AB080_20230622_152205": "AB080 (learning)", "MH021_20250311_110321": "MH021 (expert)"}
SESSION_COLORS = {"AB080_20230622_152205": "#1f77b4", "MH021_20250311_110321": "#d62728"}


def load_results(test_name: str) -> list[dict]:
    with open(HERE / f"{test_name}_small_scale_results.json") as f:
        return json.load(f)


def plot_pvalue_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, test_name in zip(axes, TESTS):
        results = load_results(test_name)
        for i, sess in enumerate(results):
            p = np.array(sess["p_raw"])
            q = np.array(sess["p_fdr"])
            x = i + rng.uniform(-0.12, 0.12, size=len(p))
            color = SESSION_COLORS[sess["session_id"]]
            ax.scatter(x, p, color=color, s=36, alpha=0.85, edgecolor="none",
                       label=SESSION_LABELS[sess["session_id"]] if ax is axes[0] else None)
            sig_fdr = q < 0.05
            if sig_fdr.any():
                ax.scatter(x[sig_fdr], p[sig_fdr], facecolor="none", edgecolor="black", s=90, linewidth=1.3, zorder=5)
            n_raw = int((p < 0.05).sum())
            n_fdr = int(np.nansum(q < 0.05))
            ax.text(i, -0.08, f"n={len(p)}\nraw<.05: {n_raw}\nFDR<.05: {n_fdr}", ha="center", va="top", fontsize=8)
        ax.axhline(0.05, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([])
        ax.set_title(TEST_LABELS[test_name], fontsize=11)
        ax.set_ylim(-0.02, 1.0)
    axes[0].set_ylabel("raw p-value (per unit)")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Single-cell test small-scale checkpoint: 20-unit subsets, 2 sessions\n"
                  "black-ringed = FDR<0.05; dashed line = raw p=0.05", fontsize=10)
    fig.subplots_adjust(bottom=0.26, top=0.82, wspace=0.15)
    out_path = HERE / "006_pvalue_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_example_modality_unit() -> None:
    results = load_results("modality")
    best = None
    for sess in results:
        p = np.array(sess["p_raw"])
        idx = int(np.argmin(p))
        if best is None or p[idx] < best["p"]:
            best = {"session_id": sess["session_id"], "day_stage": sess["day_stage"],
                     "cluster_id": sess["cluster_id"][idx], "p": float(p[idx])}
    print(f"Most significant Modality-test unit in the small-scale subset: "
          f"session={best['session_id']} cluster_id={best['cluster_id']} p={best['p']:.4f}")

    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    prepped = prep_session(DATASET_ROOT, best["session_id"], sessions_tbl, trials_tbl)
    candidate = prepped["trials"][prepped["trials"]["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
    is_whisker = (candidate["trial_type"] == "whisker_trial").to_numpy()
    response = candidate["lick_flag"].to_numpy().astype(bool)
    t1_rewarded = candidate["t1_rewarded"].to_numpy().astype(bool)
    start_time = candidate["start_time"].to_numpy()

    shard = load_spike_shard(DATASET_ROOT / "spikes" / best["session_id"])
    dense_idx = np.where(shard["cluster_ids"] == best["cluster_id"])[0]
    unit_spikes = np.sort(shard["spike_times_seconds"][shard["spike_clusters"] == dense_idx[0]])
    rates = unit_rates_for_trials(unit_spikes, start_time, is_whisker, EVOKED_WINDOW)
    valid = ~np.isnan(rates)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    strata_labels = []
    positions = []
    pos = 0
    rng = np.random.default_rng(1)
    for resp_val in (False, True):
        for t1_val in (False, True):
            m = valid & (response == resp_val) & (t1_rewarded == t1_val)
            for is_wh, color, tag in ((True, "#8c564b", "whisker"), (False, "#2ca02c", "auditory")):
                vals = rates[m & (is_whisker == is_wh)]
                if len(vals) == 0:
                    pos += 1
                    continue
                x = pos + rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(x, vals, color=color, alpha=0.7, s=22,
                           label=tag if pos == 0 else None)
                ax.scatter([pos], [np.mean(vals)], color="black", marker="_", s=300, linewidth=2, zorder=5)
                pos += 1
            strata_labels.append(f"resp={int(resp_val)}\nt-1 rew={int(t1_val)}")
            pos += 0.6

    tick_positions = [1.5 + i * 2.6 for i in range(4)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(strata_labels, fontsize=8)
    ax.set_ylabel(f"firing rate in evoked window [{EVOKED_WINDOW[0]*1000:.0f},{EVOKED_WINDOW[1]*1000:.0f}]ms (Hz)")
    ax.set_title(f"Most significant Modality-test unit (small-scale subset)\n"
                 f"{SESSION_LABELS[best['session_id']]}, cluster_id={best['cluster_id']}, p={best['p']:.4f}\n"
                 f"brown=whisker, green=auditory, black dash=stratum mean", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path = HERE / "006_example_modality_unit.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot_pvalue_summary()
    plot_example_modality_unit()
