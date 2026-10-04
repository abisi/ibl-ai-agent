"""Pseudo-population whole-brain hit/miss decoding, within cohort, whole-
session only (user request 2026-09-13: "try a whole-brain pseudo-population
decoding run (with enough subsampling), within cohort, of the hit/miss
condition (whole_session only)").

Instead of decoding within one session's own units (like `024_master_sweep.py`),
this pools units across every eligible learning-stage whole-brain session
within one cohort (R+ or R- separately -- pooling across cohorts would beg
the question, since R+/R- differ in exactly the condition being decoded) into
one large pseudo-population, and builds pseudo-trials by independently
bootstrap-resampling (with replacement) one real hit/miss trial per session
for each pseudo-trial slot, reusing the SAME sampled trial index across every
time bin within a draw (so a pseudo-trial's own within-trial temporal
structure, to the extent shared trial-level state exists, stays intact for
that session's units). See the assumptions discussed with the user this
session: no true trial-order alignment across sessions (matching is by
condition label only), cross-session noise correlations are destroyed by
construction (only within-session units keep real trial-by-trial covariance),
and cross-session/animal stationarity of the hit/miss-locked response is
assumed.

First-pass / exploratory configuration (deliberately lighter than
024_master_sweep.py's per-session sweep, to get a fast read on "does pooling
help" before investing more compute): coarser 20ms-stride bin grid (same
-200/+600ms window, so still directly comparable on the x-axis to the
existing per-session whole-brain hit/miss curves), no label-shuffle null this
first pass (compare directly against the 0.5 chance line), single-threaded
(the hitmiss/modality/perfstate area_group sweeps are already saturating
CPU this session).

Usage: python 033_pseudopopulation_hitmiss_wholebrain.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import (  # noqa: E402
    AREA_LABELS_PATH,
    add_whole_brain_column,
    area_units,
    causal_bin_edges,
    data_sufficiency_ok,
    decode_curve,
    hitmiss_session_list,
    load_session_unit_spikes,
    prep_hitmiss_trials,
    select_fixed_c,
    sliding_bin_population_matrices,
    wide_window_matrix_from_bins,
)

OUT_DIR = Path(__file__).resolve().parent
STIM_WINDOW = (-0.2, 0.6)
BIN_WIDTH = 0.05
STRIDE = 0.02  # coarser than 024's 0.005 -- first-pass cost containment
WIDE_DEAD_ZONE = (-0.010, 0.005)
N_PSEUDO_DRAWS = 15
N_PSEUDO_PER_CLASS = 30
INNER_N_REPEATS = 2
INNER_N_FOLDS = 5
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def load_cohort_sessions(dataset_root, sessions_tbl, trials_tbl, area_labels, cohort: str, bin_edges) -> list[dict]:
    hm = hitmiss_session_list(sessions_tbl)
    hm = hm[(hm.day_stage == "learning") & (hm.reward_group == cohort)]
    pooled = []
    for r in hm.itertuples():
        trials = prep_hitmiss_trials(dataset_root, r.session_id, sessions_tbl, trials_tbl)
        if trials is None or len(trials) == 0:
            continue
        unit_ids = area_units(r.session_id, "whole_brain", "All units", area_labels)
        y = trials["lick_flag"].to_numpy().astype(bool)
        ok, reason = data_sufficiency_ok(len(unit_ids), y, min_trials_per_class=N_PSEUDO_PER_CLASS // 4)
        if not ok:
            continue
        unit_spikes = load_session_unit_spikes(dataset_root, r.session_id)
        is_whisker = np.ones(len(trials), dtype=bool)
        start_time = trials["start_time"].to_numpy()
        matrices = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=WIDE_DEAD_ZONE)
        pooled.append(dict(
            session_id=r.session_id, subject_id=r.subject_id, n_units=len(unit_ids),
            hit_idx=np.where(y)[0], miss_idx=np.where(~y)[0], matrices=matrices,
        ))
    return pooled


def run_cohort(cohort: str, pooled: list[dict], n_bins: int, rng: np.random.Generator) -> np.ndarray:
    total_units = sum(p["n_units"] for p in pooled)
    real_curves = []
    for draw in range(N_PSEUDO_DRAWS):
        t0 = time.time()
        draw_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        # Sample pseudo-trial indices once per session/class -- reused across
        # every bin below so a pseudo-trial's per-session real trial choice
        # (and whatever within-trial temporal structure it carries) is fixed
        # for the whole draw, not resampled bin-by-bin.
        sampled = [
            (draw_rng.choice(p["hit_idx"], size=N_PSEUDO_PER_CLASS, replace=True),
             draw_rng.choice(p["miss_idx"], size=N_PSEUDO_PER_CLASS, replace=True))
            for p in pooled
        ]
        y_pseudo = np.array([True] * N_PSEUDO_PER_CLASS + [False] * N_PSEUDO_PER_CLASS)
        pseudo_matrices = []
        for b in range(n_bins):
            X = np.empty((2 * N_PSEUDO_PER_CLASS, total_units))
            col = 0
            for p, (hit_samples, miss_samples) in zip(pooled, sampled):
                nu = p["n_units"]
                X[:N_PSEUDO_PER_CLASS, col:col + nu] = p["matrices"][b][hit_samples, :]
                X[N_PSEUDO_PER_CLASS:, col:col + nu] = p["matrices"][b][miss_samples, :]
                col += nu
            pseudo_matrices.append(X)
        X_wide = wide_window_matrix_from_bins(pseudo_matrices)
        C = select_fixed_c(X_wide, y_pseudo, draw_rng)
        curve = decode_curve(pseudo_matrices, y_pseudo, C, draw_rng, n_repeats=INNER_N_REPEATS, n_folds=INNER_N_FOLDS)
        real_curves.append(curve)
        print(f"  [{cohort}] draw {draw + 1}/{N_PSEUDO_DRAWS}: C={C:.4g}, peak_acc={np.nanmax(curve):.3f}, "
              f"{time.time() - t0:.1f}s", flush=True)
    return np.stack(real_curves)


def plot_panel(ax, bin_labels_ms, mean_curve, sem_curve, color, title, label=None):
    ax.plot(bin_labels_ms, mean_curve, color=color, lw=2.0, label=label)
    ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.16, lw=0)
    ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
    ax.set_ylim(0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped
    ax.set_box_aspect(1)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("time from start_time (ms)", fontsize=8)
    ax.set_ylabel("balanced accuracy", fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    bin_edges = causal_bin_edges(STIM_WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
    (OUT_DIR / "033_bin_edges.json").write_text(json.dumps(bin_edges))
    print(f"bin grid: {len(bin_edges)} causal bins, window {STIM_WINDOW}, stride {STRIDE}", flush=True)

    results = {}
    for cohort in ("R+", "R-"):
        print(f"[{cohort}] loading + preparing pooled sessions...", flush=True)
        pooled = load_cohort_sessions(dataset_root, sessions_tbl, trials_tbl, area_labels, cohort, bin_edges)
        total_units = sum(p["n_units"] for p in pooled)
        print(f"[{cohort}] {len(pooled)} sessions pooled, {total_units} total units", flush=True)
        rng = np.random.default_rng(abs(hash(cohort)) % (2**31))
        curves = run_cohort(cohort, pooled, len(bin_edges), rng)
        results[cohort] = dict(curves=curves, n_sessions=len(pooled), n_units=total_units,
                                session_ids=[p["session_id"] for p in pooled])

        tag = cohort.replace("+", "plus").replace("-", "minus")
        pd.DataFrame({"draw": np.arange(len(curves)), "curve": [c.tolist() for c in curves]}).assign(
            cohort=cohort, n_sessions=len(pooled), n_units=total_units,
            session_ids=json.dumps([p["session_id"] for p in pooled]),
        ).to_parquet(OUT_DIR / f"033_pseudopop_hitmiss_whole_wholebrain_{tag}.parquet", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, cohort in zip(axes[:2], ("R+", "R-")):
        r = results[cohort]
        mean_curve = np.nanmean(r["curves"], axis=0)
        sem_curve = np.nanstd(r["curves"], axis=0) / np.sqrt(r["curves"].shape[0])
        plot_panel(ax, bin_labels_ms, mean_curve, sem_curve, COHORT_COLOR[cohort],
                   f"{cohort} pseudo-population\n(n={r['n_sessions']} sessions, {r['n_units']} units, {N_PSEUDO_DRAWS} draws)")

    ax = axes[2]
    for cohort in ("R+", "R-"):
        r = results[cohort]
        mean_curve = np.nanmean(r["curves"], axis=0)
        sem_curve = np.nanstd(r["curves"], axis=0) / np.sqrt(r["curves"].shape[0])
        plot_panel(ax, bin_labels_ms, mean_curve, sem_curve, COHORT_COLOR[cohort], "R+ & R- overlaid", label=cohort)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("Whole-brain pseudo-population hit/miss decoding, within cohort, whole session", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT_DIR / "033_pseudopop_hitmiss_whole_wholebrain.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
