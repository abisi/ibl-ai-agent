"""Neuron-count control for the time-resolved hitmiss decoding pipeline
(user request 2026-09-23: "build a control for the number of neurons per
recording per area ... time-resolved accuracy curves with increasing
number of neurons, e.g. in steps of 100"; scoped to `whole_brain` only,
example-session pilot first -- user confirmed via AskUserQuestion).

For each of 5 example sessions (`EXAMPLE_MICE`, same picks as `068`,
learning-stage, `whole_brain` scheme), subsamples the unit pool to
N = 100, 200, ..., up to `N_CAP` (or the session's own unit count if
smaller), plus one extra point at the session's TRUE full unit count if
that's above `N_CAP` -- so the grid stays bounded (at most 10 steps + 1
endpoint) even for the largest sessions (AB158 has 2704 whole_brain
units; a dense 100-step grid all the way there would be ~27x the cost for
a pilot). `N_REPEATS_SUBSAMPLE` independent random draws per N are
averaged, to damp which-specific-units-got-picked noise.

Reuses `024_master_sweep.py`'s exact hitmiss/stim/whole_brain recipe
(causal 50ms bins, 5ms stride, `WIDE_DEAD_ZONE`, `select_fixed_c_pooled`/
`decode_curve_pooled`) -- same bin grid and estimator, so curve SHAPE is
directly comparable to the production `hitmiss_stim_whole_brain` figures.

**Deliberately real-accuracy only, no null shuffles** -- this is a control
for how accuracy scales with N, not a re-run of the significance test at
every N (that would multiply cost by `N_SHUF`, ~25x, for a question this
control doesn't need to answer). `decode_curve_pooled`'s own `n_repeats`
is also reduced from production's 5 to 3 for the same cost reason -- pilot
numbers, not production ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]  # same picks as 068
STIM_WINDOW = (-0.2, 0.6)
BIN_WIDTH = 0.05
STRIDE = 0.005
WIDE_DEAD_ZONE = (-0.010, 0.005)
N_STEP = 100
N_CAP = 1000  # dense grid stops here; one extra point at the session's true full count beyond this
N_REPEATS_SUBSAMPLE = 3  # independent unit draws per N, averaged
N_REPEATS_CURVE = 3  # decode_curve_pooled's own repeat count (production uses 5; reduced for this pilot)


def process_one_session(args: tuple) -> dict:
    mouse, session_id, reward_group, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, causal_bin_edges, decode_curve_pooled,
        load_session_unit_spikes, prep_hitmiss_trials, select_fixed_c_pooled, sliding_bin_population_matrices,
        wide_window_matrix_from_bins,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    bin_edges = causal_bin_edges(STIM_WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])

    trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
    if trials is None or len(trials) == 0:
        return dict(mouse=mouse, session_id=session_id, ok=False)
    y = trials["lick_flag"].to_numpy().astype(bool)
    is_whisker = np.ones(len(trials), dtype=bool)
    start_time = trials["start_time"].to_numpy()

    all_unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    n_total = len(all_unit_ids)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)

    n_grid = list(range(N_STEP, min(n_total, N_CAP) + 1, N_STEP))
    if n_total > N_CAP:
        n_grid.append(n_total)
    elif n_total not in n_grid:
        n_grid.append(n_total)

    curves_by_n = {}
    for n in n_grid:
        draw_curves = []
        for draw in range(N_REPEATS_SUBSAMPLE):
            rng = np.random.default_rng(abs(hash((session_id, n, draw))) % (2**31))
            unit_ids = rng.choice(all_unit_ids, size=n, replace=False) if n < n_total else all_unit_ids
            matrices = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=WIDE_DEAD_ZONE)
            X_wide = wide_window_matrix_from_bins(matrices)
            C = select_fixed_c_pooled(X_wide, y, rng)
            real = decode_curve_pooled(matrices, y, C, rng, n_repeats=N_REPEATS_CURVE)
            draw_curves.append(real)
        curves_by_n[n] = np.mean(draw_curves, axis=0)
        print(f"{mouse}/{session_id} N={n}: peak_acc={curves_by_n[n].max():.3f}", flush=True)

    return dict(mouse=mouse, session_id=session_id, reward_group=reward_group, ok=True,
                n_total=n_total, bin_labels_ms=bin_labels_ms, curves_by_n=curves_by_n)


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    learning = hm[hm["day_stage"] == "learning"]

    tasks = []
    for mouse in EXAMPLE_MICE:
        rows = learning[learning.subject_id == mouse]
        if len(rows) == 0:
            print(f"{mouse}: no learning-stage session, skipped")
            continue
        r = rows.iloc[0]
        tasks.append((mouse, r["session_id"], r.get("reward_group", "?"), SCRIPTS_DIR))
    print(f"{len(tasks)} example sessions", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), 10)) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("ok"):
                results.append(res)
    results.sort(key=lambda r: EXAMPLE_MICE.index(r["mouse"]))
    print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)

    # --- Figure 1: one panel per session, accuracy(t) curves color-graded by N ---
    fig1, axes1 = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2), constrained_layout=True, squeeze=False)
    axes1 = axes1[0]
    cmap = plt.get_cmap("viridis")
    for ax, res in zip(axes1, results):
        ns = sorted(res["curves_by_n"].keys())
        for i, n in enumerate(ns):
            color = cmap(i / max(1, len(ns) - 1))
            label = f"N={n}" + (" (full)" if n == res["n_total"] else "")
            ax.plot(res["bin_labels_ms"], res["curves_by_n"][n], color=color, lw=1.3, label=label)
        ax.axhline(0.5, color="#888888", lw=0.8, linestyle=":")
        ax.axvspan(-10, 5, color="#dddddd", alpha=0.5, zorder=0)  # dead zone
        ax.set_title(f"{res['mouse']} [{res['reward_group']}]\nn_total={res['n_total']}", fontsize=9.5)
        ax.set_xlabel("time from stim onset (ms)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    axes1[0].set_ylabel("balanced accuracy\n(hit vs miss)", fontsize=9)
    fig1.suptitle("Time-resolved hitmiss accuracy vs N units (whole_brain, real accuracy only, no null)", fontsize=12)
    fig1_path = OUT_DIR / "073_hitmiss_timeresolved_nsubsample_curves.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: summary -- peak accuracy vs N, one line per session ---
    fig2, ax2 = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for res in results:
        ns = sorted(res["curves_by_n"].keys())
        peaks = [res["curves_by_n"][n].max() for n in ns]
        ax2.plot(ns, peaks, marker="o", markersize=4, lw=1.3, label=f"{res['mouse']} (n_total={res['n_total']})")
    ax2.axhline(0.5, color="#888888", lw=0.8, linestyle=":")
    ax2.set_xlabel("N units subsampled", fontsize=9)
    ax2.set_ylabel("peak balanced accuracy over the time-resolved curve", fontsize=9)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("Peak accuracy vs N units, whole_brain, 5 example sessions", fontsize=11)
    fig2_path = OUT_DIR / "073_hitmiss_timeresolved_nsubsample_peakaccuracy.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")
    print("DONE_073")


if __name__ == "__main__":
    main()
