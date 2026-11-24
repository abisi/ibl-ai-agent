"""SuppFigure1_movements -- evoked movements on every trial type and outcome of whisker day 0 (moved out of
SuppFigure1_behaviour, user 2026-11-22: tongue added, lick-aligned traces added, one style for the statistics).
One compute_<key>(ctx) per panel, returning (data frame for the panel CSV, info for the caption).

Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/SuppFigure1_movements/compute.py [--date D]
then build.py figures/SuppFigure1_movements.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import shared_behaviour  # noqa: E402
from common import load_spec, snapshot_dir  # noqa: E402

SPEC = load_spec(FIG_DIR)
_spec = importlib.util.spec_from_file_location("fig1_compute", FIG_DIR.parent / "Figure1_behaviour" / "compute.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

STIM_WINDOW = {"whisker_speed": (-0.1, 0.5), "jaw_opening": (-0.1, 0.5), "tongue_out": (-0.1, 0.5),
               "nose_speed": (-0.1, 0.5), "pupil_area": (-1.0, 3.0), "nose_distance": (-0.1, 0.5),
               "tongue_distance": (-0.1, 0.5)}
LICK_WINDOW = {"whisker_speed": (-0.3, 0.3), "jaw_opening": (-0.3, 0.3), "tongue_out": (-0.3, 0.3),
               "nose_speed": (-0.3, 0.3), "pupil_area": (-2.0, 2.0), "nose_distance": (-0.3, 0.3),
               "tongue_distance": (-0.3, 0.3)}  # shared_movement_peth v12 lick_window_s
MIN_MICE = 5


def _evoked(ctx, signal):
    """Stimulus-aligned `signal` on every trial type x outcome (whisker hit / miss, auditory hit / miss, false alarm /
    correct rejection) and lick-aligned on licked trials (first lick = start_time + computed_reaction_time), learners
    with dual-view video, early licks dropped (shared_movement_peth v11, dlc_helpers); each trial minus its pre-stimulus
    mean, mean over trials per session, then over sessions per mouse. Cluster-based permutation test R+ vs R- per
    alignment x trial type x outcome (>= MIN_MICE mice per cohort)."""
    import shared_movement_peth
    sessions = set(fig1.day0_table(ctx["table"], learners_only=True).session_id)
    peth = shared_movement_peth.load()
    peth = peth[(peth.signal == signal) & peth.session_id.isin(sessions)]
    keep = ((peth["align"] == "stimulus") & peth.time_s.between(*STIM_WINDOW[signal])) | \
           ((peth["align"] == "lick") & peth.time_s.between(*LICK_WINDOW[signal]))
    peth = peth[keep]
    d = (peth.groupby(["mouse_id", "cohort", "signal", "unit", "align", "trial_type", "lick", "time_s"])
         .agg(value=("value_delta", "mean"), n_trials=("n_trials", "sum")).reset_index())
    d["cluster_sig"], d["cluster_p"] = False, np.nan
    clusters, n = {}, {}
    for (al, tt, lick), g in d.groupby(["align", "trial_type", "lick"]):
        nm = g.groupby("cohort").mouse_id.nunique()
        key = f"{al} {tt} lick={lick}"
        n[key] = {c: int(nm.get(c, 0)) for c in ("R+", "R-")}
        if min(n[key].values()) < MIN_MICE:
            continue
        clusters[key] = fig1.cluster_test(g, "time_s", "value")
        d.loc[g.index, "cluster_sig"] = fig1.mark_clusters(g, "time_s", clusters[key])
        d.loc[g.index, "cluster_p"] = fig1.cluster_p(g, "time_s", clusters[key])
    text = "; ".join(f"{k}, {c['start']:.2f} to {c['end']:.2f} s (P = {c['p']:.3f})"
                     for k, cl in clusters.items() for c in cl if c["p"] < 0.05) or "none"
    return d, {"n": fig1.n_by_cohort(d), "n_by_condition": n, "clusters": clusters, "cluster_text": text,
               "stim_window_s": list(STIM_WINDOW[signal]), "lick_window_s": list(LICK_WINDOW[signal]),
               "params": shared_movement_peth.PARAMS}


def compute_evoked_whisker_speed(ctx):
    return _evoked(ctx, "whisker_speed")


def compute_evoked_jaw_opening(ctx):
    return _evoked(ctx, "jaw_opening")


def compute_evoked_tongue_out(ctx):
    return _evoked(ctx, "tongue_out")


def compute_evoked_nose_speed(ctx):
    return _evoked(ctx, "nose_speed")


def compute_evoked_nose_distance(ctx):  # user 2026-11-23
    return _evoked(ctx, "nose_distance")


def compute_evoked_tongue_distance(ctx):  # user 2026-11-23: lick-aligned only, frames with the tongue out, no baseline
    return _evoked(ctx, "tongue_distance")


def compute_evoked_pupil_area(ctx):
    return _evoked(ctx, "pupil_area")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*")
    ap.add_argument("--date")
    a = ap.parse_args()
    ctx = {"table": shared_behaviour.load()}
    out = snapshot_dir(SPEC, a.date)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    for panel in SPEC["panels"]:
        pid, key = panel["id"], panel.get("key", panel["id"])
        fn = globals().get(f"compute_{key}")
        if fn is None or "data" not in panel or (a.panels and pid not in a.panels and key not in a.panels):
            continue
        df, info = fn(ctx)
        df.to_csv(out / "data" / panel["data"], index=False)
        meta[pid] = info
        print(pid, key, panel["data"], len(df), "rows", flush=True)
    meta_file.write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
