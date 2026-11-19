"""Rerun the single-neuron ROC analysis (roc_utils_new.compute_unit_roc) with firing RATES instead of spike counts.

Why: roc_utils_new._get_counts returns baseline-corrected spike COUNTS in each class's window. Several ROC types compare
windows of different lengths (hit / CR / miss post window +5..+35 ms = 30 ms vs spontaneous-lick post window 0..200 ms,
e.g. *_hit_vs_spontaneous, spontaneous_licks_vs_cr; passive pre window 35 ms vs post 30 ms), so counts are biased by
window length. Rates (count / window length, baseline scaled the same way) remove that bias; for same-window
comparisons the AUC is unchanged (a constant scale factor).

Implementation: roc_utils_new._get_counts is wrapped at runtime (the user's code file is not modified); everything else
(baseline correction, windows, permutations, alpha, analysis lists) is identical. Trial lick_time is not used by the ROC
(hits / misses are stimulus-aligned; spontaneous licks come from piezo_lick_times), so the NWB lick_time offset does not
apply here.
Sessions: every day-0 session in the rastermap variant(s) given (default: learners + all mice, good+mua).
Output -> combined_results_ks4/<mouse>/whisker_0/roc_analysis_rates/<mouse>_roc_results_new.csv (+ config.json)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/roc_analysis"))
from roc_analysis import roc_utils_new as ru                           # noqa: E402

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
VARIANTS = ["qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin",
            "qc_good_mua__learncat_all"]
SUBDIR = "roc_analysis_rates"

_orig_get_counts = ru._get_counts


def _duration(event, col):
    if col == "post_spikes":
        return ru.SPONT_POST_DURATION if event == "spontaneous_licks" else ru.POST_DURATION
    if col == "pre_spikes":
        return ru.SPONT_PRE_DURATION if event == "spontaneous_licks" else ru.PRE_DURATION
    if col == "baseline_pre_spikes":
        return ru.BASELINE_DURATION
    return None


def _get_rates(unit_data, event, context, col="post_spikes", apply_baseline=True):
    """same as roc_utils_new._get_counts, divided by the window length -> spikes/s"""
    x = _orig_get_counts(unit_data, event, context, col=col, apply_baseline=apply_baseline)
    d = _duration(event, col)
    return x / d if (d and len(x)) else x


ru._get_counts = _get_rates


def sessions():
    out = set()
    for v in VARIANTS:
        f = sorted((RES / "rastermap_variants" / v).glob("rastermap_clustering/*/*/*/neuron_metadata.csv"))
        if f:
            out |= set(pd.read_csv(f[0], usecols=["session_id"]).session_id)
    return sorted(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--n-workers", type=int, default=48)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    todo = a.sessions or sessions()
    print(f"[024] {len(todo)} sessions -> <mouse>/whisker_0/{SUBDIR}/", flush=True)
    for i, s in enumerate(todo):
        m = s.split("_")[0]
        out = RES / m / "whisker_0" / SUBDIR
        if (out / f"{m}_roc_results_new.csv").exists() and not a.overwrite:
            print(f"[024] {i + 1}/{len(todo)} {s}: exists, skip", flush=True)
            continue
        t0 = time.time()
        try:
            ru.compute_unit_roc(str(NWB / f"{s}.nwb"), str(out), make_psth_plots=False, n_workers=a.n_workers)
            json.dump(dict(session_id=s, measure="rate (spikes/s) = baseline-corrected count / window length",
                           wrapper="024_roc_rates.py wraps roc_utils_new._get_counts",
                           durations_s=dict(post=ru.POST_DURATION, pre=ru.PRE_DURATION,
                                            spont_post=ru.SPONT_POST_DURATION, spont_pre=ru.SPONT_PRE_DURATION,
                                            baseline=ru.BASELINE_DURATION)),
                      open(out / f"{m}_roc_rates_note.json", "w"), indent=2)
            print(f"[024] {i + 1}/{len(todo)} {s}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception as e:
            print(f"[024] {i + 1}/{len(todo)} {s}: FAILED {type(e).__name__}: {e}", flush=True)
    print("ALL DONE", flush=True)
