"""Day-0 exact learning curves with the paper's session-end cut (user 2026-11-23: "are these using the entire sessions or
removing the last 10%?"). Trials from the stored exact curve files (combined_results_ks4/<mouse>/whisker_0/learning_curve/
<mouse>_whisker_0_<type>_learning_curve_exact.h5: not passive, perf != 6, early_lick == 0). Session end (as Figure 1
compute.session_end_time): first whisker trial + keep x (last - first whisker trial); trials of every type after it are
dropped and the curves REFITTED on the cut session (grid HMM at sigma = 1, the paper model; lt_lib of
ssl-learning-trial-identification, identical to the stored curves when keep = 1). keep = 0.85 is the paper's rule (last 15 %
cut, decided 2026-11-17).

curves(mouse, keep) -> dict(outcomes, start, p_mean, p_low (10 %), p_high (90 %), p_chance (no-stim posterior mixed in time
at the whisker trials), p_auditory (same for auditory), n_cut) or None.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "projects" / "ssl-learning-trial-identification" / "exploratory-analyses"))
sys.path.insert(0, str(REPO / "scripts"))
import lt_lib as LT  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root() / "combined_results_ks4"


def _load(mouse, tt):
    f = ROOT / mouse / "whisker_0" / "learning_curve" / f"{mouse}_whisker_0_{tt}_learning_curve_exact.h5"
    return pd.read_hdf(f).iloc[0] if f.exists() else None


def _fit(y):
    g, _ = LT.forward_backward(np.asarray(y).astype(int), 1.0)
    return g


def curves(mouse, keep=0.85):
    w = _load(mouse, "whisker_trial")
    if w is None:
        return None
    tw = np.asarray(w.start_time, float)
    end = tw.min() + keep * (tw.max() - tw.min())
    out = {}
    for tt, k in (("whisker_trial", "w"), ("no_stim_trial", "n"), ("auditory_trial", "a")):
        e = w if tt == "whisker_trial" else _load(mouse, tt)
        if e is None:
            out[k] = None
            continue
        t = np.asarray(e.start_time, float)
        m = t <= end + 1e-9
        out[k] = (np.asarray(e.outcomes).astype(int)[m], t[m])
    yw, sw = out["w"]
    gw = _fit(yw)
    lo, hi = LT.quantiles(gw, [0.1, 0.9])
    res = dict(outcomes=yw, start=sw, p_mean=gw @ LT.P_GRID, p_low=lo, p_high=hi, n_cut=int(len(tw) - len(yw)))
    for k, name in (("n", "p_chance"), ("a", "p_auditory")):
        if out[k] is None or len(out[k][0]) < 2:
            res[name] = np.full(len(yw), np.nan)
        else:
            g = _fit(out[k][0])
            res[name] = LT.interp_marginals(g, out[k][1], sw) @ LT.P_GRID
    return res
