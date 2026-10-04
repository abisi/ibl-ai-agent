"""107 -- "Decodability learning curves": how hit/miss decodability evolves trial by trial, outside the
split/placebo framework (user request 2026-09-25: "Is there a simpler way to see how hit/miss decoding changes
as trials go?" -> "yes").

Per session (whole brain; all learning-stage whisker trials, disengaged trials kept, no gates):
 A. One decoder for the whole session: StandardScaler + L2 logistic, class_weight='balanced', C chosen once
    (select_fixed_c_pooled); N_REP x stratified 5-fold CV -> every trial gets held-out predictions; per trial
    'correct' = held-out prediction equals the true class in the majority of repeats (0/1).
    The 0/1 correctness sequence is smoothed with the SAME learning-curve model as behaviour (random walk on
    logit p, sigma = 1, exact grid posterior, lt_lib), separately for hit trials and miss trials (each on its
    own trial positions), both interpolated to all trial positions and averaged -> class-balanced P(correct)
    curve with 80% band (average of the two bands, approximate). Chance = 0.5.
 B. Drift control: the same, after a linear (non-wrapping) shift of the hit/miss labels by a random 10-50%
    of the session -> shift-null curve.
 C. Sliding-window decoder: windows of WIN trials, step STEP; retrained in each window; stratified CV with
    n_folds = min(5, minority count); windows with < 3 hits or < 3 misses left empty; balanced accuracy at the
    window centre.
Behaviour (sigma = 1 whisker + FA curves, cohort-specific learning trial 020) plotted above on the same
trial axis. Windows: sensory (5-50 ms) and sensory minus baseline.
Outputs: 107_decodability_curves.pkl, figures/whole_brain/learning/107_decodability_curves_<window>.png
"""

from __future__ import annotations

import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS)
OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
sys.path.insert(0, str(LTP / "exploratory-analyses"))
EXAMPLES = ["AB119", "AB125", "MH029", "AB085", "MH018", "AB120", "MH030", "AB159"]
WINDOWS = {"sensory": (0.005, 0.050), "baseline": (-0.200, -0.010)}
WIN_USE = ["sensory", "sensory_minus_base"]
N_REP, SIGMA, WIN, STEP = 20, 1.0, 30, 5
DZ = (-0.010, 0.005)


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def clf(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear", C=C, class_weight="balanced",
                                                              max_iter=1000))


def heldout_correct(X, y, C, rng):
    from sklearn.model_selection import StratifiedKFold
    nf = min(5, int(y.sum()), int((~y).sum()))
    votes = np.zeros(len(y))
    for _ in range(N_REP):
        for tr, te in StratifiedKFold(nf, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, y):
            votes[te] += clf(C).fit(X[tr], y[tr]).predict(X[te]) == y[te]
    return (votes / N_REP > 0.5).astype(int), votes / N_REP


def smooth_balanced(correct, y):
    import lt_lib as L
    n = len(y)
    out = {}
    for cls, m in (("hit", y), ("miss", ~y)):
        idx = np.where(m)[0]
        g, _ = L.forward_backward(correct[idx], SIGMA)
        lo, hi = L.quantiles(g, [0.1, 0.9])
        mean = g @ L.P_GRID
        out[cls] = tuple(np.interp(np.arange(n), idx, v) for v in (mean, lo, hi))
    bal = tuple((a + b) / 2 for a, b in zip(out["hit"], out["miss"]))
    return bal, out


def sliding(X, y, C, rng):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    cen, acc = [], []
    for s in range(0, len(y) - WIN + 1, STEP):
        Xw, yw = X[s:s + WIN], y[s:s + WIN]
        cen.append(s + WIN / 2)
        k = min(5, int(yw.sum()), int((~yw).sum()))
        if min(yw.sum(), (~yw).sum()) < 3:
            acc.append(np.nan)
            continue
        sc = []
        for _ in range(5):
            pred = np.empty(len(yw), bool)
            for tr, te in StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1 << 31))).split(Xw, yw):
                pred[te] = clf(C).fit(Xw[tr], yw[tr]).predict(Xw[te])
            sc.append(balanced_accuracy_score(yw, pred))
        acc.append(np.mean(sc))
    return np.array(cen), np.array(acc)


def process(args):
    sid, X, y, C, seed = args
    rng = np.random.default_rng(seed)
    corr, frac = heldout_correct(X, y, C, rng)
    bal, per = smooth_balanced(corr, y)
    n = len(y)
    k = int(rng.integers(int(0.1 * n), int(0.5 * n) + 1))
    if rng.random() < 0.5:
        Xs, ys, offs = X[:n - k], y[k:], 0
    else:
        Xs, ys, offs = X[k:], y[:n - k], k
    corr_s, _ = heldout_correct(Xs, ys, C, rng)
    bal_s, _ = smooth_balanced(corr_s, ys)
    null_curve = np.full(n, np.nan)
    null_curve[offs:offs + len(ys)] = bal_s[0]
    cen, acc = sliding(X, y, C, rng)
    return dict(correct=corr, frac=frac, bal=bal, per=per, null=null_curve, slide_c=cen, slide_acc=acc)


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        load_session_unit_spikes, prep_hitmiss_trials, select_fixed_c_pooled, sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    tab = pd.read_csv(LTP / "artifacts" / "020_lt_eval.csv").set_index("session_id")
    sids = [next(s for s in tab.index if s.startswith(p)) for p in EXAMPLES]
    jobs, meta = [], {}
    for sid in sids:
        tr = prep_hitmiss_trials(root, sid, st, tt)
        y = tr["lick_flag"].to_numpy().astype(bool)
        t = tr["start_time"].to_numpy()
        cw = _active_trials_from_whisker_onset_for_curve(sid, tt)
        cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
        lt = tab.loc[sid, "lt_cohort"]
        units = area_units(sid, "whole_brain", "All units", labels)
        mats = sliding_bin_population_matrices(load_session_unit_spikes(root, sid), units, t, np.ones(len(y), bool),
                                               [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=DZ)
        feats = {"sensory": mats[0], "sensory_minus_base": mats[0] - mats[1]}
        meta[sid] = dict(reward_group=tab.loc[sid, "reward_group"], group=tab.loc[sid, "group"], y=y,
                         lt_dec=int(np.sum(t < cw_t[int(lt)])) if not pd.isna(lt) else np.nan, n_units=len(units),
                         cw_t=cw_t, t=t)
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        for w in WIN_USE:
            C = select_fixed_c_pooled(feats[w], y, rng, n_folds=5)
            jobs.append(((sid, w), feats[w], y, C, int(rng.integers(1 << 31))))
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=_init) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(process, [(k, X, y, C, s) for k, X, y, C, s in jobs])))
    pickle.dump(dict(res=res, meta=meta), open(OUT / "107_decodability_curves.pkl", "wb"))
    plot(res, meta)


def plot(res, meta):
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _s = importlib.util.spec_from_file_location("q034", OUT / "034_area_window_quant_grid.py")
    q034 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(q034)
    curves = pickle.load(open(LTP / "artifacts" / "018_curves_sigma1.pkl", "rb"))
    col = {"R+": "#00B400", "R-": "#C800C8"}
    for w in WIN_USE:
        sids = list(meta)
        fig, axes = plt.subplots(2, len(sids), figsize=(4.3 * len(sids), 7.4), constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.3]))
        for c, sid in enumerate(sids):
            m, r = meta[sid], res[(sid, w)]
            rg, y = m["reward_group"], m["y"]
            n = len(y)
            x = np.arange(n)
            # behaviour mapped onto the decoded-trial axis (curve index -> decoded trial index by time)
            cu = curves[sid]["sigma1"]
            xb = np.searchsorted(m["t"], m["cw_t"])
            ax = axes[0, c]
            ax.fill_between(xb, cu["p_low80"], cu["p_high80"], color=col[rg], alpha=0.2, lw=0)
            ax.plot(xb, cu["p_mean"], color=col[rg], lw=1.3, label="whisker lick P (sigma=1)")
            ax.plot(xb, cu["fa_time"], color="#555555", lw=1, ls="--", label="FA")
            ax.scatter(x, np.where(y, 1.06, -0.06), s=2, color="k", marker="|")
            ax.set_title(f"{sid[:5]} {rg} {m['group']} ({n} trials, {int(y.sum())} hits, {m['n_units']} units)", fontsize=8.5)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(fontsize=6.5, frameon=False, loc="upper right")
            ax = axes[1, c]
            mean, lo, hi = r["bal"]
            ax.fill_between(x, lo, hi, color="k", alpha=0.15, lw=0)
            ax.plot(x, mean, color="k", lw=2, label="P(correct decoding), class-balanced")
            ax.plot(x, r["per"]["hit"][0], color=col[rg], lw=0.9, ls="--", label="hit trials")
            ax.plot(x, r["per"]["miss"][0], color="#777777", lw=0.9, ls="--", label="miss trials")
            ax.plot(x, r["null"], color="#bbbbbb", lw=1.5, label="shift null (labels slid)")
            ax.plot(r["slide_c"], r["slide_acc"], color="#ff7f0e", marker="o", ms=2.5, lw=1, label=f"sliding {WIN}-trial decoder")
            ax.scatter(x, np.where(r["correct"] == 1, 1.03, -0.03), s=2, color="k", marker="|")
            ax.axhline(0.5, color="#888888", ls=":")
            ax.set_ylim(-0.06, 1.06)
            ax.set_xlabel("whisker trial (decoded)")
            ax.legend(fontsize=6, frameon=False, loc="lower right")
            for a in axes[:, c]:
                if not pd.isna(m["lt_dec"]):
                    a.axvline(m["lt_dec"], color="#1f77b4", lw=1.8)
                a.set_xlim(-1, n)
        axes[1, 0].set_ylabel("decodability")
        fig.suptitle(f"Decodability learning curves ({w}, whole brain): one cross-validated decoder per session, per-trial "
                     f"correct/incorrect (ticks) smoothed with the behavioural learning-curve model (sigma = 1), hits and misses "
                     f"separately then averaged; grey = shift null; orange = sliding-window decoder; blue = behavioural LT",
                     fontsize=10)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"107_decodability_curves_{w}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    print("plotted")


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        d = pickle.load(open(OUT / "107_decodability_curves.pkl", "rb"))
        plot(d["res"], d["meta"])
    else:
        main()
