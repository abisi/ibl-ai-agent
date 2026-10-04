"""121 -- Example sessions: continuous single-trial decoder outputs (held-out log-probability of the true class and
session-scaled signed margin) for hit vs miss, compared with the behavioural learning curves and learning trials
(user 2026-09-30: "show for example sessions how one could get the log prob or session-scaled margin and visualize
example test predictions ... then compare with learning curves"). Exploratory preview before a large test.

Per session (8 example sessions of 107; whole brain; learning-stage whisker trials from prep_hitmiss_trials, disengaged
trials KEPT; y = lick):
  features  mean rate per unit in the window (5-50 ms or 5-100 ms after the stimulus; dead zone -10..+5 ms);
  decoder   pipeline StandardScaler -> L2 logistic regression (liblinear, class_weight='balanced'); C chosen INSIDE each
            training fold by an inner stratified 3-fold CV on log-loss over C_GRID (nested; no test-trial leakage;
            scaler fitted on the training fold only);
  outer CV  repeated stratified K-fold, K = min(5, minority count), N_REP repeats with new random fold assignments;
            every trial is held out once per repeat -> N_REP held-out outputs per trial;
  outputs   f = decision value (signed distance from the boundary in the decoder's units = w.x + b; the LDA-like
            projection); p_hit = P(lick | x);
  per trial (averaged over repeats):
            log-prob score  = log2 P(true class | x) + 1  ["bits above chance": 0 = chance (P = 0.5), 1 = certain
                              correct, negative = confidently wrong]; averaged as the mean of log2 P over repeats;
            scaled margin   = s * f / SD_session(f), s = +1 for hits, -1 for misses; SD over all held-out outputs of the
                              session -> comparable across sessions; > 0 = correct side of the boundary;
            correct         = held-out prediction correct in the majority of repeats (107's 0/1 metric, for reference);
  smoothing class-balanced: Gaussian kernel (SMOOTH_SD trials) over the trial index separately on hit and miss trials,
            interpolated to every trial, then averaged (class composition changes along the session);
  drift null linear-shift null: labels shifted by 10-50% of the session (non-wrapping, N_SHIFT shifts), the SAME
            nested-CV decoding re-run, per-trial scores mapped to the neural trial they sit on, averaged over shifts.
Behaviour: whisker-lick and no-stim (FA) learning curves (ssl-learning-trial-identification 018, sigma = 1) on the same
trial axis; learning trials of several definitions (013) as vertical lines.
Outputs: figures/121_singletrial_scores_examples_<window>.pdf/.png/.svg, 121_singletrial_scores_examples.pkl
Run (haas, repo root): python .../121_singletrial_scores_examples.py
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

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
EXAMPLES = ["AB119", "AB125", "MH029", "AB085", "MH018", "AB120", "MH030", "AB159"]
WINDOWS = {"5-50ms": (0.005, 0.050), "5-100ms": (0.005, 0.100)}
DZ = (-0.010, 0.005)
N_REP, N_SHIFT, SMOOTH_SD = 20, 10, 5.0
LT_SHOW = {"L0 stored": ("#999999", ":"), "L5 whisker CP": ("#1f77b4", "-"), "L6 joint CP": ("#d62728", "-"),
           "lenient cascade": ("#ff7f0e", "--")}


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def heldout(X, y, rng, C_grid):
    """Nested repeated CV -> per-trial arrays (N_REP x n) of decision values f and P(hit)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    k = min(5, int(y.sum()), int((~y).sum()))
    F, P = np.full((N_REP, n), np.nan), np.full((N_REP, n), np.nan)
    if k < 2:
        return F, P
    for r in range(N_REP):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, y):
            ki = min(3, int(y[tr].sum()), int((~y[tr]).sum()))
            pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear",
                                                                      class_weight="balanced", max_iter=1000))
            if ki >= 2:
                g = GridSearchCV(pipe, {"logisticregression__C": list(C_grid)}, scoring="neg_log_loss",
                                 cv=StratifiedKFold(ki, shuffle=True, random_state=int(rng.integers(1 << 31))))
                m = g.fit(X[tr], y[tr]).best_estimator_
            else:
                m = pipe.set_params(logisticregression__C=1.0).fit(X[tr], y[tr])
            F[r, te] = m.decision_function(X[te])
            P[r, te] = m.predict_proba(X[te])[:, 1]
    return F, P


def scores(F, P, y):
    s = np.where(y, 1.0, -1.0)
    p_true = np.where(y[None, :], P, 1 - P)
    logp = np.nanmean(np.log2(np.clip(p_true, 1e-6, 1)), 0) + 1
    sd = np.nanstd(F)
    margin = s * np.nanmean(F, 0) / (sd if sd > 0 else 1)
    correct = (np.nanmean((F > 0) == y[None, :], 0) > 0.5).astype(float)
    return dict(logp=logp, margin=margin, correct=correct, f_mean=np.nanmean(F, 0) / (sd if sd > 0 else 1))


def smooth_balanced(v, y, sd=SMOOTH_SD):
    n = len(v)
    x = np.arange(n)
    out = []
    for m in (y, ~y):
        idx = np.where(m & np.isfinite(v))[0]
        if len(idx) < 2:
            out.append(np.full(n, np.nan))
            continue
        w = np.exp(-0.5 * ((x[:, None] - idx[None, :]) / sd) ** 2)
        out.append((w @ v[idx]) / w.sum(1))
    return np.nanmean(np.stack(out), 0), out


def process(args):
    key, X, y, seed = args
    sys.path.insert(0, SCRIPTS)
    import warnings
    warnings.filterwarnings("ignore")
    from ssl_timeresolved_decoding import C_GRID
    rng = np.random.default_rng(seed)
    F, P = heldout(X, y, rng, C_GRID)
    real = scores(F, P, y)
    n = len(y)
    null = {k: np.full((N_SHIFT, n), np.nan) for k in ("logp", "margin")}
    for i in range(N_SHIFT):
        k = int(rng.integers(max(1, int(0.1 * n)), int(0.5 * n) + 1))
        if rng.random() < 0.5:
            Xs, ys, off = X[: n - k], y[k:], 0            # neural trials 0..n-k-1
        else:
            Xs, ys, off = X[k:], y[: n - k], k            # neural trials k..n-1
        if min(ys.sum(), (~ys).sum()) < 3:
            continue
        Fs, Ps = heldout(Xs, ys, rng, C_GRID)
        s_ = scores(Fs, Ps, ys)
        for m in null:
            null[m][i, off:off + len(ys)] = smooth_balanced(s_[m], ys)[0]
    out = dict(real=real, F=F, P=P, null={m: np.nanmean(v, 0) for m, v in null.items()})
    for m in ("logp", "margin", "correct"):
        out[f"{m}_smooth"], out[f"{m}_smooth_cls"] = smooth_balanced(real[m], y)
    return key, out


def main():
    os.chdir(OUT.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    sess = T.hitmiss_session_list(st)
    sess = sess[sess.day_stage == "learning"].set_index("session_id")
    lt = pd.read_csv(LTP / "artifacts" / "013_learning_trials_all_methods.csv").set_index("session_id")
    sids = [next(s for s in sess.index if s.startswith(p)) for p in EXAMPLES]
    jobs, meta = [], {}
    for sid in sids:
        tr = T.prep_hitmiss_trials(root, sid, st, tt)
        y = (tr.lick_flag == 1).to_numpy()
        t = tr.start_time.to_numpy()
        units = T.area_units(sid, "whole_brain", "All units", labels)
        mats = T.sliding_bin_population_matrices(T.load_session_unit_spikes(root, sid), units, t, np.ones(len(y), bool),
                                                 list(WINDOWS.values()), dead_zone=DZ)
        cw = T._active_trials_from_whisker_onset_for_curve(sid, tt)
        cw_t = cw.loc[cw.trial_type == "whisker_trial", "start_time"].to_numpy()
        lts = {}
        for dname in LT_SHOW:
            v = lt[dname].get(sid, np.nan) if dname in lt.columns else np.nan
            lts[dname] = int(np.sum(t < cw_t[int(v)])) if pd.notna(v) and int(v) < len(cw_t) else np.nan
        meta[sid] = dict(reward_group=sess.loc[sid, "reward_group"], y=y, t=t, cw_t=cw_t, n_units=len(units), lts=lts,
                         category=lt["L6 category"].get(sid, "") if "L6 category" in lt.columns else "")
        for w, X in zip(WINDOWS, mats):
            ok = ~np.isnan(X).any(1)
            assert ok.all(), "NaN rows"
            jobs.append(((sid, w), X, y, zlib.crc32(f"{sid}|{w}".encode())))
    with ProcessPoolExecutor(len(jobs), initializer=_init) as ex:
        res = dict(ex.map(process, jobs))
    pickle.dump(dict(res=res, meta=meta), open(OUT / "121_singletrial_scores_examples.pkl", "wb"))
    plot(res, meta)


def plot(res, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5,
                         "ytick.labelsize": 5, "legend.fontsize": 5, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    curves = pickle.load(open(LTP / "artifacts" / "018_curves_sigma1.pkl", "rb"))
    col = {"R+": "#00B400", "R-": "#C800C8"}
    for w in WINDOWS:
        sids = list(meta)
        fig, axes = plt.subplots(4, len(sids), figsize=(15, 7.6), sharex="col",
                                 gridspec_kw=dict(height_ratios=[0.8, 1, 1, 1]))
        fig.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.07, hspace=0.18, wspace=0.28)
        for c, sid in enumerate(sids):
            m, r = meta[sid], res[(sid, w)]
            rg, y = m["reward_group"], m["y"]
            n = len(y)
            x = np.arange(n)
            ax = axes[0, c]
            cu = curves[sid]["sigma1"]
            xb = np.searchsorted(m["t"], m["cw_t"])
            ax.fill_between(xb, cu["p_low80"], cu["p_high80"], color=col[rg], alpha=0.2, lw=0)
            ax.plot(xb, cu["p_mean"], color=col[rg], lw=1.1, label="whisker lick")
            ax.plot(xb, cu["fa_time"], color="#555555", lw=0.9, ls="--", label="no-stim lick (FA)")
            ax.scatter(x, np.where(y, 1.05, -0.05), s=1.5, color="k", marker="|")
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(f"{sid[:5]} {'R+' if rg == 'R+' else 'R−'} ({m['category']})\n{n} trials, {int(y.sum())} hits, "
                         f"{m['n_units']} units", color=col[rg])
            # example test predictions: per-trial held-out decision value (session-scaled), hits vs misses
            ax = axes[1, c]
            fm = r["real"]["f_mean"]
            ax.scatter(x[y], fm[y], s=3, color=col[rg], lw=0, label="hit trials")
            ax.scatter(x[~y], fm[~y], s=3, color="0.45", lw=0, label="miss trials")
            ax.axhline(0, color="k", lw=0.5)
            # log-prob bits above chance
            ax = axes[2, c]
            ax.scatter(x, r["real"]["logp"], s=1.5, color="0.7", lw=0)
            ax.plot(x, r["logp_smooth"], color="k", lw=1.2, label="class-balanced smooth")
            ax.plot(x, r["null"]["logp"], color="#bbbbbb", lw=1.2, label="shift null")
            ax.axhline(0, color="0.5", lw=0.5, ls=":")
            ax.set_ylim(-1.5, 1.05)
            # scaled margin
            ax = axes[3, c]
            ax.scatter(x, r["real"]["margin"], s=1.5, color="0.7", lw=0)
            ax.plot(x, r["margin_smooth"], color="k", lw=1.2, label="class-balanced smooth")
            ax.plot(x, r["null"]["margin"], color="#bbbbbb", lw=1.2, label="shift null")
            ax.axhline(0, color="0.5", lw=0.5, ls=":")
            ax.set_ylim(-3, 3)
            ax.set_xlabel("whisker trial")
            for a in axes[:, c]:
                for dname, (lc, ls) in LT_SHOW.items():
                    v = m["lts"].get(dname, np.nan)
                    if np.isfinite(v):
                        a.axvline(v, color=lc, ls=ls, lw=0.9)
                a.set_xlim(-1, n)
        axes[0, 0].set_ylabel("P(lick)")
        axes[1, 0].set_ylabel("held-out decision value\n(session SD units)")
        axes[2, 0].set_ylabel("log₂ P(true class) + 1\n(bits above chance)")
        axes[3, 0].set_ylabel("signed margin\n(session SD units)")
        axes[0, 0].legend(loc="center left", frameon=False)
        axes[1, 0].legend(loc="lower left", frameon=False, markerscale=2)
        axes[2, 0].legend(loc="lower left", frameon=False)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=lc, ls=ls, lw=1) for lc, ls in LT_SHOW.values()], list(LT_SHOW), loc="lower center",
                   ncol=len(LT_SHOW), frameon=False, bbox_to_anchor=(0.5, 0.0), title="learning trial", title_fontsize=6)
        fig.suptitle(f"Single-trial hit vs miss decoder outputs, whole brain, {w} after the stimulus (one nested-CV decoder "
                     f"per session, {N_REP} repeats of stratified K-fold; C chosen inside each training fold)\n"
                     "row 2: held-out decision value per trial; rows 3-4: per-trial score (grey dots), class-balanced "
                     f"Gaussian smooth (sd {SMOOTH_SD:.0f} trials, black) and linear-shift null ({N_SHIFT} shifts, grey)",
                     fontsize=6.5, y=0.995)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / "figures" / f"121_singletrial_scores_examples_{w}.{ext}", dpi=250)
        plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        d = pickle.load(open(OUT / "121_singletrial_scores_examples.pkl", "rb"))
        plot(d["res"], d["meta"])
    else:
        main()
