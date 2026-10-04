"""111 -- Neural-behaviour coupling model v2, all learning-stage sessions (autonomous run 2026-09-28, following
the v1 example run 108 and the improvement plan agreed there: drift-corrected neural score, behavioural
lick-tendency offset, minimum class counts per epoch, common-window linear shift null, group-level tests).

Question: does the whole-brain sensory-window signal predict licking more strongly after learning?

Per session (whisker trials, all kept incl. disengaged; whole brain; windows sensory 5-50 ms and sensory minus
baseline -200..-10 ms; decoder unchanged from the pipeline: StandardScaler + L2 logistic, one C per session/window):
 1. Drift correction (inside each CV training fold, no test labels used): per unit, OLS of the feature on a
    Legendre polynomial of trial position (degree DETREND_DEG) plus the hit/miss indicator; the fitted polynomial
    part is subtracted from training and held-out trials. (A centred running mean was tried first and rejected:
    it also removes the local class mean, anti-correlating z with the local hit rate.)
 2. Neural score z_i: cross-validated decision value (class-balanced, N_REP x stratified 5-fold, averaged), z-scored.
 3. Behavioural lick tendency o_i: logit of the leave-one-out smoothed lick probability of the random-walk
    learning-curve model (sigma = 1, exact grid; trial i's own lick excluded), used as a fixed offset.
 4. Coupling GLM (ridge-stabilised Newton, lambda = RIDGE on non-intercept terms):
        logit P(lick_i) = o_i + b0 + b1 z_i + b2 s_i + b3 z_i s_i
    s = post (1 at/after the split) for version "lt" (learners: cohort-specific learning trial, 020; non-learners:
    session midpoint), or s = trial position in [0, 1] for version "pos". b3 = change in coupling.
    Version "lt" requires >= MIN_CLASS hits and misses on both sides of the split (full session).
 5. Common-window linear shift null (Harris 2020): S = max(S_MIN, round(S_FRAC * n)); neural trials fixed to
    [S, n - S); behaviour (lick, s, offset) taken from trial i + k for k in [-S, S], |k| >= K_MIN; decoder retrained
    and GLM refitted per shift. The real value is k = 0 on the same window. b3_z = (b3 - null mean) / null SD.
Also stored: drift check = within-class correlation of z with trial index, with and without detrending.
Version "lt" is SECONDARY: the common window cuts the pre-learning epoch of early learners, so it uses the full
session with a truncating linear shift null (LT_SHIFTS shifts of 10-40% of the session, both directions).
Group tests (session = unit; one learning session per mouse): b3_z vs 0 (Wilcoxon | one-sample t, plus mouse-block
sign-flip), R+ vs R- and learners vs non-learners (Mann-Whitney | Welch).
Outputs: 111_coupling_v2.parquet (per session x window x version, full provenance), 111_coupling_v2_stats.csv,
figures/whole_brain/learning/111_coupling_v2.pdf/.png
Run (haas): python 111_coupling_model_v2.py [plot]
"""

from __future__ import annotations

import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
OUT_PATH = OUT / "111_coupling_v2.parquet"
WINDOWS = {"sensory": (0.005, 0.050), "baseline": (-0.200, -0.010)}
WIN_USE = ["sensory", "sensory_minus_base"]
DZ = (-0.010, 0.005)
DETREND_DEG = 3
N_REP = 5
SIGMA = 1.0
RIDGE = 0.01
MIN_CLASS = 5
S_FRAC, S_MIN, K_MIN = 0.2, 15, 10
LT_SHIFTS = 30
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "24"))
COL = {"R+": "#00B400", "R-": "#C800C8"}


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


# ------------------------------------------------------------------------------------------ building blocks
def poly_basis(tpos, deg=DETREND_DEG):
    from numpy.polynomial import legendre
    return np.column_stack([legendre.legval(tpos, np.eye(deg + 1)[k]) for k in range(1, deg + 1)])


def fold_detrend(Xtr, Xte, ytr, Btr, Bte):
    """Class-aware slow-drift removal fitted on the training fold only."""
    D = np.column_stack([np.ones(len(Xtr)), Btr, ytr.astype(float)])
    coef, *_ = np.linalg.lstsq(D, Xtr, rcond=None)
    trend = coef[1:1 + Btr.shape[1]]
    return Xtr - Btr @ trend, Xte - Bte @ trend


def loo_lick_logit(y, sigma=SIGMA):
    """Leave-one-out predictive logit of P(lick_i) under the random-walk learning-curve model."""
    sys.path.insert(0, str(LTP / "exploratory-analyses"))
    import lt_lib as L
    n, G = len(y), len(L.X_GRID)
    T = L._transition(sigma)
    lik = np.where(np.asarray(y)[:, None] == 1, L.P_GRID[None, :], 1 - L.P_GRID[None, :])
    alpha = np.empty((n, G))
    a = np.full(G, 1.0 / G) * lik[0]
    alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * lik[t]
        alpha[t] = a / a.sum()
    beta = np.empty((n, G))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        b = T @ (lik[t + 1] * beta[t + 1])
        beta[t] = b / b.sum()
    p = np.empty(n)
    for i in range(n):
        pred = np.full(G, 1.0 / G) if i == 0 else alpha[i - 1] @ T
        post = pred * beta[i]
        p[i] = (post / post.sum()) @ L.P_GRID
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def cv_score(X, y, C, rng, tpos=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    nf = min(5, int(y.sum()), int((~y).sum()))
    if nf < 2:
        return None
    z = np.zeros(len(y))
    for _ in range(N_REP):
        for tr, te in StratifiedKFold(nf, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear", C=C, class_weight="balanced",
                                                                   max_iter=1000))
            if tpos is None:
                Xtr, Xte = X[tr], X[te]
            else:
                B = poly_basis(tpos)
                Xtr, Xte = fold_detrend(X[tr], X[te], y[tr], B[tr], B[te])
            z[te] += m.fit(Xtr, y[tr]).decision_function(Xte)
    z /= N_REP
    return (z - z.mean()) / (z.std() + 1e-12)


def glm_offset(z, s, y, off):
    """Ridge-stabilised Newton fit of logit P = off + b0 + b1 z + b2 s + b3 z s."""
    D = np.column_stack([np.ones(len(z)), z, s, z * s])
    b = np.zeros(4)
    lam = np.r_[0, RIDGE, RIDGE, RIDGE]
    for _ in range(60):
        eta = np.clip(off + D @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = D.T @ (y - p) - lam * b
        H = (D * (p * (1 - p))[:, None]).T @ D + np.diag(lam)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


# ---------------------------------------------------------------------------------------------- per session
def process(args):
    sid, subject_id, rg, lcat = args
    sys.path.insert(0, SCRIPTS)
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        load_session_unit_spikes, prep_hitmiss_trials, select_fixed_c_pooled, sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    tab = pd.read_csv(LTP / "artifacts" / "020_lt_eval.csv").set_index("session_id")
    base = dict(session_id=sid, mouse_id=subject_id, reward_group=rg, learning_category=lcat,
                group=tab["group"].get(sid, None), lt_curve=tab["lt_cohort"].get(sid, np.nan))
    tr = prep_hitmiss_trials(root, sid, st, tt)
    if tr is None or base["group"] is None:
        return [dict(base, skipped_reason="no trials or no learning-trial table entry")]
    y = tr["lick_flag"].to_numpy().astype(bool)
    n = len(y)
    if min(y.sum(), (~y).sum()) < 2 * MIN_CLASS:
        return [dict(base, n_trials=n, n_hits=int(y.sum()), skipped_reason="too few hits or misses in session")]
    t = tr["start_time"].to_numpy()
    cw = _active_trials_from_whisker_onset_for_curve(sid, tt)
    cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
    learner = base["group"] == "learner" and not pd.isna(base["lt_curve"])
    split = int(np.sum(t < cw_t[int(base["lt_curve"])])) if learner else n // 2
    s_lt = (np.arange(n) >= split).astype(float)
    s_pos = np.arange(n) / (n - 1)
    off = loo_lick_logit(y.astype(int))
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    units = area_units(sid, "whole_brain", "All units", labels)
    mats = sliding_bin_population_matrices(load_session_unit_spikes(root, sid), units, t, np.ones(n, bool),
                                           [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=DZ)
    feats = {"sensory": mats[0], "sensory_minus_base": mats[0] - mats[1]}
    S = max(S_MIN, int(round(S_FRAC * n)))
    W = np.arange(S, n - S)
    shifts = [k for k in range(-S, S + 1) if k == 0 or abs(k) >= K_MIN]
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    rows = []
    for w in WIN_USE:
        X = feats[w]
        tp_full = np.linspace(-1, 1, n)
        C = select_fixed_c_pooled(X, y, rng, n_folds=5)
        z_full = cv_score(X, y, C, rng, tp_full)
        z_raw = cv_score(feats[w], y, C, rng)

        def within_class_r(z):
            """Drift check: correlation of z with trial index WITHIN hits and within misses (mean of the two), so
            that the legitimate trial-time structure of the hit/miss labels themselves does not count as drift."""
            idx = np.arange(n)
            return float(np.nanmean([np.corrcoef(z[m], idx[m])[0, 1] for m in (y, ~y) if m.sum() > 3]))
        r_det, r_raw = within_class_r(z_full), within_class_r(z_raw)
        res = {v: {} for v in ("lt", "pos")}
        # primary ("pos"): common-window linear shift null, neural trials fixed to W
        for k in shifts:
            yb, sp, ob = y[W + k], s_pos[W + k], off[W + k]
            if min(yb.sum(), (~yb).sum()) < MIN_CLASS:
                continue
            z = cv_score(X[W], yb, C, rng, tp_full[W])
            if z is not None:
                res["pos"][k] = glm_offset(z, sp, yb.astype(float), ob)
        # secondary ("lt"): full session (the common window would cut the pre-learning epoch of early learners);
        # truncating linear shift null, LT_SHIFTS shifts of 10-40% of the session in both directions
        pre, post = s_lt == 0, s_lt == 1

        def lt_fit(Xn, tpn, yb, sl, ob):
            if min(yb[sl == 0].sum(), (~yb[sl == 0]).sum(), yb[sl == 1].sum(), (~yb[sl == 1]).sum()) < MIN_CLASS:
                return None
            z = cv_score(Xn, yb, C, rng, tpn)
            return None if z is None else glm_offset(z, sl, yb.astype(float), ob)
        b_real = lt_fit(X, tp_full, y, s_lt, off)
        if b_real is not None:
            res["lt"][0] = b_real
            for j, k in enumerate(sorted({int(round(f * n)) for f in np.linspace(0.1, 0.4, LT_SHIFTS // 2)})):
                for sgn in (1, -1):
                    if sgn > 0:   # neural trials 0..n-k-1 paired with behaviour k..n-1
                        nsl, bsl = slice(0, n - k), slice(k, n)
                    else:         # neural trials k..n-1 paired with behaviour 0..n-k-1
                        nsl, bsl = slice(k, n), slice(0, n - k)
                    b = lt_fit(X[nsl], tp_full[nsl], y[bsl], s_lt[bsl], off[bsl])
                    if b is not None:
                        res["lt"][sgn * k] = b
        yw = y[W]
        cnt = dict(pre_hit=int(y[pre].sum()), pre_miss=int((~y[pre]).sum()), post_hit=int(y[post].sum()),
                   post_miss=int((~y[post]).sum()), window_hits=int(yw.sum()), window_misses=int((~yw).sum()))
        for v in ("lt", "pos"):
            r = res[v]
            common = dict(base, window=w, version=v, split_ref=("learning_trial" if learner else "midpoint") if v == "lt" else "position",
                          split_trial=split if v == "lt" else np.nan, n_trials=n, n_hits=int(y.sum()), n_units=len(units), C=C,
                          null_kind="common-window linear shift" if v == "pos" else "truncating linear shift (full session)",
                          S=S, n_window=len(W) if v == "pos" else n, detrend_deg=DETREND_DEG, sigma=SIGMA, ridge=RIDGE, n_rep=N_REP,
                          r_within_class_z_trial_detrended=r_det, r_within_class_z_trial_raw=r_raw, **cnt)
            if 0 not in r:
                rows.append(dict(common, skipped_reason="real fit not possible (class counts in window/epochs)"))
                continue
            nb = np.array([b for k, b in r.items() if k != 0])
            b = r[0]
            ok = len(nb) >= 10
            rows.append(dict(common, b0=b[0], b1=b[1], b2=b[2], b3=b[3], coupling_pre=b[1], coupling_post=b[1] + b[3],
                             n_null=len(nb), null_b3_mean=nb[:, 3].mean() if len(nb) else np.nan,
                             null_b3_sd=nb[:, 3].std(ddof=1) if len(nb) > 1 else np.nan,
                             b3_z=(b[3] - nb[:, 3].mean()) / nb[:, 3].std(ddof=1) if ok else np.nan,
                             b3_pct=float((nb[:, 3] < b[3]).mean()) if ok else np.nan,
                             null_b1_mean=nb[:, 1].mean() if len(nb) else np.nan,
                             skipped_reason=None if ok else f"only {len(nb)} valid shifts"))
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    sess = hitmiss_session_list(pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    print(f"[111] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, (r.session_id, r.subject_id, r.reward_group, r.learning_category)): r.session_id
                for r in todo.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                new = pd.DataFrame(f.result())
            except Exception as e:  # keep going; record the failure
                new = pd.DataFrame([dict(session_id=futs[f], skipped_reason=f"error: {e!r}")])
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            print(f"[{i}/{len(futs)}] {futs[f]} {time.time() - t0:.0f}s", flush=True)
    summarize()


# ------------------------------------------------------------------------------------------------ summary
def summarize():
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, ttest_rel, wilcoxon
    _s = importlib.util.spec_from_file_location("p110", OUT / "110_publication_figures.py")
    p110 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(p110)
    p110.style()
    d = pd.read_parquet(OUT_PATH)
    print("skipped:", d[d.skipped_reason.notna()].skipped_reason.str[:60].value_counts().to_dict())
    ok = d[d.skipped_reason.isna()].copy()
    rng = np.random.default_rng(0)

    def signflip(v, mice, n=5000):
        v, mice = np.asarray(v, float), np.asarray(mice)
        m = np.isfinite(v)
        v, mice = v[m], mice[m]
        um = np.unique(mice)
        idx = np.searchsorted(um, mice)
        obs = v.mean()
        null = np.array([(v * rng.choice([-1, 1], len(um))[idx]).mean() for _ in range(n)])
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))

    def one(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return (np.nan, np.nan) if len(v) < 3 else (wilcoxon(v).pvalue, ttest_1samp(v, 0).pvalue)

    def two(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        return (np.nan, np.nan) if min(len(a), len(b)) < 3 else (mannwhitneyu(a, b).pvalue, ttest_ind(a, b, equal_var=False).pvalue)

    rows = []
    groups = [("R+ learners", "R+", "learner"), ("R+ non-learners", "R+", "non_learner"), ("R- learners", "R-", "learner"),
              ("R- non-learners", "R-", "non_learner")]
    for (w, v), g in ok.groupby(["window", "version"]):
        for lab, rg, grp in groups:
            s = g[(g.reward_group == rg) & (g.group == grp)]
            pw, pt = one(s.b3_z)
            pc = ttest_rel(s.coupling_post, s.coupling_pre).pvalue if len(s) >= 3 else np.nan
            rows.append(dict(window=w, version=v, comparison=f"{lab}: b3_z vs 0", n=len(s), mean=s.b3_z.mean(), median=s.b3_z.median(),
                             p_nonparam=pw, p_param=pt, p_mouse_signflip=signflip(s.b3_z, s.mouse_id) if len(s) >= 3 else np.nan,
                             mean_coupling_pre=s.coupling_pre.mean(), mean_coupling_post=s.coupling_post.mean(), p_coupling_paired_t=pc))
        for lab, a, b in (("learners R+ vs R-", g[(g.reward_group == "R+") & (g.group == "learner")], g[(g.reward_group == "R-") & (g.group == "learner")]),
                          ("R+ learners vs non-learners", g[(g.reward_group == "R+") & (g.group == "learner")], g[(g.reward_group == "R+") & (g.group == "non_learner")]),
                          ("R- learners vs non-learners", g[(g.reward_group == "R-") & (g.group == "learner")], g[(g.reward_group == "R-") & (g.group == "non_learner")]),
                          ("all learners vs non-learners", g[g.group == "learner"], g[g.group == "non_learner"])):
            pm, pw_ = two(a.b3_z, b.b3_z)
            rows.append(dict(window=w, version=v, comparison=f"{lab} (b3_z)", n=len(a) + len(b), mean_a=a.b3_z.mean(), mean_b=b.b3_z.mean(),
                             n_a=len(a), n_b=len(b), p_nonparam=pm, p_param=pw_))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "111_coupling_v2_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string())

    # figure
    fig = plt.figure(figsize=(7.2, 6.4))
    fig.suptitle("Neural-behaviour coupling v2: does the whole-brain signal predict licking more strongly after learning?",
                 fontsize=8.5, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.955, "logit P(lick) = behavioural tendency (LOO curve) + b0 + b1 z + b2 s + b3 z s;  z = drift-corrected "
                          "cross-validated neural score;  b3_z = b3 vs common-window linear-shift null;  brackets: MW | Welch",
             fontsize=5.6, color="#444444")
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.98, top=0.88, bottom=0.09, hspace=0.75, wspace=0.45)
    xs = {"R+ learners": 0, "R+ non-learners": 1, "R- learners": 2.4, "R- non-learners": 3.4}
    for r, w in enumerate(WIN_USE):
        for c, v in enumerate(("lt", "pos")):
            ax = fig.add_subplot(gs[r, c])
            g = ok[(ok.window == w) & (ok.version == v)]
            for lab, rg, grp in groups:
                s = g[(g.reward_group == rg) & (g.group == grp)].b3_z.dropna()
                p110.strip(ax, xs[lab], s, COL[rg], filled=grp == "learner")
            brk = []
            for lab, a, b in (("R+", "R+ learners", "R+ non-learners"), ("R-", "R- learners", "R- non-learners")):
                ga = g[(g.reward_group == lab) & (g.group == "learner")].b3_z
                gb = g[(g.reward_group == lab) & (g.group == "non_learner")].b3_z
                pm, pw_ = two(ga, gb)
                brk.append((xs[a], xs[b], f"{p110.fmt_s(pm)} | {p110.fmt_s(pw_)}"))
            ax.axhline(0, color="#999999", lw=0.5, ls="--")
            ax.set_xticks(list(xs.values()), ["L", "NL", "L", "NL"])
            ax.set_xlabel("R+            R-")
            ax.set_ylabel("b3_z (coupling change vs null)")
            p110.draw_brackets(ax, brk)
            ttl = "before vs after learning trial\n(non-learners: midpoint)" if v == "lt" else "trial position (0 to 1)"
            ax.set_title(f"{w}: {ttl}", fontsize=6)
            p110.letter(ax, "abcdef"[r * 3 + c], dx=-0.3)
        ax = fig.add_subplot(gs[r, 2])
        g = ok[(ok.window == w) & (ok.version == "lt")]
        for lab, rg, grp in groups:
            s = g[(g.reward_group == rg) & (g.group == grp)]
            x0 = xs[lab] * 1.3
            for _, q in s.iterrows():
                ax.plot([x0, x0 + 0.8], [q.coupling_pre, q.coupling_post], color=COL[rg], lw=0.35, alpha=0.35 if grp == "learner" else 0.2)
            for j, col in enumerate(("coupling_pre", "coupling_post")):
                ax.errorbar(x0 + 0.8 * j, s[col].mean(), yerr=s[col].std() / np.sqrt(max(len(s), 1)), color="k", marker="o", ms=2.5,
                            capsize=1.5, lw=0.7, zorder=3)
        ax.axhline(0, color="#999999", lw=0.5, ls="--")
        ax.set_xticks([xs[k] * 1.3 + 0.4 for k in xs], ["R+ L", "R+ NL", "R- L", "R- NL"])
        ax.set_ylabel("coupling b1 (pre) to b1 + b3 (post)")
        ax.set_title(f"{w}: coupling before -> after", fontsize=6)
        p110.letter(ax, "abcdef"[r * 3 + 2], dx=-0.3)
    fdir = OUT / "figures" / "whole_brain" / "learning"
    for ext in ("pdf", "png"):
        fig.savefig(fdir / f"111_coupling_v2.{ext}", dpi=300)
    dr = ok.drop_duplicates(["session_id", "window"])
    print("drift check median |r(z, trial)|: raw", dr.r_within_class_z_trial_raw.abs().median().round(3), "detrended", dr.r_within_class_z_trial_detrended.abs().median().round(3))
    print("group counts (lt, sensory):", ok[(ok.window == "sensory") & (ok.version == "lt")].groupby(["reward_group", "group"]).size().to_dict())


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        summarize()
    else:
        main()
