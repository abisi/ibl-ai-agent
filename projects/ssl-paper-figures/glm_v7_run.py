"""Shared input glm_v7 (user 2026-11-17: "with the new GLM, also regress reaction time, multi-head GLM"): the glm_v6 lick
GLM refitted on the current NWBs (DLC reconversion of 2026-11-16) plus a reaction-time head.

Two heads on ONE design matrix (glm_v5 load_session: Myriam Hamon's glm_behavior_v3 predictors of each whisker trial of
whisker day 0, learners) and ONE set of cross-validation splits (glm_v5 contiguous blocks: 5 blocks of whisker trials in
time order, gap 3 trials, 10 repeats with shifted block boundaries):
  - lick head = glm_v6 exactly (L2 logistic, one C per session chosen on the other sessions, running-mean baseline,
    drop-and-refit and shuffle importance); same file names as glm_v6, so Figure 1 can point at glm_v7
  - RT head = ridge regression of the reaction time (ms) of the hits in each block. Hits = licked whisker trials without
    an early lick, RT 0-1 s (behaviour table computed_reaction_time, as Figure 1 I). Train = hits of the training trials,
    test = hits of the test block. One ridge alpha per session, chosen on the other sessions (as C in glm_v6).
    Score (also in ms: MAE / RMSE and their gain over predicting the training mean, user 2026-11-17): held-out R2
    against the training-fold mean RT (r2_vs_train = 1 - SSE / SSE of predicting the training mean;
    with contiguous blocks this is the fair baseline, as pseudo-R2 is for the lick head), plus the usual pooled R2 and
    the correlation of observed and predicted RT. Baseline: linear regression on the causal mean RT of the previous 5
    hits. Importance: drop-and-refit change in r2_vs_train per glm_v5 predictor group.
Third head (user 2026-11-17): jaw-opening onset of the same hits (jaw_tables), files <mouse>_<session>_jaw_*, jaw_summary.csv;
`--heads jaw` fits it alone on the cached design matrices.
Outputs: Figures/_cache/glm_v7/ = glm_v6 files + per session <mouse>_<session>_rt_{metrics.csv, coefficients.csv,
nested.csv, predictions.parquet}; c_curves.csv, alpha_curves.csv, c_choice.csv, alpha_choice.csv,
all_sessions_summary.csv (lick), rt_summary.csv (RT), config.json; design matrices in _design/ (npz, reused by stage C).
Run on haas (repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_v7_run.py [--n-jobs 100]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_v5_run as v5  # noqa: E402
import glm_v6_run as v6  # noqa: E402
import shared_behaviour  # noqa: E402
from common import CACHE_DIR, session_nwbs  # noqa: E402
from glm_cv_compare import block_splits  # noqa: E402
from glm_v5_compare import run_mean_feature  # noqa: E402

import os  # noqa: E402

# pilot runs: GLM_V7_OUT=glm_v7_pilot (read at import, so the parent and the workers agree)
# SSL_DAY0_KEEP (user 2026-11-25, _alt): whisker trials after first + keep x (last - first) dropped, out glm_v7_keepNN
OUT = CACHE_DIR / (os.environ.get("GLM_V7_OUT", "glm_v7") + shared_behaviour.CUT_TAG)
DESIGN = OUT / "_design"
ALPHA_GRID = np.logspace(-1, 5, 13)
RT_MIN_HITS = 20
RT_BASELINE_K = 5


def _patch():
    """In each worker: glm_v5/glm_v6 write to glm_v7 and read the cached design matrix (module globals set in the parent
    are not seen by loky workers)."""
    v5.OUT = v6.OUT = OUT
    if not hasattr(v5, "orig_load_session"):  # loky reuses workers: stage A keeps the real loader (a module-level
        v5.orig_load_session = v5.load_session  # alias would be re-resolved by name to the patched one)

    def load_cached(glm, nwb):
        f = DESIGN / f"{Path(nwb).stem}.npz"
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=True)
        return z["X"], z["y"], z["t"], list(z["names"])
    v5.load_session = load_cached


def rt_for_trials(t, rt_table):
    """RT (ms) of each whisker trial start time t (nan when not a valid hit)."""
    st, rt = rt_table
    i = np.clip(np.searchsorted(st, t), 1, len(st) - 1)
    j = np.where(np.abs(st[i - 1] - t) < np.abs(st[i] - t), i - 1, i)
    return np.where(np.abs(st[j] - t) < 0.01, rt[j], np.nan)


def rt_scores(y, p, ref):
    sse = np.sum((y - p) ** 2)
    return {"rt_r2_vs_train": 1 - sse / np.sum((y - ref) ** 2), "rt_r2": 1 - sse / np.sum((y - y.mean()) ** 2),
            "rt_corr": np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan,
            # user 2026-11-17: errors in ms too; gain = error of predicting the training mean minus the model's error
            # (ms; > 0 = the model predicts better)
            "rt_mae_ms": np.mean(np.abs(y - p)), "rt_rmse_ms": np.sqrt(np.mean((y - p) ** 2)),
            "train_mean_mae_ms": np.mean(np.abs(y - ref)), "train_mean_rmse_ms": np.sqrt(np.mean((y - ref) ** 2)),
            "rt_mae_gain_ms": np.mean(np.abs(y - ref)) - np.mean(np.abs(y - p)),
            "rt_rmse_gain_ms": np.sqrt(np.mean((y - ref) ** 2)) - np.sqrt(np.mean((y - p) ** 2))}


def rt_cv(X, rt, valid, alpha, off, cols=None):
    """One repeat of the block CV over all whisker trials; fit and test on the hits. Returns predictions, the training
    mean of each test hit and (cols) the drop-and-refit predictions per group."""
    n = len(rt)
    P, REF = np.full(n, np.nan), np.full(n, np.nan)
    drop = {g: np.full(n, np.nan) for g in (cols or {})}
    coefs = []
    for k, (tr, te) in enumerate(block_splits(n, v5.N_FOLDS, offset=off, gap=v5.GAP)):
        tr, te = tr[valid[tr]], te[valid[te]]
        if len(tr) < RT_MIN_HITS // 2 or not len(te):
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = Ridge(alpha=alpha).fit(Xtr, rt[tr])
        P[te], REF[te] = m.predict(Xte), rt[tr].mean()
        coefs.append((k, m.coef_, m.intercept_))
        for g, idx in (cols or {}).items():
            keep = [i for i in range(X.shape[1]) if i not in set(idx)]
            drop[g][te] = Ridge(alpha=alpha).fit(Xtr[:, keep], rt[tr]).predict(Xte[:, keep])
    return P, REF, drop, coefs


def stage_a(mouse, cohort, nwb, src, rt_table):
    """Load the design matrix once (saved for stage C), then the lick C curve (glm_v6) and the RT alpha curve."""
    glm = v5.glm_module(src)
    try:
        loaded = getattr(v5, "orig_load_session", v5.load_session)(glm, nwb)
    except Exception as e:  # noqa: BLE001 - reported and skipped, as glm_v5
        print("skip", Path(nwb).stem, type(e).__name__, e)
        return None, None
    if loaded is None:  # not whisker day 0
        return None, None
    X, y, t, names = loaded
    if shared_behaviour.DAY0_KEEP:  # predictors are causal, so cutting the tail leaves the kept rows unchanged
        k = t <= t.min() + shared_behaviour.DAY0_KEEP * (t.max() - t.min())
        X, y, t = X[k], y[k], t[k]
    DESIGN.mkdir(parents=True, exist_ok=True)
    np.savez(DESIGN / f"{Path(nwb).stem}.npz", X=X, y=y, t=t, names=np.array(names, dtype=object))
    _patch()
    lick = v6.c_curve(mouse, cohort, nwb, src) if len(np.unique(y)) == 2 else None
    return lick, alpha_rows(mouse, cohort, nwb, X, y, t, rt_table)


def alpha_rows(mouse, cohort, nwb, X, y, t, rt_table):
    """Stage A of a regression head: held-out scores per ridge alpha (3 repeats)."""
    rt = rt_for_trials(t, rt_table) if rt_table is not None else np.full(len(y), np.nan)
    valid = (y == 1) & np.isfinite(rt)
    rows = []
    if valid.sum() >= RT_MIN_HITS:
        for rep, off in enumerate(v6.offsets_for(len(y), v6.N_REP_A)):
            for a in ALPHA_GRID:
                P, REF, _, _ = rt_cv(X, rt, valid, a, off)
                ok = np.isfinite(P)
                rows.append({"mouse_id": mouse, "session_id": Path(nwb).stem, "cohort": cohort, "repeat": rep,
                             "alpha": a, **rt_scores(rt[ok], P[ok], REF[ok])})
    return pd.DataFrame(rows) if rows else None


def cached_alpha_rows(mouse, cohort, nwb, table):
    _patch()
    loaded = v5.load_session(None, nwb)
    return None if loaded is None or table is None else alpha_rows(mouse, cohort, nwb, *loaded[:3], table)


def rt_fit(mouse, cohort, nwb, alpha, rt_table, prefix="rt"):
    """Regression head (RT, or jaw onset with prefix "jaw": same code, values in ms from the stimulus)."""
    _patch()
    loaded = v5.load_session(None, nwb)
    if loaded is None or rt_table is None:
        return None
    X, y, t, names = loaded
    session = Path(nwb).stem
    rt = rt_for_trials(t, rt_table)
    valid = (y == 1) & np.isfinite(rt)
    if valid.sum() < RT_MIN_HITS:
        return None
    n = len(y)
    cols = v5.group_columns(names)
    hits = np.flatnonzero(valid)
    prev = pd.Series(rt[hits]).shift(1).rolling(RT_BASELINE_K, min_periods=1).mean().to_numpy()
    base_feat = np.full(n, np.nan)
    base_feat[hits] = prev
    rep_rows, drop_rows, coef_rows, preds = [], [], [], []
    for rep, off in enumerate(v6.offsets_for(n, v5.N_REPEATS)):
        P, REF, drop, coefs = rt_cv(X, rt, valid, alpha, off, cols)
        B = np.full(n, np.nan)  # baseline: running mean RT of the previous hits (first hit: training mean)
        for tr, te in block_splits(n, v5.N_FOLDS, offset=off, gap=v5.GAP):
            tr, te = tr[valid[tr]], te[valid[te]]
            if len(tr) < RT_MIN_HITS // 2 or not len(te):
                continue
            mu = rt[tr].mean()
            ftr, fte = np.nan_to_num(base_feat[tr], nan=mu)[:, None], np.nan_to_num(base_feat[te], nan=mu)[:, None]
            B[te] = LinearRegression().fit(ftr, rt[tr]).predict(fte)
        ok = np.isfinite(P)
        full = rt_scores(rt[ok], P[ok], REF[ok])
        rep_rows.append({"repeat": rep, "n_hits_predicted": int(ok.sum()), **full,
                         **{f"baseline_{k}": v for k, v in rt_scores(rt[ok], B[ok], REF[ok]).items()}})
        for g in cols:
            d = rt_scores(rt[ok], drop[g][ok], REF[ok])
            drop_rows.append({"repeat": rep, "group": g, "delta_rt_r2_vs_train": full["rt_r2_vs_train"] - d["rt_r2_vs_train"],
                              "delta_rt_mae_ms": d["rt_mae_ms"] - full["rt_mae_ms"],  # > 0: dropping costs accuracy
                              "delta_rt_rmse_ms": d["rt_rmse_ms"] - full["rt_rmse_ms"]})
        for k, c, b in coefs:
            coef_rows.append(pd.DataFrame({"repeat": rep, "fold": k, "feature": names, "coefficient_ms_per_sd": c,
                                           "intercept_ms": b}))
        preds.append(pd.DataFrame({"repeat": rep, "trial": np.arange(n), "start_time": t, "lick": y, "rt_ms": rt,
                                   "rt_pred_ms": P, "rt_baseline_ms": B, "train_mean_rt_ms": REF}))
    rm = pd.DataFrame(rep_rows)
    metrics = {"mouse_id": mouse, "session_id": session, "cohort": cohort, "n_trials": n, "n_hits_rt": int(valid.sum()),
               "rt_mean_ms": float(rt[valid].mean()), "rt_sd_ms": float(rt[valid].std()), "alpha": alpha,
               **rm.drop(columns=["repeat"]).mean().to_dict()}
    metrics["delta_rt_r2_vs_baseline"] = metrics["rt_r2_vs_train"] - metrics["baseline_rt_r2_vs_train"]
    metrics["delta_rt_mae_ms_vs_baseline"] = metrics["baseline_rt_mae_ms"] - metrics["rt_mae_ms"]  # > 0: GLM better
    metrics["delta_rt_rmse_ms_vs_baseline"] = metrics["baseline_rt_rmse_ms"] - metrics["rt_rmse_ms"]
    stem = OUT / mouse / f"{mouse}_{session}_{prefix}_"
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(f"{stem}metrics.csv", index=False)
    pd.concat(coef_rows).groupby("feature", sort=False)[["coefficient_ms_per_sd"]].mean().reset_index().to_csv(
        f"{stem}coefficients.csv", index=False)
    dm = pd.DataFrame(drop_rows).groupby("group", sort=False)[["delta_rt_r2_vs_train", "delta_rt_mae_ms",
                                                                "delta_rt_rmse_ms"]].mean().reset_index()
    dm.insert(0, "model", "Without " + dm.pop("group"))
    dm.to_csv(f"{stem}nested.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(f"{stem}predictions.parquet")
    return metrics


def lick_fit(mouse, cohort, nwb, src, c_star):
    _patch()
    return v6.fit_fixed(mouse, cohort, nwb, src, c_star)


def lick_baseline(mouse, cohort, nwb, src):
    _patch()
    return v6.baseline(mouse, cohort, nwb, src)


def loso(curves, key, score):
    per = curves.groupby(["session_id", key])[score].mean().unstack()
    return {sid: float(per.drop(index=sid).mean().idxmax()) for sid in per.index}, per.mean()


def rt_tables():
    tab = shared_behaviour.load()
    d0 = tab[(tab.behavior == "whisker") & (tab.day == 0) & (tab.trial_type == "whisker_trial")]
    out = {}
    for sid, g in d0.groupby("session_id"):
        g = g.sort_values("start_time")
        ok = (g.lick_flag == 1) & (g.early_lick == 0) & g.computed_reaction_time.between(0, 1)
        out[sid] = (g.start_time.to_numpy(float), np.where(ok, g.computed_reaction_time * 1000, np.nan))
    return out


def jaw_tables(rts):
    """User 2026-11-17: jaw-opening onset of the hits (behaviour_analysis dlc_jaw_onset_times.pkl per mouse,
    jaw_dlc_onset in s from the stimulus; dlc_helpers jaw_onset_times, regenerated 2026-11-12), in ms. Same hits as the
    RT head (valid RT 0-1 s, no early lick) and onset 0-1 s after the stimulus (pre-stimulus onsets left out)."""
    from common import axel_bisi_root
    out = {}
    for f in sorted((axel_bisi_root() / "combined_results_ks4").glob("*/dlc_jaw_onset_times.pkl")):
        d = pd.read_pickle(f)
        d = d[(d.day == 0) & (d.behavior == "whisker") & (d.trial_type == "whisker_trial") & (d.context != "passive")]
        for sid, g in d.groupby("session_id"):
            if sid not in rts:
                continue
            g = g.sort_values("start_time")
            st = g.start_time.to_numpy(float)
            jaw = g.jaw_dlc_onset.to_numpy(float) * 1000
            ok = np.isfinite(rt_for_trials(st, rts[sid])) & (jaw >= 0) & (jaw <= 1000)
            out[sid] = (st, np.where(ok, jaw, np.nan))
    return out


def jaw_head(jobs, rts, par):
    """Third head on the cached design matrices (glm_v7 stage A must have run)."""
    jaws = jaw_tables(rts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cur = [r for r in par(delayed(cached_alpha_rows)(*j, jaws.get(Path(j[2]).stem)) for j in jobs) if r is not None]
    cur = pd.concat(cur, ignore_index=True)
    cur.to_csv(OUT / "alpha_curves_jaw.csv", index=False)
    choice, mean = loso(cur, "alpha", "rt_r2_vs_train")
    pd.Series(choice, name="alpha_star").rename_axis("session_id").reset_index().to_csv(OUT / "alpha_choice_jaw.csv",
                                                                                        index=False)
    print("jaw alpha*:", pd.Series(choice).value_counts().to_dict(), "| R2 by alpha:", mean.round(4).to_dict())
    keep = [j for j in jobs if Path(j[2]).stem in choice]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = par(delayed(rt_fit)(*j, choice[Path(j[2]).stem], jaws.get(Path(j[2]).stem), "jaw") for j in keep)
    df = pd.DataFrame([r for r in rows if r])
    df.to_csv(OUT / "jaw_summary.csv", index=False)
    print(df.groupby("cohort")[["n_hits_rt", "rt_mean_ms", "rt_sd_ms", "rt_r2_vs_train", "rt_corr", "rt_mae_ms",
                                "train_mean_mae_ms", "rt_mae_gain_ms", "rt_rmse_gain_ms",
                                "delta_rt_mae_ms_vs_baseline"]].mean().round(4).T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=100)  # haas default (skill ssl-analyze 9b)
    ap.add_argument("--mice", nargs="*")
    ap.add_argument("--heads", default="lick,rt,jaw", help="jaw alone reuses the cached design matrices")
    a = ap.parse_args()
    ref = shared_behaviour.mouse_reference()
    ref = ref[ref.learning_category.isin(["good", "moderate"])]
    if a.mice:
        ref = ref[ref.mouse_id.isin(a.mice)]
    src = v5.glm_source_dir()
    rts = rt_tables()
    jobs = [(r.mouse_id, r.cohort, str(f)) for r in ref.itertuples() for f in session_nwbs(r.mouse_id)]
    OUT.mkdir(parents=True, exist_ok=True)
    par = Parallel(n_jobs=a.n_jobs, verbose=1)
    if a.heads == "jaw":
        return jaw_head(jobs, rts, par)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = par(delayed(stage_a)(*j, src, rts.get(Path(j[2]).stem)) for j in jobs)
    curves = pd.concat([l for l, _ in res if l is not None], ignore_index=True)
    acurves = pd.concat([r for _, r in res if r is not None], ignore_index=True)
    curves.to_csv(OUT / "c_curves.csv", index=False)
    acurves.to_csv(OUT / "alpha_curves.csv", index=False)
    c_choice, c_mean = loso(curves, "C", "ll_per_trial")
    a_choice, a_mean = loso(acurves, "alpha", "rt_r2_vs_train")
    pd.Series(c_choice, name="C_star").rename_axis("session_id").reset_index().to_csv(OUT / "c_choice.csv", index=False)
    pd.Series(a_choice, name="alpha_star").rename_axis("session_id").reset_index().to_csv(OUT / "alpha_choice.csv",
                                                                                          index=False)
    print("lick C*:", pd.Series(c_choice).value_counts().to_dict(), "| RT alpha*:", pd.Series(a_choice).value_counts().to_dict())
    print("mean held-out RT R2 (vs training mean) by alpha:", a_mean.round(4).to_dict())
    keep = [j for j in jobs if Path(j[2]).stem in c_choice]
    keep_rt = [j for j in jobs if Path(j[2]).stem in a_choice]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = par(delayed(lick_fit)(*j, src, c_choice[Path(j[2]).stem]) for j in keep)
        bases = par(delayed(lick_baseline)(*j, src) for j in keep)
        rt_rows = par(delayed(rt_fit)(*j, a_choice[Path(j[2]).stem], rts.get(Path(j[2]).stem)) for j in keep_rt)
    base = pd.DataFrame([b for b in bases if b])
    base.to_csv(OUT / "baseline_metrics.csv", index=False)
    df = pd.DataFrame([r for r in fits if r]).merge(base, on="session_id", how="left")
    df["C_star"] = df.session_id.map(c_choice)
    df["delta_ll_vs_baseline"] = df.test_ll_per_trial - df.baseline_test_ll_per_trial
    df["delta_auc_vs_baseline"] = df.test_auc - df.baseline_test_auc
    for r in df.itertuples():  # as glm_v6: baseline columns into each results_metrics.csv
        f = OUT / r.mouse_id / f"{r.mouse_id}_{r.session_id}_results_metrics.csv"
        if f.exists():
            m = pd.read_csv(f)
            for c in [c for c in df.columns if c.startswith(("baseline_", "delta_", "C_star"))]:
                m[c] = getattr(r, c)
            m.to_csv(f, index=False)
    df.to_csv(OUT / "all_sessions_summary.csv", index=False)
    rt = pd.DataFrame([r for r in rt_rows if r])
    rt.to_csv(OUT / "rt_summary.csv", index=False)
    json.dump({"design": "glm_v5 load_session (glm_behavior_v3 predictors), whisker trials of whisker day 0, learners",
               "splits": {"n_folds": v5.N_FOLDS, "n_repeats": v5.N_REPEATS, "gap": v5.GAP, "contiguous_blocks": True},
               "lick_head": "glm_v6: L2 logistic, C leave-one-session-out on held-out LL per trial",
               "C_grid": v6.C_GRID.tolist(),
               "rt_head": "ridge on RT (ms) of hits (no early lick, RT 0-1 s), alpha leave-one-session-out on held-out R2 "
                          "vs training-fold mean; same blocks as the lick head",
               "alpha_grid": ALPHA_GRID.tolist(), "rt_min_hits": RT_MIN_HITS,
               "rt_baseline": f"linear on causal mean RT of the previous {RT_BASELINE_K} hits",
               "n_sessions_lick": len(df), "n_sessions_rt": len(rt)}, open(OUT / "config.json", "w"), indent=1)
    print(df.groupby("cohort")[["test_auc", "test_pseudo_r2", "delta_ll_vs_baseline"]].mean().round(4).T.to_string())
    print(rt.groupby("cohort")[["n_hits_rt", "rt_r2_vs_train", "rt_r2", "rt_corr", "baseline_rt_r2_vs_train",
                                "delta_rt_r2_vs_baseline", "rt_mae_ms", "train_mean_mae_ms", "rt_mae_gain_ms",
                                "rt_rmse_gain_ms", "delta_rt_mae_ms_vs_baseline"]].mean().round(4).T.to_string())
    if "jaw" in a.heads.split(","):
        jaw_head(jobs, rts, par)


if __name__ == "__main__":
    main()
