"""Shared input glm_v6 (user 2026-11-09): glm_v5 with ONE penalty per session chosen across the other sessions, and a
'recent licking' baseline model.

Why: in glm_v5 the per-fold inner CV (about 25 trials per inner block) chose C unstably (17% of folds at the grid's
strongest end, others weak) and predicted worse than a fixed strong penalty (glm_v5_compare.py).

Stage A (C curves): for each session and each C on C_GRID, the glm_v5 block CV (5 contiguous blocks, gap 3, 3 repeats)
gives the pooled held-out log-likelihood per trial.
Stage B (leave-one-session-out C): for each session, C* = the C maximising the mean held-out log-likelihood of all OTHER
sessions (so the session's own data never chooses its penalty).
Stage C (fit): glm_v5 run_session with C fixed at C* (10 repeats; metrics, per-trial predictions, weights, shuffled
delta AUC, drop-and-refit delta log-likelihood), plus the baseline: a one-predictor logistic regression on the causal
running mean of the previous 10 whisker-trial licks, same splits, scored the same way; written to baseline_metrics.csv
and as baseline_* columns of results_metrics.csv (with delta_ll_vs_baseline, delta_auc_vs_baseline).
Outputs: Figures/_cache/glm_v6/ (glm_v5 file names) + c_curves.csv + c_choice.csv.
Run on haas (repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_v6_run.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_v5_run as v5  # noqa: E402
from common import CACHE_DIR, session_nwbs  # noqa: E402
from glm_cv_compare import block_splits  # noqa: E402
from glm_v5_compare import run_mean_feature  # noqa: E402
from shared_behaviour import mouse_reference  # noqa: E402

OUT = CACHE_DIR / "glm_v6"
C_GRID = np.logspace(-4, 2, 13)
N_REP_A = 3


def offsets_for(n, n_rep):
    rng = np.random.default_rng(7)  # the glm_v5 offsets
    return [0] + [int(rng.integers(n)) for _ in range(n_rep - 1)]


def load(mouse, nwb, src):
    glm = v5.glm_module(src)
    try:
        return v5.load_session(glm, nwb)
    except Exception:  # noqa: BLE001 - as glm_v5: reported there, skipped here
        return None


def c_curve(mouse, cohort, nwb, src):
    loaded = load(mouse, nwb, src)
    if loaded is None:
        return None
    X, y, t, names = loaded
    if len(np.unique(y)) < 2:
        return None
    n, rows = len(y), []
    for rep, off in enumerate(offsets_for(n, N_REP_A)):
        P = {c: np.full(n, np.nan) for c in C_GRID}
        thr = np.full(n, np.nan)
        for tr, te in block_splits(n, v5.N_FOLDS, offset=off, gap=v5.GAP):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            thr[te] = y[tr].mean()
            for c in C_GRID:
                P[c][te] = v5._fit(Xtr, y[tr], c).predict_proba(Xte)[:, 1]
        ok = np.isfinite(thr)
        for c in C_GRID:
            s = v5.scores(y[ok], P[c][ok], thr[ok], thr[ok])
            rows.append({"mouse_id": mouse, "session_id": Path(nwb).stem, "cohort": cohort, "repeat": rep, "C": c,
                         "ll_per_trial": s["test_ll_per_trial"], "auc": s["test_auc"]})
    return pd.DataFrame(rows)


def baseline(mouse, cohort, nwb, src):
    loaded = load(mouse, nwb, src)
    if loaded is None:
        return None
    X, y, t, names = loaded
    n, rows = len(y), []
    rm = run_mean_feature(y)[:, None]
    for rep, off in enumerate(offsets_for(n, v5.N_REPEATS)):
        P, thr = np.full(n, np.nan), np.full(n, np.nan)
        for tr, te in block_splits(n, v5.N_FOLDS, offset=off, gap=v5.GAP):
            if len(np.unique(y[tr])) < 2:
                continue
            thr[te] = y[tr].mean()
            P[te] = v5._fit(rm[tr], y[tr], 1.0).predict_proba(rm[te])[:, 1]
        ok = np.isfinite(thr)
        rows.append({"repeat": rep, **v5.scores(y[ok], P[ok], thr[ok], thr[ok])})
    m = pd.DataFrame(rows).mean(numeric_only=True)
    return {"session_id": Path(nwb).stem, **{f"baseline_{k}": float(m[k]) for k in
                                              ("test_auc", "test_balanced_accuracy", "test_ll_per_trial", "test_pseudo_r2")}}


def fit_fixed(mouse, cohort, nwb, src, c_star):
    v5.OUT = OUT
    v5.choose_C = lambda Xtr, ytr: c_star  # leave-one-session-out penalty, same for every fold and repeat
    return v5.run_session(mouse, cohort, nwb, src)


def main():
    ref = mouse_reference()
    ref = ref[ref.learning_category.isin(["good", "moderate"])]
    src = v5.glm_source_dir()
    jobs = [(r.mouse_id, r.cohort, str(f)) for r in ref.itertuples() for f in session_nwbs(r.mouse_id)]
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = pd.concat([r for r in Parallel(n_jobs=24)(delayed(c_curve)(*j, src) for j in jobs) if r is not None],
                           ignore_index=True)
    curves.to_csv(OUT / "c_curves.csv", index=False)
    per = curves.groupby(["session_id", "C"]).ll_per_trial.mean().unstack()  # sessions x C
    choice = {}
    for sid in per.index:
        others = per.drop(index=sid).mean()
        choice[sid] = float(others.idxmax())
    pd.Series(choice, name="C_star").rename_axis("session_id").reset_index().to_csv(OUT / "c_choice.csv", index=False)
    print("leave-one-session-out C*:", pd.Series(choice).value_counts().to_dict())
    print("mean held-out LL per trial by C:", per.mean().round(4).to_dict())
    keep = [j for j in jobs if Path(j[2]).stem in choice]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = Parallel(n_jobs=24)(delayed(fit_fixed)(*j, src, choice[Path(j[2]).stem]) for j in keep)
        bases = Parallel(n_jobs=24)(delayed(baseline)(*j, src) for j in keep)
    rows = [r for r in fits if r]
    base = pd.DataFrame([b for b in bases if b])
    base.to_csv(OUT / "baseline_metrics.csv", index=False)
    df = pd.DataFrame(rows).merge(base, on="session_id", how="left")
    df["C_star"] = df.session_id.map(choice)
    df["delta_ll_vs_baseline"] = df.test_ll_per_trial - df.baseline_test_ll_per_trial
    df["delta_auc_vs_baseline"] = df.test_auc - df.baseline_test_auc
    for r in df.itertuples():  # baseline columns into each session's results_metrics.csv (read by Figure 1)
        f = OUT / r.mouse_id / f"{r.mouse_id}_{r.session_id}_results_metrics.csv"
        if f.exists():
            m = pd.read_csv(f)
            for c in [c for c in df.columns if c.startswith(("baseline_", "delta_", "C_star"))]:
                m[c] = getattr(r, c)
            m.to_csv(f, index=False)
    df.to_csv(OUT / "all_sessions_summary.csv", index=False)
    json.dump({"C_grid": C_GRID.tolist(), "n_folds": v5.N_FOLDS, "n_repeats": v5.N_REPEATS, "gap": v5.GAP,
               "c_choice": "leave-one-session-out, max mean held-out LL per trial (stage A: 3 repeats)",
               "baseline": "logistic on causal running mean of previous 10 whisker-trial licks", "n_sessions": len(df)},
              open(OUT / "config.json", "w"), indent=1)
    print(df.groupby("cohort")[["test_auc", "test_balanced_accuracy", "test_ll_per_trial", "test_pseudo_r2",
                                "baseline_test_auc", "baseline_test_ll_per_trial", "delta_ll_vs_baseline",
                                "delta_auc_vs_baseline"]].mean().round(4).T.to_string())
    from scipy import stats
    for c in ("R+", "R-"):
        d = df[df.cohort == c]
        print(c, "delta LL vs baseline > 0 in", f"{np.mean(d.delta_ll_vs_baseline > 0):.0%}",
              "Wilcoxon p", f"{stats.wilcoxon(d.delta_ll_vs_baseline).pvalue:.2g}",
              "| delta AUC p", f"{stats.wilcoxon(d.delta_auc_vs_baseline).pvalue:.2g}")


if __name__ == "__main__":
    main()
