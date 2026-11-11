"""Is the nested CV of glm_v5 sound, does it help, and how good is the model? (user 2026-11-09)

Same sessions, design matrix and splits as glm_v5 (5 contiguous blocks, gap 3, 3 repeats with shifted blocks here),
held-out scores pooled per repeat then averaged (AUC relative to the training lick rate, log-likelihood per trial,
balanced accuracy at the training lick rate). Models:
  nested      : C chosen by inner 5-block CV on log-loss (glm_v5)
  C=0.01/1/100: fixed penalty, all predictors
  base_rate   : intercept only (training lick rate)
  run_mean    : one-predictor logistic regression on the causal running mean of the previous 10 whisker-trial licks
                (a 'slow state' baseline: how much do the GLM predictors add beyond recent licking?)
Also: where the inner CV puts C (fraction of folds at the grid edges), from glm_v5 coefficients_folds.
Output: Figures/_cache/glm_v5_compare/metrics.csv, c_choice.csv, summary printed.
Run on haas (repo root): OMP_NUM_THREADS=1 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_v5_compare.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, session_nwbs  # noqa: E402
import glm_v5_run as v5  # noqa: E402
from glm_cv_compare import block_splits  # noqa: E402
from shared_behaviour import mouse_reference  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = CACHE_DIR / "glm_v5_compare"
N_REP = 3


def run_mean_feature(y, k=10):
    """Causal mean of the previous k outcomes (0.5 before the first trial)."""
    c = np.r_[0, np.cumsum(y)]
    n = np.arange(len(y))
    lo = np.maximum(0, n - k)
    cnt = n - lo
    return np.where(cnt > 0, (c[n] - c[lo]) / np.maximum(cnt, 1), 0.5)


def run_session(mouse, cohort, nwb, src):
    glm = v5.glm_module(src)
    try:
        loaded = v5.load_session(glm, nwb)
    except Exception as e:  # noqa: BLE001
        return None
    if loaded is None:
        return None
    X, y, t, names = loaded
    if len(np.unique(y)) < 2:
        return None
    n = len(y)
    rm = run_mean_feature(y)[:, None]
    rng = np.random.default_rng(7)
    offsets = [0] + [int(rng.integers(n)) for _ in range(N_REP - 1)]
    models = ["nested", "C=0.01", "C=1", "C=100", "base_rate", "run_mean"]
    rows = []
    for rep, off in enumerate(offsets):
        P = {m: np.full(n, np.nan) for m in models}
        thr = np.full(n, np.nan)
        for tr, te in block_splits(n, v5.N_FOLDS, offset=off, gap=v5.GAP):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            thr[te] = y[tr].mean()
            try:
                C = v5.choose_C(Xtr, y[tr])
            except ValueError:
                C = 1.0
            for m, c in (("nested", C), ("C=0.01", 0.01), ("C=1", 1.0), ("C=100", 100.0)):
                P[m][te] = v5._fit(Xtr, y[tr], c).predict_proba(Xte)[:, 1]
            P["base_rate"][te] = y[tr].mean()
            P["run_mean"][te] = v5._fit(rm[tr], y[tr], 1.0).predict_proba(rm[te])[:, 1]
        ok = np.isfinite(thr)
        for m in models:
            s = v5.scores(y[ok], P[m][ok], thr[ok], thr[ok])
            rows.append({"mouse_id": mouse, "session_id": Path(nwb).stem, "cohort": cohort, "repeat": rep, "model": m,
                         "n_trials": n, **s})
    return pd.DataFrame(rows)


def main():
    ref = mouse_reference()
    ref = ref[ref.learning_category.isin(["good", "moderate"])]
    src = v5.glm_source_dir()
    jobs = [(r.mouse_id, r.cohort, str(f)) for r in ref.itertuples() for f in session_nwbs(r.mouse_id)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = [r for r in Parallel(n_jobs=24)(delayed(run_session)(*j, src) for j in jobs) if r is not None]
    df = pd.concat(res, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "metrics.csv", index=False)
    per = df.groupby(["session_id", "cohort", "model"])[["test_auc", "test_balanced_accuracy", "test_ll_per_trial",
                                                          "test_pseudo_r2"]].mean().reset_index()
    print(per.groupby("model")[["test_auc", "test_balanced_accuracy", "test_ll_per_trial", "test_pseudo_r2"]]
          .agg(["mean", "sem"]).round(4).to_string())
    w = per.pivot_table(index="session_id", columns="model", values="test_ll_per_trial")
    from scipy import stats
    for m in ["C=0.01", "C=1", "C=100", "base_rate", "run_mean"]:
        d = (w["nested"] - w[m]).dropna()
        print(f"nested - {m}: LL/trial {d.mean():+.4f} (median {d.median():+.4f}), nested better in {np.mean(d > 0):.0%}, "
              f"Wilcoxon p={stats.wilcoxon(d).pvalue:.2g}")
    cf = []
    for f in (CACHE_DIR / "glm_v5").glob("*/*_coefficients_folds.csv"):
        c = pd.read_csv(f, usecols=["repeat", "fold", "C"]).drop_duplicates()
        cf.append(c.assign(session=f.name))
    cf = pd.concat(cf)
    grid = v5.C_VALUES
    cf.to_csv(OUT / "c_choice.csv", index=False)
    print("C chosen (log10) quantiles:", np.round(np.log10(cf.C).quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy(), 2),
          f"; at lowest grid value {np.mean(np.isclose(cf.C, grid[0])):.0%}, at highest {np.mean(np.isclose(cf.C, grid[-1])):.0%}")


if __name__ == "__main__":
    main()
