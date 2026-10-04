"""109 -- Classifier comparison for hit/miss decoding (user 2026-09-26: "Would SVM with RBF be better?",
"Explain shrinkage LDA", "Test on sessions").

All learning-stage sessions (whole brain, all whisker trials, disengaged kept, no subsampling), windows
sensory (5-50 ms) and sensory minus baseline. For each session x window: N_REP repeats of stratified
n-fold CV (n = min(5, minority count)); within a repeat ALL models see the SAME folds (paired comparison).
Every hyperparameter is chosen INSIDE the training fold (inner 3-fold stratified CV on balanced accuracy),
so no model gets test-fold information. Class balance: class_weight="balanced" / LDA priors 0.5-0.5.
Models (all behind a per-fold StandardScaler, as in the main pipeline):
  logreg      L2 logistic, C in C_GRID (nested)
  slda        LDA, lsqr solver, Ledoit-Wolf shrinkage (no tuning), priors 0.5/0.5
  pca_logreg  PCA (min(20, n_train-1) comps, fit in fold) + L2 logistic, C nested
  rbf_svm     SVC(rbf), C in {0.1, 1, 10, 100} x gamma in {0.1, 1, 10}/n_features (nested)
  logreg_pipe the current pipeline convention: L2 logistic WITHOUT class weights, one C per session from
              select_fixed_c_pooled on all trials (reference only; not nested)
Metrics: balanced accuracy of pooled held-out predictions per repeat -> mean and SD across repeats;
fit time. Stats: each model vs logreg, paired over sessions (Wilcoxon signed-rank + paired t), per window.
Outputs: 109_classifier_comparison.parquet (session x window x model, full provenance), 109_classifier_stats.csv,
figures/whole_brain/learning/109_classifier_comparison.png/.pdf
Run: python 109_classifier_comparison.py [plot]
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

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS)
OUT = Path(__file__).resolve().parent
OUT_PATH = OUT / "109_classifier_comparison.parquet"
LT_TABLE = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "020_lt_eval.csv"
EXAMPLES = ["AB119", "AB125", "MH029", "AB085", "MH018", "AB120", "MH030", "AB159"]
WINDOWS = {"sensory": (0.005, 0.050), "baseline": (-0.200, -0.010)}
WIN_USE = ["sensory", "sensory_minus_base"]
MODELS = ["logreg", "slda", "pca_logreg", "rbf_svm", "logreg_pipe"]
N_REP = 5
DZ = (-0.010, 0.005)
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "24"))
ONLY_EXAMPLES = os.environ.get("SSL_ONLY_EXAMPLES") == "1"  # run / summarize the 8 example sessions only


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def build(name, n_feat, n_train, C_pipe):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from ssl_timeresolved_decoding import C_GRID
    inner = StratifiedKFold(3, shuffle=True, random_state=0)

    def gs(pipe, grid):
        return GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy", n_jobs=1)
    lr = LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=1000)
    if name == "logreg":
        return gs(make_pipeline(StandardScaler(), lr), {"logisticregression__C": list(C_GRID)})
    if name == "slda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]))
    if name == "pca_logreg":
        k = int(min(20, n_train - 1, n_feat))
        return gs(make_pipeline(StandardScaler(), PCA(k), lr), {"logisticregression__C": list(C_GRID)})
    if name == "rbf_svm":
        return gs(make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
                  {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [g / n_feat for g in (0.1, 1, 10)]})
    if name == "logreg_pipe":
        return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear", C=C_pipe, max_iter=1000))
    raise ValueError(name)


def process(args):
    sid, subject_id, rg, lcat = args
    sys.path.insert(0, SCRIPTS)
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes, prep_hitmiss_trials,
        select_fixed_c_pooled, sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    tab = pd.read_csv(LT_TABLE).set_index("session_id")
    base = dict(session_id=sid, mouse_id=subject_id, reward_group=rg, learning_category=lcat,
                group=tab["group"].get(sid, None), is_example=sid[:5] in EXAMPLES)
    tr = prep_hitmiss_trials(root, sid, st, tt)
    if tr is None:
        return [dict(base, skipped_reason="no usable trials")]
    y = tr["lick_flag"].to_numpy().astype(bool)
    n_min = int(min(y.sum(), (~y).sum()))
    if n_min < 6:
        return [dict(base, n_trials=len(y), n_hits=int(y.sum()), skipped_reason=f"minority class {n_min} < 6")]
    units = area_units(sid, "whole_brain", "All units", labels)
    mats = sliding_bin_population_matrices(load_session_unit_spikes(root, sid), units, tr["start_time"].to_numpy(),
                                           np.ones(len(y), bool), [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=DZ)
    feats = {"sensory": mats[0], "sensory_minus_base": mats[0] - mats[1]}
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    nf = min(5, n_min)
    rows = []
    for w in WIN_USE:
        X = feats[w]
        C_pipe = select_fixed_c_pooled(X, y, rng, n_folds=5)
        scores = {m: [] for m in MODELS}
        tfit = {m: 0.0 for m in MODELS}
        for _ in range(N_REP):
            splits = list(StratifiedKFold(nf, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, y))
            for m in MODELS:
                pred = np.empty(len(y), bool)
                for trn, te in splits:
                    t0 = time.time()
                    clf = build(m, X.shape[1], len(trn), C_pipe).fit(X[trn], y[trn])
                    pred[te] = clf.predict(X[te])
                    tfit[m] += time.time() - t0
                scores[m].append(balanced_accuracy_score(y, pred))
        for m in MODELS:
            rows.append(dict(base, window=w, model=m, bal_acc=float(np.mean(scores[m])), bal_acc_sd=float(np.std(scores[m])),
                             bal_acc_reps=list(map(float, scores[m])), fit_s=tfit[m], n_trials=len(y), n_hits=int(y.sum()),
                             n_units=len(units), n_folds=nf, n_rep=N_REP, C_pipe=C_pipe, skipped_reason=None))
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    sess = hitmiss_session_list(pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    if ONLY_EXAMPLES:
        todo = todo[todo.session_id.str[:5].isin(EXAMPLES)]
    print(f"[109] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, (r.session_id, r.subject_id, r.reward_group, r.learning_category)): r.session_id
                for r in todo.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            new = pd.DataFrame(f.result())
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            print(f"[{i}/{len(futs)}] {futs[f]} {new.skipped_reason.iloc[0] or 'ok'} {time.time() - t0:.0f}s", flush=True)
    summarize()


def summarize():
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, wilcoxon
    _s = importlib.util.spec_from_file_location("q034", OUT / "034_area_window_quant_grid.py")
    q034 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(q034)
    d = pd.read_parquet(OUT_PATH)
    print("skipped:", d[d.skipped_reason.notna()].groupby("skipped_reason").size().to_dict())
    d = d[d.skipped_reason.isna()]
    if ONLY_EXAMPLES:
        d = d[d.is_example]
    col = {"R+": "#00B400", "R-": "#C800C8"}
    lab = {"logreg": "L2 logistic\n(nested C)", "slda": "shrinkage LDA\n(Ledoit-Wolf)", "pca_logreg": "PCA(20) +\nlogistic",
           "rbf_svm": "RBF-SVM\n(nested C, gamma)", "logreg_pipe": "current pipeline\n(C on all trials,\nno class weights)"}
    rows = []
    fig, axes = plt.subplots(len(WIN_USE), 3, figsize=(19, 5.6 * len(WIN_USE)), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.5, 1.5, 0.8]))
    for r, w in enumerate(WIN_USE):
        p = d[d.window == w].pivot_table(index=["session_id", "reward_group", "is_example"], columns="model", values="bal_acc").reset_index()
        sd = d[d.window == w].pivot_table(index="session_id", columns="model", values="bal_acc_sd")
        ft = d[d.window == w].groupby("model").fit_s.median()
        # A: absolute balanced accuracy per model
        ax = axes[r, 0]
        for k, m in enumerate(MODELS):
            v = p[m].to_numpy()
            ax.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=10, lw=0, alpha=0.6,
                       c=[col[g] for g in p.reward_group])
            ax.errorbar(k + 0.3, v.mean(), yerr=v.std() / np.sqrt(len(v)), color="k", marker="o", capsize=3)
            ax.text(k + 0.3, 1.03, f"{v.mean():.3f}", ha="center", fontsize=8)
        for _, row in p[p.is_example].iterrows():
            ax.plot(range(len(MODELS)), row[MODELS].to_numpy(float), color="#555555", lw=0.6, alpha=0.6)
        ax.axhline(0.5, color="#888888", ls=":")
        ax.set_xticks(range(len(MODELS)), [lab[m] for m in MODELS], fontsize=8)
        ax.set_ylim(0.25, 1.07)
        ax.set_ylabel(f"{w}\nbalanced accuracy (held-out)")
        ax.set_title(f"Per-session balanced accuracy (n = {len(p)} sessions; dots R+ green / R- magenta; grey lines = 8 examples)",
                     fontsize=9)
        # B: paired difference vs logreg
        ax = axes[r, 1]
        for k, m in enumerate([m for m in MODELS if m != "logreg"]):
            dd = (p[m] - p["logreg"]).to_numpy()
            pw, pt = wilcoxon(dd).pvalue, ttest_rel(p[m], p["logreg"]).pvalue
            dsd = (sd[m] - sd["logreg"]).dropna().to_numpy()
            rows.append(dict(window=w, model=m, vs="logreg", n=len(dd), mean_diff=dd.mean(), median_diff=np.median(dd),
                             frac_better=(dd > 0).mean(), p_wilcoxon=pw, p_paired_t=pt, mean_acc=p[m].mean(),
                             mean_acc_logreg=p["logreg"].mean(), median_sd_diff=np.median(dsd), median_fit_s=ft[m]))
            ax.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(dd)), dd, s=10, lw=0, alpha=0.6,
                       c=[col[g] for g in p.reward_group])
            ax.errorbar(k + 0.3, dd.mean(), yerr=dd.std() / np.sqrt(len(dd)), color="k", marker="o", capsize=3)
            ax.text(k, ax.get_ylim()[1] if k else 0.0, "", fontsize=1)
            ax.annotate(f"mean {dd.mean():+.3f}\nbetter in {100 * (dd > 0).mean():.0f}%\nW p={pw:.2g}, t p={pt:.2g}",
                        (k, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7.5)
        ax.axhline(0, color="#888888", ls=":")
        ax.set_xticks(range(len(MODELS) - 1), [lab[m] for m in MODELS if m != "logreg"], fontsize=8)
        lim = np.nanmax(np.abs(ax.get_ylim()))
        ax.set_ylim(-lim, lim * 1.6)
        ax.set_ylabel("difference vs L2 logistic (nested C)")
        ax.set_title("Paired difference vs L2 logistic (same folds); Wilcoxon signed-rank + paired t", fontsize=9)
        # C: stability (SD across CV repeats) and fit time
        ax = axes[r, 2]
        med = [np.median(sd[m].dropna()) for m in MODELS]
        ax.barh(range(len(MODELS)), med, color="#7f7f7f")
        for k, m in enumerate(MODELS):
            ax.text(med[k], k, f" {med[k]:.3f}  ({ft[m]:.1f}s fit)", va="center", fontsize=7.5)
        ax.set_yticks(range(len(MODELS)), [lab[m].replace("\n", " ") for m in MODELS], fontsize=7.5)
        ax.set_xlim(0, max(med) * 1.9)
        ax.set_xlabel("median SD of balanced accuracy\nacross CV repeats (lower = more stable)")
        ax.set_title("Stability and total fit time per session", fontsize=9)
    fig.suptitle("Hit/miss decoding classifier comparison (whole brain, learning-stage sessions, all trials, class-balanced, "
                 f"{N_REP} x stratified CV with identical folds per model, hyperparameters nested in the training fold)", fontsize=11)
    q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"109_classifier_comparison{'_examples' if ONLY_EXAMPLES else ''}.png", dpi=170, bbox_inches="tight")
    stt = pd.DataFrame(rows)
    stt.to_csv(OUT / f"109_classifier_stats{'_examples' if ONLY_EXAMPLES else ''}.csv", index=False)
    pd.set_option("display.width", 250)
    print(stt.round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        summarize()
    else:
        main()
