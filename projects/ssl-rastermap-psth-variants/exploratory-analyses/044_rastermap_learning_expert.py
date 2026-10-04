"""Rastermap variant combining LEARNING (day 0) and EXPERT (day > 0) sessions of ALL mice (standard settings).

Settings as the long-window lick-fixed runs: --qc good (default) or good_mua units, all learning categories, passive conditions included,
reward time bins included, locality 0.25, grid_upsample 10, clusters = rastermap k-means nodes, corrected lick time,
config.yaml windows (long). Tables loaded with day_to_analyze='all' (separate cache _cache/tables_all_days.pkl).
Units of a session lacking a condition get NaN PSTHs and are dropped by the build (finite-row filter).

Learning vs expert representation (session = unit of analysis; a mouse can contribute a learning and expert sessions):
    F_sk  = fraction of session s's neurons in cluster k
    per cluster: learning vs expert sessions, Mann-Whitney U AND Welch t, BH-FDR across clusters (significant = both)
    PERMANOVA (Euclidean, pseudo-F) of the session F profiles by stage, 9999 label permutations
    -> stats/stage_f_matrix.npz, stats/stage_cluster_stats.csv, stats/stage_permanova.txt
The matrix summary then adds a 'Learning share' column (learning share of mean F_sk; filled = significant) next to
the R+ share column (population_matrix_summary picks the stage files up automatically).
Output: rastermap_variants/qc_<qc>__learncat_all__loc0p25_grid10_nodes__lickfix_longwin__learning_expert/
"""
import argparse
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_ap = argparse.ArgumentParser()
_ap.add_argument("--qc", default="good", choices=["good", "good_mua"])
_args = _ap.parse_args()
sys.argv = [sys.argv[0], "--src", f"qc_{_args.qc}__learncat_all", "--locality", "0.25"]
g10 = importlib.import_module("006_grid10_node_clusters")        # installs the k-means-node fit_rastermap wrapper
drv = g10.drv
import rastermap_psth.rastermap_utils as rmu                     # noqa: E402
import rastermap_psth.run_clustering_new as rcn                  # noqa: E402
import rastermap_psth.build_feature_matrix as bfm                # noqa: E402
from rastermap_psth.population_matrix_summary import run_matrix_summary  # noqa: E402

NAME = f"qc_{_args.qc}__learncat_all__loc0p25_grid10_nodes__lickfix_longwin__learning_expert"
QC = ["good"] if _args.qc == "good" else ["good", "mua"]
OVERRIDES = dict(unit_quality_label=QC, correct_lick_time=True)
N_PERM = 9999
MIN_NEURONS_SESSION = 10


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def permanova(X, groups, n_perm, rng):
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    n = len(X); labs = np.unique(groups)

    def pseudo_f(g):
        ss_t = D[np.triu_indices(n, 1)].sum() / n
        ss_w = sum(D[np.ix_(g == l, g == l)][np.triu_indices((g == l).sum(), 1)].sum() / (g == l).sum() for l in labs)
        return ((ss_t - ss_w) / (len(labs) - 1)) / (ss_w / (n - len(labs)))
    f0 = pseudo_f(groups)
    null = np.array([pseudo_f(rng.permutation(groups)) for _ in range(n_perm)])
    return f0, (1 + np.sum(null >= f0)) / (n_perm + 1)


def stage_stats(md, unit_table):
    lab = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    K = int(lab.cluster_label.max()) + 1
    day = unit_table.groupby("session_id")["day"].first()
    lab["day"] = lab.session_id.map(day)
    lab["stage"] = np.where(lab.day == 0, "learning", "expert")
    n_s = lab.groupby("session_id").size()
    keep = n_s[n_s >= MIN_NEURONS_SESSION].index
    lab = lab[lab.session_id.isin(keep)]
    F = pd.crosstab(lab.session_id, lab.cluster_label, normalize="index").reindex(columns=range(K), fill_value=0.0)
    meta = lab.groupby("session_id").agg(stage=("stage", "first"), mouse_id=("mouse_id", "first"),
                                         reward_group=("reward_group", "first"), day=("day", "first"),
                                         n_neurons=("unit_ids", "size")).reindex(F.index)
    g = meta.stage.to_numpy()
    rows = []
    for k in range(K):
        a, b = F.loc[g == "learning", k], F.loc[g == "expert", k]
        rows.append(dict(cluster=k + 1, mean_f_learning=a.mean(), mean_f_expert=b.mean(),
                         learning_share=a.mean() / (a.mean() + b.mean()) if a.mean() + b.mean() > 0 else np.nan,
                         p_mannwhitney=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
                         p_welch_t=stats.ttest_ind(a, b, equal_var=False).pvalue))
    S = pd.DataFrame(rows)
    S["p_mannwhitney_fdr"], S["p_welch_t_fdr"] = bh(S.p_mannwhitney), bh(S.p_welch_t)
    S["significant_both"] = (S.p_mannwhitney_fdr < 0.05) & (S.p_welch_t_fdr < 0.05)
    out = md / "stats"; out.mkdir(exist_ok=True)
    S.to_csv(out / "stage_cluster_stats.csv", index=False)
    np.savez(out / "stage_f_matrix.npz", f_matrix=F.to_numpy(), stage=g, session_ids=F.index.to_numpy(),
             mouse_ids=meta.mouse_id.to_numpy(), reward_groups=meta.reward_group.to_numpy(), days=meta.day.to_numpy())
    meta.to_csv(out / "stage_sessions.csv")
    f0, p = permanova(F.to_numpy(), g, N_PERM, np.random.default_rng(0))
    both_mice = meta.groupby("mouse_id").stage.nunique()
    txt = (f"Learning vs expert sessions: PERMANOVA (Euclidean on session cluster-fraction profiles F_sk)\n"
           f"F-statistic : {f0:.4f}\np-value     : {p:.4g}\npermutations: {N_PERM} (unrestricted session labels)\n"
           f"sessions    : {int((g == 'learning').sum())} learning, {int((g == 'expert').sum())} expert "
           f"(>= {MIN_NEURONS_SESSION} neurons)\n"
           f"mice        : {meta.mouse_id.nunique()} ({int((both_mice == 2).sum())} contribute both stages)\n"
           f"clusters significant (MWU & Welch FDR<0.05): {S.significant_both.sum()} "
           f"(learning-enriched {[int(c) for c in S[S.significant_both & (S.learning_share > .5)].cluster]}, "
           f"expert-enriched {[int(c) for c in S[S.significant_both & (S.learning_share <= .5)].cluster]})\n")
    (out / "stage_permanova.txt").write_text(txt)
    print(txt, flush=True)
    return S


if __name__ == "__main__":
    t0 = time.time()
    drv.DAY_TO_ANALYZE = "all"
    drv.CACHE = drv.VARIANTS_ROOT / "_cache" / "tables_all_days.pkl"
    tables = drv.load_tables()
    # the all-days cache contains exact duplicate rows (2-9 copies per unit in 33 sessions, found 2026-10-02):
    # drop identical duplicates so each neuron / trial / lick enters once
    keys = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
    ut = tables["unit_table"]
    ut = ut.assign(_cid=ut.cluster_id.astype(str))
    dup_u = ut.duplicated(subset=[k for k in keys if k != "cluster_id"] + ["_cid"])
    print(f"unit_table: dropping {int(dup_u.sum())} duplicate rows of {len(ut)}", flush=True)
    ut = ut[~dup_u].drop(columns="_cid")
    for name in ("trial_table", "lick_df"):
        tb = tables[name]
        cols = [c for c in tb.columns if tb[c].map(lambda x: isinstance(x, (list, np.ndarray))).sum() == 0]
        dup = tb.duplicated(subset=cols)
        print(f"{name}: dropping {int(dup.sum())} duplicate rows of {len(tb)}", flush=True)
        tables[name] = tb[~dup]
    assert "day" in ut.columns, "unit_table has no 'day' column"
    print("units per day:", ut.groupby("day").size().to_dict(), "| sessions per day:",
          ut.groupby("day").session_id.nunique().to_dict(), flush=True)
    vroot = drv.VARIANTS_ROOT / NAME
    vroot.mkdir(parents=True, exist_ok=True)
    rmu.DEFAULT_CFG["locality"] = 0.25
    bfm.MOUSE_INFO = drv.mouse_info_for("all", vroot)
    print(f"=== {NAME}: build feature matrix with {OVERRIDES} ===", flush=True)
    res = bfm.run_build_feature_matrix(ut.copy(), tables["trial_table"].copy(), lick_df=tables["lick_df"].copy(),
                                       config_path=drv.CONFIG_PATH, out_root=vroot, **OVERRIDES)
    g10.CAPTURE.clear()
    rres = rcn.run_rastermap(data_folder=res["data_folder"], config_path=drv.CONFIG_PATH, **OVERRIDES)
    md = pathlib.Path(rres["method_dir"])
    d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    assert np.array_equal(d["cluster_labels"].astype(int), g10.CAPTURE[-1]["node"]), "labels != k-means nodes"
    cfg = drv.load_cfg(drv.CONFIG_PATH, **OVERRIDES)
    drv.set_conditions(cfg)
    drv.fig_flow_umap_cv(md, cfg)
    drv.run_stats_only(md, cfg, ut, tables["trial_table"])
    drv.fig_cluster_mean_matrix_with_fmk(md, cfg, NAME)
    S = stage_stats(md, ut)
    run_matrix_summary(md, drv.CONFIG_PATH, OVERRIDES, title_prefix=f"{NAME}: ")
    json.dump(dict(variant=NAME, config_overrides=OVERRIDES, locality=0.25, grid_upsample=10, learning_category="all",
                   day_to_analyze="all", n_neurons=int(len(d["unit_ids"])), method_dir=str(md),
                   stage_stats=dict(n_sig_both=int(S.significant_both.sum()), min_neurons_per_session=MIN_NEURONS_SESSION,
                                    tests="session-level F_sk: Mann-Whitney U and Welch t, BH-FDR; PERMANOVA 9999"),
                   runtime_min=round((time.time() - t0) / 60, 1)),
              open(vroot / "variant_provenance.json", "w"), indent=2)
    print("ALL DONE", flush=True)
