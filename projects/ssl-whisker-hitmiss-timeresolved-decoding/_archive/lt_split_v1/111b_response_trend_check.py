"""111b -- Diagnostic for 111 (coupling v2): is the within-class decline of the neural score over the session a
change of the whisker response itself? Per learning-stage session (whole brain, whisker trials), population mean
rate in the sensory window (5-50 ms), in the baseline window (-200..-10 ms) and their difference, averaged over
units after per-unit z-scoring across trials; Spearman correlation with trial index WITHIN hits and within misses.
Group tests: vs 0 (Wilcoxon | one-sample t) per cohort x learner group.
Output: 111b_response_trend.parquet, printed summary.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
LT_TABLE = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "020_lt_eval.csv"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def process(args):
    sid, subject_id, rg = args
    sys.path.insert(0, SCRIPTS)
    from scipy.stats import spearmanr
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes, prep_hitmiss_trials,
        sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    tr = prep_hitmiss_trials(root, sid, st, tt)
    if tr is None:
        return []
    y = tr["lick_flag"].to_numpy().astype(bool)
    units = area_units(sid, "whole_brain", "All units", add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH)))
    S, B = sliding_bin_population_matrices(load_session_unit_spikes(root, sid), units, tr["start_time"].to_numpy(),
                                           np.ones(len(y), bool), [(0.005, 0.050), (-0.200, -0.010)], dead_zone=(-0.010, 0.005))
    zs = lambda M: np.nanmean((M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-9), 1)  # noqa: E731
    feats = {"sensory": zs(S), "baseline": zs(B), "sensory_minus_base": zs(S - B)}
    idx = np.arange(len(y))
    row = dict(session_id=sid, mouse_id=subject_id, reward_group=rg, n_trials=len(y), n_hits=int(y.sum()), n_units=len(units))
    for f, v in feats.items():
        for lab, m in (("hit", y), ("miss", ~y)):
            row[f"rho_{f}_{lab}"] = spearmanr(v[m], idx[m]).correlation if m.sum() >= 8 else np.nan
    return [row]


def main():
    from scipy.stats import ttest_1samp, wilcoxon
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    sess = hitmiss_session_list(pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"]
    with ProcessPoolExecutor(24, initializer=_init) as ex:
        res = list(ex.map(process, [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples()]))
    d = pd.DataFrame([r for rr in res for r in rr])
    tab = pd.read_csv(LT_TABLE)[["session_id", "group"]]
    d = d.merge(tab, on="session_id", how="left")
    d.to_parquet(OUT / "111b_response_trend.parquet", index=False)
    rows = []
    for (rg, grp), g in d.groupby(["reward_group", "group"]):
        for c in [c for c in d.columns if c.startswith("rho_")]:
            v = g[c].dropna()
            if len(v) >= 3:
                rows.append(dict(group=f"{rg} {grp}", measure=c[4:], n=len(v), mean_rho=v.mean(), p_wilcoxon=wilcoxon(v).pvalue,
                                 p_t=ttest_1samp(v, 0).pvalue))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
