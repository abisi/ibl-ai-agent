"""009b -- Two-sided mouse-level cohort-permutation p-values from the 009 window-permutation outputs (whole brain, abstract
windows), written to artifacts/009_perm_wb_abstract_pvalues.csv (the table read by 006 / 013-015).
Statistic per target x window: diff = d(R+) - d(R-) (d = accuracy - linear-shift null, averaged over iterations); null =
the same with R+ / R- labels permuted at the mouse level; p = (#|null| >= |observed| + 1) / (n_perm + 1).
2026-10-02: the pre-lick window changed from -150..0 to -100..0 ms (user); modality_lick is taken from the rerun
(009_cohort_perm_windows_wb_abstract_pl100.parquet), hit/miss from the original run (windows unchanged). The previous table is
kept as 009_perm_wb_abstract_pvalues_pl150.csv.
Run (haas, repo root): python .../009b_perm_pvalues.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"


def pvals(d):
    rows = []
    for (t, w), g in d.groupby(["target", "window"]):
        obs = g[g.labels != "permuted at mouse level"]
        nul = g[g.labels == "permuted at mouse level"]
        if not len(obs):
            continue
        o = obs.iloc[0]
        diff = o.d_a - o.d_b
        nd = (nul.d_a - nul.d_b).to_numpy()
        rows.append(dict(target=t, window=w, n_rplus=int(o.n_a), n_rminus=int(o.n_b), d_rplus=o.d_a, d_rminus=o.d_b, diff=diff,
                         null_sd=float(nd.std()), n_perm=len(nd), p_perm=float((np.sum(np.abs(nd) >= abs(diff)) + 1) / (len(nd) + 1)),
                         frac_null_ge=float(np.mean(nd >= diff))))
    return pd.DataFrame(rows)


def main():
    old = ART / "009_perm_wb_abstract_pvalues.csv"
    if old.exists() and not (ART / "009_perm_wb_abstract_pvalues_pl150.csv").exists():
        shutil.copy(old, ART / "009_perm_wb_abstract_pvalues_pl150.csv")
    a = pd.read_parquet(ART / "009_cohort_perm_windows_wb_abstract.parquet")
    b = pd.read_parquet(ART / "009_cohort_perm_windows_wb_abstract_pl100.parquet")
    d = pd.concat([a[a.target == "hitmiss"], b[b.target == "modality_lick"]], ignore_index=True)
    P = pvals(d)
    P.to_csv(old, index=False)
    print(P.to_string(index=False))


if __name__ == "__main__":
    main()
