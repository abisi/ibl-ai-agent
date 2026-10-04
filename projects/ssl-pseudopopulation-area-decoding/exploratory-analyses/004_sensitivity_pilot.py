"""004 -- Sensitivity pilot (user request 2026-09-29): hit/miss, R+, whole brain; neurons per session (5 vs 20) x
training pseudo-trials per class (20 vs 100); everything else as 002 (10 mice, 3-fold CV, T_test 100, 10 paired
shifts, 100 iterations -- PILOT counts). The n5/T20 configuration is pilot 2 (002_pseudo_hitmiss_R+.parquet).
Per configuration: real and paired-null mean curves with 2.5-97.5% bands, the 5th percentile of d = real - null per
bin (above chance when > 0), onset (4 of 5 bins), spread of real across iterations (median width of the 2.5-97.5%
band after stimulus onset) and sensory-window d.
Outputs: ../artifacts/004_sensitivity_hitmiss_R+.csv, figures/004_sensitivity_hitmiss_R+.pdf/.png
Run (repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/004_sensitivity_pilot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"
CONFIGS = [("n5_t20", ""), ("n5_t100", "_sens_n5_t100"), ("n20_t20", "_sens_n20_t20"), ("n20_t100", "_sens_n20_t100")]


def onset(above, t, k=5, m=4):
    for i in range(len(above) - k + 1):
        if above[i:i + k].sum() >= m:
            return t[i]
    return np.nan


def main():
    import ssl_timeresolved_decoding as T
    t = np.array([e[1] for e in T.causal_bin_edges((-0.2, 0.6), bin_width=0.05, stride=0.005)])
    post = t > 0
    sens = (t >= 0.005) & (t <= 0.05)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), constrained_layout=True, sharey="row")
    rows = []
    for j, (lab, suf) in enumerate(CONFIGS):
        f = ART / f"002_pseudo_hitmiss_R+{suf}.parquet"
        ax, ax2 = axes[0, j], axes[1, j]
        if not f.exists():
            ax.set_title(f"{lab}: missing")
            continue
        d = pd.read_parquet(f)
        d = d[d.skipped_reason.isna() & (d.area == "All units")]
        R = np.stack(d.curve.map(np.asarray).to_numpy())
        N = np.stack(d.null_mean_curve.map(np.asarray).to_numpy())
        D = R - N
        for M, c, name in ((N, "#777777", "paired null"), (R, "#00B400", "real")):
            ax.fill_between(t * 1000, np.percentile(M, 2.5, 0), np.percentile(M, 97.5, 0), color=c, alpha=0.25, lw=0)
            ax.plot(t * 1000, M.mean(0), color=c, lw=1.2, label=name)
        above = np.percentile(D, 5, 0) > 0
        on = onset(above, t)
        ax.plot(np.where(above, t * 1000, np.nan), np.full(len(t), 1.02), color="k", lw=2)
        if np.isfinite(on):
            ax.axvline(on * 1000, color="k", ls=":", lw=0.8)
        ax.set_title(f"{lab} (n={len(R)})", fontsize=9)
        ax.set_ylim(0.35, 1.05)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0.5, color="#999999", lw=0.5, ls="--")
        ax2.fill_between(t * 1000, np.percentile(D, 5, 0), np.percentile(D, 95, 0), color="#555555", alpha=0.25, lw=0)
        ax2.plot(t * 1000, D.mean(0), color="k", lw=1.1)
        ax2.plot(t * 1000, np.percentile(D, 5, 0), color="#c0392b", lw=0.8)
        ax2.axhline(0, color="#999999", lw=0.5, ls="--")
        ax2.axvline(0, color="k", lw=0.5)
        ax2.set_xlabel("time from stimulus (ms)")
        dw = D[:, sens].mean(1)
        rows.append(dict(config=lab, n_iterations=len(R), n_neurons=int(d.n_neurons.iloc[0]), t_train=int(d.t_train.iloc[0]),
                         peak=R.mean(0).max(), onset_ms=on * 1000, frac_bins_above=above.mean(),
                         real_band_width_post=float(np.median(np.percentile(R, 97.5, 0)[post] - np.percentile(R, 2.5, 0)[post])),
                         d_sensory=dw.mean(), d5_sensory=np.percentile(dw, 5), p_sensory=float((dw <= 0).mean()),
                         null_mean=N.mean()))
    axes[0, 0].set_ylabel("balanced accuracy")
    axes[1, 0].set_ylabel("real - paired null\n(mean, 5-95%; red: 5th pct)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Sensitivity pilot: hit/miss, R+, whole brain, learning day -- neurons per session x training pseudo-trials "
                 "per class (10 mice, T_test 100, 10 paired shifts, 100 iterations: PILOT counts)", fontsize=9)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"004_sensitivity_hitmiss_R+.{ext}", dpi=200)
    S = pd.DataFrame(rows)
    S.to_csv(ART / "004_sensitivity_hitmiss_R+.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
