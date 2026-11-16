"""Positive / negative control on synthetic spikes: do RRR, PCA->CCA and ensemble Granger recover a known A -> B
coupling (20-ms lag) and its direction, stay at the trial-shuffle null without coupling, and ignore a shared
stimulus-locked PSTH (removed by the condition-mean residuals)?

Simulation: 30 trials x 160 bins of 10 ms; area A latent = AR(1) fluctuation; area B latent = c x A latent lagged by
2 bins + its own AR(1); both areas also share a strong evoked transient at t = 0 (identical on every trial: must not
create communication). 20 Poisson units per area with random loadings. Coupling c in {0, 0.25, 0.5, 1}.
Output: combined_results_ks4/ssl-communication-methods-pilot/simulation/{simulation.csv, simulation.png}
Run (haas): OMP_NUM_THREADS=1 python projects/ssl-communication-methods-pilot/001_simulation_validation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402

OUT = cl.OUT / "simulation"


def simulate(c, rng, n_trials=30, n_bins=160, n_units=20, lag=2):
    def ar1(shape, phi=0.8):
        x = np.zeros(shape)
        e = rng.normal(size=shape)
        for t in range(1, shape[1]):
            x[:, t] = phi * x[:, t - 1] + e[:, t]
        return x / x.std()
    la = ar1((n_trials, n_bins))
    lb = 0.7 * ar1((n_trials, n_bins))
    lb[:, lag:] += c * la[:, :-lag]
    evoked = np.exp(-0.5 * ((np.arange(n_bins) - 50) / 3) ** 2) * 2.0  # shared PSTH at bin 50, same every trial
    def spikes(lat):
        w = rng.normal(0.6, 0.3, n_units)
        base = np.log(rng.uniform(5, 15, n_units) * 0.01)
        rate = np.exp(base[None, None, :] + w[None, None, :] * 0.5 * lat[:, :, None] + evoked[None, :, None])
        return rng.poisson(rate).astype(float)
    return spikes(la), spikes(lb)


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [0.0, 0.25, 0.5, 1.0]:
        for rep in range(5):
            fa, fb = simulate(c, rng)
            cond = np.array(["x"] * len(fa))
            # one 100-ms window (10 bins) around the evoked transient, residualised; samples = trial x bin
            wa = cl.residualise(fa[:, 45:55, :], cond)
            wb = cl.residualise(fb[:, 45:55, :], cond)
            perm = rng.permutation(len(wa))
            r = cl.rrr_cv(wa, wb, rng)
            rs = cl.rrr_cv(wa[perm], wb, rng)
            cc = cl.pca_cca_cv(wa, wb, rng)
            ccs = cl.pca_cca_cv(wa[perm], wb, rng)
            za = cl.population_pcs(fa, cond, 3)
            zb = cl.population_pcs(fb, cond, 3)
            t_index = np.arange(20, 140)
            ab, ba = cl.granger_ensemble(za, zb, 3, t_index)
            ab_r, ba_r = cl.granger_ensemble(za[:, ::-1], zb[:, ::-1], 3, t_index)
            null = [cl.granger_ensemble(za[rng.permutation(len(za))], zb, 3, t_index)[0] for _ in range(20)]
            rows.append(dict(coupling=c, rep=rep, rrr=r["perf_rrr"], rrr_shuffle=rs["perf_rrr"], rank=r["rank"],
                             cca=cc["cc1"], cca_shuffle=ccs["cc1"], gc_ab=ab, gc_ba=ba, gc_ab_rev=ab_r, gc_ba_rev=ba_r,
                             gc_null=np.mean(null), net_trgc=(ab - ba) - (ab_r - ba_r)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation.csv", index=False)
    g = df.groupby("coupling").agg(["mean", "sem"])
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))
    for ax, (cols, title) in zip(axs, [(["rrr", "rrr_shuffle"], "RRR (CV 1-NSE)"), (["cca", "cca_shuffle"], "PCA->CCA (CV cc1)"),
                                        (["gc_ab", "gc_ba", "gc_null", "net_trgc"], "Granger (A->B, B->A, null, net TRGC)")]):
        for col in cols:
            ax.errorbar(g.index, g[col]["mean"], g[col]["sem"], marker="o", ms=3, label=col, capsize=0)
        ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel("True coupling A -> B")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "simulation.png", dpi=200)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
