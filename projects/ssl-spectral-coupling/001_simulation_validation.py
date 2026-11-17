"""Synthetic validation of spec_lib measures (known ground truth, Poisson units in 5-ms bins, 1-s quiet-like windows).

Scenarios (40 trials, 15 units per area, 20 shuffles, 30 simulated sessions each):
  null        independent areas;
  ff          A -> B: B's rate follows A's 30-Hz latent with a 10-ms lag;
  common      an unrecorded 30-Hz latent drives A and B at zero lag (no A-B interaction);
  common_rec  as common, but 30 "rest" units also follow the latent (recorded common input).
Expected: msc / cancoh excess at 30 Hz in ff, common, common_rec; PSI and gc_ab - gc_ba > 0 only in ff;
imc ~ 0 at zero lag; pcoh excess removed in common_rec.
Output: OUT/simulation/001_simulation.csv, 001_simulation.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_lib as sl  # noqa: E402

N_TR, N_U, N_SH, N_SIM, T = 40, 15, 20, 30, 200
F0, GAIN, BASE = 30.0, 0.6, 8.0


def ar2(n_tr, T, f0, rng, r=0.95):
    """Narrow-band latent around f0 (AR(2)), unit SD, [trial, time]."""
    a1, a2 = 2 * r * np.cos(2 * np.pi * f0 * sl.BIN), -r ** 2
    x = np.zeros((n_tr, T + 100))
    e = rng.standard_normal(x.shape)
    for t in range(2, T + 100):
        x[:, t] = a1 * x[:, t - 1] + a2 * x[:, t - 2] + e[:, t]
    x = x[:, 100:]
    return x / x.std()


def pop(latent, n, rng):
    rate = BASE * sl.BIN * np.exp(GAIN * latent[:, :, None] * rng.uniform(0.5, 1.5, n)[None, None, :])
    return rng.poisson(rate).astype(np.float32)


def simulate(kind, rng):
    la, lb = ar2(N_TR, T, F0, rng), ar2(N_TR, T, F0, rng)
    rest_lat = ar2(N_TR, T, F0, rng)
    if kind == "ff":
        lb = 0.7 * np.roll(la, 2, axis=1) + 0.7 * lb
        lb /= lb.std()
    elif kind.startswith("common"):
        c = ar2(N_TR, T, F0, rng)
        la, lb = (0.7 * c + 0.7 * la), (0.7 * c + 0.7 * lb)
        rest_lat = c if kind == "common_rec" else rest_lat
    A, B, Z = pop(la, N_U, rng), pop(lb, N_U, rng), pop(rest_lat, 30, rng)
    return A, B, Z


def main():
    rows = []
    idx = np.arange(T)
    for kind in ("null", "ff", "common", "common_rec"):
        for s in range(N_SIM):
            rng = np.random.default_rng(1000 * s + len(kind))
            A, B, Z = simulate(kind, rng)
            lab = np.zeros(N_TR)
            sa, sb = sl.area_signals(sl.residuals(A, lab)), sl.area_signals(sl.residuals(B, lab))
            zp = sl.pcs(sl.residuals(Z, lab), sl.N_REST_PC)
            FA, f = sl.tfft(sa, idx, 3.0)
            FB, _ = sl.tfft(sb, idx, 3.0)
            FZ, _ = sl.tfft(zp, idx, 3.0)
            xa = sa[:, :, 0] - sa[:, :, 0].mean(1, keepdims=True)
            xb = sb[:, :, 0] - sb[:, :, 0].mean(1, keepdims=True)
            perms = [np.arange(N_TR)] + [rng.permutation(N_TR) for _ in range(N_SH)]
            bands = {"b30": (24, 36)}
            spec, psi = sl.pair_all(FA, FB, FZ, xa, xb, f, perms, bands)
            for m, v in spec.items():
                obs, nul = sl.band_means(v[0], f, bands)["b30"], np.mean([sl.band_means(x, f, bands)["b30"] for x in v[1:]])
                sd = np.std([sl.band_means(x, f, bands)["b30"] for x in v[1:]])
                rows.append(dict(scenario=kind, sim=s, measure=m, value=obs, null_mean=nul, z=(obs - nul) / (sd + 1e-12)))
            rows.append(dict(scenario=kind, sim=s, measure="psi", value=psi["b30"][0], null_mean=psi["b30"][1:].mean(),
                             z=(psi["b30"][0] - psi["b30"][1:].mean()) / (psi["b30"][1:].std() + 1e-12)))
    d = pd.DataFrame(rows)
    d["excess"] = d.value - d.null_mean
    od = sl.OUT / "simulation"
    od.mkdir(parents=True, exist_ok=True)
    d.to_csv(od / "001_simulation.csv", index=False)
    summ = d.groupby(["scenario", "measure"]).agg(excess=("excess", "mean"), frac_z2=("z", lambda z: np.mean(z > 2)),
                                                  frac_zneg2=("z", lambda z: np.mean(z < -2))).unstack(0)
    print(summ.round(3).to_string())
    gc = d[d.measure.isin(["gc_ab", "gc_ba"])].pivot_table(index=["scenario", "sim"], columns="measure", values="excess")
    print("direction gc_ab - gc_ba > 0 fraction:", (gc.gc_ab - gc.gc_ba > 0).groupby(level=0).mean().round(2).to_dict())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = ["msc", "cancoh", "pcoh", "imc", "psi", "gc_ab", "gc_ba"]
    fig, axs = plt.subplots(1, len(ms), figsize=(2.2 * len(ms), 2.4))
    for ax, m in zip(axs, ms):
        sub = d[d.measure == m]
        data = [sub[sub.scenario == k].excess for k in ("null", "ff", "common", "common_rec")]
        ax.boxplot(data, widths=0.6, showfliers=False)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks([1, 2, 3, 4], ["null", "A->B", "common", "common\n(recorded)"], rotation=60, fontsize=7)
        ax.set_title(m, fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("excess over shuffle (24-36 Hz)", fontsize=8)
    fig.tight_layout()
    fig.savefig(od / "001_simulation.png", dpi=200)
    fig.savefig(od / "001_simulation.pdf")


if __name__ == "__main__":
    main()
