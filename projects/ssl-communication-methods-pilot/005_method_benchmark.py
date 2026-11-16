"""Benchmark of inter-areal communication methods on synthetic spikes with known ground truth (user request 2026-11-06:
"benchmarking when possible methods; include CCA, pCCA"), before extending the pilot to the full dataset.

Ground truth: three recorded areas A, B, C (+ an unrecorded area H), 10-ms bins, 60 bins per trial. Each area has one
communication latent and 2 private latents (AR(1), phi 0.8); Poisson units (5-15 Hz) load on them. All areas share an
evoked transient at 350 ms. Scenarios (coupling strength c):
  none           no interaction
  ff             A -> B, 20-ms lag                     (truth: A -> B)
  fb             B -> A, 20-ms lag                     (truth: B -> A)
  bidir          A -> B (20 ms) and B -> A (40 ms)     (truth: both)
  common_rec     C -> A (10 ms) and C -> B (30 ms), C recorded   (truth: no A-B interaction; conditioning can fix)
  common_hidden  H -> A (10 ms) and H -> B (30 ms), H unrecorded (truth: none; nothing can fix)
  gain           trial-wise shared gain of the evoked response in A and B (arousal-like; truth: none)
Analysis window: 100 ms (10-ms bins) or 200 ms (20-ms bins) around the evoked transient, condition-mean residuals.
Methods (comm_lib), each against the same 40 trial shuffles of A:
  RRR (same bin), lagged RRR A(t-1..t-L) -> B(t) and B -> A (Semedo 2022), lagged RRR conditioned on the target past
  and C (lagRRR|C, Granger-like), PCA->CCA, pCCA | C (partial_CCA, C PCs at
  lags 0..L), lagged CCA scan (lead asymmetry), Granger A <-> B with time-reversal (TRGC), conditional Granger | C.
  DLAG: not benchmarked (comm_lib.DLAG_NOTE).
Grid: main = scenarios x c in {0.25, 0.5, 1} (+ none) x bins {10, 20 ms}, 30 trials, 20 units; sweep = ff (c 0.5) and
none x trials {15, 30, 60} x units {20, 40} x bins {10, 20 ms}. 30 repetitions per cell.
Outputs: OUT/benchmark/{benchmark_raw.parquet, benchmark_summary.csv, figures/*.png}
Run (haas): OMP_NUM_THREADS=1 python projects/ssl-communication-methods-pilot/005_method_benchmark.py [--n-proc 32]
            [--quick] [--replot]
"""

import argparse
import itertools
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402

OUT = cl.OUT / "benchmark"
SCENARIOS = ["none", "ff", "fb", "bidir", "common_rec", "common_hidden", "gain"]
COUPLINGS = [0.25, 0.5, 1.0]
N_REP, N_SH, N_BINS, EVOKED_BIN = 30, 40, 60, 35
# per bin size: window bins, max lag (bins) for lagged RRR / CCA scan / pCCA Z, Granger window (bins), Granger order
BIN_CFG = {10: dict(win=np.arange(30, 40), lag=5, gwin=15, order=3),
           20: dict(win=np.arange(13, 23), lag=3, gwin=8, order=3)}
METHODS = {  # method: (truth-checked quantity, kind)
    "RRR": "undirected", "lagRRR": "directed", "CCA": "undirected", "pCCA|C": "undirected",
    "lagCCA": "directed", "Granger": "directed", "Granger|C": "directed",
}


def simulate(scenario, c, rng, n_trials, n_units):
    T = N_BINS

    def ar1(n=1):
        x = np.zeros((n_trials, T, n))
        e = rng.normal(size=x.shape)
        for t in range(1, T):
            x[:, t] = 0.8 * x[:, t - 1] + e[:, t]
        return x / x.std((0, 1), keepdims=True)

    comm = {k: ar1()[..., 0] for k in "ABCH"}
    priv = {k: ar1(2) for k in "ABC"}

    def add_lag(dst, src, lag, w):
        comm[dst][:, lag:] += w * comm[src][:, :-lag]

    if scenario == "ff":
        add_lag("B", "A", 2, c)
    elif scenario == "fb":
        add_lag("A", "B", 2, c)
    elif scenario == "bidir":
        a0 = comm["A"].copy()
        add_lag("A", "B", 4, c)
        comm["B"][:, 2:] += c * a0[:, :-2]
    elif scenario in ("common_rec", "common_hidden"):
        s = "C" if scenario == "common_rec" else "H"
        add_lag("A", s, 1, c)
        add_lag("B", s, 3, c)
    evoked = 2.0 * np.exp(-0.5 * ((np.arange(T) - EVOKED_BIN) / 3) ** 2)
    gain = np.ones(n_trials)
    if scenario == "gain":
        gain = np.exp(c * 0.5 * rng.normal(size=n_trials))  # same trial gain in A and B

    def spikes(k, g):
        lat = np.concatenate([comm[k][..., None], priv[k]], 2)  # [trial, time, 3]
        w = rng.normal(0, 0.3, (3, n_units))
        w[0] = rng.normal(0.6, 0.3, n_units)  # communication latent: the strongest loading
        base = np.log(rng.uniform(5, 15, n_units) * 0.01)
        drive = base[None, None] + 0.5 * lat @ w + g[:, None, None] * evoked[None, :, None]
        return rng.poisson(np.exp(drive)).astype(float)

    oracle = comm["H" if scenario == "common_hidden" else "C"][..., None]  # true common driver (latent)
    return spikes("A", gain), spikes("B", gain), spikes("C", np.ones(n_trials)), oracle


def rebin(x, f):
    return x if f == 1 else x[:, : x.shape[1] // f * f].reshape(x.shape[0], -1, f, x.shape[2]).sum(2)


def evaluate(fa, fb, fc, oracle, bin_ms, rng, prep="standard"):
    """All methods + their trial-shuffle nulls on one simulated session. prep: 'standard' condition-mean residuals or
    'gain' (comm_lib.residualise_gain). Returns {quantity: (value, null array)}."""
    cfg = BIN_CFG[bin_ms]
    f = bin_ms // 10
    fa, fb, fc, oracle = rebin(fa, f), rebin(fb, f), rebin(fc, f), rebin(oracle, f)
    cond = np.zeros(len(fa))
    res_fn = cl.residualise if prep == "standard" else cl.residualise_gain
    A, B, C = (res_fn(x, cond) for x in (fa, fb, fc))
    idx, L, order = cfg["win"], cfg["lag"], cfg["order"]
    def pcs(x, k):  # top-k PCs of the (prep) residuals, unit variance
        flat = x.reshape(-1, x.shape[2])
        _, _, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
        z = (flat - flat.mean(0)) @ Vt[:k].T
        return (z / (z.std(0) + 1e-12)).reshape(x.shape[0], x.shape[1], k)

    za, zb, zc = (pcs(x, cl.CONFIG["granger_pcs"]) for x in (A, B, C))
    Z = cl.lag_stack(pcs(C, 5), idx, range(L + 1))
    zo = (oracle - oracle.mean((0, 1))) / oracle.std()
    Zo = cl.lag_stack(zo, idx, range(L + 1))
    lo = max(order, int(idx[0] + len(idx) / 2 - cfg["gwin"] / 2))
    t_index = np.arange(lo, lo + cfg["gwin"])
    t_rev = np.sort(za.shape[1] - 1 - t_index)
    lags = range(1, L + 1)

    def run(perm):
        Ap, zap = A[perm], za[perm]
        out = {}
        out["RRR"] = cl.rrr_cv(Ap[:, idx], B[:, idx], rng)["perf_rrr"]
        ff = cl.rrr_cv(cl.lag_stack(Ap, idx, lags), B[:, idx], rng)["perf_rrr"]
        fbk = cl.rrr_cv(cl.lag_stack(B, idx, lags), Ap[:, idx], rng)["perf_rrr"]
        out["lagRRR_ab"], out["lagRRR_ba"], out["lagRRR_dir"], out["lagRRR_sum"] = ff, fbk, ff - fbk, ff + fbk
        cff = cl.lagged_rrr_conditional(Ap, B, idx, L, rng, Z)["perf_rrr"]
        cfb = cl.lagged_rrr_conditional(B, Ap, idx, L, rng, Z)["perf_rrr"]
        out["lagRRR|C_ab"], out["lagRRR|C_ba"], out["lagRRR|C_dir"], out["lagRRR|C_sum"] = cff, cfb, cff - cfb, cff + cfb
        out["CCA"] = cl.pca_cca_cv(Ap[:, idx], B[:, idx], rng)["cc1"]
        out["pCCA|C"] = cl.pcca_cv(Ap[:, idx], B[:, idx], Z, rng)["cc1"]
        out["CCA_lib"] = cl.pcca_cv(Ap[:, idx], B[:, idx], None, rng)["cc1"]
        out["pCCA|oracle"] = cl.pcca_cv(Ap[:, idx], B[:, idx], Zo, rng)["cc1"]
        sc = cl.lagged_cca_scan(Ap, B, idx, L, rng)
        out["lagCCA_dir"], out["lagCCA_peak"], out["lagCCA_max"] = sc["asym"], sc["peak_lag"], np.nanmax(sc["cc1"])
        ab, ba = cl.granger_ensemble(zap, zb, order, t_index)
        ab_r, ba_r = cl.granger_ensemble(zap[:, ::-1], zb[:, ::-1], order, t_rev)
        out["Granger_ab"], out["Granger_ba"], out["Granger_dir"] = ab, ba, (ab - ba) - (ab_r - ba_r)
        out["Granger_sum"] = ab + ba
        cab, cba = cl.granger_conditional(zap, zb, zc, order, t_index)
        cab_r, cba_r = cl.granger_conditional(zap[:, ::-1], zb[:, ::-1], zc[:, ::-1], order, t_rev)
        out["Granger|C_ab"], out["Granger|C_ba"], out["Granger|C_dir"] = cab, cba, (cab - cba) - (cab_r - cba_r)
        out["Granger|C_sum"] = cab + cba
        oab, oba = cl.granger_conditional(zap, zb, zo, order, t_index)
        oab_r, oba_r = cl.granger_conditional(zap[:, ::-1], zb[:, ::-1], zo[:, ::-1], order, t_rev)
        out["Granger|oracle_ab"], out["Granger|oracle_ba"] = oab, oba
        out["Granger|oracle_dir"] = (oab - oba) - (oab_r - oba_r)
        out["Granger|oracle_sum"] = oab + oba
        return out

    true = run(np.arange(len(A)))
    nulls = [run(rng.permutation(len(A))) for _ in range(N_SH)]
    return {k: (v, np.array([n[k] for n in nulls], float)) for k, v in true.items()}


def one_job(job):
    scen, c, bin_ms, n_trials, n_units, rep, grid, prep = job
    seed = zlib.crc32(repr((scen, c, bin_ms, n_trials, n_units, rep)).encode())  # same data for both preps
    rng = np.random.default_rng(seed)
    fa, fb, fc, oracle = simulate(scen, c, rng, n_trials, n_units)
    res = evaluate(fa, fb, fc, oracle, bin_ms, rng, prep)
    rows = []
    for q, (v, nl) in res.items():
        rows.append(dict(grid=grid, prep=prep, scenario=scen, coupling=c, bin_ms=bin_ms, n_trials=n_trials, n_units=n_units,
                         rep=rep, seed=seed, quantity=q, value=float(v), null_mean=float(np.nanmean(nl)),
                         null_sd=float(np.nanstd(nl)), p_upper=float((1 + np.sum(nl >= v)) / (1 + len(nl))),
                         p_lower=float((1 + np.sum(nl <= v)) / (1 + len(nl)))))
    return rows


def jobs(quick):
    reps = range(3 if quick else N_REP)
    out = []
    for bin_ms in (10, 20):
        for prep in ("standard", "gain"):
            out += [("none", 0.0, bin_ms, 30, 20, r, "main", prep) for r in reps]
            out += [(s, c, bin_ms, 30, 20, r, "main", prep) for s in SCENARIOS[1:] for c in COUPLINGS for r in reps]
        for s, c in (("ff", 0.5), ("none", 0.0)):
            out += [(s, c, bin_ms, nt, nu, r, "sweep", "standard")
                    for nt, nu in itertools.product((15, 30, 60), (20, 40)) if not (nt == 30 and nu == 20) for r in reps]
    return out


# ---------------------------------------------------------------- summary and figures

SIG = 0.05
# detection of an A-B interaction: directed methods use the summed A->B + B->A quantity against its own null
DETECT = {"RRR": "RRR", "lagRRR": "lagRRR_sum", "lagRRR|C": "lagRRR|C_sum", "CCA": "CCA", "CCA_lib": "CCA_lib",
          "pCCA|C": "pCCA|C", "pCCA|oracle": "pCCA|oracle", "lagCCA": "lagCCA_max", "Granger": "Granger_sum",
          "Granger|C": "Granger|C_sum", "Granger|oracle": "Granger|oracle_sum"}
DIRECTION = {"lagRRR": "lagRRR_dir", "lagRRR|C": "lagRRR|C_dir", "lagCCA": "lagCCA_dir", "Granger": "Granger_dir",
             "Granger|C": "Granger|C_dir", "Granger|oracle": "Granger|oracle_dir"}
KEY = ["grid", "prep", "scenario", "coupling", "bin_ms", "n_trials", "n_units", "rep"]


def detection_p(df, m):
    """Per simulated session: p of an A-B interaction for method m; directed methods: either direction, Bonferroni x2."""
    q = DETECT[m]
    if isinstance(q, str):
        return df[df.quantity == q].set_index(KEY).p_upper
    a = df[df.quantity == q[0]].set_index(KEY).p_upper
    b = df[df.quantity == q[1]].set_index(KEY).p_upper
    return np.minimum(1, 2 * np.minimum(a, b))


def summarise(df):
    """Per method x scenario x coupling x bin (main grid): detection rate (A-B interaction above the shuffle null, any
    direction), and for directed methods the rate of a significant A->B lead and of a significant B->A lead."""
    rows = []
    d = df[df.grid == "main"]
    for (prep, scen, c, bin_ms), g in d.groupby(["prep", "scenario", "coupling", "bin_ms"]):
        for m, q in DETECT.items():
            p = detection_p(g, m)
            x = g[g.quantity == (q if isinstance(q, str) else q[0])]
            r = dict(prep=prep, scenario=scen, coupling=c, bin_ms=bin_ms, method=m, n=len(p),
                     detect=float((p < SIG).mean()),
                     z=float(((x.value - x.null_mean) / x.null_sd.replace(0, np.nan)).median()))
            if m in DIRECTION:
                y = g[g.quantity == DIRECTION[m]]
                r["lead_AB"] = float((y.p_upper < SIG).mean())
                r["lead_BA"] = float((y.p_lower < SIG).mean())
            rows.append(r)
    return pd.DataFrame(rows)


COLORS = {"RRR": "#4c72b0", "lagRRR": "#1b3f7a", "lagRRR|C": "#6a3d9a", "CCA": "#dd8452", "CCA_lib": "#f0b98a", "pCCA|C": "#c44e52",
          "pCCA|oracle": "#c44e52", "lagCCA": "#937860", "Granger": "#55a868", "Granger|C": "#2d6b3c",
          "Granger|oracle": "#2d6b3c"}
STYLE = {m: (":" if "oracle" in m else "-") for m in COLORS}


def figures(df, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meths = list(DETECT)
    # 1. detection rate vs coupling per scenario; rows: standard residuals, gain-corrected residuals
    for bin_ms in (10, 20):
        fig, axs = plt.subplots(2, 6, figsize=(15, 5.6), sharey=True)
        for row, prep in enumerate(("standard", "gain")):
            s = summ[(summ.bin_ms == bin_ms) & (summ.prep == prep)]
            none = s[s.scenario == "none"].set_index("method")
            for ax, scen in zip(axs[row], SCENARIOS[1:]):
                for m in meths:
                    x = s[(s.scenario == scen) & (s.method == m)].sort_values("coupling")
                    ax.plot([0] + list(x.coupling), [none.loc[m, "detect"]] + list(x.detect), STYLE[m], marker="o",
                            ms=3, lw=1.2, color=COLORS[m], label=m)
                ax.axhline(SIG, color="0.5", lw=0.6, ls=":")
                truth = "true A-B interaction" if scen in ("ff", "fb", "bidir") else "NO A-B interaction"
                ax.set_title(f"{scen} ({truth})\n{prep} residuals", fontsize=8)
                ax.set_xlabel("coupling")
            axs[row, 0].set_ylabel("fraction of sims with\nA-B interaction (p<0.05)")
        axs[0, -1].legend(fontsize=6, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
        fig.suptitle(f"Detection of an A-B interaction ({bin_ms}-ms bins, 30 trials, 20 units/area, 30 sims per point; "
                     "directed methods: either direction, Bonferroni x2; dotted: conditioned on the true driver)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"01_detection_{bin_ms}ms.png", dpi=200)
        plt.close(fig)
    # 2. direction at coupling 1: significant A->B lead (up) and B->A lead (down) per scenario
    dm = list(DIRECTION)
    fig, axs = plt.subplots(2, 6, figsize=(15, 5), sharey=True)
    for row, bin_ms in enumerate((10, 20)):
        s = summ[(summ.bin_ms == bin_ms) & summ.method.isin(dm) & (summ.prep == "standard")]
        for ax, scen in zip(axs[row], SCENARIOS[1:]):
            x = s[(s.scenario == scen) & (s.coupling == 1.0)].set_index("method").loc[dm]
            pos = np.arange(len(dm))
            ax.bar(pos, x.lead_AB, color=[COLORS[m] for m in dm], width=0.7)
            ax.bar(pos, -x.lead_BA, color=[COLORS[m] for m in dm], width=0.7, alpha=0.45, hatch="//")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(pos, dm, rotation=45, fontsize=7, ha="right")
            ax.set_title(f"{scen} ({bin_ms} ms)", fontsize=8)
        axs[row, 0].set_ylabel("A->B lead (up)\nB->A lead (down, hatched)")
    fig.suptitle("Directionality at coupling 1: fraction of sims with a significant lead (p<0.05 vs shuffles). "
                 "Truth: ff A->B, fb B->A, bidir both, common_* / gain none", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_direction.png", dpi=200)
    plt.close(fig)
    # 3. power and false-positive rate vs trials / units (sweep + matching main cells)
    sw = df[(df.prep == "standard") & df.coupling.isin([0.0, 0.5]) & df.scenario.isin(["ff", "none"])]
    sw = sw[(sw.grid == "sweep") | ((sw.n_trials == 30) & (sw.n_units == 20))]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharey="row")
    for col, bin_ms in enumerate((10, 20)):
        for row, scen in enumerate(("ff", "none")):
            ax = axs[row, col]
            for m in meths:
                if "oracle" in m or m == "CCA_lib":
                    continue
                for nu, ls in ((20, "-"), (40, "--")):
                    p = detection_p(sw[(sw.scenario == scen) & (sw.bin_ms == bin_ms) & (sw.n_units == nu)], m)
                    r = (p < SIG).groupby(level="n_trials").mean()
                    ax.plot(r.index, r.values, ls, marker="o", ms=3, color=COLORS[m],
                            label=f"{m}, {nu} units" if (row, col) == (0, 1) else None)
            ax.axhline(SIG, color="0.5", lw=0.6, ls=":")
            ax.set_xticks([15, 30, 60])
            ax.set_title(f"{'power (ff, c=0.5)' if scen == 'ff' else 'false positives (none)'}, {bin_ms}-ms bins",
                         fontsize=8)
            ax.set_xlabel("trials per condition")
    axs[0, 1].legend(fontsize=5.5, frameon=False, ncol=2, bbox_to_anchor=(1, 1), loc="upper left")
    axs[0, 0].set_ylabel("detection rate")
    axs[1, 0].set_ylabel("detection rate")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_power_trials_units.png", dpi=200)
    plt.close(fig)
    # 4. gain correction: detection with standard vs gain-corrected residuals (20-ms bins)
    cells = [("none", 0.0), ("ff", 0.5), ("ff", 1.0), ("gain", 0.5), ("gain", 1.0), ("common_rec", 1.0)]
    mm = [m for m in meths if "oracle" not in m and m != "CCA_lib"]
    fig, axs = plt.subplots(1, len(cells), figsize=(15, 2.8), sharey=True)
    s = summ[summ.bin_ms == 20]
    for ax, (scen, c) in zip(axs, cells):
        pos = np.arange(len(mm))
        for k, prep in enumerate(("standard", "gain")):
            x = s[(s.scenario == scen) & (s.coupling == c) & (s.prep == prep)].set_index("method").loc[mm]
            ax.bar(pos + (k - 0.5) * 0.38, x.detect, 0.38, color=[COLORS[m] for m in mm], alpha=1 if k == 0 else 0.45,
                   hatch=None if k == 0 else "//", label=prep)
        ax.axhline(SIG, color="0.5", lw=0.6, ls=":")
        ax.set_xticks(pos, mm, rotation=45, fontsize=7, ha="right")
        ax.set_title(f"{scen}, c={c}", fontsize=8)
    axs[0].set_ylabel("detection rate")
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Trial-wise gain correction (hatched) vs standard condition-mean residuals (solid), 20-ms bins",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_gain_correction.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if args.quick else ""
    raw = OUT / f"benchmark_raw{sfx}.parquet"
    if not args.replot:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        js = jobs(args.quick)
        t0, rows = time.time(), []
        with Pool(args.n_proc) as pool:
            for i, r in enumerate(pool.imap_unordered(one_job, js, chunksize=2)):
                rows += r
                if i % 50 == 0:
                    print(f"{i + 1}/{len(js)} {time.time() - t0:.0f} s", flush=True)
        df = pd.DataFrame(rows)
        df.attrs["config"] = dict(n_rep=N_REP, n_shuffles=N_SH, bin_cfg={k: {kk: (vv.tolist() if hasattr(vv, "tolist")
                                  else vv) for kk, vv in v.items()} for k, v in BIN_CFG.items()},
                                  comm_config=cl.CONFIG, dlag=cl.DLAG_NOTE)
        df.to_parquet(raw)
    df = pd.read_parquet(raw)
    summ = summarise(df)
    summ.to_csv(OUT / f"benchmark_summary{sfx}.csv", index=False)
    figures(df, summ)
    print(summ[(summ.coupling.isin([0, 1.0]))].round(2).to_string())


if __name__ == "__main__":
    main()
