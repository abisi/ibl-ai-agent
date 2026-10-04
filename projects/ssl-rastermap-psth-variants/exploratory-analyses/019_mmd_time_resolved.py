"""Time-resolved reward-free MMD^2 (localise the R+ vs R- difference in time), per single condition.

Sliding windows (WIN_S wide, STEP_S step) over the reward-free bins of a condition (016 cuts). Per window and CV
direction: features = the window's bins (low-dimensional, no PCA); RBF kernel, sigma^2 = median heuristic on the
FIT split; mouse-level permutation MMD^2 on the TEST split. In parallel, an amplitude-only test: 1-D feature = each
neuron's mean response in the window (same kernel / permutation machinery). No multiple-comparison correction;
`significant_both` = p < 0.05 in both CV directions.
Output -> <method_dir>/<run-dir>/time_resolved/{time_resolved_mmd.csv, time_resolved_<cond>.png/pdf}
"""
import argparse
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/rastermap_psth"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import cluster_mmd_analysis_new as mmd                                    # noqa: E402
d017 = importlib.import_module("017_mmd_method_figures")

WIN_S, STEP_S = 0.03, 0.01
CONDS = ["Whisker miss", "Auditory hit", "Auditory hit (lick)", "Auditory post", "Whisker hit", "Whisker hit (lick)",
         "Spont. lick", "Whisker pre"]
RP, RM = d017.RP, d017.RM
DIRS = ("fit_odd_test_even", "fit_even_test_odd")


def window_test(fit, test, D, n_perm, seed):
    sigma2 = mmd.median_sq_dist(fit, seed=seed)
    S = mmd.kernel_mouse_sums(test, D["codes"], len(D["uniq"]), sigma2)
    return mmd.mouse_block_permutation_test(S, D["codes"], D["is_pos"], n_perm=n_perm,
                                            rng=np.random.default_rng(seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--conds", nargs="*", default=CONDS)
    a = ap.parse_args()
    _, _, md = d017.d016.paths(a.variant)
    out = md / a.run_dir / "time_resolved"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, cond in enumerate(a.conds):
        Ds = {d: d017.prepare(a.variant, cond, d, None) for d in DIRS}
        p0 = Ds[DIRS[0]]["psth"][0]
        t_all, bin_s = p0["t"], Ds[DIRS[0]]["bin_s"]
        keep = np.ones(len(t_all), bool) if p0["cut"] is None else (t_all + bin_s / 2 <= p0["cut"] + 1e-9)
        t = t_all[keep]
        assert Ds[DIRS[0]]["fit"].shape[1] == len(t), (cond, Ds[DIRS[0]]["fit"].shape, len(t))
        dt = float(np.median(np.diff(t)))
        wlen, wstep = max(1, int(round(WIN_S / dt))), max(1, int(round(STEP_S / dt)))
        for wi, s0 in enumerate(range(0, len(t) - wlen + 1, wstep)):
            sl = slice(s0, s0 + wlen)
            for di, d in enumerate(DIRS):
                D = Ds[d]
                seed = 100000 * ci + 10 * wi + di
                fit, test = D["fit"][:, sl], D["test"][:, sl]
                full = window_test(fit, test, D, a.n_perm, seed)
                amp = window_test(fit.mean(1, keepdims=True), test.mean(1, keepdims=True), D, a.n_perm, seed + 5)
                m = test.mean(1)
                rows.append(dict(condition=cond, t_start=float(t[s0]), t_center=float(t[sl].mean()),
                                 t_end=float(t[s0 + wlen - 1]), direction=d,
                                 mmd2_z=full["mmd2_z"], p_value=full["p_value"],
                                 amp_mmd2_z=amp["mmd2_z"], amp_p_value=amp["p_value"],
                                 mean_rplus=float(m[D["is_pos"][D["codes"]]].mean()),
                                 mean_rminus=float(m[~D["is_pos"][D["codes"]]].mean())))
        print(f"[019] {cond}: {wi + 1} windows of {wlen} bins", flush=True)
        df_c = pd.DataFrame([r for r in rows if r["condition"] == cond])
        fig_condition(df_c, Ds[DIRS[0]], t, cond, out)
    df = pd.DataFrame(rows)
    df.to_csv(out / "time_resolved_mmd.csv", index=False)
    summ = (df.groupby(["condition", "t_center"])
            .agg(z_mean=("mmd2_z", "mean"), p_max=("p_value", "max"), amp_z_mean=("amp_mmd2_z", "mean"),
                 amp_p_max=("amp_p_value", "max")).reset_index())
    summ["significant_both"], summ["amp_significant_both"] = summ.p_max < 0.05, summ.amp_p_max < 0.05
    summ.to_csv(out / "time_resolved_summary.csv", index=False)
    for c, g in summ.groupby("condition", sort=False):
        sig = g[g.significant_both].t_center.round(3).tolist()
        asig = g[g.amp_significant_both].t_center.round(3).tolist()
        print(f"{c:>20s}: full sig windows {sig}\n{'':>20s}  amplitude sig windows {asig}")
    print("ALL DONE")


def fig_condition(dfc, D, t, cond, out):
    import rastermap_psth.population_matrix_summary as pms
    pms.use_pub_font()
    fs = 6.5
    plt.rcParams.update({"font.size": fs, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(60 / 25.4, 95 / 25.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.2, 1, 1], hspace=0.25))
    p = D["psth"][0]; coh = D["cohorts"]
    for lab, col in (("R+", RP), ("R-", RM)):
        mx = p["X"][coh == lab]; mu, se = mx.mean(0), mx.std(0) / np.sqrt(len(mx))
        axes[0].fill_between(p["t"], mu - se, mu + se, color=col, alpha=0.25, lw=0)
        axes[0].plot(p["t"], mu, color=col, lw=0.8, label=lab)
    if p["cut"] is not None:
        axes[0].axvspan(p["cut"], p["t"][-1], color="0.88", lw=0, zorder=0)
    axes[0].set_ylabel("population mean"); axes[0].legend(frameon=False, fontsize=fs - 0.5, loc="upper right")
    for ax, zc, pc, ttl in ((axes[1], "mmd2_z", "p_value", "MMD² (response profile in window)"),
                            (axes[2], "amp_mmd2_z", "amp_p_value", "Amplitude only (window mean)")):
        for d, col in zip(DIRS, ("0.35", "#d9822b")):
            g = dfc[dfc.direction == d]
            ax.plot(g.t_center, g[zc], color=col, lw=0.8, marker="o", ms=1.5,
                    label="odd → even" if d == DIRS[0] else "even → odd")
        both = dfc.groupby("t_center")[pc].max() < 0.05
        top = dfc[zc].max()
        ax.scatter(both.index[both.values], np.full(both.sum(), top * 1.1 + 0.3), marker="s", s=4, color="k")
        ax.axhline(0, color="0.6", lw=0.4); ax.set_ylabel("z"); ax.set_title(ttl, fontsize=fs, loc="left")
    axes[1].legend(frameon=False, fontsize=fs - 1, loc="upper left")
    for ax in axes:
        ax.axvline(0, color="0.5", lw=0.4, ls=":")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[2].set_xlabel(f"time from {'lick' if 'lick' in cond.lower() else 'stimulus'} (s)")
    fig.suptitle(f"{cond}: time-resolved R+ vs R−\n(■ p<0.05 both halves, {int(WIN_S * 1000)} ms windows)",
                 fontsize=fs + 0.5, x=0.02, ha="left")
    tag = cond.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out / f"time_resolved_{tag}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / f"time_resolved_{tag}.pdf", bbox_inches="tight")
    plt.close(fig); plt.rcdefaults()


if __name__ == "__main__":
    main()
