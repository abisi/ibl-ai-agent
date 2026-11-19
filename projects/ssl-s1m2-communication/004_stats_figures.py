"""Readouts, statistics and figures: SSL S1-M2 time-shifted CCA vs the paper (001 reference).

Per session x pair x condition x kind (total / noise): paper readout (s1m2_lib.readout) for feedforward (delays 0..+15
ms), feedback (-15..0 ms) and the trial-shuffle map; peak delay of the baseline-subtracted map in 0..100 ms.
Pairs: SSp-bfd vs wMO (MO-wM1 + MO-wM2 units pooled, main), MO-wM1, MO-wM2. In-sample CCV + 10-fold CV CCV (v2).
Stats (session = unit): within cohort x stage, value vs 0 (Wilcoxon, mouse-level sign flip), FF - FB (Wilcoxon),
FF - shuffle (Wilcoxon); R+ vs R- per stage: Mann-Whitney + Welch. Uncorrected (exploratory).

Run (local, after pulling nothing: reads the NAS): python 004_stats_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

import s1m2_lib as L

def _dirs():
    v = L.CONFIG["version"]
    return L.OUT / f"maps_v{v}", L.OUT / f"figures_v{v}"
PAIRS = ["SSp-bfd__wMO", "SSp-bfd__MO-wM1", "SSp-bfd__MO-wM2"]
COL = {"R+": "#00B400", "R-": "#C800C8"}
LS = {"learning": "-", "expert": "--"}
CONDS = ["All", "W_all", "W_hit", "W_miss", "A_hit", "NS_all"]


def sign_flip_p(df, col, n=10000, seed=0):
    """Mouse-level sign flip: mean per mouse, flip signs of mouse means."""
    m = df.groupby("mouse_id")[col].mean().dropna().to_numpy()
    if len(m) < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1, 1], (n, len(m))) * m).mean(1))
    return float((np.sum(null >= abs(m.mean())) + 1) / (n + 1))


def wil(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def load():
    """v2 files: one per (session, pair, condition) with total / noise in-sample maps, shuffle maps and CV maps."""
    rows, maps, traces = [], {}, {}
    T = D = None
    MAPS, _ = _dirs()
    RO = L.CONFIG["readout"]
    for f in sorted(MAPS.glob("*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        t, d, dcv = z["times"], z["delays"], z["cv_delays"]
        T, D = t, d
        b = (t >= RO["baseline_s"][0] - 1e-9) & (t <= RO["baseline_s"][1] + 1e-9)
        e = (t >= RO["early_s"][0] - 1e-9) & (t <= RO["early_s"][1] + 1e-9)
        for kind in ("total", "noise"):
            c, s = z[f"{kind}__ccv"].astype(float), z[f"{kind}__shuffle"].astype(float)
            cv = z[f"{kind}__cv"].astype(float)
            ff, ff_tr, ff_raw = L.readout(c, d, t, RO["ff_delays_s"])
            fb, fb_tr, _ = L.readout(c, d, t, RO["fb_delays_s"])
            sff, sff_tr, _ = L.readout(s, d, t, RO["ff_delays_s"])
            cff, cff_tr, cff_raw = L.readout(cv, dcv, t, RO["ff_delays_s"])
            cfb, cfb_tr, _ = L.readout(cv, dcv, t, RO["fb_delays_s"])
            # option a (artefact-free windows only)
            ffc, ffc_tr, _ = L.readout_clean(c, d, t, RO["ff_delays_s"])
            fbc, fbc_tr, _ = L.readout_clean(c, d, t, RO["fb_delays_s"])
            cffc, cffc_tr, _ = L.readout_clean(cv, dcv, t, RO["ff_delays_s"])
            cb = c - np.nanmean(c[:, b], axis=1, keepdims=True)
            prof = np.nanmean(cb[:, e], axis=1)
            win = np.abs(d) <= 0.05
            peak = float(d[win][np.nanargmax(prof[win])]) if np.isfinite(prof[win]).any() else np.nan
            rows.append(dict(meta, kind=kind, FF=ff, FB=fb, FF_shuffle=sff, FF_minus_FB=ff - fb,
                             FF_minus_shuffle=ff - sff, CV_FF=cff, CV_FB=cfb, CV_FF_minus_FB=cff - cfb,
                             FF_clean=ffc, FB_clean=fbc, FF_minus_FB_clean=ffc - fbc, CV_FF_clean=cffc,
                             baseline_ccv=float(np.nanmean(ff_raw[b])), baseline_cv=float(np.nanmean(cff_raw[b])),
                             peak_delay_ms=peak * 1000))
            key = (meta["session_id"], meta["pair"], meta["condition"], kind)
            maps[key] = cb
            traces[key] = (ff_tr, fb_tr, sff_tr, cff_tr, cfb_tr, ffc_tr, cffc_tr)
    df = pd.DataFrame(rows)
    hr, fa = df.hit_rate_W.clip(0.01, 0.99), df.fa_rate.clip(0.01, 0.99)
    df["dprime_W"] = norm.ppf(hr) - norm.ppf(fa)
    return df, maps, traces, T, D


def stats_table(df):
    out = []
    for (pair, cond, kind, stage), g in df.groupby(["pair", "condition", "kind", "stage"]):
        for coh in ("R+", "R-"):
            h = g[g.cohort == coh]
            r = dict(pair=pair, condition=cond, kind=kind, stage=stage, cohort=coh, n_sessions=len(h),
                     n_mice=h.mouse_id.nunique())
            r["baseline_ccv_mean"] = h.baseline_ccv.mean()
            for col in ("FF", "FB", "FF_minus_FB", "FF_minus_shuffle", "CV_FF", "CV_FB", "CV_FF_minus_FB",
                        "FF_clean", "FF_minus_FB_clean", "CV_FF_clean"):
                r[f"{col}_mean"] = h[col].mean()
                r[f"{col}_wilcoxon_p"] = wil(h[col])
                r[f"{col}_signflip_p"] = sign_flip_p(h, col)
            r["peak_delay_ms_median"] = h.peak_delay_ms.median()
            out.append(r)
        for col in ("FF", "CV_FF", "FF_clean", "CV_FF_clean"):
            a, b = g[g.cohort == "R+"][col].dropna(), g[g.cohort == "R-"][col].dropna()
            if len(a) >= 3 and len(b) >= 3:
                out.append(dict(pair=pair, condition=cond, kind=kind, stage=stage, cohort=f"R+ vs R- ({col})",
                                n_sessions=len(a) + len(b), **{f"{col}_mean": a.mean() - b.mean()},
                                mannwhitney_p=stats.mannwhitneyu(a, b).pvalue,
                                welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return pd.DataFrame(out)


def fig_maps(maps, T, D, df, pair, conds=("W_all", "All"), kind="total"):
    groups = [(st, co) for st in ("learning", "expert") for co in ("R+", "R-")]
    fig, axs = plt.subplots(len(groups), len(conds), figsize=(3.4 * len(conds), 2.6 * len(groups)),
                            constrained_layout=True, squeeze=False)
    win = np.abs(D) <= 0.05
    tm = (T >= -0.15) & (T <= 0.25)
    for i, (st, co) in enumerate(groups):
        sess = df[(df.pair == pair) & (df.stage == st) & (df.cohort == co)].session_id.unique()
        for j, cond in enumerate(conds):
            ax = axs[i, j]
            ms = [maps[(s, pair, cond, kind)] for s in sess if (s, pair, cond, kind) in maps]
            if not ms:
                ax.set_axis_off()
                continue
            m = np.nanmean(ms, axis=0)[np.ix_(win, tm)]
            v = np.nanmax(np.abs(m))
            im = ax.imshow(m, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                           extent=[T[tm][0] * 1e3, T[tm][-1] * 1e3, D[win][0] * 1e3, D[win][-1] * 1e3])
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{co} {st}, {cond} (n={len(ms)})", fontsize=9, color=COL[co])
            ax.set_xlabel("Time from stimulus (ms)")
            ax.set_ylabel("Delay (ms; >0 S1 leads)")
            fig.colorbar(im, ax=ax, shrink=0.8, label="CCV - baseline")
    fig.suptitle(f"{pair.replace('__', ' vs ')}, {kind} CCV (baseline-subtracted per delay)")
    return fig


def fig_paper_maps():
    tr = np.load(L.OUT / "paper_reference" / "traces.npz")
    D = tr["delays"]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.4), constrained_layout=True)
    win = np.abs(D) <= 0.05
    for i, kind in enumerate(("Total", "Noise")):
        for j, cond in enumerate(("All", "Go", "NoGo")):
            m = tr[f"{kind}_{cond}_map_mean"]
            T = tr[f"{kind}_{cond}_t"]
            tm = (T >= -0.15) & (T <= 0.25)
            b = (T >= -0.25) & (T <= 0)
            m = (m - np.nanmean(m[:, b], axis=1, keepdims=True))[np.ix_(win, tm)]
            v = np.nanmax(np.abs(m))
            im = axs[i, j].imshow(m, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                                  extent=[T[tm][0] * 1e3, T[tm][-1] * 1e3, D[win][0] * 1e3, D[win][-1] * 1e3])
            axs[i, j].axhline(0, color="k", lw=0.5)
            axs[i, j].axvline(0, color="k", lw=0.5)
            axs[i, j].set_title(f"Paper: {kind}, {cond} (n=50)", fontsize=9)
            axs[i, j].set_xlabel("Time from air puff (ms)")
            axs[i, j].set_ylabel("Delay (ms; >0 S1 leads)")
            fig.colorbar(im, ax=axs[i, j], shrink=0.8, label="CCV - baseline")
    fig.suptitle("Ranjbar-Slamloo & Kamigaki 2026: S1-M2 CCV maps from their Zenodo data")
    return fig


PAPER_OF = {"W_all": "Go", "W_hit": "hit", "W_miss": "miss", "A_hit": "NoGo", "All": "All", "NS_all": "NoGo"}
GROUPS = [("learning", "R+"), ("learning", "R-"), ("expert", "R+"), ("expert", "R-")]


def fig_traces(traces, T, df, pair="SSp-bfd__wMO", kind="total", conds=("All", "W_all", "W_hit", "W_miss", "A_hit")):
    """Rows: paper (their maps), SSL in-sample CCV, SSL 10-fold CV CCV; FF (delays 0..+15 ms), baseline-subtracted."""
    tr = np.load(L.OUT / "paper_reference" / "traces.npz")
    pk = "Total" if kind == "total" else "Noise"
    fig, axs = plt.subplots(3, len(conds), figsize=(3.2 * len(conds), 7.6), constrained_layout=True, sharex=True)
    for j, cond in enumerate(conds):
        ax = axs[0, j]
        cc = PAPER_OF[cond]
        for direction, ls in (("FF", "-"), ("FB", ":")):
            x = tr[f"{pk}_{cc}_{direction}"]
            m, se = np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(len(x))
            tp = tr[f"{pk}_{cc}_t"]
            ax.plot(tp * 1e3, m, "k", ls=ls, label=direction)
            ax.fill_between(tp * 1e3, m - se, m + se, color="k", alpha=0.15)
        ax.set_title(f"Paper, {cc} trials (n=50)", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        for i, (idx, lab) in enumerate(((0, "SSL in-sample CCV"), (3, "SSL 10-fold CV CCV")), start=1):
            ax = axs[i, j]
            for st, co in GROUPS:
                sess = df[(df.pair == pair) & (df.stage == st) & (df.cohort == co)].session_id.unique()
                x = [traces[(s, pair, cond, kind)][idx] for s in sess if (s, pair, cond, kind) in traces]
                if not x:
                    continue
                x = np.array(x)
                mm, se = np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(len(x))
                ax.plot(T * 1e3, mm, color=COL[co], ls=LS[st], lw=1.2, label=f"{co} {st} (n={len(x)})")
                ax.fill_between(T * 1e3, mm - se, mm + se, color=COL[co], alpha=0.12)
            ax.set_title(f"{lab}, {cond}", fontsize=9)
            if j == 0:
                ax.legend(fontsize=6, frameon=False)
    for ax in axs.ravel():
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.axvspan(0, 100, color="gold", alpha=0.12, lw=0)
        ax.set_xlim(-250, 500)
    for ax in axs[-1]:
        ax.set_xlabel("Time from stimulus (ms)")
    for ax in axs[:, 0]:
        ax.set_ylabel("CCV minus baseline")
    fig.suptitle(f"S1 -> M2 coupling over time ({kind}; S1 leading by 0-15 ms; {pair.replace('__', ' vs ')}). "
                 "Yellow: 0-100 ms readout window")
    return fig


def fig_readouts(df, pair="SSp-bfd__wMO", kind="total", conds=("All", "W_all", "W_hit", "W_miss", "A_hit")):
    """One value per session: mean CCV over 0-100 ms after the stimulus minus the -250..0 ms baseline, S1 leading by
    0..15 ms. Grey band: paper mean +- s.e.m. (50 sessions) for the matching trial type."""
    ref = pd.read_parquet(L.OUT / "paper_reference" / "per_session.parquet")
    pk = "Total" if kind == "total" else "Noise"
    rows = (("FF", "In-sample CCV (paper measure)"), ("CV_FF", "10-fold cross-validated CCV"))
    fig, axs = plt.subplots(len(rows), len(conds), figsize=(2.7 * len(conds), 6.4), constrained_layout=True,
                            squeeze=False)
    rng = np.random.default_rng(0)
    for i, (col, lab) in enumerate(rows):
        for j, cond in enumerate(conds):
            ax = axs[i, j]
            if col == "FF":
                v = ref[(ref["corr"] == pk) & (ref.condition == PAPER_OF[cond]) & (ref.direction == "FF")].value
                sem = v.std() / np.sqrt(len(v))
                ax.axhspan(v.mean() - sem, v.mean() + sem, color="0.85", lw=0)
                ax.axhline(v.mean(), color="0.5", lw=1)
            ann = []
            for x0, (st, co) in enumerate(GROUPS):
                h = df[(df.pair == pair) & (df.stage == st) & (df.cohort == co) & (df.condition == cond)
                       & (df.kind == kind)]
                v = h[col].dropna()
                if not len(v):
                    continue
                ax.scatter(x0 + rng.uniform(-.15, .15, len(v)), v, s=12, c=COL[co], alpha=.55,
                           marker="o" if st == "learning" else "^", lw=0)
                ax.errorbar(x0 + .3, v.mean(), v.std() / np.sqrt(len(v)), color=COL[co], capsize=3,
                            marker="o" if st == "learning" else "^", mec="k")
                p = wil(v)
                ann.append((x0, f"n={len(v)}" + (f"\np={p:.2g}" if np.isfinite(p) else "")))
            ax.axhline(0, color="k", lw=0.5)
            top = ax.get_ylim()[1]
            for x0, txt in ann:
                ax.text(x0, top, txt, ha="center", va="top", fontsize=6)
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels([f"{co}\n{st}" for st, co in GROUPS], fontsize=7)
            ax.set_title(cond + (f" (grey: paper {PAPER_OF[cond]})" if col == "FF" else ""), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{lab}\n0-100 ms minus baseline")
    fig.suptitle(f"Per-session S1 -> M2 readout ({kind}; {pair.replace('__', ' vs ')}); p: Wilcoxon vs 0")
    return fig


def fig_dprime(df, pair="SSp-bfd__wMO", kind="total", cond="All"):
    ref = pd.read_parquet(L.OUT / "paper_reference" / "per_session.parquet")
    pk = "Total" if kind == "total" else "Noise"
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    r = ref[(ref["corr"] == pk) & (ref.condition == "All") & (ref.direction == "FF")].dropna(subset=["dprime"])
    rr = stats.pearsonr(r.value, r.dprime)
    axs[0].scatter(r.value, r.dprime, c="k", s=10)
    axs[0].set_title(f"Paper, All trials: r={rr[0]:.2f}, p={rr[1]:.1g}, n={len(r)}", fontsize=9)
    axs[0].set_ylabel("d-prime (Go hit vs NoGo FA)")
    for ax, st in zip(axs[1:], ("learning", "expert")):
        g = df[(df.pair == pair) & (df.stage == st) & (df.condition == cond) & (df.kind == kind)]
        txt = []
        for co in ("R+", "R-"):
            h = g[g.cohort == co].dropna(subset=["FF", "dprime_W"])
            ax.scatter(h.FF, h.dprime_W, c=COL[co], s=14)
            if len(h) >= 4:
                q = stats.pearsonr(h.FF, h.dprime_W)
                txt.append(f"{co} r={q[0]:.2f} p={q[1]:.2g} n={len(h)}")
        ax.set_title(f"SSL {st}, {cond}\n" + "; ".join(txt), fontsize=8)
        ax.set_ylabel("d-prime (whisker hit vs no-stim FA)")
    for ax in axs:
        ax.set_xlabel("FF CCV, 0-100 ms minus baseline")
    return fig


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v3", choices=list(L.PROFILES))
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    L.set_profile(a.profile)
    _, FIG = _dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    df, maps, traces, T, D = load()
    df.to_parquet(L.OUT / f"readouts_v{L.CONFIG['version']}.parquet")
    st = stats_table(df)
    st.to_csv(L.OUT / f"stats_v{L.CONFIG['version']}.csv", index=False)
    show = st[st.condition.isin(["All", "W_all", "W_hit", "W_miss", "A_hit"]) & (st.pair == "SSp-bfd__wMO")]
    pd.set_option("display.width", 300)
    cols = ["condition", "kind", "stage", "cohort", "n_sessions", "baseline_ccv_mean", "FF_mean", "FF_wilcoxon_p",
            "FF_signflip_p", "FF_minus_FB_wilcoxon_p", "FF_minus_shuffle_wilcoxon_p", "CV_FF_mean", "CV_FF_wilcoxon_p",
            "peak_delay_ms_median", "mannwhitney_p", "welch_p"]
    print(show[[c for c in cols if c in show]].round(4).to_string())
    if a.stats_only:
        return
    fig_paper_maps().savefig(FIG / "00_paper_maps.png", dpi=150)
    for pair in PAIRS:
        for kind in ("total", "noise"):
            fig_maps(maps, T, D, df, pair, conds=("All", "W_all", "W_hit", "W_miss", "A_hit"), kind=kind).savefig(
                FIG / f"01_maps_{pair}_{kind}.png", dpi=130)
    for kind in ("total", "noise"):
        fig_traces(traces, T, df, kind=kind).savefig(FIG / f"02_traces_{kind}.png", dpi=140)
        for pair in PAIRS:
            fig_readouts(df, pair, kind).savefig(FIG / f"03_readouts_{pair}_{kind}.png", dpi=140)
        fig_dprime(df, kind=kind).savefig(FIG / f"04_dprime_{kind}.png", dpi=140)
    plt.close("all")


if __name__ == "__main__":
    main()
