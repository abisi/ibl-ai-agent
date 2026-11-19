"""Readouts, statistics and figures: SSL S1-M2 time-shifted CCA vs the paper (001 reference).

Per session x pair x condition x kind (total / noise): paper readout (s1m2_lib.readout) for feedforward (delays 0..+15
ms), feedback (-15..0 ms) and the trial-shuffle map; peak delay of the baseline-subtracted map in 0..100 ms.
"pooled" pair = the mean over the session's available M2 areas (MO-wM1, MO-wM2).
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

MAPS, FIG = L.OUT / "maps", L.OUT / "figures"
COL = {"R+": "#00B400", "R-": "#C800C8"}
LS = {"learning": "-", "expert": "--"}
CONDS = ["All", "W_all", "W_hit", "W_miss", "A_hit", "NS_all"]
RO = L.CONFIG["readout"]


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
    rows, maps, traces = [], {}, {}
    for f in sorted(MAPS.glob("*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        t, d = z["times"], z["delays"]
        b = (t >= RO["baseline_s"][0] - 1e-9) & (t <= RO["baseline_s"][1] + 1e-9)
        e = (t >= RO["early_s"][0] - 1e-9) & (t <= RO["early_s"][1] + 1e-9)
        for cond in CONDS:
            for kind in ("total", "noise"):
                k = f"{cond}__{kind}__ccv"
                if k not in z.files:
                    continue
                c, s = z[k].astype(float), z[f"{cond}__{kind}__shuffle"].astype(float)
                ff, ff_tr, ff_raw = L.readout(c, d, t, RO["ff_delays_s"])
                fb, fb_tr, _ = L.readout(c, d, t, RO["fb_delays_s"])
                sff, sff_tr, _ = L.readout(s, d, t, RO["ff_delays_s"])
                cb = c - np.nanmean(c[:, b], axis=1, keepdims=True)
                prof = np.nanmean(cb[:, e], axis=1)
                win = np.abs(d) <= 0.05
                peak = float(d[win][np.nanargmax(prof[win])]) if np.isfinite(prof[win]).any() else np.nan
                rows.append(dict(meta, condition=cond, kind=kind, n_trials=int(z[f"{cond}__n_trials"]),
                                 FF=ff, FB=fb, FF_shuffle=sff, FF_minus_FB=ff - fb, FF_minus_shuffle=ff - sff,
                                 baseline_ccv=float(np.nanmean(ff_raw[b])), peak_delay_ms=peak * 1000))
                key = (meta["session_id"], meta["pair"], cond, kind)
                maps[key] = cb
                traces[key] = (ff_tr, fb_tr, sff_tr)
        T, D = t, d
    df = pd.DataFrame(rows)
    # d' (whisker hit vs no-stim false alarm), rates clipped as in log-linear correction bounds
    hr, fa = df.hit_rate_W.clip(0.01, 0.99), df.fa_rate.clip(0.01, 0.99)
    df["dprime_W"] = norm.ppf(hr) - norm.ppf(fa)
    return df, maps, traces, T, D


def pooled(df):
    num = ["FF", "FB", "FF_shuffle", "FF_minus_FB", "FF_minus_shuffle", "baseline_ccv", "peak_delay_ms", "n_trials"]
    keys = ["session_id", "mouse_id", "day", "stage", "cohort", "learner", "condition", "kind", "dprime_W"]
    p = df.groupby(keys, dropna=False)[num].mean().reset_index()
    p["pair"] = "SSp-bfd__wMO (pooled)"
    return pd.concat([df, p], ignore_index=True)


def stats_table(df):
    out = []
    for (pair, cond, kind, stage), g in df.groupby(["pair", "condition", "kind", "stage"]):
        for coh in ("R+", "R-"):
            h = g[g.cohort == coh]
            r = dict(pair=pair, condition=cond, kind=kind, stage=stage, cohort=coh, n_sessions=len(h),
                     n_mice=h.mouse_id.nunique())
            for col in ("FF", "FB", "FF_minus_FB", "FF_minus_shuffle"):
                r[f"{col}_mean"] = h[col].mean()
                r[f"{col}_wilcoxon_p"] = wil(h[col])
                r[f"{col}_signflip_p"] = sign_flip_p(h, col)
            r["peak_delay_ms_median"] = h.peak_delay_ms.median()
            out.append(r)
        a, b = g[g.cohort == "R+"].FF.dropna(), g[g.cohort == "R-"].FF.dropna()
        if len(a) >= 3 and len(b) >= 3:
            out.append(dict(pair=pair, condition=cond, kind=kind, stage=stage, cohort="R+ vs R-",
                            n_sessions=len(a) + len(b), FF_mean=a.mean() - b.mean(),
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


def fig_traces(traces, T, df, kind="total", conds=("W_all", "W_hit", "W_miss", "A_hit")):
    tr = np.load(L.OUT / "paper_reference" / "traces.npz")
    pairs = ["SSp-bfd__MO-wM1", "SSp-bfd__MO-wM2"]
    fig, axs = plt.subplots(len(pairs) + 1, len(conds), figsize=(3.3 * len(conds), 2.6 * (len(pairs) + 1)),
                            constrained_layout=True, sharex=True)
    pk = "Total" if kind == "total" else "Noise"
    pc = {"W_all": "Go", "W_hit": "hit", "W_miss": "miss", "A_hit": "NoGo"}
    for j, cond in enumerate(conds):
        ax = axs[0, j]
        cc = pc[cond]
        for direction, ls in (("FF", "-"), ("FB", ":")):
            x = tr[f"{pk}_{cc}_{direction}"]
            m, se = np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(len(x))
            tp = tr[f"{pk}_{cc}_t"]
            ax.plot(tp * 1e3, m, "k", ls=ls, label=f"paper {cc} {direction}")
            ax.fill_between(tp * 1e3, m - se, m + se, color="k", alpha=0.15)
        ax.set_title(f"Paper {cc} ({pk.lower()})", fontsize=9)
        ax.legend(fontsize=6)
        for i, pair in enumerate(pairs, start=1):
            ax = axs[i, j]
            for st in ("learning", "expert"):
                for co in ("R+", "R-"):
                    sess = df[(df.pair == pair) & (df.stage == st) & (df.cohort == co)].session_id.unique()
                    x = [traces[(s, pair, cond, kind)][0] for s in sess if (s, pair, cond, kind) in traces]
                    if not x:
                        continue
                    x = np.array(x)
                    m, se = np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(len(x))
                    ax.plot(T * 1e3, m, color=COL[co], ls=LS[st], lw=1.2, label=f"{co} {st} n={len(x)}")
                    ax.fill_between(T * 1e3, m - se, m + se, color=COL[co], alpha=0.12)
            ax.set_title(f"{pair.split('__')[1]}: {cond} FF ({kind})", fontsize=9)
            ax.legend(fontsize=6)
    for ax in axs.ravel():
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.axvspan(0, 100, color="gold", alpha=0.1)
        ax.set_xlim(-250, 600)
    for ax in axs[-1]:
        ax.set_xlabel("Time from stimulus (ms)")
    for ax in axs[:, 0]:
        ax.set_ylabel("CCV - baseline (delays 0..15 ms)")
    return fig


def fig_readouts(df, kind="total", cond="W_all"):
    ref = pd.read_parquet(L.OUT / "paper_reference" / "per_session.parquet")
    pk = "Total" if kind == "total" else "Noise"
    pairs = ["SSp-bfd__MO-wM1", "SSp-bfd__MO-wM2", "SSp-bfd__wMO (pooled)"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for ax, col in zip(axs, ("FF", "FB")):
        x0 = 0
        for pc, lab in (("Go", "paper Go"), ("NoGo", "paper NoGo")):
            v = ref[(ref["corr"] == pk) & (ref.condition == pc) & (ref.direction == col)].value
            ax.scatter(np.full(len(v), x0) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v, s=8, c="k", alpha=.5)
            ax.errorbar(x0, v.mean(), v.std() / np.sqrt(len(v)), color="k", capsize=3, marker="o")
            ax.text(x0, ax.get_ylim()[1] if False else v.max(), lab, ha="center", va="bottom", fontsize=7)
            x0 += 1
        ticks, labels = [0, 1], ["paper Go", "paper NoGo"]
        for pair in pairs:
            x0 += 0.5
            for st in ("learning", "expert"):
                for co in ("R+", "R-"):
                    v = df[(df.pair == pair) & (df.stage == st) & (df.cohort == co) & (df.condition == cond)
                           & (df.kind == kind)][col].dropna()
                    if len(v):
                        ax.scatter(np.full(len(v), x0) + np.random.default_rng(1).uniform(-.15, .15, len(v)), v, s=10,
                                   c=COL[co], alpha=.6, marker="o" if st == "learning" else "^")
                        ax.errorbar(x0, v.mean(), v.std() / np.sqrt(max(len(v), 1)), color=COL[co], capsize=3,
                                    marker="o" if st == "learning" else "^", mec="k")
                    ticks.append(x0)
                    labels.append(f"{pair.split('__')[1].replace(' (pooled)', '*')}\n{co} {st[:5]} n={len(v)}")
                    x0 += 1
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
        ax.set_ylabel(f"{col} CCV, 0-100 ms minus baseline")
        ax.set_title(f"{col} ({'delays 0..+15 ms' if col == 'FF' else 'delays -15..0 ms'}), {kind}, SSL {cond}",
                     fontsize=9)
    return fig


def fig_dprime(df, kind="total", cond="W_all"):
    ref = pd.read_parquet(L.OUT / "paper_reference" / "per_session.parquet")
    pk = "Total" if kind == "total" else "Noise"
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    r = ref[(ref["corr"] == pk) & (ref.condition == "All") & (ref.direction == "FF")].dropna(subset=["dprime"])
    rr = stats.pearsonr(r.value, r.dprime)
    axs[0].scatter(r.value, r.dprime, c="k", s=10)
    axs[0].set_title(f"Paper All: r={rr[0]:.2f}, p={rr[1]:.1g}, n={len(r)}", fontsize=9)
    axs[0].set_ylabel("d'")
    for ax, st in zip(axs[1:], ("learning", "expert")):
        g = df[(df.pair == "SSp-bfd__wMO (pooled)") & (df.stage == st) & (df.condition == cond) & (df.kind == kind)]
        txt = []
        for co in ("R+", "R-"):
            h = g[g.cohort == co].dropna(subset=["FF", "dprime_W"])
            ax.scatter(h.FF, h.dprime_W, c=COL[co], s=14)
            if len(h) >= 4:
                q = stats.pearsonr(h.FF, h.dprime_W)
                txt.append(f"{co} r={q[0]:.2f} p={q[1]:.2g} n={len(h)}")
        ax.set_title(f"SSL {st}, pooled pair, {cond}\n" + "; ".join(txt), fontsize=8)
        ax.set_ylabel("d' (whisker hit vs no-stim FA)")
    for ax in axs:
        ax.set_xlabel("FF CCV, 0-100 ms minus baseline")
    return fig


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, maps, traces, T, D = load()
    df = pooled(df)
    df.to_parquet(L.OUT / "readouts.parquet")
    st = stats_table(df)
    st.to_csv(L.OUT / "stats.csv", index=False)
    show = st[(st.condition.isin(["W_all", "All", "W_hit", "A_hit"]))]
    pd.set_option("display.width", 250)
    cols = ["pair", "condition", "kind", "stage", "cohort", "n_sessions", "FF_mean", "FF_wilcoxon_p", "FF_signflip_p",
            "FB_mean", "FF_minus_FB_wilcoxon_p", "FF_minus_shuffle_wilcoxon_p", "peak_delay_ms_median",
            "mannwhitney_p", "welch_p"]
    print(show[[c for c in cols if c in show]].round(4).to_string())
    fig_paper_maps().savefig(FIG / "00_paper_maps.png", dpi=150)
    for pair in ("SSp-bfd__MO-wM1", "SSp-bfd__MO-wM2"):
        for kind in ("total", "noise"):
            fig_maps(maps, T, D, df, pair, conds=("W_all", "W_hit", "W_miss", "A_hit", "All"), kind=kind).savefig(
                FIG / f"01_maps_{pair}_{kind}.png", dpi=130)
    for kind in ("total", "noise"):
        fig_traces(traces, T, df, kind).savefig(FIG / f"02_traces_{kind}.png", dpi=140)
        for cond in ("W_all", "W_hit", "All"):
            fig_readouts(df, kind, cond).savefig(FIG / f"03_readouts_{kind}_{cond}.png", dpi=140)
        fig_dprime(df, kind).savefig(FIG / f"04_dprime_{kind}.png", dpi=140)
    plt.close("all")


if __name__ == "__main__":
    main()
