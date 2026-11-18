"""Exploratory R+ vs R- statistics on the v2 runs (user 2026-11-18): no multiple-comparison correction, mouse level.

Inputs:  OUT/v2_<align>/sessions/*_{bands,spectra}.parquet (002_run_spikes.py --align stim|jaw)
Unit:    mouse (sessions of a mouse averaged; whisker day 0 has one session per mouse)
Values:  excess = value - shuffle-null mean
Focus:   msc (5-ms bins, multitaper) and corr50 (Pearson r of area sum residuals in 50-ms bins)
Scopes:  between = pairs of different areas; within = two disjoint unit halves of the same area ("<area> (within)")
Tests:   Mann-Whitney + Welch, uncorrected; a result is flagged when both p < 0.05; cohorts need >= 3 mice
Pops:    all mice, learners
Outputs: OUT/v2_<align>/stats/<pop>/{pair_tests,within_tests,omnibus_tests}.csv
         OUT/v2_<align>/figures/<pop>/{e1_msc_spectra,e2_time_resolved,e3_within_areas,e4_pairs_<window>}.{png,pdf,svg}
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-spectral-coupling/007_exploratory_stats.py --align stim
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_lib as sl  # noqa: E402

WITHIN = " (within)"
FOCUS = ("msc", "corr50")
COL = {"R+": "#00B400", "R-": "#C800C8"}
MIN_MICE = 3
MAIN = {"stim": [("quiet", "ALL"), ("early", "W_lick"), ("early", "W_nolick"), ("execution", "W_lick"),
                 ("early", "A_hit"), ("execution", "A_hit")],
        "jaw": [("pre", "W_lick"), ("post", "W_lick"), ("pre", "A_hit"), ("post", "A_hit")]}
KEYS = ["level", "window", "condition", "measure", "band"]


def two_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_MICE or len(b) < MIN_MICE:
        return dict(n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=np.nan, mean_Rminus=np.nan, diff=np.nan,
                    p_mw=np.nan, t_welch=np.nan, p_welch=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, pw = stats.ttest_ind(a, b, equal_var=False)
        pm = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    return dict(n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=a.mean(), mean_Rminus=b.mean(), diff=a.mean() - b.mean(),
                p_mw=pm, t_welch=t, p_welch=pw)


def load(align):
    d = pd.concat([pd.read_parquet(p) for p in sorted((sl.OUT / f"v2_{align}" / "sessions").glob("*_bands.parquet"))
                   if "_quick" not in p.name], ignore_index=True)
    d = d[d.measure.isin(FOCUS) | d.measure.isin(["rate", "power"])]
    d["excess"] = d.value - d.null_mean
    d["within"] = d.source.astype(str).str.endswith(WITHIN)
    d["area"] = d.source.astype(str).str.replace(WITHIN, "", regex=False)
    d["pair"] = np.where(d.within, d.area, d.source.astype(str) + " | " + d.target.astype(str))
    return d


def mouse_level(d, cols):
    return d.groupby(["mouse_id", "cohort", "learner"] + cols, observed=True, dropna=False).excess.mean().reset_index()


def test_by(m, by):
    rows = []
    for k, g in m.groupby(by, observed=True):
        r = dict(zip(by, k if isinstance(k, tuple) else (k,)))
        r.update(two_tests(g[g.cohort == "R+"].excess, g[g.cohort == "R-"].excess))
        rows.append(r)
    t = pd.DataFrame(rows)
    t["both_p05"] = (t.p_mw < 0.05) & (t.p_welch < 0.05)
    return t


# ------------------------------------------------------------------------------------------------ figures

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.family": "Arial"})
    return plt


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")


def mean_sem(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (np.nan, np.nan)


def star(ax, x, p_mw, p_w, y):
    if p_mw < 0.05 and p_w < 0.05:
        ax.plot(x, y, "k*", ms=4)
    elif min(p_mw, p_w) < 0.05:
        ax.plot(x, y, marker="o", mfc="none", mec="0.4", ms=3, ls="none")


def fig_spectra(align, pop, mice, plt, od):
    """MSC excess spectra (group level), between (mean over pairs) and within (mean over areas), per main cell."""
    sp = pd.concat([pd.read_parquet(p) for p in sorted((sl.OUT / f"v2_{align}" / "sessions").glob("*_spectra.parquet"))
                    if "_quick" not in p.name], ignore_index=True)
    sp = sp[(sp.measure == "msc") & sp.mouse_id.isin(mice)]
    sp["excess"] = sp.value - sp.null_mean
    sp["within"] = sp.source.astype(str).str.endswith(WITHIN)
    cells = MAIN[align]
    fig, axs = plt.subplots(2, len(cells), figsize=(2.3 * len(cells), 4.4), squeeze=False)
    for j, (w, c) in enumerate(cells):
        for i, wi in enumerate((False, True)):
            ax = axs[i, j]
            x = sp[(sp.window == w) & (sp.condition == c) & (sp.within == wi)]
            m = x.groupby(["mouse_id", "cohort", "freq"]).excess.mean().reset_index()
            for coh in ("R+", "R-"):
                g = m[m.cohort == coh].pivot(index="mouse_id", columns="freq", values="excess")
                if g.empty:
                    continue
                mu, se = g.mean(), g.sem()
                ax.plot(mu.index, mu, color=COL[coh], lw=1, label=f"{coh} (n={len(g)})")
                ax.fill_between(mu.index, mu - se, mu + se, color=COL[coh], alpha=0.25, lw=0)
            top = ax.get_ylim()[1]
            for fq, g in m.groupby("freq"):
                t = two_tests(g[g.cohort == "R+"].excess, g[g.cohort == "R-"].excess)
                if np.isfinite(t["p_mw"]):
                    star(ax, fq, t["p_mw"], t["p_welch"], top)
            ax.axhline(0, color="0.6", lw=0.5, ls=":")
            ax.set_title(f"{c}, {w}\n{'within area' if wi else 'between areas'}", fontsize=8)
            ax.set_xlabel("frequency (Hz)")
            if j == 0:
                ax.set_ylabel("MSC excess over shuffle")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Coherence spectra, mouse level ({pop}); * both tests p < 0.05, o one test (uncorrected)", fontsize=9)
    fig.tight_layout()
    save(fig, od / "e1_msc_spectra")
    plt.close(fig)


def fig_time_resolved(tr, align, pop, plt, od):
    """Time-resolved MSC (bands) and corr50, between and within, group level, per condition."""
    conds = [c for c in (["W_lick", "W_nolick", "A_hit", "NS_cr"] if align == "stim" else ["W_lick", "A_hit", "NS_fa"])
             if c in set(tr.condition)]
    rows = [("msc", b) for b in ("beta", "gamma_low", "gamma_high")] + [("corr50", "all")]
    for scope in ("between", "within"):
        fig, axs = plt.subplots(len(rows), len(conds), figsize=(2.4 * len(conds), 1.7 * len(rows)), squeeze=False)
        for i, (meas, band) in enumerate(rows):
            for j, c in enumerate(conds):
                ax = axs[i, j]
                x = tr[(tr.measure == meas) & (tr.band == band) & (tr.condition == c) & (tr.within == (scope == "within"))]
                m = x.groupby(["mouse_id", "cohort", "t"]).excess.mean().reset_index()
                for coh in ("R+", "R-"):
                    g = m[m.cohort == coh].pivot(index="mouse_id", columns="t", values="excess")
                    if g.empty:
                        continue
                    mu, se = g.mean(), g.sem()
                    ax.plot(mu.index, mu, color=COL[coh], lw=1, label=f"{coh} (n={len(g)})")
                    ax.fill_between(mu.index, mu - se, mu + se, color=COL[coh], alpha=0.25, lw=0)
                if m.empty:
                    continue
                top = ax.get_ylim()[1]
                for t, g in m.groupby("t"):
                    r = two_tests(g[g.cohort == "R+"].excess, g[g.cohort == "R-"].excess)
                    if np.isfinite(r["p_mw"]):
                        star(ax, t, r["p_mw"], r["p_welch"], top)
                ax.axvline(0, color="0.5", lw=0.5, ls="--")
                ax.axhline(0, color="0.6", lw=0.5, ls=":")
                if i == 0:
                    ax.set_title(c, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{meas}{'' if band == 'all' else ' ' + band}\nexcess")
                if i == len(rows) - 1:
                    ax.set_xlabel("window centre from " + ("stimulus" if align == "stim" else "jaw onset") + " (s)")
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=6)
        fig.suptitle(f"Time-resolved coupling, {scope} area groups, mouse level ({pop}); 200-ms windows; "
                     "* both tests p < 0.05, o one test (uncorrected)", fontsize=9)
        fig.tight_layout()
        save(fig, od / f"e2_time_resolved_{scope}")
        plt.close(fig)


def fig_within(wt, align, pop, plt, od):
    """Within-area coupling per area group: R+ - R- difference with markers, msc beta / low gamma and corr50."""
    cells = MAIN[align]
    rows = [("msc", "beta"), ("msc", "gamma_low"), ("corr50", "all")]
    x = wt[(wt.level == "group")]
    areas = sorted(x.area.unique())
    fig, axs = plt.subplots(len(rows), len(cells), figsize=(2.3 * len(cells), 0.25 * len(areas) * len(rows) + 1),
                            squeeze=False, sharey=True)
    for i, (meas, band) in enumerate(rows):
        for j, (w, c) in enumerate(cells):
            ax = axs[i, j]
            g = x[(x.measure == meas) & (x.band == band) & (x.window == w) & (x.condition == c)].set_index("area")
            for k, a in enumerate(areas):
                if a not in g.index or not np.isfinite(g.loc[a, "diff"]):
                    continue
                r = g.loc[a]
                colr = COL["R+"] if r["diff"] > 0 else COL["R-"]
                ax.barh(k, r["diff"], color=colr, alpha=0.8 if r.both_p05 else 0.3)
                if r.both_p05:
                    ax.text(r["diff"], k, " *", va="center", fontsize=7)
            ax.axvline(0, color="0.5", lw=0.5)
            ax.set_yticks(range(len(areas)))
            ax.set_yticklabels(areas, fontsize=6)
            if i == 0:
                ax.set_title(f"{c}, {w}", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("R+ − R− (excess)")
            if j == 0:
                ax.set_ylabel(f"{meas}{'' if band == 'all' else ' ' + band}")
    fig.suptitle(f"Within-area coupling (two disjoint unit halves), R+ − R−, mouse level ({pop}); "
                 "* both tests p < 0.05 (uncorrected)", fontsize=9)
    fig.tight_layout()
    save(fig, od / "e3_within_areas")
    plt.close(fig)


def fig_pairs(pt, align, pop, plt, od):
    """Between-area pair matrices (group level): t (Welch) of R+ vs R-, msc beta / low gamma / corr50, main cells."""
    x = pt[pt.level == "group"]
    areas = sorted({a for p in x.pair for a in p.split(" | ")})
    ix = {a: i for i, a in enumerate(areas)}
    rows = [("msc", "beta"), ("msc", "gamma_low"), ("corr50", "all")]
    cells = MAIN[align]
    fig, axs = plt.subplots(len(rows), len(cells), figsize=(2.6 * len(cells), 2.6 * len(rows)), squeeze=False)
    for i, (meas, band) in enumerate(rows):
        for j, (w, c) in enumerate(cells):
            ax = axs[i, j]
            M = np.full((len(areas), len(areas)), np.nan)
            g = x[(x.measure == meas) & (x.band == band) & (x.window == w) & (x.condition == c)]
            for r in g.itertuples():
                a, b = r.pair.split(" | ")
                M[ix[a], ix[b]] = M[ix[b], ix[a]] = r.t_welch
                if r.both_p05:
                    ax.text(ix[b], ix[a], "*", ha="center", va="center", fontsize=7)
                    ax.text(ix[a], ix[b], "*", ha="center", va="center", fontsize=7)
            im = ax.imshow(M, cmap="PiYG", vmin=-4, vmax=4)
            ax.set_xticks(range(len(areas)))
            ax.set_yticks(range(len(areas)))
            ax.set_xticklabels(areas, rotation=90, fontsize=5)
            ax.set_yticklabels(areas if j == 0 else [], fontsize=5)
            ax.set_title(f"{meas}{'' if band == 'all' else ' ' + band}: {c}, {w}", fontsize=7)
    fig.colorbar(im, ax=axs, shrink=0.4, label="Welch t (green R+ > R−, magenta R− > R+)")
    fig.suptitle(f"Between-area pairs, mouse level ({pop}); * both tests p < 0.05 (uncorrected)", fontsize=9)
    save(fig, od / "e4_pairs")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", choices=["stim", "jaw"], required=True)
    args = ap.parse_args()
    d = load(args.align)
    plt = setup_mpl()
    root = sl.OUT / f"v2_{args.align}"
    for pop in ("all", "learners"):
        x = d if pop == "all" else d[d.learner]
        sd, fd = root / "stats" / pop, root / "figures" / pop
        sd.mkdir(parents=True, exist_ok=True)
        coupling = x[x.measure.isin(FOCUS)]
        ep = coupling[~coupling.window.str.startswith("tr_")]
        mp = mouse_level(ep, KEYS + ["pair", "within", "area"])
        pt = test_by(mp[~mp.within], KEYS + ["pair"])
        wt = test_by(mp[mp.within], KEYS + ["area"])
        # omnibus: per mouse, mean over its pairs (between) or areas (within), then test
        rows = []
        for scope, sub in (("between", mp[~mp.within]), ("within", mp[mp.within])):
            om = sub.groupby(["mouse_id", "cohort"] + KEYS, observed=True).excess.mean().reset_index()
            t = test_by(om, KEYS)
            t.insert(0, "scope", scope)
            rows.append(t)
        ot = pd.concat(rows, ignore_index=True)
        pt.to_csv(sd / "pair_tests.csv", index=False)
        wt.to_csv(sd / "within_tests.csv", index=False)
        ot.to_csv(sd / "omnibus_tests.csv", index=False)
        n_m = x.groupby("cohort").mouse_id.nunique().to_dict()
        print(f"{args.align} {pop}: mice {n_m}; pair tests {len(pt)} (both p<.05: {int(pt.both_p05.sum())}, "
              f"{pt.both_p05.mean():.3f}); within tests {len(wt)} ({int(wt.both_p05.sum())}); "
              f"omnibus {len(ot)} ({int(ot.both_p05.sum())})", flush=True)
        main_cells = ot[ot.apply(lambda r: (r.window, r.condition) in MAIN[args.align], axis=1) & (ot.level == "group")]
        print(main_cells[main_cells.both_p05][["scope", "window", "condition", "measure", "band", "mean_Rplus",
                                               "mean_Rminus", "p_mw", "p_welch"]].round(4).to_string(), flush=True)
        # time-resolved table (group level)
        tr = coupling[coupling.window.str.startswith("tr_") & (coupling.level == "group")].copy()
        tr["t"] = tr.window.str[3:].astype(float)
        trm = mouse_level(tr, KEYS + ["t", "within"])
        fig_spectra(args.align, pop, set(x.mouse_id), plt, fd)
        fig_time_resolved(trm, args.align, pop, plt, fd)
        fig_within(wt, args.align, pop, plt, fd)
        fig_pairs(pt, args.align, pop, plt, fd)
        tt = test_by(trm, KEYS + ["t", "within"])
        tt.to_csv(sd / "time_resolved_tests.csv", index=False)


if __name__ == "__main__":
    main()
