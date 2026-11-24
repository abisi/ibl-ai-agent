"""Statistics + figures for the v3 baseline run (009): within- and across-area coupling in quiet time, R+ vs R-.
No multiple-comparison correction (user 2026-11-18: overview of effects first).

Per (level, condition, kind, area / pair, measure, band): session excess over the segment-shuffle null (value for
sync / rate), R+ vs R- with Mann-Whitney + Welch; learners and all mice; >= 3 sessions per cohort.
Omnibus per session (mean over its areas / pairs) with an OLS adjustment for quiescence (realised quiet window, 008)
and unit count. Within vs across: per session and area, within excess vs the mean across excess of its pairs (Wilcoxon).
Outputs: OUT/v3_baseline/stats/<pop>/*.csv, figures/<pop>/*.{png,pdf}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-spectral-coupling/010_baseline_stats.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_lib as sl  # noqa: E402

OUT = sl.OUT / "v3_baseline"
COL = {"R+": "#00B400", "R-": "#C800C8"}
MIN_S = 3
BANDS = list(sl.BANDS)
KEYS = ["level", "condition", "kind", "pair", "measure", "band"]
WITHIN_M = ["popcoh_within", "msc_within_matched", "msc_within", "pcoh_within", "sync"]
ACROSS_M = ["popcoh", "msc_matched", "msc", "pcoh", "imc"]


def two_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = dict(n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=np.nan, mean_Rminus=np.nan, diff=np.nan, p_mw=np.nan,
               p_welch=np.nan)
    if len(a) < MIN_S or len(b) < MIN_S:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.update(mean_Rplus=a.mean(), mean_Rminus=b.mean(), diff=a.mean() - b.mean(),
                   p_mw=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
                   p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue)
    return out


def load():
    d = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "sessions").glob("*_bands.parquet")) if "_quick" not in p.name],
                  ignore_index=True)
    d["excess"] = np.where(d.null_mean.notna(), d.value - d.null_mean, d.value)
    d["pair"] = np.where(d.kind == "across", d.source.astype(str) + " | " + d.target.astype(str), d.source.astype(str))
    q = pd.read_csv(OUT / "quiescence" / "sessions.csv")[["session_id", "quiet_beh_median", "quiet_min"]]
    return d.merge(q, on="session_id", how="left")


def cohort_tests(d):
    rows = []
    for k, g in d.groupby(KEYS):
        rows.append(dict(zip(KEYS, k), **two_tests(g.loc[g.cohort == "R+", "excess"], g.loc[g.cohort == "R-", "excess"])))
    return pd.DataFrame(rows)


def ols_cohort(y, cohort, quiet, nunits):
    """cohort coefficient (R+ - R-) and p with and without quiescence + log unit-count covariates."""
    import statsmodels.api as sm
    df = pd.DataFrame(dict(y=y, rp=(cohort == "R+").astype(float), q=quiet, n=np.log(nunits))).dropna()
    out = {}
    for name, cols in (("raw", ["rp"]), ("adj", ["rp", "q", "n"])):
        m = sm.OLS(df.y, sm.add_constant(df[cols])).fit()
        out[f"b_{name}"], out[f"p_{name}"] = m.params["rp"], m.pvalues["rp"]
    out["p_quiet"] = m.pvalues["q"]
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d0 = load()
    n_ses = d0.session_id.nunique()
    for pop in ("learners", "all"):
        d = d0[d0.learner] if pop == "learners" else d0
        so, fo = OUT / "stats" / pop, OUT / "figures" / pop
        so.mkdir(parents=True, exist_ok=True)
        fo.mkdir(parents=True, exist_ok=True)
        ct = cohort_tests(d)
        ct.to_csv(so / "cohort_tests.csv", index=False)

        # omnibus per session
        om = []
        for (lev, cond, kind, meas, band), g in d[d.kind != "area"].groupby(["level", "condition", "kind", "measure", "band"]):
            s = g.groupby(["session_id", "cohort"]).agg(y=("excess", "mean"), q=("quiet_beh_median", "first"),
                                                         n=("n_units_a", "mean")).reset_index()
            r = two_tests(s.loc[s.cohort == "R+", "y"], s.loc[s.cohort == "R-", "y"])
            try:
                r.update(ols_cohort(s.y.to_numpy(), s.cohort.to_numpy(), s.q.to_numpy(), s.n.to_numpy()))
            except Exception:  # noqa: BLE001
                pass
            om.append(dict(level=lev, condition=cond, kind=kind, measure=meas, band=band, **r))
        om = pd.DataFrame(om)
        om.to_csv(so / "omnibus.csv", index=False)

        # within vs across (per session and area)
        wa = []
        for (lev, cond, band), g in d.groupby(["level", "condition", "band"]):
            w = g[g.measure == "popcoh_within"].set_index(["session_id", "source"]).excess
            a = g[g.measure == "popcoh"]
            if not len(w) or not len(a):
                continue
            aa = pd.concat([a[["session_id", "source", "excess"]],
                            a[["session_id", "target", "excess"]].rename(columns={"target": "source"})])
            aa = aa.groupby(["session_id", "source"]).excess.mean()
            j = pd.concat([w.rename("within"), aa.rename("across")], axis=1).dropna().reset_index()
            j = j.merge(d[["session_id", "cohort"]].drop_duplicates(), on="session_id")
            for c in ("R+", "R-", "both"):
                jj = j if c == "both" else j[j.cohort == c]
                s = jj.groupby("session_id")[["within", "across"]].mean()
                if len(s) >= MIN_S:
                    wa.append(dict(level=lev, condition=cond, band=band, cohort=c, n=len(s), within=s.within.mean(),
                                   across=s.across.mean(), ratio=(s.within / s.across).median(),
                                   p_wilcoxon=stats.wilcoxon(s.within, s.across).pvalue))
        pd.DataFrame(wa).to_csv(so / "within_vs_across.csv", index=False)

        # ---- figures (area group, pooled) ----
        g = d[(d.level == "group") & (d.condition == "pooled")]
        # 1. within-area coupling per area, R+ vs R-, per band (popcoh_within and msc_within_matched)
        for meas in ("popcoh_within", "msc_within_matched", "sync"):
            gm = g[g.measure == meas]
            areas = sorted(gm.source.unique(), key=lambda a: -gm[gm.source == a].excess.mean())
            fig, axs = plt.subplots(len(BANDS), 1, figsize=(max(8, 0.55 * len(areas)), 2.0 * len(BANDS)), sharex=True)
            rng = np.random.default_rng(0)
            for ax, band in zip(axs, BANDS):
                for i, a in enumerate(areas):
                    for k, c in enumerate(("R+", "R-")):
                        v = gm[(gm.source == a) & (gm.band == band) & (gm.cohort == c)].excess.dropna()
                        x = i + (k - 0.5) * 0.35
                        ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=COL[c], alpha=0.5, lw=0)
                        if len(v):
                            ax.plot([x - 0.13, x + 0.13], [v.median()] * 2, color="k", lw=1)
                    r = ct[(ct.level == "group") & (ct.condition == "pooled") & (ct.pair == a) & (ct.measure == meas)
                           & (ct.band == band)]
                    if len(r) and r.p_mw.iloc[0] < 0.05:
                        ax.text(i, 0.92, "*", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
                ax.set_ylabel(band, fontsize=7)
                ax.axhline(0, color="0.7", lw=0.5)
                ax.spines[["top", "right"]].set_visible(False)
            axs[-1].set_xticks(range(len(areas)), areas, rotation=45, ha="right", fontsize=7)
            fig.suptitle(f"{meas} (excess over shuffle), quiet baseline, area groups, {pop}; * MW p < 0.05 uncorrected",
                         fontsize=9)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(fo / f"within_{meas}.{ext}", dpi=180)
            plt.close(fig)
        # 2. across matrices: R+, R-, difference, -log10 p (popcoh, msc_matched) per band
        for meas in ("popcoh", "msc_matched", "pcoh"):
            ga = g[g.measure.isin([meas, meas + "_within" if meas != "pcoh" else "pcoh_within"])]
            areas = sorted(set(ga.source) | set(ga.target))
            fig, axs = plt.subplots(len(BANDS), 4, figsize=(17, 3.6 * len(BANDS)))
            for r_, band in enumerate(BANDS):
                M = {c: np.full((len(areas), len(areas)), np.nan) for c in ("R+", "R-")}
                P = np.full((len(areas), len(areas)), np.nan)
                gb = ga[ga.band == band]
                for (s_, t_), gg in gb.groupby(["source", "target"]):
                    i, j = areas.index(s_), areas.index(t_)
                    for c in M:
                        M[c][i, j] = M[c][j, i] = gg.loc[gg.cohort == c, "excess"].mean()
                    P[i, j] = P[j, i] = two_tests(gg.loc[gg.cohort == "R+", "excess"], gg.loc[gg.cohort == "R-", "excess"])["p_mw"]
                off = ~np.eye(len(areas), dtype=bool)
                vmax = np.nanpercentile(np.r_[M["R+"][off], M["R-"][off]], 98) if np.isfinite(M["R+"][off]).any() else 1
                D = M["R+"] - M["R-"]
                dm = np.nanmax(np.abs(D[off])) if np.isfinite(D[off]).any() else 1
                for ax, (img, cmap, lo, hi, ttl) in zip(axs[r_], [
                        (M["R+"], "viridis", 0, vmax, "R+"), (M["R-"], "viridis", 0, vmax, "R−"),
                        (D, "PiYG", -dm, dm, "R+ − R−"), (-np.log10(P), "Greys", 0, 3, "−log10 p (MW)")]):
                    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
                    ax.set_xticks(range(len(areas)), areas, rotation=90, fontsize=5)
                    ax.set_yticks(range(len(areas)), areas, fontsize=5)
                    ax.set_title(f"{ttl}, {band}", fontsize=8)
                    plt.colorbar(im, ax=ax, fraction=0.04)
            fig.suptitle(f"{meas} excess, quiet baseline, area groups ({pop}); diagonal = within area; uncorrected",
                         fontsize=10)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(fo / f"matrix_{meas}.{ext}", dpi=150)
            plt.close(fig)
        # 3. p-value histograms (cohort tests, area group + acronym, pooled), per measure
        meas_all = WITHIN_M + ACROSS_M
        fig, axs = plt.subplots(2, len(meas_all) // 2, figsize=(18, 6))
        for ax, m in zip(axs.ravel(), meas_all):
            for lev, ls in (("group", "-"), ("acronym", "--")):
                p = ct[(ct.measure == m) & (ct.condition == "pooled") & (ct.level == lev)].p_mw.dropna()
                if len(p):
                    ax.hist(p, bins=20, range=(0, 1), histtype="step", density=True, ls=ls,
                            label=f"{lev} (n={len(p)}, {100 * (p < 0.05).mean():.0f} % < .05)")
            ax.axhline(1, color="0.6", lw=0.5)
            ax.set_title(m, fontsize=8)
            ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"R+ vs R− p-values (MW), quiet baseline, all bands ({pop}); uniform = no effect", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fo / f"pvalue_hist.{ext}", dpi=150)
        plt.close(fig)
        # 4. within-area spectra per cohort (popcoh_within) for the 6 best-sampled groups + omnibus
        sp = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "sessions").glob("*_spectra.parquet"))
                        if "_quick" not in p.name], ignore_index=True)
        sp = sp[(sp.condition == "pooled") & (sp.measure.isin(["popcoh_within", "popcoh"]))]
        if pop == "learners":
            sp = sp[sp.learner]
        sp["excess"] = sp.value - sp.null_mean
        top = sp[sp.measure == "popcoh_within"].groupby("source").session_id.nunique().nlargest(8).index
        fig, axs = plt.subplots(2, 5, figsize=(17, 6), sharex=True)
        for ax, a in zip(axs.ravel(), list(top) + ["all areas (within)", "all pairs (across)"]):
            for c in ("R+", "R-"):
                if a == "all areas (within)":
                    s = sp[(sp.measure == "popcoh_within") & (sp.cohort == c)].groupby(["session_id", "freq"]).excess.mean()
                elif a == "all pairs (across)":
                    s = sp[(sp.measure == "popcoh") & (sp.cohort == c)].groupby(["session_id", "freq"]).excess.mean()
                else:
                    s = sp[(sp.measure == "popcoh_within") & (sp.source == a) & (sp.cohort == c)].set_index(
                        ["session_id", "freq"]).excess
                s = s.unstack("freq")
                mu, se = s.mean(), s.sem()
                ax.plot(mu.index, mu, color=COL[c], lw=1, label=f"{c} (n={len(s)})")
                ax.fill_between(mu.index, mu - se, mu + se, color=COL[c], alpha=0.2, lw=0)
            ax.set_xscale("log")
            ax.set_title(a, fontsize=8)
            ax.legend(fontsize=6, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel("frequency (Hz)")
        fig.suptitle(f"population coherency (excess) in quiet time, area groups ({pop}; mean ± s.e.m.)", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fo / f"spectra_popcoh.{ext}", dpi=150)
        plt.close(fig)
        print(pop, "sessions", d.session_id.nunique(), "of", n_ses, flush=True)
        print(om[(om.condition == "pooled") & om.measure.isin(["popcoh", "popcoh_within", "msc_matched",
                                                                 "msc_within_matched", "pcoh", "pcoh_within"])]
              [["level", "kind", "measure", "band", "n_Rplus", "n_Rminus", "mean_Rplus", "mean_Rminus", "p_mw",
                "p_welch", "p_raw", "p_adj", "p_quiet"]].round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
