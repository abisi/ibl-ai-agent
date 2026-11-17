"""Statistics and figures for ssl-spectral-coupling (pre-registered tests H1-H3 in question.md).

Inputs: OUT/<source>/sessions/*_{bands,spectra}.parquet (002, or the AP-band source).
Per population (learners = primary, all = sensitivity):
  pair tests      per (level, window, condition, measure, band, pair): session excess (value - shuffle mean), R+ vs R-
                  Mann-Whitney + Welch, both cohorts >= 3 sessions; BH over pairs within each cell
  contrasts       whisker - auditory per session (W_lick - A_hit, W_nolick - A_hit) in early / execution
  omnibus         one value per session = mean excess over its cross-area pairs, raw and pair-centred (each pair's
                  across-session mean removed first, so cohorts are not compared on which pairs they happen to have)
  spectra         MSC / cancoh / pcoh excess spectra (group level), session mean over pairs; frequency-cluster
                  permutation (Welch t, |t| > 2, cluster mass, 2000 cohort-label permutations) on the omnibus and per pair
  time-resolved   omnibus excess per band over 200-ms windows, cluster permutation over time
  confounds       firing rate and sum-signal power per area group by cohort
Outputs: OUT/<source>/stats/<population>/*.csv, OUT/<source>/figures/<population>/*.png|pdf|svg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_lib as sl  # noqa: E402

COL = {"R+": "#00B400", "R-": "#C800C8"}
COUPLING = ["msc", "cancoh", "pcoh", "imc", "psi", "gc_ab", "gc_ba", "gc_dir"]
PRIMARY = {("quiet", "ALL"), ("early", "W_lick"), ("early", "W_nolick"), ("execution", "W_lick")}
EVENT_CONDS = ["W_lick", "W_nolick", "A_hit", "NS_cr"]
N_PERM = 2000
WITHIN = " (within)"


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def two_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=np.nan, mean_Rminus=np.nan, diff=np.nan,
                    p_mw=np.nan, p_welch=np.nan, t_welch=np.nan)
    return dict(n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=a.mean(), mean_Rminus=b.mean(), diff=a.mean() - b.mean(),
                p_mw=stats.mannwhitneyu(a, b).pvalue, t_welch=stats.ttest_ind(a, b, equal_var=False).statistic,
                p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue)


def load(source):
    od = sl.OUT / source / "sessions"
    files = sorted(p for p in od.glob("*_bands.parquet") if "_quick" not in p.name)
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    sp = [p for p in od.glob("*_spectra.parquet") if "_quick" not in p.name]
    s = pd.concat([pd.read_parquet(p) for p in sp], ignore_index=True) if sp else pd.DataFrame()
    assert not d.duplicated(["session_id", "level", "condition", "window", "source", "target", "measure", "band"]).any()
    d["excess"] = d.value - d.null_mean
    d.loc[d.measure.isin(["power", "rate"]), "excess"] = d.value
    # GC direction index: excess a->b minus excess b->a
    gc = d[d.measure.isin(["gc_ab", "gc_ba"])].pivot_table(
        index=["session_id", "mouse_id", "cohort", "learner", "level", "condition", "window", "source", "target", "band"],
        columns="measure", values="excess").reset_index()
    gc["excess"] = gc.gc_ab - gc.gc_ba
    gc["measure"] = "gc_dir"
    d = pd.concat([d, gc.drop(columns=["gc_ab", "gc_ba"])], ignore_index=True)
    d["within"] = d.source.astype(str).str.endswith(WITHIN)
    d["pair"] = d.source.astype(str) + " | " + d.target.astype(str)
    return d, s


def pair_tests(d):
    rows = []
    x = d[~d.within & d.measure.isin(COUPLING)]
    for key, g in x.groupby(["level", "window", "condition", "measure", "band", "pair"], observed=True):
        r = two_tests(g[g.cohort == "R+"].excess, g[g.cohort == "R-"].excess)
        rows.append(dict(zip(["level", "window", "condition", "measure", "band", "pair"], key), **r))
    t = pd.DataFrame(rows)
    t = t[t.p_mw.notna()].copy()
    for c in ("p_mw", "p_welch"):
        t["q" + c[1:]] = t.groupby(["level", "window", "condition", "measure", "band"])[c].transform(bh)
    return t


def contrasts(d):
    """Per session and pair: whisker - auditory (A_hit) difference of excess, in early / execution."""
    x = d[~d.within & d.measure.isin(COUPLING) & d.window.isin(["early", "execution"])]
    p = x.pivot_table(index=["session_id", "cohort", "level", "window", "measure", "band", "pair"],
                      columns="condition", values="excess").reset_index()
    out = []
    for w in ("W_lick", "W_nolick"):
        if w in p and "A_hit" in p:
            q = p[["session_id", "cohort", "level", "window", "measure", "band", "pair"]].copy()
            q["excess"] = p[w] - p["A_hit"]
            q["condition"] = f"{w}-A_hit"
            out.append(q.dropna(subset=["excess"]))
    return pd.concat(out, ignore_index=True)


def session_omnibus(d, by=("level", "window", "condition", "measure", "band")):
    x = d[~d.within & d.measure.isin(COUPLING)].copy()
    x["centred"] = x.excess - x.groupby(list(by) + ["pair"]).excess.transform("mean")
    return x.groupby(["session_id", "cohort"] + list(by), observed=True).agg(
        raw=("excess", "mean"), centred=("centred", "mean"), n_pairs=("pair", "nunique")).reset_index()


def omnibus_tests(o, by=("level", "window", "condition", "measure", "band")):
    rows = []
    for key, g in o.groupby(list(by), observed=True):
        for v in ("raw", "centred"):
            r = two_tests(g[g.cohort == "R+"][v], g[g.cohort == "R-"][v])
            rows.append(dict(zip(by, key), value=v, **r))
    return pd.DataFrame(rows)


def cluster_perm(M, lab, rng, thr=2.0, n_perm=N_PERM):
    """M [session, position] (NaN-free columns), lab bool (R+). Welch-t clusters (|t| > thr, sign-specific); returns
    observed t, list of (start, stop, mass, p) with p from the max |mass| permutation distribution."""
    def tvals(lb):
        a, b = M[lb], M[~lb]
        return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-30)

    def clusters(t):
        out = []
        for sgn in (1, -1):
            above = sgn * t > thr
            i = 0
            while i < len(t):
                if above[i]:
                    j = i
                    while j < len(t) and above[j]:
                        j += 1
                    out.append((i, j, float(t[i:j].sum())))
                    i = j
                else:
                    i += 1
        return out
    t = tvals(lab)
    obs = clusters(t)
    null = np.zeros(n_perm)
    for k in range(n_perm):
        c = clusters(tvals(rng.permutation(lab)))
        null[k] = max([abs(m) for *_, m in c], default=0.0)
    return t, [(i, j, m, (1 + np.sum(null >= abs(m))) / (n_perm + 1)) for i, j, m in obs]


# ---------------------------------------------------------------- figures

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "Arial",
                         "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 6})
    return plt


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + ext), dpi=300 if ext == "png" else None, bbox_inches="tight")


def nice(name):
    return (str(name).replace("SSp-w", "Ss-whisker").replace("SSp-orofacial", "Ss-orofacial")
            .replace("Motor and frontal areas", "Motor-frontal").replace("Striatum and pallidum", "Striatum-pallidum"))


MEAS_LABEL = {"msc": "coherence (MSC)", "cancoh": "canonical coherence", "pcoh": "partial coherence | rest",
              "imc": "imaginary coherency", "psi": "phase-slope index", "gc_dir": "Granger A->B minus B->A"}


def fig_spectra(s, plt, od, pop, sessions, rng):
    """Omnibus excess spectra by cohort, group level: rows = measures, columns = window x condition."""
    if s.empty:
        return []
    s = s[s.session_id.isin(sessions.session_id) & ~s.source.astype(str).str.endswith(WITHIN) & (s.level == "group")]
    s = s.assign(excess=s.value - s.null_mean)
    cells = [("quiet", "ALL")] + [(w, c) for w in ("early", "execution") for c in ("W_lick", "W_nolick", "A_hit")]
    meas = ["msc", "cancoh", "pcoh"]
    fig, axs = plt.subplots(len(meas), len(cells), figsize=(1.45 * len(cells), 1.3 * len(meas)), squeeze=False)
    rows = []
    for j, (w, c) in enumerate(cells):
        for i, m in enumerate(meas):
            ax = axs[i, j]
            g = s[(s.window == w) & (s.condition == c) & (s.measure == m)]
            M = g.groupby(["session_id", "cohort", "freq"]).excess.mean().unstack("freq")
            M = M.loc[:, M.notna().all()]
            if M.shape[0] < 6:
                ax.set_axis_off()
                continue
            lab = (M.index.get_level_values("cohort") == "R+")
            for coh, mk in (("R+", lab), ("R-", ~lab)):
                mu, se = M[mk].mean(), M[mk].sem()
                ax.plot(M.columns, mu, color=COL[coh], lw=0.8, label=f"{coh} (n={mk.sum()})")
                ax.fill_between(M.columns, mu - se, mu + se, color=COL[coh], alpha=0.25, lw=0, edgecolor="none")
            ax.axhline(0, color="0.6", lw=0.4, ls=":")
            if lab.sum() >= 3 and (~lab).sum() >= 3:
                t, cl = cluster_perm(M.to_numpy(), lab, rng, n_perm=500)
                for a, b, mass, p in cl:
                    rows.append(dict(window=w, condition=c, measure=m, f_lo=M.columns[a], f_hi=M.columns[b - 1],
                                     mass=mass, p=p))
                    if p < 0.05:
                        ax.plot([M.columns[a], M.columns[b - 1]], [ax.get_ylim()[1]] * 2, color="k", lw=1.2)
            if i == 0:
                ax.set_title(f"{w}\n{c}")
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel(f"{MEAS_LABEL[m]}\nexcess over shuffle")
    axs[0, 0].legend(frameon=False)
    fig.suptitle(f"Area-group coupling spectra, mean over pairs per session ({pop}); bar = cluster p < 0.05",
                 fontsize=7)
    fig.tight_layout()
    save(fig, od / "fig1_omnibus_spectra")
    plt.close(fig)
    return rows


def fig_pair_matrices(t, plt, od, pop, level="group", measure="msc"):
    """R+ - R- excess per pair (upper triangle), one panel per primary cell x band; * p_mw < 0.05, ** q < 0.05."""
    cells = [("quiet", "ALL"), ("early", "W_lick"), ("early", "W_nolick"), ("early", "A_hit"),
             ("execution", "W_lick"), ("execution", "A_hit")]
    bands = list(sl.BANDS)
    x = t[(t.level == level) & (t.measure == measure)]
    if x.empty:
        return
    areas = sorted({p for pr in x.pair for p in pr.split(" | ")})
    n = len(areas)
    fig, axs = plt.subplots(len(cells), len(bands), figsize=(1.5 * len(bands) + 0.6, 1.45 * len(cells)), squeeze=False)
    vmax = np.nanpercentile(np.abs(x["diff"]), 97) if x["diff"].notna().any() else 1
    for i, (w, c) in enumerate(cells):
        for j, b in enumerate(bands):
            ax = axs[i, j]
            g = x[(x.window == w) & (x.condition == c) & (x.band == b)]
            M = np.full((n, n), np.nan)
            for r in g.itertuples():
                a1, a2 = r.pair.split(" | ")
                M[areas.index(a1), areas.index(a2)] = M[areas.index(a2), areas.index(a1)] = r.diff
                mark = "**" if r.q_mw < 0.05 and r.q_welch < 0.05 else ("*" if r.p_mw < 0.05 and r.p_welch < 0.05 else "")
                if mark:
                    ax.text(areas.index(a2), areas.index(a1), mark, ha="center", va="center", fontsize=5)
            im = ax.imshow(M, cmap="PiYG_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(n), [nice(a) for a in areas] if i == len(cells) - 1 else [""] * n, rotation=90,
                          fontsize=4)
            ax.set_yticks(range(n), [nice(a) for a in areas] if j == 0 else [""] * n, fontsize=4)
            if i == 0:
                ax.set_title(b)
            if j == 0:
                ax.set_ylabel(f"{w} {c}", fontsize=6)
    cb = fig.colorbar(im, ax=axs, shrink=0.4)
    cb.set_label(f"R+ minus R- excess {MEAS_LABEL.get(measure, measure)}\n(green = R+ higher)")
    fig.suptitle(f"{level} pairs, {measure}, {pop}; * MW and Welch p < 0.05, ** both BH q < 0.05 over pairs", fontsize=7)
    save(fig, od / f"fig2_pairs_{level}_{measure}")
    plt.close(fig)


def fig_pvalues(t, plt, od, pop):
    """Distribution of pair-level Mann-Whitney p values per level x measure (uniform = no effect)."""
    levels = sorted(t.level.unique())
    meas = ["msc", "cancoh", "pcoh", "imc", "psi", "gc_dir"]
    fig, axs = plt.subplots(len(levels), len(meas), figsize=(1.3 * len(meas), 1.2 * len(levels)), squeeze=False)
    for i, lv in enumerate(levels):
        for j, m in enumerate(meas):
            ax = axs[i, j]
            g = t[(t.level == lv) & (t.measure == m)]
            for (w, c), col in zip([("quiet", "ALL"), ("early", "W_lick"), ("early", "W_nolick"), ("execution", "W_lick")],
                                   ["0.2", "#f7b519", "#b07a00", "#e05a00"]):
                p = g[(g.window == w) & (g.condition == c)].p_mw.dropna()
                if len(p):
                    ax.hist(p, bins=20, range=(0, 1), histtype="step", density=True, color=col, lw=0.7,
                            label=f"{w} {c} ({(p < 0.05).mean():.0%} < .05)")
            ax.axhline(1, color="0.6", lw=0.4, ls=":")
            ax.set_title(f"{lv}: {m}")
            ax.set_xlabel("pair p (Mann-Whitney)")
            ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=4)
    fig.suptitle(f"Pair-level R+ vs R- p-value distributions ({pop}); flat at 1 = chance", fontsize=7)
    fig.tight_layout()
    save(fig, od / "fig3_pvalue_distributions")
    plt.close(fig)


def fig_time_resolved(d, plt, od, pop, rng):
    x = d[(d.level == "group") & d.window.str.startswith("tr_") & ~d.within & (d.measure == "msc")]
    if x.empty:
        return []
    x = x.assign(tc=x.window.str[3:].astype(float))
    bands = ["beta", "gamma_low", "gamma_high"]
    fig, axs = plt.subplots(len(bands), len(EVENT_CONDS), figsize=(1.6 * len(EVENT_CONDS), 1.3 * len(bands)),
                            squeeze=False)
    rows = []
    for j, c in enumerate(EVENT_CONDS):
        for i, b in enumerate(bands):
            ax = axs[i, j]
            g = x[(x.condition == c) & (x.band == b)]
            M = g.groupby(["session_id", "cohort", "tc"]).excess.mean().unstack("tc").dropna()
            if M.shape[0] < 6:
                ax.set_axis_off()
                continue
            lab = M.index.get_level_values("cohort") == "R+"
            for coh, mk in (("R+", lab), ("R-", ~lab)):
                mu, se = M[mk].mean(), M[mk].sem()
                ax.plot(M.columns, mu, color=COL[coh], lw=0.8, label=f"{coh} (n={mk.sum()})")
                ax.fill_between(M.columns, mu - se, mu + se, color=COL[coh], alpha=0.25, lw=0, edgecolor="none")
            ax.axvline(0, color="0.4", lw=0.5, ls="--")
            ax.axhline(0, color="0.6", lw=0.4, ls=":")
            if lab.sum() >= 3 and (~lab).sum() >= 3:
                _, cl = cluster_perm(M.to_numpy(), lab, rng, n_perm=500)
                for a, bb, mass, p in cl:
                    rows.append(dict(condition=c, band=b, t_lo=M.columns[a], t_hi=M.columns[bb - 1], mass=mass, p=p))
                    if p < 0.05:
                        ax.plot([M.columns[a], M.columns[bb - 1]], [ax.get_ylim()[1]] * 2, color="k", lw=1.2)
            if i == 0:
                ax.set_title(c)
            ax.set_xlabel("window centre from stimulus (s)")
            ax.set_ylabel(f"{b}\nexcess coherence")
    axs[0, 0].legend(frameon=False)
    fig.suptitle(f"Time-resolved area-group coherence (200-ms windows), mean over pairs ({pop})", fontsize=7)
    fig.tight_layout()
    save(fig, od / "fig4_time_resolved_msc")
    plt.close(fig)
    return rows


def fig_confounds(d, plt, od, pop):
    x = d[(d.level == "group") & (d.window == "quiet") & (d.condition == "ALL")]
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0))
    for ax, (m, b, lab) in zip(axs, [("rate", "all", "firing rate (Hz), quiet window"),
                                    ("power", "gamma_low", "sum-signal power 30-60 Hz, quiet")]):
        g = x[(x.measure == m) & (x.band == b) & ~x.within]
        areas = sorted(g.source.unique())
        for k, coh in enumerate(("R+", "R-")):
            gg = g[g.cohort == coh].groupby("source").excess
            mu, se = gg.mean().reindex(areas), gg.sem().reindex(areas)
            ax.errorbar(np.arange(len(areas)) + (k - 0.5) * 0.25, mu, se, fmt="o", ms=2, lw=0.6, color=COL[coh],
                        label=coh)
        ax.set_xticks(range(len(areas)), [nice(a) for a in areas], rotation=60, ha="right", fontsize=5)
        ax.set_ylabel(lab)
    axs[0].legend(frameon=False)
    fig.suptitle(f"Confound checks per area group, mean +- s.e.m. over sessions ({pop})", fontsize=7)
    fig.tight_layout()
    save(fig, od / "fig5_confounds")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="spikes")
    args = ap.parse_args()
    d_all, s_all = load(args.source)
    rng = np.random.default_rng(0)
    plt = setup_mpl()
    sess = d_all[["session_id", "cohort", "learner"]].drop_duplicates()
    for pop in ("learners", "all"):
        sessions = sess[sess.learner] if pop == "learners" else sess
        d = d_all[d_all.session_id.isin(sessions.session_id)]
        so = sl.OUT / args.source / "stats" / pop
        fo = sl.OUT / args.source / "figures" / pop
        so.mkdir(parents=True, exist_ok=True)
        epochs = d[~d.window.str.startswith("tr_")]
        t = pair_tests(epochs)
        t.to_csv(so / "pair_tests.csv", index=False)
        ct = contrasts(epochs)
        tc = pair_tests(ct.assign(within=False, learner=True))
        tc.to_csv(so / "pair_tests_whisker_minus_auditory.csv", index=False)
        om = session_omnibus(epochs)
        om.to_csv(so / "omnibus_sessions.csv", index=False)
        ot = omnibus_tests(om)
        ot.to_csv(so / "omnibus_tests.csv", index=False)
        oc = omnibus_tests(session_omnibus(ct.assign(within=False)))
        oc.to_csv(so / "omnibus_tests_whisker_minus_auditory.csv", index=False)
        # summary of pre-registered primary cells
        prim = t[t.apply(lambda r: (r.window, r.condition) in PRIMARY, axis=1)]
        summ = prim.groupby(["level", "window", "condition", "measure", "band"]).agg(
            n_pairs=("pair", "size"), frac_p05_both=("p_mw", lambda p: np.mean((p < 0.05) & (prim.loc[p.index, "p_welch"] < 0.05))),
            n_q05_both=("q_mw", lambda q: int(((q < 0.05) & (prim.loc[q.index, "q_welch"] < 0.05)).sum()))).reset_index()
        summ.to_csv(so / "primary_summary.csv", index=False)
        cl_spec = fig_spectra(s_all, plt, fo, pop, sessions, rng)
        pd.DataFrame(cl_spec).to_csv(so / "spectra_clusters.csv", index=False)
        for lv in ("group", "acronym"):
            for m in ("msc", "pcoh", "gc_dir"):
                fig_pair_matrices(t, plt, fo, pop, level=lv, measure=m)
        fig_pvalues(t, plt, fo, pop)
        cl_tr = fig_time_resolved(d, plt, fo, pop, rng)
        pd.DataFrame(cl_tr).to_csv(so / "time_resolved_clusters.csv", index=False)
        fig_confounds(d, plt, fo, pop)
        # confound tests: rate / power per area group by cohort
        cf = d[(d.measure.isin(["rate", "power"])) & (d.level == "group") & ~d.within]
        rows = []
        for key, g in cf.groupby(["window", "condition", "measure", "band", "source"]):
            rows.append(dict(zip(["window", "condition", "measure", "band", "area"], key),
                             **two_tests(g[g.cohort == "R+"].excess, g[g.cohort == "R-"].excess)))
        pd.DataFrame(rows).to_csv(so / "confound_tests.csv", index=False)
        print(pop, "sessions", sessions.cohort.value_counts().to_dict())
        print(summ[(summ.level == "group") & summ.measure.isin(["msc", "cancoh", "pcoh"])].to_string(index=False))
        print(ot[(ot.measure == "msc")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
