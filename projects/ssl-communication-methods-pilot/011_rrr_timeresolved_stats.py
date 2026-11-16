"""Statistics and figures of the time-resolved RRR run (010): R+ vs R- communication between area groups.

Per session (unit: the session, one per mouse at day 0): mean over draws of the excess over the trial-shuffle null
(held-out R^2, communication fraction) and of the chosen rank. A pair (source -> target) enters a cohort with >= 3
sessions, and a cohort comparison needs both cohorts to pass (within-cohort inclusion rule of the project).
Tests, per pair and condition:
  - time-resolved: cluster-based permutation over window centres (Welch t, cluster mass, 1,000 permutations of cohort
    labels); BH-FDR over pairs (within condition and quantity) on each pair's smallest cluster p;
  - epochs (mean over window centres): pre -0.4..-0.1 s, early 0..0.2 s, late 0.25..0.6 s; Mann-Whitney and Welch
    (both reported), BH-FDR over pairs within condition x epoch on the Mann-Whitney p;
  - modality interaction: per session W_lick minus A_hit (late and early epochs), R+ vs R-;
  - within cohort: excess > 0 (Wilcoxon) per epoch.
Populations: learners (primary; sheet learning_category good/moderate) and all mice (sensitivity).
Outputs: OUT/rrr_timeresolved/{session_level.parquet, stats_epochs.csv, stats_clusters.csv, summary.md, figures/}.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-communication-methods-pilot/011_rrr_timeresolved_stats.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comm_lib as cl  # noqa: E402

RUN = cl.OUT / "rrr_timeresolved"
COL = {"R+": "#00B400", "R-": "#C800C8"}
COHORTS = ["R+", "R-"]
CONDS = ["W_lick", "W_nolick", "A_hit"]
COND_LAB = {"W_lick": "Whisker lick", "W_nolick": "Whisker no lick", "A_hit": "Auditory hit"}
EPOCHS = {"pre": (-0.4, -0.1), "early": (0.0, 0.2), "late": (0.25, 0.6)}
QUANT = {"r2": "Held-out R² (excess)", "cf": "Communication fraction (excess)", "rank": "Rank"}
MIN_SESSIONS = 3
SHORT = {"Motor and frontal areas": "Mot-Fr", "Striatum and pallidum": "Str-Pal", "SSp-w": "SSp-w",
         "SSp-orofacial": "SSp-of", "Hippocampus": "HPC", "Midbrain": "MB", "Thalamus": "TH", "PPC": "PPC",
         "Retrosplenial areas": "RSP", "Auditory areas": "AUD", "Pons and medulla": "P-M",
         "Somatosensory cortex": "SSp"}
PALETTE_NAME = {"Somatosensory cortex": "Somatosensory-whisker", "SSp-w": "Somatosensory-whisker", "SSp-orofacial": "Somatosensory-orofacial",
                "Striatum and pallidum": "Striatum", "PPC": "Posterior parietal areas"}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def area_colors(groups):
    try:
        from ephys_utilities.allen_utils import allen_utils as au
        pal = au.get_custom_area_groups_colors()
    except Exception as err:  # noqa: BLE001
        print("area palette unavailable:", err)
        pal = {}
    return {g: pal.get(PALETTE_NAME.get(g, g), "0.3") for g in groups}


# ---------------------------------------------------------------- data

def load():
    rows, metas = [], []
    for f in sorted((RUN / "sessions").glob("*.parquet")):
        if f.stem.endswith("_quick"):
            continue
        d = pd.read_parquet(f)
        if len(d):
            rows.append(d)
        metas.append(json.load(open(str(f).replace(".parquet", "_meta.json"))))
    d = pd.concat(rows, ignore_index=True)
    d["excess"] = np.where(d.quantity == "rank", d.value, d.value - d.null_mean)
    s = d.groupby(["session_id", "mouse_id", "cohort", "learner", "condition", "source", "target", "window_center",
                   "quantity"], as_index=False).agg(excess=("excess", "mean"), value=("value", "mean"),
                                                     null_mean=("null_mean", "mean"))
    return s, metas


def included(s):
    """(condition, source, target) with >= MIN_SESSIONS sessions in both cohorts."""
    n = s[s.quantity == "r2"].groupby(["condition", "source", "target", "cohort"]).session_id.nunique().unstack()
    n = n.reindex(columns=COHORTS).fillna(0)
    return n, n[(n >= MIN_SESSIONS).all(axis=1)].index


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        o = np.argsort(p[ok])
        r = p[ok][o] * ok.sum() / np.arange(1, ok.sum() + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        q[np.flatnonzero(ok)[o]] = np.minimum(r, 1)
    return q


def two_group(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def cluster_test(wide, cohort, n_perm=1000, alpha=0.05, seed=0):
    """wide: sessions x window centres. Welch t per window, clusters of contiguous same-sign p < alpha, mass = sum t,
    null = max |mass| over cohort-label permutations."""
    data = wide.to_numpy(float)
    xs = wide.columns.to_numpy()

    def tv(lab):
        with np.errstate(all="ignore"):
            r = stats.ttest_ind(data[lab == "R+"], data[lab == "R-"], axis=0, equal_var=False, nan_policy="omit")
        return np.asarray(r.statistic, float), np.asarray(r.pvalue, float)

    def cl_(t, p):
        sig, out, i = np.nan_to_num(p, nan=1) < alpha, [], 0
        while i < len(t):
            if not sig[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(t) and sig[j + 1] and np.sign(t[j + 1]) == np.sign(t[i]):
                j += 1
            out.append((i, j, float(np.nansum(t[i:j + 1]))))
            i = j + 1
        return out

    obs = cl_(*tv(cohort))
    rng = np.random.default_rng(seed)
    null = np.array([max([abs(c[2]) for c in cl_(*tv(rng.permutation(cohort)))], default=0) for _ in range(n_perm)])
    return [{"start": float(xs[a]), "end": float(xs[b]), "mass": m, "p": float((1 + np.sum(null >= abs(m))) / (1 + n_perm))}
            for a, b, m in obs]


# ---------------------------------------------------------------- tests

def run_tests(s, keep):
    ep_rows, cl_rows = [], []
    for (cond, a, b) in keep:
        g = s[(s.condition == cond) & (s.source == a) & (s.target == b)]
        for q in QUANT:
            gq = g[g.quantity == q]
            wide = gq.pivot_table(index=["session_id", "cohort"], columns="window_center", values="excess")
            for c in cluster_test(wide, wide.index.get_level_values("cohort").to_numpy()):
                cl_rows.append(dict(condition=cond, source=a, target=b, quantity=q, **c))
            for ep, (lo, hi) in EPOCHS.items():
                e = gq[gq.window_center.between(lo - 1e-9, hi + 1e-9)].groupby(["session_id", "cohort"]).excess.mean()
                r, m = e.xs("R+", level="cohort"), e.xs("R-", level="cohort")
                mw, we = two_group(r, m)
                ep_rows.append(dict(condition=cond, source=a, target=b, quantity=q, epoch=ep, n_rplus=len(r), n_rminus=len(m),
                                    mean_rplus=r.mean(), mean_rminus=m.mean(), diff=r.mean() - m.mean(),
                                    sem_rplus=r.sem(), sem_rminus=m.sem(), mannwhitney_p=mw, welch_p=we,
                                    rplus_vs0_p=stats.wilcoxon(r).pvalue if len(r) > 5 and q != "rank" else np.nan,
                                    rminus_vs0_p=stats.wilcoxon(m).pvalue if len(m) > 5 and q != "rank" else np.nan))
    ep = pd.DataFrame(ep_rows)
    ep["mw_q"] = np.nan
    for _, idx in ep.groupby(["condition", "quantity", "epoch"]).groups.items():
        ep.loc[idx, "mw_q"] = bh(ep.loc[idx, "mannwhitney_p"])
    cl_ = pd.DataFrame(cl_rows, columns=["condition", "source", "target", "quantity", "start", "end", "mass", "p"])
    best = cl_.groupby(["condition", "source", "target", "quantity"]).p.min().rename("min_cluster_p").reset_index()
    best["cluster_q"] = np.nan
    for _, idx in best.groupby(["condition", "quantity"]).groups.items():
        best.loc[idx, "cluster_q"] = bh(best.loc[idx, "min_cluster_p"])
    return ep, cl_.merge(best, on=["condition", "source", "target", "quantity"], how="left")


def interaction(s, keep):
    """W_lick minus A_hit per session (r2 excess, epoch means), R+ vs R-."""
    rows = []
    pairs = {(a, b) for c, a, b in keep if c == "W_lick"} & {(a, b) for c, a, b in keep if c == "A_hit"}
    for a, b in sorted(pairs):
        g = s[(s.source == a) & (s.target == b) & (s.quantity == "r2")]
        for ep, (lo, hi) in EPOCHS.items():
            e = g[g.window_center.between(lo - 1e-9, hi + 1e-9)].groupby(["session_id", "cohort", "condition"]).excess.mean()
            d = (e.xs("W_lick", level="condition") - e.xs("A_hit", level="condition")).dropna()
            r, m = d.xs("R+", level="cohort"), d.xs("R-", level="cohort")
            mw, we = two_group(r, m)
            rows.append(dict(source=a, target=b, epoch=ep, n_rplus=len(r), n_rminus=len(m), w_minus_a_rplus=r.mean(),
                             w_minus_a_rminus=m.mean(), mannwhitney_p=mw, welch_p=we))
    out = pd.DataFrame(rows)
    if len(out):
        out["mw_q"] = np.nan
        for _, idx in out.groupby("epoch").groups.items():
            out.loc[idx, "mw_q"] = bh(out.loc[idx, "mannwhitney_p"])
    return out


# ---------------------------------------------------------------- figures

def _groups(s):
    order = ["Somatosensory cortex", "SSp-w", "SSp-orofacial", "Motor and frontal areas", "PPC", "Retrosplenial areas", "Auditory areas",
             "Hippocampus", "Striatum and pallidum", "Thalamus", "Midbrain", "Pons and medulla"]
    present = set(s.source) | set(s.target)
    return [g for g in order if g in present] + sorted(present - set(order))


def fig_matrices(ep, groups, acol, q, pdf, title):
    """Rows: conditions; columns: epochs; cell = R+ minus R- (source rows, target columns); ** BH q < 0.05, * p < 0.05."""
    e = ep[ep.quantity == q]
    lim = np.nanpercentile(np.abs(e["diff"]), 98) if len(e) else 1
    fig, axes = plt.subplots(len(CONDS), len(EPOCHS), figsize=(7.2, 2.3 * len(CONDS)), squeeze=False)
    for i, cond in enumerate(CONDS):
        for j, epn in enumerate(EPOCHS):
            a = axes[i, j]
            M = np.full((len(groups), len(groups)), np.nan)
            sub = e[(e.condition == cond) & (e.epoch == epn)]
            for r in sub.itertuples():
                si, ti = groups.index(r.source), groups.index(r.target)
                M[si, ti] = r.diff
                mark = "**" if r.mw_q < 0.05 else "*" if r.mannwhitney_p < 0.05 else ""
                if mark:
                    a.text(ti, si, mark, ha="center", va="center", fontsize=6)
            im = a.imshow(M, cmap="PiYG", vmin=-lim, vmax=lim)  # PiYG: green = R+ > R- (positive), magenta = R- > R+
            im.cmap.set_bad("0.93")
            a.set_xticks(range(len(groups)), [SHORT.get(g, g) for g in groups], rotation=90)
            a.set_yticks(range(len(groups)), [SHORT.get(g, g) for g in groups])
            for t, g in zip(a.get_xticklabels(), groups):
                t.set_color(acol[g])
            for t, g in zip(a.get_yticklabels(), groups):
                t.set_color(acol[g])
            a.set_title(f"{COND_LAB[cond]}, {epn}", fontsize=6.5)
            if j == 0:
                a.set_ylabel("Source")
            if i == len(CONDS) - 1:
                a.set_xlabel("Target")
    cb = fig.colorbar(im, ax=axes, shrink=0.4, location="right")
    cb.set_label(f"R+ − R- ({QUANT[q]})")
    fig.suptitle(title + "\nEpochs: " + ", ".join(f"{k} {v[0]:+.2f} to {v[1]:+.2f} s" for k, v in EPOCHS.items()) + "\n(green: R+ > R-, magenta: R- > R+; * Mann-Whitney p < 0.05, ** BH q < 0.05; grey: < 3 sessions per cohort)",
                 fontsize=7)
    pdf.savefig(fig)
    return fig


def fig_timecourses(s, cl_, keep, cond, q, groups, acol, pdf, title, top=None):
    pairs = sorted({(a, b) for c, a, b in keep if c == cond}, key=lambda p: (groups.index(p[0]), groups.index(p[1])))
    best = cl_[(cl_.condition == cond) & (cl_.quantity == q)].groupby(["source", "target"]).p.min()
    if top:
        pairs = sorted(pairs, key=lambda p: best.get(p, 1))[:top]
    nc = 6
    nr = int(np.ceil(len(pairs) / nc)) if pairs else 1
    fig, axes = plt.subplots(nr, nc, figsize=(7.2, 1.25 * nr + 0.6), sharex=True, squeeze=False)
    g = s[(s.condition == cond) & (s.quantity == q)]
    for k, (a, b) in enumerate(pairs):
        ax = axes.flat[k]
        gp = g[(g.source == a) & (g.target == b)]
        for c in COHORTS:
            m = gp[gp.cohort == c].groupby("window_center").excess
            ax.plot(m.mean().index, m.mean(), color=COL[c], lw=0.8)
            ax.fill_between(m.mean().index, m.mean() - m.sem(), m.mean() + m.sem(), color=COL[c], alpha=0.25, lw=0)
        ax.axvline(0, color="0.6", lw=0.4, ls=":")
        ax.axhline(0, color="0.6", lw=0.4)
        cls = cl_[(cl_.condition == cond) & (cl_.quantity == q) & (cl_.source == a) & (cl_.target == b) & (cl_.p < 0.05)]
        top_y = ax.get_ylim()[1]
        sig = False
        for r in cls.itertuples():
            ax.plot([r.start - 0.025, r.end + 0.025], [top_y, top_y], color="k", lw=1.5, solid_capstyle="butt")
            sig = sig or r.cluster_q < 0.05
        n = gp.groupby("cohort").session_id.nunique()
        ax.set_title(f"{SHORT.get(a, a)} → {SHORT.get(b, b)} ({n.get('R+', 0)}/{n.get('R-', 0)})",
                     fontsize=6, fontweight="bold" if len(cls) else "normal", color="k")
        if sig:
            for sp in ax.spines.values():
                sp.set_color("k")
                sp.set_linewidth(1.2)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Time from stimulus (s)")
    axes[0, 0].set_ylabel(QUANT[q])
    fig.suptitle(f"{title}: {COND_LAB[cond]}, {QUANT[q]}, mean ± s.e.m. over sessions (R+ green, R- magenta; n R+/R-).\n"
                 "Bars: cohort cluster p < 0.05; bold title: cluster p < 0.05; framed: BH q < 0.05 across pairs.",
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    return fig


def fig_overview(s, keep, pdf, title):
    """Mean over all included pairs: excess R^2, CF and rank time courses per cohort and condition."""
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True)
    for j, cond in enumerate(CONDS):
        kp = {(a, b) for c, a, b in keep if c == cond}
        g = s[(s.condition == cond).to_numpy() & np.array([p in kp for p in zip(s.source, s.target)])]
        for i, q in enumerate(QUANT):
            ax = axes[i, j]
            per = g[g.quantity == q].groupby(["session_id", "cohort", "window_center"]).excess.mean().reset_index()
            for c in COHORTS:
                m = per[per.cohort == c].groupby("window_center").excess
                ax.plot(m.mean().index, m.mean(), color=COL[c])
                ax.fill_between(m.mean().index, m.mean() - m.sem(), m.mean() + m.sem(), color=COL[c], alpha=0.25, lw=0)
            wide = per.pivot_table(index=["session_id", "cohort"], columns="window_center", values="excess")
            for c in cluster_test(wide, wide.index.get_level_values("cohort").to_numpy()):
                if c["p"] < 0.05:
                    ax.plot([c["start"] - 0.025, c["end"] + 0.025], [ax.get_ylim()[1]] * 2, color="k", lw=1.5)
            ax.axvline(0, color="0.6", lw=0.4, ls=":")
            if q != "rank":
                ax.axhline(0, color="0.6", lw=0.4)
            if i == 0:
                ax.set_title(COND_LAB[cond])
            if j == 0:
                ax.set_ylabel(QUANT[q])
            if i == 2:
                ax.set_xlabel("Time from stimulus (s)")
    fig.suptitle(f"{title}: mean over included pairs per session, then mean ± s.e.m. over sessions; bars: cohort cluster p < 0.05",
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    return fig


def fig_coverage(n, groups, acol, pdf, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    w = n.loc["W_lick"] if "W_lick" in n.index.get_level_values(0) else n.iloc[:0]
    for a, c in zip(axes, COHORTS):
        M = np.full((len(groups), len(groups)), np.nan)
        for (src, tgt), v in w[c].items():
            M[groups.index(src), groups.index(tgt)] = v
        a.imshow(M, cmap="Greys", vmin=0, vmax=np.nanmax(M) if np.isfinite(M).any() else 1)
        for i in range(len(groups)):
            for j in range(len(groups)):
                if np.isfinite(M[i, j]):
                    a.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=5,
                           color="w" if M[i, j] > np.nanmax(M) / 2 else "k")
        a.set_xticks(range(len(groups)), [SHORT.get(g, g) for g in groups], rotation=90)
        a.set_yticks(range(len(groups)), [SHORT.get(g, g) for g in groups])
        for t, g in zip(list(a.get_xticklabels()) + list(a.get_yticklabels()), groups + groups):
            t.set_color(acol[g])
        a.set_title(f"{c}: sessions per pair (whisker lick)", color=COL[c])
        a.set_xlabel("Target")
        a.set_ylabel("Source")
    fig.suptitle(f"{title}: coverage (>= 20 units per group, >= 20 trials per condition)", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    return fig


# ---------------------------------------------------------------- main

def md(df):
    """DataFrame -> markdown table (no tabulate on haas)."""
    head = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    warnings.simplefilter("ignore")
    s_all, metas = load()
    for tag, s in (("learners", s_all[s_all.learner]), ("all_mice", s_all)):
        out = RUN / tag
        (out / "figures").mkdir(parents=True, exist_ok=True)
        n, keep = included(s)
        ep, cl_ = run_tests(s, keep)
        it = interaction(s, keep)
        s.to_parquet(out / "session_level.parquet")
        n.to_csv(out / "sessions_per_pair.csv")
        ep.to_csv(out / "stats_epochs.csv", index=False)
        cl_.to_csv(out / "stats_clusters.csv", index=False)
        it.to_csv(out / "stats_whisker_minus_auditory.csv", index=False)
        groups = _groups(s)
        acol = area_colors(groups)
        title = f"RRR, {tag.replace('_', ' ')} ({s.session_id.nunique()} sessions: "
        title += ", ".join(f"{c} {s[s.cohort == c].session_id.nunique()}" for c in COHORTS) + ")"
        with PdfPages(out / "figures" / "rrr_timeresolved.pdf") as pdf:
            figs = {"overview": fig_overview(s, keep, pdf, title),
                    "coverage": fig_coverage(n, groups, acol, pdf, title)}
            for q in ("r2", "cf", "rank"):
                figs[f"matrices_{q}"] = fig_matrices(ep, groups, acol, q, pdf, f"{title}, {QUANT[q]}")
            for cond in CONDS:
                figs[f"time_{cond}_r2"] = fig_timecourses(s, cl_, keep, cond, "r2", groups, acol, pdf, title)
            for name, f in figs.items():
                f.savefig(out / "figures" / f"{name}.png", dpi=200)
                plt.close(f)
        sig_cl = cl_[(cl_.p < 0.05) & (cl_.quantity == "r2")].sort_values("p")
        lines = [f"# Time-resolved RRR, {tag}", "", title, "",
                 f"Included (condition, pair) cells: {len(keep)}; groups: {', '.join(groups)}", "",
                 "## Cohort clusters (excess held-out R², p < 0.05)", "",
                 md(sig_cl[["condition", "source", "target", "start", "end", "mass", "p", "cluster_q"]].round(4))
                 if len(sig_cl) else "none", "",
                 "## Epoch tests with BH q < 0.05 (Mann-Whitney)", "",
                 md(ep[(ep.mw_q < 0.05)].round(4)) if (ep.mw_q < 0.05).any() else "none", "",
                 "## Whisker lick minus auditory hit, R+ vs R- (p < 0.05)", "",
                 md(it[it.mannwhitney_p < 0.05].round(4)) if len(it) and (it.mannwhitney_p < 0.05).any()
                 else "none"]
        (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
