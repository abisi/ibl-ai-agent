"""Statistics and figures of the CD-projection coupling (014; Chen et al. 2024 style) and comparison with the RRR / CCA /
partial-CCA v2 run (012 / 013). User request 2026-11-09: "do something similar with the prelick CD or the hit-miss CD,
then compare".

Measures per session (the unit), unordered area pair, trial set and window (Fisher z for averaging, back-transformed):
  r            Spearman correlation of the CD projections (Chen et al.'s measure when the trial set is pooled);
  r_excess     r minus the within-class permutation null mean;
  r_partial    after regressing out the top 5 PCs of the rest of the recording (pair-specific part);
  r_vs_gain    r minus the correlation along the population-mean axis (task-axis specificity beyond shared gain).
Epochs (window centres; windows 200 ms): hitmiss (stimulus-aligned) pre -0.4..-0.2, early 0.1..0.2, late 0.4..0.6 s;
prelick (lick-aligned) pre -0.5..-0.35, early (pre-lick) -0.1..0.0, late (post-lick) 0.15..0.3 s. Rise = epoch - pre.
Tests (013 test_table): within cohort != 0 (Wilcoxon, BH over pairs), R+ vs R- (Mann-Whitney and Welch, BH over pairs),
on the pre level and the rises; pooled over each session's pairs with a bootstrap 95% CI of the cohort difference;
cluster permutation over windows on the pooled time courses. Inclusion: >= 3 sessions per cohort.
Comparison with RRR v2 (same sessions, units and groups): per session and unordered pair, the RRR excess R^2 (and CCA /
partial CCA cc1; mean of the two directions) early rise on W_lick vs the hit-miss CD coupling early rise on W_hit
(Spearman over session x pair); and the pooled cohort tests of each method side by side.
Outputs: OUT/cd_coupling/{learners,all_mice}/: session_level.parquet, stats_pairs.csv, stats_pooled.csv,
stats_clusters.csv, dprime.csv, compare_rrr.csv, summary.md, figures/*.png + cd_coupling.pdf.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-communication-methods-pilot/015_cd_coupling_stats.py
"""

import importlib.util
import os
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s13 = _load("rrr_v2_stats", "013_rrr_timeresolved_v2_stats.py")
v1 = s13.v1
RUN = cl.OUT / os.environ.get("CD_RUN", "cd_coupling")  # cd_coupling_seven: seven-area run
COL, COHORTS, SHORT = v1.COL, v1.COHORTS, v1.SHORT
SETS = {"hitmiss": ["W_hit", "W_miss", "hm_pooled"], "prelick": ["WH", "AH", "SL", "ahsl_pooled"]}
SET_LAB = {"W_hit": "Whisker hits", "W_miss": "Whisker misses", "hm_pooled": "Hits + misses (Chen)",
           "WH": "Whisker hits", "AH": "Auditory hits", "SL": "Spontaneous licks", "ahsl_pooled": "AH + SL (Chen)"}
CD_LAB = {"hitmiss": "Hit-miss CD (stimulus-aligned)", "prelick": "Pre-lick CD AH − SL (lick-aligned)"}
EPOCHS = {"hitmiss": {"pre": (-0.4, -0.2), "early": (0.1, 0.2), "late": (0.4, 0.6)},
          "prelick": {"pre": (-0.5, -0.35), "early": (-0.1, 0.0), "late": (0.15, 0.3)}}
MEASURES = {"r": "Spearman r (CD projections)", "r_excess": "r − permutation null", "r_partial": "r | rest of recording",
            "r_vs_gain": "r − r(population mean axis)"}
MIN_SESSIONS = 3
plt.rcParams.update(v1.plt.rcParams)


# ---------------------------------------------------------------- data

def load():
    rows, dps = [], []
    for f in sorted((RUN / "sessions").glob("*.parquet")):
        if f.stem.endswith("_dprime"):
            dps.append(pd.read_parquet(f))
        else:
            d = pd.read_parquet(f)
            if len(d):
                rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    z = lambda x: np.arctanh(np.clip(x, -0.999, 0.999))  # noqa: E731
    d = d.assign(r_excess=z(d.r) - z(d.r_null_mean), r_vs_gain=z(d.r) - z(d.r_gain), r=z(d.r), r_partial=z(d.r_partial))
    long = d.melt(id_vars=["session_id", "mouse_id", "cohort", "learner", "cd", "trial_set", "area_a", "area_b",
                           "window_center"], value_vars=list(MEASURES), var_name="quantity", value_name="excess")
    # 013 helpers use (quantity, condition, source, target); condition = cd:trial_set; source/target = the pair
    long = long.rename(columns={"area_a": "source", "area_b": "target"})
    long["condition"] = long.cd + ":" + long.trial_set
    return long.dropna(subset=["excess"]), pd.concat(dps, ignore_index=True)


def epochs(m, by):
    out = []
    for cd, eps in EPOCHS.items():
        mc = m[m.condition.str.startswith(cd + ":")]
        parts = []
        for ep, (lo, hi) in eps.items():
            sel = mc.window_center.between(lo - 1e-9, hi + 1e-9)
            parts.append(mc[sel].groupby(["session_id", "cohort"] + by).excess.mean().rename(ep))
        out.append(pd.concat(parts, axis=1).reset_index())
    e = pd.concat(out, ignore_index=True)
    e["rise_early"], e["rise_late"] = e.early - e.pre, e.late - e.pre
    return e


def tanh_(x):
    return np.tanh(x)


# ---------------------------------------------------------------- figures

def fig_overview(pool, clus, title, pdf):
    """Pooled over pairs per session: r (solid), r | rest (dashed), r on the population-mean axis (dotted)."""
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), squeeze=False)
    for i, cd in enumerate(SETS):
        for j in range(4):
            ax = axes[i, j]
            if j >= len(SETS[cd]):
                ax.axis("off")
                continue
            ts = SETS[cd][j]
            cond = f"{cd}:{ts}"
            for q, ls in (("r", "-"), ("r_partial", "--")):
                g = pool[(pool.quantity == q) & (pool.condition == cond)]
                for c in COHORTS:
                    mm = g[g.cohort == c].groupby("window_center").excess
                    mu, se = mm.mean(), mm.sem()
                    ax.plot(mu.index, tanh_(mu), color=COL[c], ls=ls, lw=0.9 if ls == "-" else 0.7)
                    if q == "r":
                        ax.fill_between(mu.index, tanh_(mu - se), tanh_(mu + se), color=COL[c], alpha=0.2, lw=0)
            for r in clus[(clus.quantity == "r") & (clus.condition == cond) & (clus.p < 0.05)].itertuples():
                ax.plot([r.start - 0.025, r.end + 0.025], [ax.get_ylim()[1]] * 2, color="k", lw=1.5, solid_capstyle="butt")
            for lo, hi in EPOCHS[cd].values():
                ax.axvspan(lo - 0.1, hi + 0.1, color="0.5", alpha=0.06, lw=0)
            ax.axvline(0, color="0.6", lw=0.4, ls=":")
            ax.axhline(0, color="0.6", lw=0.4)
            ax.set_title(SET_LAB[ts], fontsize=6.5)
            ax.set_xlabel("Time from stimulus (s)" if cd == "hitmiss" else "Time from first lick (s)", fontsize=6)
            if j == 0:
                ax.set_ylabel(f"{CD_LAB[cd]}\nr across areas", fontsize=6)
    fig.suptitle(f"{title}: CD-projection coupling, mean over each session's pairs, then over sessions (± s.e.m.)." + chr(10) +
                 "Solid: r; dashed: r | rest of recording (top 5 PCs); bars: cohort cluster p < 0.05", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    return fig


def fig_dprime(dp, title, pdf, acol):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), squeeze=False)
    for ax, cd in zip(axes[0], SETS):
        g = dp[dp.cd == cd]
        areas = v1._groups(g.rename(columns={"area": "source"}).assign(target=g.area))
        for k, a in enumerate(areas):
            for c, dx in (("R+", -0.15), ("R-", 0.15)):
                v = g[(g.area == a) & (g.cohort == c)].cd_dprime
                ax.scatter(np.full(len(v), k + dx) + np.random.default_rng(0).uniform(-0.05, 0.05, len(v)), v, s=3,
                           color=COL[c], alpha=0.5, lw=0)
                if len(v):
                    ax.errorbar(k + dx, v.mean(), v.sem(), fmt="o", ms=3, color=COL[c], mfc="white", elinewidth=0.8)
            r_, m_ = g[(g.area == a) & (g.cohort == "R+")].cd_dprime, g[(g.area == a) & (g.cohort == "R-")].cd_dprime
            mw, we = v1.two_group(r_, m_)
            if np.isfinite(mw) and mw < 0.05:
                ax.text(k, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, "*", ha="center")
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xticks(range(len(areas)), [SHORT.get(a, a) for a in areas], rotation=60, fontsize=5.5)
        for t, a in zip(ax.get_xticklabels(), areas):
            t.set_color(acol.get(a, "k"))
        ax.set_ylabel("Held-out d' along the CD")
        ax.set_title(CD_LAB[cd], fontsize=6.5)
    fig.suptitle(f"{title}: is the CD present per area (leave-one-out d', sessions; * R+ vs R- Mann-Whitney p < 0.05)",
                 fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    pdf.savefig(fig)
    return fig


def fig_compare(cmp_, title, pdf):
    meths = [c for c in ("r2", "cca", "pcca", "prrr", "dr2") if c in cmp_]
    fig, axes = plt.subplots(1, max(len(meths), 1), figsize=(7.2, 2.4), squeeze=False)
    for ax, q in zip(axes[0], meths):
        ok = cmp_[[q, "cd_r", "cohort"]].dropna()
        for c in COHORTS:
            o = ok[ok.cohort == c]
            ax.scatter(o[q], o.cd_r, s=3, color=COL[c], alpha=0.4, lw=0)
        rho = stats.spearmanr(ok[q], ok.cd_r).statistic if len(ok) > 10 else np.nan
        ax.axhline(0, color="0.6", lw=0.4)
        ax.axvline(0, color="0.6", lw=0.4)
        ax.set_xlabel(f"{s13.MEASURES[q]}: W_lick early rise")
        ax.set_ylabel("Hit-miss CD r (z): W_hit early rise")
        ax.set_title(f"Spearman ρ = {rho:.2f} (n = {len(ok)} session × pair)", fontsize=6)
    fig.suptitle(f"{title}: CD coupling vs subspace methods on the same sessions and area pairs", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    return fig


def compare_rrr(e_pair, tag):
    """Per session x unordered pair: RRR / CCA / pCCA early rise (W_lick, mean of both directions) vs hit-miss CD r."""
    f = cl.OUT / os.environ.get("COMM_RUN", "rrr_timeresolved_v2") / tag / "epochs_session.parquet"
    if not f.exists():
        return pd.DataFrame()
    r = pd.read_parquet(f)
    r = r[(r.condition == "W_lick") & r.quantity.isin(["r2", "cca", "pcca", "prrr", "dr2"]) & (r.source != "all") & (r.target != "all")]
    r = r.assign(a=np.minimum(r.source, r.target), b=np.maximum(r.source, r.target))
    r = r.groupby(["session_id", "cohort", "a", "b", "quantity"]).rise_early.mean().unstack("quantity").reset_index()
    c = e_pair[(e_pair.quantity == "r") & (e_pair.condition == "hitmiss:W_hit")]
    c = c.rename(columns={"source": "a", "target": "b", "rise_early": "cd_r"})[["session_id", "a", "b", "cd_r"]]
    return r.merge(c, on=["session_id", "a", "b"], how="inner")


# ---------------------------------------------------------------- main

def main():
    warnings.simplefilter("ignore")
    m_all, dp_all = load()
    for tag, learners in (("learners", True), ("all_mice", False)):
        m = m_all[m_all.learner] if learners else m_all
        out = RUN / tag
        (out / "figures").mkdir(parents=True, exist_ok=True)
        n, keep = s13.included(m)
        mk = s13.keep_rows(m, keep)
        pool = s13.pooled(mk, "all")
        e_pair = epochs(mk, ["quantity", "condition", "source", "target"])
        e_pool = epochs(pool, ["quantity", "condition", "source", "target"])
        tpair = s13.test_table(e_pair)
        tpool = s13.test_table(e_pool, boot=True)
        clus = s13.cluster_table(pool, list(MEASURES))
        dp = dp_all.merge(m[["session_id", "cohort"]].drop_duplicates(), on="session_id")
        cmp_ = compare_rrr(e_pair, tag)
        mk.to_parquet(out / "session_level.parquet")
        for name, df in (("stats_pairs", tpair), ("stats_pooled", tpool), ("stats_clusters", clus), ("dprime", dp),
                         ("compare_rrr", cmp_)):
            df.to_csv(out / f"{name}.csv", index=False)
        groups = v1._groups(mk)
        acol = v1.area_colors(groups)
        title = f"CD coupling, {tag.replace('_', ' ')} ({mk.session_id.nunique()} sessions: " + \
            ", ".join(f"{c} {mk[mk.cohort == c].session_id.nunique()}" for c in COHORTS) + ")"
        s13.ALL_CONDS[:] = [f"{cd}:{ts}" for cd in SETS for ts in SETS[cd]]
        s13.COND_LAB.update({f"{cd}:{ts}": ("Hit-miss CD" if cd == "hitmiss" else "Pre-lick CD") + " − " + SET_LAB[ts] for cd in SETS for ts in SETS[cd]})
        s13.MEASURES.update(MEASURES)
        figs = {}
        with PdfPages(out / "figures" / "cd_coupling.pdf") as pdf:
            figs["overview"] = fig_overview(pool, clus, title, pdf)
            figs["dprime"] = fig_dprime(dp, title, pdf, acol)
            for q in ("r", "r_partial"):
                for var in ("pre", "rise_early"):
                    figs[f"forest_pairs_{q}_{var}"] = s13.fig_forest(tpair, "pair", q, var, title, pdf, acol)
            if len(cmp_):
                figs["compare_rrr"] = fig_compare(cmp_, title, pdf)
            for name, f in figs.items():
                f.savefig(out / "figures" / f"{name}.png", dpi=200)
                plt.close(f)
        nl = chr(10) * 2
        cols = ["condition", "variable", "n_rplus", "n_rminus", "mean_rplus", "mean_rminus", "diff", "diff_ci_lo",
                "diff_ci_hi", "mannwhitney_p", "welch_p", "rplus_wilcoxon_p", "rminus_wilcoxon_p"]
        lines = [f"# CD-projection coupling, {tag}", "", title, "",
                 "Values are Fisher z of Spearman r (r_excess, r_vs_gain: differences of z).", ""]
        for q in MEASURES:
            lines += [f"## Pooled over pairs, {MEASURES[q]}", "", v1.md(tpool[tpool.quantity == q][cols].round(4)), ""]
        cnt = tpair[tpair.variable != "pre"].groupby(["quantity", "condition", "variable"]).agg(
            n_pairs=("mw_q", "size"), rplus_ne0=("rplus_q", lambda x: int((x < 0.05).sum())),
            rminus_ne0=("rminus_q", lambda x: int((x < 0.05).sum())), cohort_p05=("mannwhitney_p", lambda x: int((x < 0.05).sum())),
            cohort_q05=("mw_q", lambda x: int((x < 0.05).sum()))).reset_index()
        lines += ["## Pairs: counts", "", v1.md(cnt), ""]
        hit = tpair[(tpair.mannwhitney_p < 0.05)].sort_values("mannwhitney_p")
        lines += ["## Pairs with R+ vs R- Mann-Whitney p < 0.05", "",
                  v1.md(hit[["quantity", "condition", "source", "target", "variable", "n_rplus", "n_rminus", "mean_rplus",
                             "mean_rminus", "mannwhitney_p", "welch_p", "mw_q"]].round(4)) if len(hit) else "none", ""]
        sc = clus[clus.p < 0.05]
        lines += ["## Pooled clusters R+ vs R- (p < 0.05)", "",
                  v1.md(sc[["quantity", "condition", "start", "end", "p"]].round(4)) if len(sc) else "none", ""]
        dps = dp.groupby(["cd", "area", "cohort"]).cd_dprime.agg(["count", "mean", "sem"]).round(3).reset_index()
        lines += ["## Held-out d' per area and cohort", "", v1.md(dps), ""]
        if len(cmp_):
            rows = []
            for q in ("r2", "cca", "pcca", "prrr", "dr2"):
                if q in cmp_:
                    ok = cmp_[[q, "cd_r"]].dropna()
                    rows.append(dict(method=q, n=len(ok), spearman_vs_cd=stats.spearmanr(ok[q], ok.cd_r).statistic))
            lines += ["## CD coupling vs RRR / CCA / pCCA (W_lick vs W_hit early rise, session x pair)", "",
                      v1.md(pd.DataFrame(rows).round(3))]
        txt = nl.join(["\n".join(lines)])
        (out / "summary.md").write_text(txt, encoding="utf-8")
        print(txt)


if __name__ == "__main__":
    main()
