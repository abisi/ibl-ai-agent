"""Statistics and figures of the time-resolved RRR v2 run (012). Question (user 2026-11-09): does communication rise
after the stimulus the same way in R+ and R- for all trial types and tested area pairs?

Measures per session (the unit; one day-0 session per mouse), mean over draws of the excess over the trial-shuffle
null: r2 (held-out R^2 of the RRR), cf (communication fraction), and the ceiling-normalised r2n / cfn = cross-area
excess / within-area ceiling of the target group. The ceiling is the session's within-area excess (15 other units of
the target group predicting the same 15 target units), averaged over windows and conditions so it is one scale per
session and target group (NaN when it is < CEIL_MIN: no within-area structure to scale by).
Methods (same draws, 012): RRR (r2, cf), PCA->CCA first canonical correlation (cca) and partial CCA conditioned on
the top 5 PCs of the rest of the recording (pcca); each also ceiling-normalised (ccan, pccan). Method comparison:
Spearman correlation between methods over session x pair x condition x window excess values and over session x pair
rises; per method, the share of pairs whose rise is != 0 within each cohort and the cohort tests.
Conditions: W_lick, W_nolick, A_hit, and the within-session contrasts WL-AH (W_lick minus A_hit: same action and
reward, other modality) and WL-WN (W_lick minus W_nolick: same stimulus, lick or not); a contrast is computed per
session, pair and window before any averaging, so it cancels whatever is fixed in the session (probes, sub-areas).
Epochs (window centres; windows are 300 ms): pre -0.4..-0.2 s (windows end <= -0.05 s), early 0.2..0.3 s (0.05-0.45 s),
late 0.5..0.7 s (0.35-0.85 s). Rise = epoch minus pre, per session.
Tests, per (measure, condition, pair): within each cohort, rise != 0 (Wilcoxon signed-rank and one-sample t);
R+ vs R- on the rise and on the pre level (Mann-Whitney and Welch, both reported); BH-FDR over pairs within
measure x condition x variable (Mann-Whitney p for cohorts, Wilcoxon p within cohort). Time-resolved: cluster-based
permutation of cohort labels (011 cluster_test). Pooled: per session mean over included pairs, and per target / per
source group (mean over the session's pairs with that target or source), same tests plus a bootstrap 95% CI of the
cohort difference in rise (sessions resampled within cohort).
Units: stable good + mua (012); trials: >= 5 per condition, min(available, 20) per draw, so the trial count differs
between sessions: summary.md reports it per cohort and condition (it biases R^2 the same way in the shuffle null, which
the excess removes to first order).
Inclusion: >= 3 sessions per cohort for a cell (both cohorts). Populations: learners (primary) and all mice.
Outputs: OUT/rrr_timeresolved_v2/{learners,all_mice}/: session_level.parquet, epochs_session.parquet, stats_pairs.csv,
stats_pooled.csv, stats_clusters.csv, summary.md, figures/*.png + rrr_v2.pdf.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-communication-methods-pilot/013_rrr_timeresolved_v2_stats.py
"""

import importlib.util
import os
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

_spec = importlib.util.spec_from_file_location("rrr_stats_v1", HERE / "011_rrr_timeresolved_stats.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

RUN = cl.OUT / os.environ.get("COMM_RUN", "rrr_timeresolved_v2")  # 016 runs: comm_v3_all, comm_v3_seven
WITHIN = " (within)"
COL, COHORTS, SHORT = v1.COL, v1.COHORTS, v1.SHORT
CONDS = ["W_lick", "W_nolick", "A_hit"]
CONTRASTS = {"WL-AH": ("W_lick", "A_hit"), "WL-WN": ("W_lick", "W_nolick")}
ALL_CONDS = CONDS + list(CONTRASTS)
COND_LAB = {"W_lick": "Whisker lick", "W_nolick": "Whisker no lick", "A_hit": "Auditory hit",
            "WL-AH": "Whisker lick − auditory hit", "WL-WN": "Whisker lick − whisker no lick"}
EPOCHS = {"pre": (-0.4, -0.2), "early": (0.2, 0.3), "late": (0.5, 0.7)}
VARS = ["pre", "rise_early", "rise_late"]
MEASURES = {"r2": "Held-out R² (excess)", "r2n": "R² / within-area ceiling", "cf": "Comm. fraction (excess)",
            "cfn": "Comm. fraction / ceiling", "cca": "CCA cc1 (excess)", "ccan": "CCA cc1 / ceiling",
            "pcca": "Partial CCA cc1 (excess)", "pccan": "Partial CCA cc1 / ceiling",
            "prrr": "Partial RRR R² (excess)", "prrrn": "Partial RRR R² / ceiling",
            "dr2": "Unique ΔR² of source (excess)", "dr2n": "Unique ΔR² / ceiling"}
METHODS = ["r2", "cca", "pcca", "prrr", "dr2"]
METHOD_SHORT = {"r2": "RRR", "cca": "CCA", "pcca": "pCCA", "prrr": "pRRR", "dr2": "ΔR²"}
PRIMARY = "r2n"
CEIL_MIN = 0.01
MIN_SESSIONS = 3
plt.rcParams.update(v1.plt.rcParams)


# ---------------------------------------------------------------- data

def load():
    rows = []
    for f in sorted((RUN / "sessions").glob("*.parquet")):
        if f.stem.endswith("_quick"):
            continue
        d = pd.read_parquet(f)
        if len(d):
            rows.append(d[d.quantity.isin(["r2", "cf", "cca", "pcca", "prrr", "dr2"])])
    d = pd.concat(rows, ignore_index=True)
    d["excess"] = d.value - d.null_mean
    ntr = d.groupby(["session_id", "cohort", "learner", "condition"]).n_trials_used.first().reset_index()
    keys = ["session_id", "mouse_id", "cohort", "learner", "condition", "source", "target", "window_center", "quantity"]
    s = d.groupby(keys, as_index=False).excess.mean()
    within = s.source.str.endswith(WITHIN)
    ceil = s[within].groupby(["session_id", "target", "quantity"]).excess.mean().rename("ceiling").reset_index()
    cross = s[~within].merge(ceil, on=["session_id", "target", "quantity"], how="left")
    norm = cross.assign(quantity=cross.quantity + "n",
                        excess=np.where(cross.ceiling >= CEIL_MIN, cross.excess / cross.ceiling, np.nan))
    m = pd.concat([cross, norm], ignore_index=True).drop(columns="ceiling").dropna(subset=["excess"])
    k2 = [k for k in keys if k != "condition"]
    w = m.pivot_table(index=k2, columns="condition", values="excess")
    con = [w[a].sub(w[b]).rename(c) for c, (a, b) in CONTRASTS.items() if {a, b} <= set(w.columns)]
    con = pd.concat(con, axis=1).stack().rename("excess").reset_index().rename(columns={"level_8": "condition"})
    m = pd.concat([m, con], ignore_index=True)
    return m, s[within].assign(target_group=lambda x: x.target), ntr


def included(m):
    n = m.groupby(["quantity", "condition", "source", "target", "cohort"]).session_id.nunique().unstack()
    n = n.reindex(columns=COHORTS).fillna(0)
    return n, set(n[(n >= MIN_SESSIONS).all(axis=1)].index)


def keep_rows(m, keep):
    idx = pd.MultiIndex.from_frame(m[["quantity", "condition", "source", "target"]])
    return m[idx.isin(list(keep))]


def epochs(m, by):
    """Per session x by: epoch means and rises."""
    out = []
    for ep, (lo, hi) in EPOCHS.items():
        sel = m.window_center.between(lo - 1e-9, hi + 1e-9)
        out.append(m[sel].groupby(["session_id", "cohort"] + by).excess.mean().rename(ep))
    e = pd.concat(out, axis=1).reset_index()
    e["rise_early"], e["rise_late"] = e.early - e.pre, e.late - e.pre
    return e


def pooled(m, how):
    """Per session mean over its included pairs: overall, per target or per source group."""
    by = {"all": [], "target": ["target"], "source": ["source"]}[how]
    p = m.groupby(["session_id", "cohort", "quantity", "condition", "window_center"] + by).excess.mean().reset_index()
    p["source"] = p["source"] if "source" in by else "all"
    p["target"] = p["target"] if "target" in by else "all"
    return p


# ---------------------------------------------------------------- tests

def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def test_table(e, boot=False):
    rows = []
    for (q, cond, a, b), g in e.groupby(["quantity", "condition", "source", "target"]):
        for v in VARS:
            r = g.loc[g.cohort == "R+", v].dropna().to_numpy()
            n_ = g.loc[g.cohort == "R-", v].dropna().to_numpy()
            if min(len(r), len(n_)) < MIN_SESSIONS:
                continue
            mw, we = v1.two_group(r, n_)
            row = dict(quantity=q, condition=cond, source=a, target=b, variable=v, n_rplus=len(r), n_rminus=len(n_),
                       mean_rplus=r.mean(), mean_rminus=n_.mean(), sem_rplus=stats.sem(r), sem_rminus=stats.sem(n_),
                       diff=r.mean() - n_.mean(), mannwhitney_p=mw, welch_p=we)
            for c, x in (("rplus", r), ("rminus", n_)):
                row[f"{c}_wilcoxon_p"] = stats.wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
                row[f"{c}_ttest_p"] = stats.ttest_1samp(x, 0).pvalue if len(x) >= 3 else np.nan
            if boot:
                row["diff_ci_lo"], row["diff_ci_hi"] = boot_diff(r, n_)
            rows.append(row)
    t = pd.DataFrame(rows)
    for col, qcol in (("mannwhitney_p", "mw_q"), ("rplus_wilcoxon_p", "rplus_q"), ("rminus_wilcoxon_p", "rminus_q")):
        t[qcol] = np.nan
        for _, idx in t.groupby(["quantity", "condition", "variable"]).groups.items():
            t.loc[idx, qcol] = v1.bh(t.loc[idx, col])
    return t


def cluster_table(m, keep_q):
    rows = []
    for (q, cond, a, b), g in m[m.quantity.isin(keep_q)].groupby(["quantity", "condition", "source", "target"]):
        wide = g.pivot_table(index=["session_id", "cohort"], columns="window_center", values="excess")
        coh = wide.index.get_level_values("cohort").to_numpy()
        if min((coh == "R+").sum(), (coh == "R-").sum()) < MIN_SESSIONS:
            continue
        for c in v1.cluster_test(wide, coh):
            rows.append(dict(quantity=q, condition=cond, source=a, target=b, **c))
    t = pd.DataFrame(rows, columns=["quantity", "condition", "source", "target", "start", "end", "mass", "p"])
    best = t.groupby(["quantity", "condition", "source", "target"]).p.min().rename("min_cluster_p").reset_index()
    best["cluster_q"] = np.nan
    for _, idx in best.groupby(["quantity", "condition"]).groups.items():
        best.loc[idx, "cluster_q"] = v1.bh(best.loc[idx, "min_cluster_p"])
    return t.merge(best, on=["quantity", "condition", "source", "target"], how="left")


# ---------------------------------------------------------------- figures

def _mark(p, q):
    return "**" if q < 0.05 else "*" if p < 0.05 else ""


def fig_overview(pool, clus, title, pdf):
    """Pooled over pairs: time courses per cohort; rows measures, columns conditions and contrasts."""
    qs = [q for q in ["r2", "r2n", "cca", "pcca", "prrr", "dr2"] if q in set(pool.quantity)]
    fig, axes = plt.subplots(len(qs), len(ALL_CONDS), figsize=(7.2, 1.6 * len(qs) + 0.6), sharex=True, squeeze=False)
    for i, q in enumerate(qs):
        for j, cond in enumerate(ALL_CONDS):
            ax = axes[i, j]
            g = pool[(pool.quantity == q) & (pool.condition == cond)]
            for c in COHORTS:
                mm = g[g.cohort == c].groupby("window_center").excess
                ax.plot(mm.mean().index, mm.mean(), color=COL[c], marker="o", ms=1.5)
                ax.fill_between(mm.mean().index, mm.mean() - mm.sem(), mm.mean() + mm.sem(), color=COL[c], alpha=0.25, lw=0)
            for r in clus[(clus.quantity == q) & (clus.condition == cond) & (clus.p < 0.05)].itertuples():
                ax.plot([r.start - 0.05, r.end + 0.05], [ax.get_ylim()[1]] * 2, color="k", lw=1.5, solid_capstyle="butt")
            for lo, hi in EPOCHS.values():
                ax.axvspan(lo - 0.15, hi + 0.15, color="0.5", alpha=0.06, lw=0)
            ax.axvline(0, color="0.6", lw=0.4, ls=":")
            ax.axhline(0, color="0.6", lw=0.4)
            if i == 0:
                ax.set_title(COND_LAB[cond].replace(" − ", "\n− "), fontsize=6.5)
            if j == 0:
                ax.set_ylabel(MEASURES[q])
            if i == len(qs) - 1:
                ax.set_xlabel("Window centre (s)")
    fig.suptitle(f"{title}\nMean over each session's included pairs, then mean ± s.e.m. over sessions; "
                 "bars: cohort cluster p < 0.05; shading: pre / early / late epochs (300-ms windows)", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    return fig


def method_agreement(mk, e_pair, tpair):
    """Spearman between methods (window values and pair rises) and per-method counts of significant rises / cohort tests."""
    rows = []
    keys = ["session_id", "condition", "source", "target", "window_center"]
    w = mk[mk.quantity.isin(METHODS) & mk.condition.isin(CONDS)].pivot_table(index=keys, columns="quantity", values="excess")
    ek = ["session_id", "condition", "source", "target"]
    r = e_pair[e_pair.quantity.isin(METHODS) & e_pair.condition.isin(CONDS)].pivot_table(index=ek, columns="quantity",
                                                                                          values="rise_early")
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            for lab, tab in (("window values", w), ("early rise per session x pair", r)):
                ok = tab[[a, b]].dropna()
                rho = stats.spearmanr(ok[a], ok[b]).statistic if len(ok) > 10 else np.nan
                rows.append(dict(comparison=f"{a} vs {b}", level=lab, n=len(ok), spearman=rho))
    cor = pd.DataFrame(rows)
    t = tpair[tpair.quantity.isin(METHODS + [m + "n" for m in METHODS]) & (tpair.variable != "pre")]
    cnt = t.groupby(["quantity", "condition", "variable"]).agg(
        n_pairs=("mw_q", "size"), rplus_rise=("rplus_q", lambda x: int((x < 0.05).sum())),
        rminus_rise=("rminus_q", lambda x: int((x < 0.05).sum())), cohort_p05=("mannwhitney_p", lambda x: int((x < 0.05).sum())),
        cohort_q05=("mw_q", lambda x: int((x < 0.05).sum()))).reset_index()
    return cor, cnt


def fig_methods(e_pair, cnt, title, pdf):
    """Left: Spearman correlation between methods of the early rise per session x pair x condition. Right: share of pairs
    whose early / late rise differs from 0 within each cohort (BH q < 0.05), per method and condition."""
    meths = [m for m in METHODS if m in set(e_pair.quantity)]
    r = e_pair[e_pair.quantity.isin(meths) & e_pair.condition.isin(CONDS)].pivot_table(
        index=["session_id", "condition", "source", "target"], columns="quantity", values="rise_early")
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1.4])
    ax = fig.add_subplot(gs[0])
    M = np.array([[stats.spearmanr(*r[[a, b]].dropna().to_numpy().T).statistic if a != b else 1.0 for b in meths]
                  for a in meths])
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(meths)):
        for j in range(len(meths)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if M[i, j] > 0.6 else "k")
    lab = [METHOD_SHORT.get(m, m) for m in meths]
    ax.set_xticks(range(len(meths)), lab, rotation=60, fontsize=5.5)
    ax.set_yticks(range(len(meths)), lab, fontsize=5.5)
    ax.set_title("Agreement (Spearman, early rise)", fontsize=6)
    for k, var in enumerate(("rise_early", "rise_late")):
        ax = fig.add_subplot(gs[k + 1])
        sub = cnt[(cnt.variable == var) & cnt.quantity.isin(meths)]
        x = np.arange(len(CONDS))
        w = 0.8 / len(meths)
        for i, q in enumerate(meths):
            g = sub[sub.quantity == q].set_index("condition").reindex(CONDS)
            xo = x - 0.4 + w * (i + 0.5)
            ax.bar(xo - w / 4, g.rplus_rise / g.n_pairs, width=w / 2, color=COL["R+"], alpha=1 - 0.15 * i, lw=0)
            ax.bar(xo + w / 4, g.rminus_rise / g.n_pairs, width=w / 2, color=COL["R-"], alpha=1 - 0.15 * i, lw=0)
            for xi, v in zip(xo, np.nanmax(np.c_[g.rplus_rise / g.n_pairs, g.rminus_rise / g.n_pairs], 1)):
                ax.text(xi, (v if np.isfinite(v) else 0) + 0.02, lab[i], rotation=90, ha="center", va="bottom", fontsize=4)
        ax.set_xticks(x, [COND_LAB[c] for c in CONDS], fontsize=5.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Pairs with rise ≠ 0 (BH q < 0.05)", fontsize=6)
        ax.set_title(var.replace("_", " "), fontsize=6)
    fig.suptitle(f"{title}: methods on the same draws (R+ green, R- magenta)", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    return fig


def fig_forest(t, level, q, var, title, pdf, acol):
    """Rise per pair (or per target / source group): R+ and R- mean ± s.e.m.; filled marker: within-cohort BH q < 0.05;
    stars at right: R+ vs R- (* p < 0.05, ** q < 0.05, Mann-Whitney)."""
    tt = t[(t.quantity == q) & (t.variable == var)]
    if level == "pair":
        tt = tt[(tt.source != "all") & (tt.target != "all")]
    labels = sorted({(r.source, r.target) for r in tt.itertuples()},
                    key=lambda p: (SHORT.get(p[0], p[0]), SHORT.get(p[1], p[1])))
    ny = max(len(labels), 1)
    fig, axes = plt.subplots(1, len(ALL_CONDS), figsize=(7.2, 0.9 + 0.11 * ny), sharey=True, squeeze=False)
    for j, cond in enumerate(ALL_CONDS):
        ax = axes[0, j]
        sub = tt[tt.condition == cond].set_index(["source", "target"])
        for k, lab in enumerate(labels):
            if lab not in sub.index:
                continue
            r = sub.loc[lab]
            for c, dy, col in (("rplus", -0.18, "R+"), ("rminus", 0.18, "R-")):
                filled = r[f"{c}_q"] < 0.05
                ax.errorbar(r[f"mean_{c}"], k + dy, xerr=r[f"sem_{c}"], fmt="o", ms=2.2, color=COL[col], elinewidth=0.6,
                            mfc=COL[col] if filled else "white", mew=0.6)
            mk = _mark(r.mannwhitney_p, r.mw_q)
            if mk:
                ax.text(1.0, k, mk, transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left")
        ax.axvline(0, color="0.6", lw=0.4)
        ax.set_title(COND_LAB[cond].replace(" − ", "\n− "), fontsize=6)
        ax.set_xlabel(var.replace('_', ' '), fontsize=5.5)  # measure in the suptitle
    def name(p):
        if level == "target":
            return f"→ {SHORT.get(p[1], p[1])}"
        if level == "source":
            return f"{SHORT.get(p[0], p[0])} →"
        return f"{SHORT.get(p[0], p[0])} → {SHORT.get(p[1], p[1])}"
    axes[0, 0].set_yticks(range(ny), [name(p) for p in labels] or [""], fontsize=5)
    axes[0, 0].set_ylim(ny - 0.5, -0.5)
    fig.suptitle(f"{title}: {var.replace('_', ' ')} per {level}, {MEASURES[q]}. R+ green, R- magenta, mean ± s.e.m. over "
                 "sessions;\nfilled: rise ≠ 0 within cohort (Wilcoxon, BH q < 0.05); * R+ vs R- Mann-Whitney p < 0.05, ** BH q < 0.05",
                 fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (0.9 + 0.11 * ny)))
    pdf.savefig(fig)
    return fig


def fig_ceiling(ceil_tc, title, pdf, acol):
    """Within-area ceiling (excess R^2 of 15 -> 15 units of the same group) per group, cohort and condition."""
    groups = v1._groups(pd.DataFrame({"source": ceil_tc.target.unique(), "target": ceil_tc.target.unique()}))
    fig, axes = plt.subplots(len(CONDS), len(groups), figsize=(7.2, 3.6), sharex=True, sharey=True, squeeze=False)
    for j, grp in enumerate(groups):
        for i, cond in enumerate(CONDS):
            ax = axes[i, j]
            g = ceil_tc[(ceil_tc.target == grp) & (ceil_tc.condition == cond) & (ceil_tc.quantity == "r2")]
            for c in COHORTS:
                mm = g[g.cohort == c].groupby("window_center").excess
                ax.plot(mm.mean().index, mm.mean(), color=COL[c])
                ax.fill_between(mm.mean().index, mm.mean() - mm.sem(), mm.mean() + mm.sem(), color=COL[c], alpha=0.25, lw=0)
            n = g.groupby("cohort").session_id.nunique()
            ax.axvline(0, color="0.6", lw=0.4, ls=":")
            ax.axhline(0, color="0.6", lw=0.4)
            if i == 0:
                ax.set_title(f"{SHORT.get(grp, grp)} ({n.get('R+', 0)}/{n.get('R-', 0)})", fontsize=6, color=acol[grp])
            if j == 0:
                ax.set_ylabel(COND_LAB[cond], fontsize=5.5)
    fig.suptitle(f"{title}: within-area ceiling (excess held-out R², 15 → 15 units of the same group), n R+/R-", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    return fig


def fig_matrix(t, groups, acol, q, var, title, pdf):
    """Rise per pair: rows conditions and contrasts; columns R+, R-, R+ − R-."""
    tt = t[(t.quantity == q) & (t.variable == var) & (t.source != "all") & (t.target != "all")]
    lim_c = np.nanpercentile(np.abs(tt[["mean_rplus", "mean_rminus"]].to_numpy()), 98) if len(tt) else 1
    lim_d = np.nanpercentile(np.abs(tt["diff"]), 98) if len(tt) else 1
    fig, axes = plt.subplots(len(ALL_CONDS), 3, figsize=(6.2, 2.0 * len(ALL_CONDS)), squeeze=False)
    for i, cond in enumerate(ALL_CONDS):
        sub = tt[tt.condition == cond]
        for j, (col, lab) in enumerate((("mean_rplus", "R+"), ("mean_rminus", "R-"), ("diff", "R+ − R-"))):
            ax = axes[i, j]
            M = np.full((len(groups), len(groups)), np.nan)
            for r in sub.itertuples():
                si, ti = groups.index(r.source), groups.index(r.target)
                M[si, ti] = getattr(r, col)
                if j < 2:
                    c = "rplus" if j == 0 else "rminus"
                    mk = "•" if getattr(r, f"{c}_q") < 0.05 else ""
                else:
                    mk = _mark(r.mannwhitney_p, r.mw_q)
                if mk:
                    ax.text(ti, si, mk, ha="center", va="center", fontsize=6)
            cmap = "PiYG" if j == 2 else "RdBu_r"
            lim = lim_d if j == 2 else lim_c
            im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
            im.cmap.set_bad("0.93")
            ax.set_xticks(range(len(groups)), [SHORT.get(g, g) for g in groups], rotation=90, fontsize=5)
            ax.set_yticks(range(len(groups)), [SHORT.get(g, g) for g in groups], fontsize=5)
            for tl, g in zip(list(ax.get_xticklabels()) + list(ax.get_yticklabels()), groups + groups):
                tl.set_color(acol[g])
            ax.set_title(f"{COND_LAB[cond]}: {lab}", fontsize=5.5)
            if j == 2 or j == 1:
                fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{title}: {var.replace('_', ' ')}, {MEASURES[q]} (rows source, columns target). • rise ≠ 0 within cohort "
                 "(BH q < 0.05);\nR+ − R-: green R+ larger, magenta R- larger; * Mann-Whitney p < 0.05, ** BH q < 0.05; "
                 "grey: < 3 sessions per cohort", fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    return fig


# ---------------------------------------------------------------- main

def summarise(tp, tpair, clus, title, n_keep):
    lines = [f"# Time-resolved RRR v2, {title}", "",
             f"Included (measure, condition, pair) cells: {n_keep}. Epochs: " +
             ", ".join(f"{k} {lo:+.1f}..{hi:+.1f} s (window centres)" for k, (lo, hi) in EPOCHS.items()), ""]
    for q in ("r2", "r2n", "cca", "pcca", "prrr", "dr2"):
        lines += [f"## Pooled over pairs, {MEASURES[q]}", ""]
        sub = tp[(tp.quantity == q) & (tp.source == "all") & (tp.target == "all")]
        lines.append(v1.md(sub[["condition", "variable", "n_rplus", "n_rminus", "mean_rplus", "mean_rminus", "diff",
                                "diff_ci_lo", "diff_ci_hi", "mannwhitney_p", "welch_p", "rplus_wilcoxon_p",
                                "rminus_wilcoxon_p"]].round(4)))
        lines.append("")
    for lvl in ("target", "source"):
        sub = tp[(tp.quantity == PRIMARY) & (tp.variable != "pre") & ((tp[lvl] != "all") & (tp[{"target": "source", "source": "target"}[lvl]] == "all"))]
        hit = sub[sub.mannwhitney_p < 0.05]
        lines += [f"## Per {lvl} group, {MEASURES[PRIMARY]}: R+ vs R- rise with Mann-Whitney p < 0.05", "",
                  v1.md(hit[["condition", "source", "target", "variable", "n_rplus", "n_rminus", "mean_rplus", "mean_rminus",
                             "mannwhitney_p", "welch_p", "mw_q"]].round(4)) if len(hit) else "none", ""]
    for q in ("r2", "r2n", "pcca", "prrr", "dr2"):
        sub = tpair[(tpair.quantity == q) & (tpair.variable != "pre")]
        lines += [f"## Pairs, {MEASURES[q]}: rise ≠ 0 within cohort (BH q < 0.05), count per condition", ""]
        cnt = sub.groupby(["condition", "variable"]).agg(n_pairs=("mw_q", "size"), rplus_rise=("rplus_q", lambda x: int((x < 0.05).sum())),
                                                       rminus_rise=("rminus_q", lambda x: int((x < 0.05).sum())),
                                                       cohort_p05=("mannwhitney_p", lambda x: int((x < 0.05).sum())),
                                                       cohort_q05=("mw_q", lambda x: int((x < 0.05).sum()))).reset_index()
        lines += [v1.md(cnt), ""]
        hit = sub[sub.mannwhitney_p < 0.05].sort_values("mannwhitney_p")
        lines += [f"### Pairs with R+ vs R- rise Mann-Whitney p < 0.05 ({MEASURES[q]})", "",
                  v1.md(hit[["condition", "source", "target", "variable", "n_rplus", "n_rminus", "mean_rplus",
                             "mean_rminus", "mannwhitney_p", "welch_p", "mw_q"]].round(4)) if len(hit) else "none", ""]
    sc = clus[(clus.p < 0.05) & (clus.source != "all")].sort_values("p")
    lines += ["## Pair clusters R+ vs R- (p < 0.05)", "",
              v1.md(sc[["quantity", "condition", "source", "target", "start", "end", "p", "cluster_q"]].round(4)) if len(sc) else "none"]
    return "\n".join(lines)


def main():
    warnings.simplefilter("ignore")
    m_all, within_all, ntr_all = load()
    for tag, learners in (("learners", True), ("all_mice", False)):
        m = m_all[m_all.learner] if learners else m_all
        within = within_all[within_all.learner] if learners else within_all
        out = RUN / tag
        (out / "figures").mkdir(parents=True, exist_ok=True)
        n, keep = included(m)
        mk = keep_rows(m, keep)
        pool = pd.concat([pooled(mk, h) for h in ("all", "target", "source")], ignore_index=True)
        e_pair = epochs(mk, ["quantity", "condition", "source", "target"])
        e_pool = epochs(pool, ["quantity", "condition", "source", "target"])
        tpair = test_table(e_pair)
        tpool = test_table(e_pool, boot=True)
        clus = pd.concat([cluster_table(mk, ["r2n", "pcca", "prrr", "dr2"]),
                          cluster_table(pool[(pool.source == "all") & (pool.target == "all")], list(MEASURES))],
                         ignore_index=True)
        mk.to_parquet(out / "session_level.parquet")
        pd.concat([e_pair, e_pool]).to_parquet(out / "epochs_session.parquet")
        n.to_csv(out / "sessions_per_cell.csv")
        tpair.to_csv(out / "stats_pairs.csv", index=False)
        tpool.to_csv(out / "stats_pooled.csv", index=False)
        clus.to_csv(out / "stats_clusters.csv", index=False)
        groups = v1._groups(mk)
        acol = v1.area_colors(sorted(set(groups) | set(within.target)))
        title = f"RRR v2, {tag.replace('_', ' ')} ({mk.session_id.nunique()} sessions: " + \
            ", ".join(f"{c} {mk[mk.cohort == c].session_id.nunique()}" for c in COHORTS) + ")"
        figs = {}
        with PdfPages(out / "figures" / "rrr_v2.pdf") as pdf:
            figs["overview"] = fig_overview(pool[(pool.source == "all") & (pool.target == "all")],
                                            clus[clus.source == "all"], title, pdf)
            figs["ceiling"] = fig_ceiling(within, title, pdf, acol)
            for lvl, tsub in (("target", tpool[(tpool.target != "all") & (tpool.source == "all")]),
                              ("source", tpool[(tpool.source != "all") & (tpool.target == "all")])):
                for var in ("rise_early", "rise_late"):
                    figs[f"forest_{lvl}_{PRIMARY}_{var}"] = fig_forest(tsub, lvl, PRIMARY, var, title, pdf, acol)
            cor, cnt = method_agreement(mk, e_pair, tpair)
            cor.to_csv(out / "method_correlations.csv", index=False)
            cnt.to_csv(out / "method_counts.csv", index=False)
            figs["methods"] = fig_methods(e_pair, cnt, title, pdf)
            for q in (PRIMARY, "pcca", "prrr", "dr2"):
                for var in (("rise_early", "rise_late", "pre") if q == PRIMARY else ("rise_early",)):
                    figs[f"forest_pairs_{q}_{var}"] = fig_forest(tpair, "pair", q, var, title, pdf, acol)
            for var in ("rise_early", "rise_late"):
                figs[f"matrix_{PRIMARY}_{var}"] = fig_matrix(tpair, groups, acol, PRIMARY, var, title, pdf)
            v1.QUANT.update(MEASURES)
            v1.COND_LAB.update(COND_LAB)
            keep_tc = [(c, a, b) for (q, c, a, b) in keep if q == PRIMARY]
            cl_tc = clus[clus.quantity == PRIMARY].assign(quantity=PRIMARY)
            for cond in ALL_CONDS:
                g = mk[mk.quantity == PRIMARY].assign(quantity=PRIMARY)
                figs[f"time_{cond}_{PRIMARY}"] = v1.fig_timecourses(g, cl_tc, keep_tc, cond, PRIMARY, groups, acol, pdf, title)
            for name, f in figs.items():
                f.savefig(out / "figures" / f"{name}.png", dpi=200)
                plt.close(f)
        ntr = ntr_all[ntr_all.learner] if learners else ntr_all
        tr_tab = ntr[ntr.session_id.isin(mk.session_id)].groupby(["condition", "cohort"]).n_trials_used.describe()
        tr_tab = tr_tab[["count", "mean", "min", "50%"]].round(1).reset_index()
        nl = chr(10) * 2
        txt = (summarise(tpool, tpair, clus, title, len(keep)) + nl + "## Method agreement (Spearman)" + nl
               + v1.md(cor.round(3)) + nl + "## Method counts (pairs)" + nl + v1.md(cnt) + nl
               + "## Trials per draw (sessions with the condition)" + nl + v1.md(tr_tab))
        (out / "summary.md").write_text(txt, encoding="utf-8")
        print(txt)


if __name__ == "__main__":
    main()
