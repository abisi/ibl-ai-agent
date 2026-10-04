"""110 -- Publication-style multi-panel figures, one per decoding analysis, whole brain -> area group
(user request 2026-09-28: "publication style quality with several - a lot of them - panels describing each
decoding analysis, from whole-brain to area group with visualization and stats").

Analyses (024 master-sweep results, current versions; lick-aligned = corrected first lick, 2026-09-27):
  hitmiss        whisker hit vs miss, stimulus-aligned, linear-shift null
  modality_stim  whisker vs auditory trial, stimulus-aligned, label-shuffle null
  modality_lick  whisker vs auditory trial (licked trials), first-lick-aligned, linear-shift null
  perfstate      high vs low performance state, stimulus-aligned, linear-shift null (whole brain, learning only)

Per figure (panels present when the data exist):
  a  method schematic (alignment, windows, classes, decoder, null, n)
  b  whole brain, learning: time-resolved balanced accuracy (mean +- SEM over sessions) per cohort, per-session
     null mean (dotted), RT / stimulus-time histogram; bars = significant clusters vs the group mouse-block
     sign-flip null (real vs one surrogate per session, N_PERM draws, cluster-mass corrected, p < 0.05)
  c  whole brain, learning: R+ minus R- curve of (accuracy - null), +- SE; bars = two-sided cluster test vs
     mouse-level permutation of cohort labels
  d  whole brain, learning: window values (accuracy - session null) per session; vs 0 (Wilcoxon + t) and
     R+ vs R- (Mann-Whitney + Welch)
  e  whole brain, learning: first vs second session half (main window), paired (Wilcoxon + paired t)
  f  whole brain: learning vs expert stage (main window), per cohort (Mann-Whitney + Welch)
  g  whole brain, expert: time course as in b
  h,i area group, learning: heatmaps of group-mean (accuracy - null) per area x time, R+ and R-; bins outside
     significant clusters (group mouse-block null) are dimmed
  j  area group, learning: onset latency (first bin reaching 50% of the peak of the group-mean accuracy - null
     curve within the latency range), 95% CI by mouse bootstrap; only areas with a significant cluster
  k  area group, learning: main-window accuracy - null per area (mean +- SEM); filled = above null
     (Wilcoxon, BH-FDR across areas, q < 0.05); * = R+ vs R- with both Mann-Whitney and Welch q < 0.05
  l  area group, expert: as k
  m  sessions per area and cohort (learning, solid; expert, hatched)
Unit of analysis: session (whole-condition rows; learning stage has one session per mouse). Areas: area_group,
same exclusions as 032/034 (EXCLUDED_OVERLAY_AREAS), >= MIN_SESS_AREA sessions in each cohort.
Outputs: figures/publication/110_<analysis>.pdf/.png (vector PDF, Arial, fonttype 42),
         110_publication_stats.csv (every test, full provenance), 110_publication_values.parquet
         (session-level window values plotted), figures/publication/110_captions.md
Run (haas): python 110_publication_figures.py [analysis ...]
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[2] / "scripts"))
from ssl_timeresolved_decoding import (  # noqa: E402
    LICK_TIME_DEFINITION, add_first_lick_time, cluster_permutation_test, group_mouseblock_permutation_null,
    prep_hitmiss_trials,
)

_s = importlib.util.spec_from_file_location("m034", OUT / "034_area_window_quant_grid.py")
m034 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m034)

FIGDIR = OUT / "figures" / "publication"
COL = {"R+": "#00B400", "R-": "#C800C8"}
COHORTS = ["R+", "R-"]
N_PERM = 1000
MIN_SESS_AREA = 5
N_BOOT = 500
SEED = 20260928

ANALYSES = {
    "hitmiss": dict(
        prefix="hitmiss_stim", title="Whisker hit vs miss decoding", align="stim",
        classes=("hit: lick after whisker stim.", "miss: no lick"), trials="all whisker trials",
        windows={"baseline": (-0.200, -0.010), "sensory": (0.005, 0.050)}, main="sensory", lat_range=(0.0, 0.3)),
    "modality_stim": dict(
        prefix="modality_stim", title="Whisker vs auditory trial decoding (stimulus-aligned)", align="stim",
        classes=("whisker trial", "auditory trial"), trials="all whisker + auditory trials",
        windows={"sensory": (0.005, 0.050)}, main="sensory", lat_range=(0.0, 0.3)),
    "modality_lick": dict(
        prefix="modality_lick", title="Whisker vs auditory trial decoding (first-lick-aligned)", align="lick",
        classes=("whisker trial + lick", "auditory trial + lick"), trials="licked whisker + auditory trials",
        windows={"pre-lick": (-0.100, 0.0), "post-lick": (0.005, 0.200)}, main="pre-lick", lat_range=(-0.6, 0.2)),
    "perfstate": dict(
        prefix="perfstate_stim", title="Performance-state decoding", align="stim",
        classes=("high-performance block", "low-performance block"), trials="trials in whisker-performance blocks",
        windows={"baseline": (-0.200, -0.010), "sensory": (0.005, 0.050)}, main="sensory", lat_range=(0.0, 0.3)),
}

STATS: list[dict] = []


def style():
    from matplotlib import font_manager
    for f in list(Path.home().glob(".local/share/fonts/arial*.ttf")) + list(Path("C:/Windows/Fonts").glob("arial*.ttf")):
        font_manager.fontManager.addfont(str(f))
    family = "Arial" if "Arial" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "legend.frameon": False,
    })


# ----------------------------------------------------------------------------------------------- data
def load_tag(tag: str):
    p, e = OUT / f"024_master_results_{tag}.parquet", OUT / f"024_bin_edges_{tag}.json"
    if not p.exists() or not e.exists():
        return None
    have = set(pq.read_schema(p).names)
    null_col = "shift_null_curves" if "shift_null_curves" in have else "null_curves"
    cols = ["session_id", "subject_id", "reward_group", "area_value", "condition_type", "condition_value",
            "skipped_reason", "n_units", "n_trials", "real_curve", "surrogate_curve", "null_curves"]
    cols += ["shift_null_curves"] if "shift_null_curves" in have else []
    cols += ["crossgen_curve_to_other"] if "crossgen_curve_to_other" in have else []
    d = pd.read_parquet(p, columns=cols)
    if "crossgen_curve_to_other" in d:
        d["xgen"] = d.crossgen_curve_to_other.map(lambda c: None if c is None else np.asarray(c, float))
        d = d.drop(columns="crossgen_curve_to_other")
    d = d[d.skipped_reason.isna() & d.condition_type.isin(["whole", "half", "perfstate"])].copy()
    if not len(d):
        return None
    if null_col == "shift_null_curves" and not d.shift_null_curves.notna().any():
        null_col = "null_curves"
    d["real"] = d.real_curve.map(lambda c: np.asarray(c, float))
    d["surr"] = d.surrogate_curve.map(lambda c: np.asarray(c, float))
    d["nullmean"] = d[null_col].map(lambda L: np.nanmean(np.stack([np.asarray(x, float) for x in L]), 0))
    d["nullsd"] = d[null_col].map(lambda L: np.nanstd(np.stack([np.asarray(x, float) for x in L]), 0))
    d["corr"] = d.real - d.nullmean
    d = d.drop(columns=[c for c in ("real_curve", "surrogate_curve", "null_curves", "shift_null_curves") if c in d])
    t = np.array([b[1] for b in json.load(open(e))])
    return dict(df=d, t=t, null_kind="linear-shift" if null_col == "shift_null_curves" else "label-shuffle", tag=tag)


def add_windows(D, windows):
    t = D["t"]
    for w, (a, b) in windows.items():
        m = (t >= a) & (t <= b)
        D["df"][f"real_{w}"] = D["df"].real.map(lambda c: np.nanmean(c[m]))
        D["df"][f"null_{w}"] = D["df"].nullmean.map(lambda c: np.nanmean(c[m]))
        D["df"][f"corr_{w}"] = D["df"][f"real_{w}"] - D["df"][f"null_{w}"]


def whole(D, area=None):
    d = D["df"][D["df"].condition_type == "whole"]
    return d if area is None else d[d.area_value == area]


# ---------------------------------------------------------------------------------------------- stats
def p1(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (np.nan, np.nan) if len(x) < 3 else (wilcoxon(x).pvalue, ttest_1samp(x, 0).pvalue)


def p2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (np.nan, np.nan) if min(len(a), len(b)) < 3 else (mannwhitneyu(a, b).pvalue, ttest_ind(a, b, equal_var=False).pvalue)


def pp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return (np.nan, np.nan) if ok.sum() < 3 else (wilcoxon(a[ok], b[ok]).pvalue, ttest_rel(a[ok], b[ok]).pvalue)


def fmt(p):
    if not np.isfinite(p):
        return "n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def rec(**k):
    STATS.append(k)


def clusters_from(obs, null, t, two_sided=False):
    res = cluster_permutation_test(obs, null, two_sided=two_sided)
    z, nd = res["observed_z"], res["null_max_cluster_mass_dist"]
    above = (np.abs(z) > 1.96) if two_sided else (z > 1.96)
    out, i = [], 0
    while i < len(z):
        if above[i] and np.isfinite(z[i]):
            s, j, m = np.sign(z[i]), i, 0.0
            while j < len(z) and above[j] and np.isfinite(z[j]) and (not two_sided or np.sign(z[j]) == s):
                m += abs(z[j])
                j += 1
            out.append(dict(i0=i, i1=j - 1, t0=t[i], t1=t[j - 1], mass=m, sign=int(s),
                            p=float((np.sum(nd >= m) + 1) / (len(nd) + 1))))
            i = j
        else:
            i += 1
    return out


def group_clusters(sub, t, rng):
    recs = [dict(subject_id=r.subject_id, real_curve=r.real, surrogate_curve=r.surr) for r in sub.itertuples()]
    obs, null = group_mouseblock_permutation_null(recs, rng, n_perm=N_PERM)
    return clusters_from(obs, null, t)


def diff_clusters(sub, t, rng):
    X = np.stack(sub["corr"].to_numpy())
    g = sub.reward_group.to_numpy()
    obs = np.nanmean(X[g == "R+"], 0) - np.nanmean(X[g == "R-"], 0)
    sg = sub.drop_duplicates("subject_id").set_index("subject_id").reward_group
    null = []
    for _ in range(N_PERM):
        mp = dict(zip(sg.index, rng.permutation(sg.to_numpy())))
        gg = sub.subject_id.map(mp).to_numpy()
        null.append(np.nanmean(X[gg == "R+"], 0) - np.nanmean(X[gg == "R-"], 0))
    return obs, clusters_from(obs, np.array(null), t, two_sided=True)


def latency(sub, t, lat_range, rng):
    """50%-of-peak rise time of the group-mean (accuracy - null) curve within lat_range; mouse bootstrap CI."""
    m = (t >= lat_range[0]) & (t <= lat_range[1])
    X = np.stack(sub["corr"].to_numpy())[:, m]
    tt = t[m]
    subj = sub.subject_id.to_numpy()
    us = np.unique(subj)

    def lat(Xs):
        c = np.nanmean(Xs, 0)
        k = int(np.nanargmax(c))
        if c[k] <= 0:
            return np.nan
        idx = np.where(c[: k + 1] >= 0.5 * c[k])[0]
        return tt[idx[0]] if len(idx) else np.nan
    est = lat(X)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(us, len(us), replace=True)
        rows = np.concatenate([np.where(subj == s)[0] for s in pick])
        boots.append(lat(X[rows]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan)
    return est, lo, hi


# ---------------------------------------------------------------------------------------------- panels
def letter(ax, s, dx=-0.18, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def shade_windows(ax, spec, scale=1000):
    for w, (a, b) in spec["windows"].items():
        ax.axvspan(a * scale, b * scale, color="#efefef", lw=0, zorder=0)


def rt_hist(sids, spec, cache):
    key = (tuple(sorted(sids)), spec["align"])
    if key in cache:
        return cache[key]
    vals = []
    for sid in sids:
        tr = prep_hitmiss_trials(ROOT, sid, ST, TT)
        if tr is None:
            continue
        h = add_first_lick_time(tr[tr.lick_flag == 1])
        v = h.reaction_time.to_numpy() * 1000
        vals.append(v if spec["align"] == "stim" else -v)
    cache[key] = np.concatenate(vals) if vals else np.array([])
    return cache[key]


def panel_timecourse(ax, D, spec, rng, stage, rtc, analysis):
    t = D["t"] * 1000
    shade_windows(ax, spec)
    top = []
    for k, rg in enumerate(COHORTS):
        sub = whole(D)[whole(D).reward_group == rg]
        if len(sub) < 3:
            continue
        X, N = np.stack(sub.real.to_numpy()), np.stack(sub.nullmean.to_numpy())
        mu, se = np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(len(X))
        ax.fill_between(t, mu - se, mu + se, color=COL[rg], alpha=0.25, lw=0)
        ax.plot(t, mu, color=COL[rg], lw=1.1, label=f"{rg} (n={len(sub)})")
        ax.plot(t, np.nanmean(N, 0), color=COL[rg], lw=0.7, ls=":")
        top.append(np.nanmax(mu + se))
        cl = group_clusters(sub, D["t"], rng)
        for c in cl:
            rec(analysis=analysis, tag=D["tag"], stage=stage, level="whole_brain", area="All units", window="time-resolved",
                condition="whole", cohort=rg, test="cluster vs group mouse-block null", n=len(sub), n_mice=sub.subject_id.nunique(),
                t_start_ms=c["t0"] * 1000, t_end_ms=c["t1"] * 1000, cluster_mass=c["mass"], p=c["p"])
        k_sig = [c for c in cl if c["p"] < 0.05]
        yb = 1.035 - 0.028 * k
        for c in k_sig:
            ax.plot([c["t0"] * 1000, c["t1"] * 1000], [yb, yb], color=COL[rg], lw=2.2, solid_capstyle="butt")
    sids = whole(D).session_id.unique()
    rt = rt_hist(sids, spec, rtc)
    if len(rt):
        h, e = np.histogram(rt, bins=np.arange(t[0] - 50, t[-1] + 1, 20))
        ax.bar(e[:-1], 0.08 * h / h.max(), width=20, bottom=0.4, align="edge", color="#b0b0b0", lw=0, zorder=1)
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0.5, color="#999999", lw=0.5, ls="--")
    ax.set_ylim(0.4, 1.06)
    ax.set_xlim(t[0] - 50, t[-1])
    ax.set_xlabel(f"time from {'stimulus' if spec['align'] == 'stim' else 'first lick'} (ms)")
    ax.set_ylabel("balanced accuracy")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.2, borderaxespad=0.2)


def panel_diff(ax, D, spec, rng, analysis):
    t = D["t"] * 1000
    sub = whole(D)
    obs, cl = diff_clusters(sub, D["t"], rng)
    Xp = np.stack(sub[sub.reward_group == "R+"]["corr"].to_numpy())
    Xm = np.stack(sub[sub.reward_group == "R-"]["corr"].to_numpy())
    se = np.sqrt(np.nanvar(Xp, 0) / len(Xp) + np.nanvar(Xm, 0) / len(Xm))
    shade_windows(ax, spec)
    ax.fill_between(t, obs - se, obs + se, color="#555555", alpha=0.25, lw=0)
    ax.plot(t, obs, color="k", lw=1.0)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.6)
    lim = np.nanmax(np.abs(np.r_[obs - se, obs + se])) * 1.35 + 0.01
    for c in cl:
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window="time-resolved",
            condition="whole", cohort="R+ - R-", test="two-sided cluster vs mouse-level cohort-label permutation",
            n=len(sub), n_mice=sub.subject_id.nunique(), t_start_ms=c["t0"] * 1000, t_end_ms=c["t1"] * 1000,
            cluster_mass=c["mass"], sign=c["sign"], p=c["p"])
        if c["p"] < 0.05:
            ax.plot([c["t0"] * 1000, c["t1"] * 1000], [lim * 0.9] * 2, color=COL["R+"] if c["sign"] > 0 else COL["R-"], lw=2.2,
                    solid_capstyle="butt")
    ax.set_ylim(-lim, lim)
    ax.set_xlim(t[0] - 50, t[-1])
    ax.set_xlabel(f"time from {'stimulus' if spec['align'] == 'stim' else 'first lick'} (ms)")
    ax.set_ylabel("R+ − R− (accuracy − null)")


def strip(ax, x, v, color, filled=True, s=6):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    j = np.random.default_rng(int(x * 100) + len(v)).uniform(-0.13, 0.13, len(v))
    ax.scatter(x + j, v, s=s, facecolor=color if filled else "white", edgecolor=color, lw=0.4, alpha=0.7, zorder=2)
    if len(v):
        ax.errorbar(x + 0.3, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0, color="k", marker="o",
                    ms=2.8, capsize=1.8, lw=0.8, zorder=3)


def fmt_s(p):
    return "n/a" if not np.isfinite(p) else "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def draw_brackets(ax, brk):
    """Significance brackets: (x_left, x_right, text); text = non-parametric | parametric p."""
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    y = y1 + 0.04 * span
    for a, b, txt in brk:
        ax.plot([a, a, b, b], [y - 0.025 * span, y, y, y - 0.025 * span], color="k", lw=0.5, clip_on=False)
        ax.text((a + b) / 2, y + 0.01 * span, txt, ha="center", va="bottom", fontsize=5.0)
    ax.set_ylim(y0 - 0.02 * span, y + 0.14 * span)


def panel_windows(ax, D, spec, analysis):
    sub = whole(D)
    ws = list(spec["windows"])
    brk = []
    for i, w in enumerate(ws):
        vals = {}
        for k, rg in enumerate(COHORTS):
            v = sub[sub.reward_group == rg][f"corr_{w}"]
            vals[rg] = v
            strip(ax, i * 2.6 + k, v, COL[rg])
            pw, pt = p1(v)
            rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition="whole",
                cohort=rg, test="accuracy - null vs 0 (Wilcoxon | one-sample t)", n=int(v.notna().sum()), n_mice=sub[sub.reward_group == rg].subject_id.nunique(),
                mean=v.mean(), p_nonparam=pw, p_param=pt)
        pm, pw_ = p2(vals["R+"], vals["R-"])
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition="whole",
            cohort="R+ vs R-", test="Mann-Whitney | Welch", n=int(sub[f"corr_{w}"].notna().sum()), n_mice=sub.subject_id.nunique(),
            mean_a=vals["R+"].mean(), mean_b=vals["R-"].mean(), p_nonparam=pm, p_param=pw_)
        brk.append((i * 2.6 + 0.15, i * 2.6 + 1.15, f"{fmt_s(pm)} | {fmt_s(pw_)}"))
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks([i * 2.6 + 0.5 for i in range(len(ws))],
                  [f"{w}\n{int(round(spec['windows'][w][0] * 1000))} to {int(round(spec['windows'][w][1] * 1000))} ms" for w in ws])
    ax.set_ylabel("accuracy − null")
    ax.set_xlim(-0.8, (len(ws) - 1) * 2.6 + 1.8)
    draw_brackets(ax, brk)


def panel_halves(ax, D, spec, analysis):
    w = spec["main"]
    d = D["df"][(D["df"].condition_type == "half")]
    brk = []
    for k, rg in enumerate(COHORTS):
        p = d[d.reward_group == rg].pivot_table(index="session_id", columns="condition_value", values=f"corr_{w}").dropna()
        if not {"first", "second"} <= set(p.columns) or len(p) < 3:
            continue
        x0 = k * 2.4
        for _, r in p.iterrows():
            ax.plot([x0, x0 + 1], [r["first"], r["second"]], color=COL[rg], lw=0.4, alpha=0.35)
        for j, c in enumerate(("first", "second")):
            ax.errorbar(x0 + j, p[c].mean(), yerr=p[c].std() / np.sqrt(len(p)), color="k", marker="o", ms=2.8, capsize=1.8, lw=0.8, zorder=3)
        pw, pt = pp(p["first"], p["second"])
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w,
            condition="half first vs second", cohort=rg, test="paired Wilcoxon | paired t", n=len(p), n_mice=len(p),
            mean_a=p["first"].mean(), mean_b=p["second"].mean(), p_nonparam=pw, p_param=pt)
        brk.append((x0, x0 + 1, f"{fmt_s(pw)} | {fmt_s(pt)}"))
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks([0, 1, 2.4, 3.4], ["1st", "2nd", "1st", "2nd"])
    ax.set_xlabel("session half   (R+ | R−)")
    ax.set_ylabel(f"accuracy − null ({w})")
    draw_brackets(ax, brk)


def panel_stage(ax, DL, DE, spec, analysis):
    w = spec["main"]
    if DE is None:
        ax.text(0.5, 0.5, "no expert-stage run", ha="center", va="center", transform=ax.transAxes, color="#777777")
        ax.set_axis_off()
        return
    brk = []
    for k, rg in enumerate(COHORTS):
        a = whole(DL)[whole(DL).reward_group == rg][f"corr_{w}"]
        b = whole(DE)[whole(DE).reward_group == rg][f"corr_{w}"]
        strip(ax, k * 2.6, a, COL[rg], filled=False)
        strip(ax, k * 2.6 + 1, b, COL[rg])
        pm, pw_ = p2(a, b)
        rec(analysis=analysis, tag=DE["tag"], stage="learning vs expert", level="whole_brain", area="All units", window=w,
            condition="whole", cohort=rg, test="Mann-Whitney | Welch (learning vs expert sessions)", n=int(a.notna().sum() + b.notna().sum()),
            n_mice=np.nan, mean_a=a.mean(), mean_b=b.mean(), p_nonparam=pm, p_param=pw_)
        brk.append((k * 2.6 + 0.15, k * 2.6 + 1.15, f"{fmt_s(pm)} | {fmt_s(pw_)}"))
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks([0.15, 1.15, 2.75, 3.75], ["learn.", "expert", "learn.", "expert"])
    ax.set_xlabel("stage   (R+ | R−)")
    ax.set_ylabel(f"accuracy − null ({w})")
    draw_brackets(ax, brk)


def area_list(D):
    d = whole(D)
    areas = m034.common_areas(d, "area_group")
    cnt = d.groupby(["area_value", "reward_group"]).session_id.nunique().unstack(fill_value=0)
    return [a for a in areas if a in cnt.index and min(cnt.loc[a].get("R+", 0), cnt.loc[a].get("R-", 0)) >= MIN_SESS_AREA]


def area_maps(D, areas, spec, rng, analysis, stage):
    t = D["t"]
    out = {}
    for rg in COHORTS:
        M, S, L = [], [], []
        for a in areas:
            sub = whole(D, a)
            sub = sub[sub.reward_group == rg]
            X = np.stack(sub["corr"].to_numpy())
            M.append(np.nanmean(X, 0))
            cl = group_clusters(sub, t, rng)
            sig = np.zeros(len(t), bool)
            for c in cl:
                rec(analysis=analysis, tag=D["tag"], stage=stage, level="area_group", area=a, window="time-resolved", condition="whole",
                    cohort=rg, test="cluster vs group mouse-block null", n=len(sub), n_mice=sub.subject_id.nunique(),
                    t_start_ms=c["t0"] * 1000, t_end_ms=c["t1"] * 1000, cluster_mass=c["mass"], p=c["p"])
                if c["p"] < 0.05:
                    sig[c["i0"]:c["i1"] + 1] = True
            S.append(sig)
            if sig.any():
                est, lo, hi = latency(sub, t, spec["lat_range"], rng)
            else:
                est = lo = hi = np.nan
            L.append((est, lo, hi))
            rec(analysis=analysis, tag=D["tag"], stage=stage, level="area_group", area=a, window=f"latency {spec['lat_range']}",
                condition="whole", cohort=rg, test="50%-of-peak latency, mouse bootstrap 95% CI", n=len(sub),
                n_mice=sub.subject_id.nunique(), latency_ms=est * 1000, ci_lo_ms=lo * 1000, ci_hi_ms=hi * 1000)
        out[rg] = (np.array(M), np.array(S), L)
    return out


def panel_heatmap(ax, maps, rg, areas, D, spec, colors, show_labels, norm):
    t = D["t"] * 1000
    M, S, _ = maps[rg]
    dt = t[1] - t[0]
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm, extent=[t[0] - dt / 2, t[-1] + dt / 2, len(areas) - 0.5, -0.5],
                   interpolation="nearest")
    veil = np.zeros(S.shape + (4,))
    veil[..., :3] = 1.0
    veil[..., 3] = np.where(S, 0.0, 0.62)
    ax.imshow(veil, aspect="auto", extent=[t[0] - dt / 2, t[-1] + dt / 2, len(areas) - 0.5, -0.5], interpolation="nearest")
    ax.axvline(0, color="k", lw=0.6)
    ticks = np.arange(np.ceil(t[0] / 200) * 200, t[-1] + 1, 200)
    ax.set_xticks(ticks if show_labels else ticks[1:])
    ax.set_yticks(range(len(areas)))
    if show_labels:
        ax.set_yticklabels(areas, fontsize=5.6)
        for tl, a in zip(ax.get_yticklabels(), areas):
            tl.set_color(colors.get(a, "#333333"))
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(f"time from {'stimulus' if spec['align'] == 'stim' else 'first lick'} (ms)")
    ax.set_title(f"{rg}", color=COL[rg], fontsize=7, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    return im


def panel_latency(ax, maps, areas, spec):
    for k, rg in enumerate(COHORTS):
        L = maps[rg][2]
        for i, (e, lo, hi) in enumerate(L):
            if np.isfinite(e):
                y = i + (-0.15 if k == 0 else 0.15)
                ax.errorbar(e * 1000, y, xerr=[[max(e - lo, 0) * 1000], [max(hi - e, 0) * 1000]] if np.isfinite(lo) else None,
                            color=COL[rg], marker="o", ms=2.8, capsize=1.5, lw=0.7)
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_yticks(range(len(areas)), [])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("onset latency (ms)\n50% of peak, 95% CI")
    ax.grid(axis="y", color="#eeeeee", lw=0.5)


def panel_area_window(ax, D, areas, spec, colors, analysis, stage):
    w = spec["main"]
    d = whole(D)
    rows = []
    for a in areas:
        for rg in COHORTS:
            v = d[(d.area_value == a) & (d.reward_group == rg)][f"corr_{w}"].dropna()
            pw, pt = p1(v)
            rows.append(dict(area=a, rg=rg, mean=v.mean(), sem=v.std() / np.sqrt(len(v)), n=len(v), pw=pw, pt=pt, v=v))
    R = pd.DataFrame(rows)
    for rg in COHORTS:
        m = R.rg == rg
        R.loc[m, "qw"] = multipletests(R.loc[m, "pw"].fillna(1), method="fdr_bh")[1]
        R.loc[m, "qt"] = multipletests(R.loc[m, "pt"].fillna(1), method="fdr_bh")[1]
    comp = []
    for a in areas:
        va, vb = R[(R.area == a) & (R.rg == "R+")].v.iloc[0], R[(R.area == a) & (R.rg == "R-")].v.iloc[0]
        pm, pw_ = p2(va, vb)
        comp.append(dict(area=a, pm=pm, pwe=pw_))
    C = pd.DataFrame(comp)
    C["qm"] = multipletests(C.pm.fillna(1), method="fdr_bh")[1]
    C["qwe"] = multipletests(C.pwe.fillna(1), method="fdr_bh")[1]
    for i, a in enumerate(areas):
        for k, rg in enumerate(COHORTS):
            r = R[(R.area == a) & (R.rg == rg)].iloc[0]
            filled = r.qw < 0.05
            ax.errorbar(i + (-0.17 if k == 0 else 0.17), r["mean"], yerr=r["sem"], fmt="o", ms=3.2, mfc=colors.get(a, "#888") if filled else "white",
                        mec=COL[rg], mew=0.9, ecolor=COL[rg], elinewidth=0.7, capsize=0)
            rec(analysis=analysis, tag=D["tag"], stage=stage, level="area_group", area=a, window=w, condition="whole", cohort=rg,
                test="accuracy - null vs 0 (Wilcoxon | one-sample t), BH-FDR across areas", n=int(r["n"]),
                n_mice=d[(d.area_value == a) & (d.reward_group == rg)].subject_id.nunique(), mean=r["mean"], p_nonparam=r.pw,
                p_param=r.pt, q_nonparam=r.qw, q_param=r.qt)
        c = C[C.area == a].iloc[0]
        rec(analysis=analysis, tag=D["tag"], stage=stage, level="area_group", area=a, window=w, condition="whole", cohort="R+ vs R-",
            test="Mann-Whitney | Welch, BH-FDR across areas", n=int(R[R.area == a].n.sum()), n_mice=d[d.area_value == a].subject_id.nunique(),
            mean_a=R[(R.area == a) & (R.rg == "R+")]["mean"].iloc[0], mean_b=R[(R.area == a) & (R.rg == "R-")]["mean"].iloc[0],
            p_nonparam=c.pm, p_param=c.pwe, q_nonparam=c.qm, q_param=c.qwe)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=55, ha="right", fontsize=5.6)
    for tl, a in zip(ax.get_xticklabels(), areas):
        tl.set_color(colors.get(a, "#333333"))
    y0, y1 = ax.get_ylim()
    top = y1 + 0.02
    for i, a in enumerate(areas):
        c = C[C.area == a].iloc[0]
        if c.qm < 0.05 and c.qwe < 0.05:
            ax.text(i, top, "*", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(y0 - 0.01, top + 0.05)
    ax.set_ylabel(f"accuracy − null ({w})")
    ax.set_xlim(-0.6, len(areas) - 0.4)


def panel_counts(ax, DL, DE, areas):
    for k, rg in enumerate(COHORTS):
        nl = [whole(DL, a)[whole(DL, a).reward_group == rg].session_id.nunique() for a in areas]
        ax.barh(np.arange(len(areas)) + (-0.2 if k == 0 else 0.2), nl, height=0.38, color=COL[rg], alpha=0.8, lw=0)
        if DE is not None:
            ne = [whole(DE, a)[whole(DE, a).reward_group == rg].session_id.nunique() if a in set(whole(DE).area_value) else 0 for a in areas]
            ax.barh(np.arange(len(areas)) + (-0.2 if k == 0 else 0.2), ne, left=nl, height=0.38, color="white", edgecolor=COL[rg],
                    hatch="//////", lw=0.4)
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels(areas, fontsize=5.3)
    ax.set_xlabel("n (hatched: expert)")


def panel_schematic(ax, spec, DL, DE):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    stim = spec["align"] == "stim"
    lo, hi = (-250, 600) if stim else (-650, 200)
    X = lambda ms: 0.03 + 0.92 * (ms - lo) / (hi - lo)  # noqa: E731
    y = 0.78
    for i, (w, (a, b)) in enumerate(spec["windows"].items()):
        ax.add_patch(Rectangle((X(a * 1000), y + 0.02), X(b * 1000) - X(a * 1000), 0.04, color=["#bdbdbd", "#6e6e6e"][i % 2], lw=0))
        xc = (X(a * 1000) + X(b * 1000)) / 2
        yl = y + 0.1 + 0.08 * (i % 2)
        ax.plot([xc, xc], [y + 0.065, yl - 0.005], color="#888888", lw=0.4)
        ax.text(xc, yl, f"{w} ({int(round(a * 1000))} to {int(round(b * 1000))} ms)", ha="center", va="bottom", fontsize=4.9)
    ax.annotate("", xy=(X(hi) + 0.02, y), xytext=(X(lo), y), arrowprops=dict(arrowstyle="-|>", lw=0.6, mutation_scale=5))
    for tick in range(int(np.ceil(lo / 200) * 200), hi + 1, 200):
        ax.plot([X(tick)] * 2, [y - 0.012, y + 0.012], color="k", lw=0.5)
        ax.text(X(tick), y - 0.03, f"{tick}", ha="center", va="top", fontsize=4.8)
    ax.plot([X(0)] * 2, [y - 0.05, y + 0.07], color="k", lw=1.3)
    ax.text(X(0), y - 0.085, "stimulus (0 ms)" if stim else "first lick (0 ms)", ha="center", va="top", fontsize=5.3, fontweight="bold")
    yb = y - 0.2
    ax.add_patch(Rectangle((X(lo + 40), yb), X(lo + 90) - X(lo + 40), 0.03, color="#4d4d4d", lw=0))
    ax.add_patch(Rectangle((X(lo + 55), yb - 0.04), X(lo + 105) - X(lo + 55), 0.03, color="#a6a6a6", lw=0))
    ax.text(X(lo + 125), yb - 0.005, "50-ms causal bins, 5-ms step (labelled at bin end)", fontsize=4.9, va="center")
    n = lambda D, rg: whole(D)[whole(D).reward_group == rg].session_id.nunique() if D is not None else 0  # noqa: E731
    txt = [f"classes: {spec['classes'][0]}  vs  {spec['classes'][1]}",
           f"trials: {spec['trials']}",
           "decoder: L2 logistic, per-fold z-scoring, 5-fold CV,",
           "    balanced accuracy, one C per session",
           f"chance: per-session {D_NULL[spec['prefix']]} null",
           "whisker artefact (-10 to +5 ms from stimulus) excluded",
           "population: all QC units (whole brain) or area group",
           f"learning: R+ {n(DL, 'R+')}, R- {n(DL, 'R-')} sessions"
           + (f";  expert: R+ {n(DE, 'R+')}, R- {n(DE, 'R-')}" if DE is not None else "")]
    if not stim:
        txt.append("first lick = start + (lick_time - response-window start)")
    ax.text(0.0, 0.47, "\n".join(txt), fontsize=5.2, va="top", linespacing=1.4)


# ---------------------------------------------------------------------------------------------- figure
D_NULL: dict[str, str] = {}


def make_figure(key, rtc):
    spec = ANALYSES[key]
    pre = spec["prefix"]
    rng = np.random.default_rng(SEED)
    DL, DE = load_tag(f"{pre}_whole_brain"), load_tag(f"{pre}_expert_whole_brain")
    AL, AE = load_tag(f"{pre}_area_group"), load_tag(f"{pre}_expert_area_group")
    for D in (DL, DE, AL, AE):
        if D is not None:
            add_windows(D, spec["windows"])
    D_NULL[pre] = DL["null_kind"]
    if AL is None and DE is None:
        fig = compact_figure(key, spec, DL, rng, rtc)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(FIGDIR / f"110_{key}.{ext}", dpi=300)
        plt.close(fig)
        keep = ["session_id", "subject_id", "reward_group", "area_value", "condition_type", "condition_value", "n_units", "n_trials"]
        keep += [c for c in DL["df"].columns if c.startswith(("real_", "null_", "corr_"))]
        print(f"[110] {key} done (compact)", flush=True)
        return DL["df"][keep].assign(analysis=key, tag=DL["tag"], level="whole_brain", stage="learning", null_kind=DL["null_kind"],
                                     lick_time_definition=None)
    fig = plt.figure(figsize=(7.2, 11.0))
    fig.suptitle(spec["title"], fontsize=9, fontweight="bold", x=0.02, ha="left", y=0.992)
    fig.text(0.02, 0.968, f"whole brain to area group, learning and expert stages; chance = per-session {DL['null_kind']} null"
             + ("; aligned to the corrected first lick" if spec["align"] == "lick" else "")
             + "; brackets: non-parametric | parametric p", fontsize=5.8, color="#444444")
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.925, bottom=0.765, wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 4, left=0.08, right=0.98, top=0.675, bottom=0.53, wspace=0.62)
    r2a = fig.add_gridspec(1, 3, left=0.2, right=0.705, top=0.445, bottom=0.27, width_ratios=[1, 1, 0.05], wspace=0.1)
    r2b = fig.add_gridspec(1, 1, left=0.8, right=0.98, top=0.445, bottom=0.27)
    r3k = fig.add_gridspec(1, 1, left=0.075, right=0.47, top=0.2, bottom=0.085)
    r3l = fig.add_gridspec(1, 1, left=0.535, right=0.73, top=0.2, bottom=0.085)
    r3m = fig.add_gridspec(1, 1, left=0.875, right=0.98, top=0.2, bottom=0.085)
    a = fig.add_subplot(r0[0]); panel_schematic(a, spec, DL, DE); letter(a, "a", dx=-0.02, dy=1.0)
    b = fig.add_subplot(r0[1]); panel_timecourse(b, DL, spec, rng, "learning", rtc, key); letter(b, "b", dx=-0.26)
    b.set_title("whole brain, learning", fontsize=6.5)
    c = fig.add_subplot(r0[2]); panel_diff(c, DL, spec, rng, key); letter(c, "c", dx=-0.26)
    c.set_title("whole brain, learning: R+ minus R-", fontsize=6.5)
    d = fig.add_subplot(r1[0]); panel_windows(d, DL, spec, key); letter(d, "d", dx=-0.4)
    d.set_title("windows, learning\nR+ vs R- (MW | Welch)", fontsize=5.8)
    e = fig.add_subplot(r1[1]); panel_halves(e, DL, spec, key); letter(e, "e", dx=-0.4)
    e.set_title(f"session halves ({spec['main']})\n1st vs 2nd (Wilcoxon | paired t)", fontsize=5.8)
    f = fig.add_subplot(r1[2]); panel_stage(f, DL, DE, spec, key); letter(f, "f", dx=-0.4)
    if DE is not None:
        f.set_title(f"learning vs expert ({spec['main']})\n(MW | Welch)", fontsize=5.8)
    g = fig.add_subplot(r1[3])
    if DE is not None:
        panel_timecourse(g, DE, spec, rng, "expert", rtc, key)
        g.set_title("whole brain, expert", fontsize=6.5)
        g.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.0, borderaxespad=0.2, fontsize=5.2)
    else:
        g.text(0.5, 0.5, "no expert-stage run", ha="center", va="center", transform=g.transAxes, color="#777777")
        g.set_axis_off()
    letter(g, "g", dx=-0.4)
    if AL is not None:
        areas = area_list(AL)
        colors = m034.get_area_color_map(areas)
        maps = area_maps(AL, areas, spec, rng, key, "learning")
        vmax = max(0.05, np.nanpercentile(np.concatenate([maps[rg][0].ravel() for rg in COHORTS]), 99))
        norm = TwoSlopeNorm(vmin=-vmax / 3, vcenter=0, vmax=vmax)
        h = fig.add_subplot(r2a[0]); im = panel_heatmap(h, maps, "R+", areas, AL, spec, colors, True, norm); letter(h, "h", dx=-0.6)
        i_ = fig.add_subplot(r2a[1]); panel_heatmap(i_, maps, "R-", areas, AL, spec, colors, False, norm); letter(i_, "i", dx=-0.06)
        cax = fig.add_subplot(r2a[2])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("")
        cb.ax.tick_params(labelsize=5.5)
        fig.text(0.2, 0.468, "area group, learning: group-mean accuracy - null (pale = outside significant clusters)", fontsize=6.3)
        j = fig.add_subplot(r2b[0]); panel_latency(j, maps, areas, spec); letter(j, "j", dx=-0.12)
        j.set_title("onset latency", fontsize=6.5)
        k = fig.add_subplot(r3k[0]); panel_area_window(k, AL, areas, spec, colors, key, "learning"); letter(k, "k", dx=-0.13)
        k.set_title(f"area group, learning ({spec['main']})", fontsize=6.3)
        l_ = fig.add_subplot(r3l[0])
        if AE is not None:
            areas_e = area_list(AE)
            panel_area_window(l_, AE, areas_e, spec, m034.get_area_color_map(areas_e), key, "expert")
            l_.set_title("area group, expert", fontsize=6.3)
        else:
            l_.text(0.5, 0.5, "no expert-stage area run", ha="center", va="center", transform=l_.transAxes, color="#777777")
            l_.set_axis_off()
        letter(l_, "l", dx=-0.25)
        m = fig.add_subplot(r3m[0]); panel_counts(m, AL, AE, areas); letter(m, "m", dx=-1.2)
        m.set_title("sessions", fontsize=6.3)
    else:
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.1, right=0.9, top=0.44, bottom=0.1)[0])
        ax.text(0.5, 0.5, "area-level decoding not run for this analysis (whole brain only)", ha="center", va="center", color="#777777")
        ax.set_axis_off()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"110_{key}.{ext}", dpi=300)
    plt.close(fig)
    vals = []
    for D, lvl, stg in ((DL, "whole_brain", "learning"), (DE, "whole_brain", "expert"), (AL, "area_group", "learning"), (AE, "area_group", "expert")):
        if D is None:
            continue
        keep = ["session_id", "subject_id", "reward_group", "area_value", "condition_type", "condition_value", "n_units", "n_trials"]
        keep += [c for c in D["df"].columns if c.startswith(("real_", "null_", "corr_")) and c not in ("real", "corr")]
        vals.append(D["df"][keep].assign(analysis=key, tag=D["tag"], level=lvl, stage=stg, null_kind=D["null_kind"],
                                         lick_time_definition=LICK_TIME_DEFINITION if spec["align"] == "lick" else None))
    print(f"[110] {key} done", flush=True)
    return pd.concat(vals, ignore_index=True)


def panel_real_vs_null(ax, D, spec, analysis):
    """Per-session real vs null window accuracy: shows where each session's empirical chance sits."""
    w = spec["main"]
    sub = whole(D)
    for rg in COHORTS:
        s = sub[sub.reward_group == rg]
        ax.scatter(s[f"null_{w}"], s[f"real_{w}"], s=7, color=COL[rg], alpha=0.75, lw=0, label=rg)
        pw, pt = pp(s[f"real_{w}"], s[f"null_{w}"])
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition="whole",
            cohort=rg, test="real vs null window accuracy (paired Wilcoxon | paired t)", n=len(s), n_mice=s.subject_id.nunique(),
            mean_a=s[f"real_{w}"].mean(), mean_b=s[f"null_{w}"].mean(), p_nonparam=pw, p_param=pt)
    lo = min(sub[f"null_{w}"].min(), sub[f"real_{w}"].min()) - 0.02
    hi = max(sub[f"null_{w}"].max(), sub[f"real_{w}"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.6, ls="--")
    ax.axvline(0.5, color="#cccccc", lw=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"session null accuracy ({w})")
    ax.set_ylabel(f"real accuracy ({w})")
    ax.legend(loc="upper left", handletextpad=0.2)


def compact_figure(key, spec, DL, rng, rtc):
    """Whole-brain-only analyses (no halves, expert or area runs): one compact figure."""
    fig = plt.figure(figsize=(7.2, 5.0))
    fig.suptitle(spec["title"], fontsize=9, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.945, f"whole brain, learning stage (no halves, expert or area-level runs for this analysis); chance = per-session "
                          f"{DL['null_kind']} null; brackets: non-parametric | parametric p", fontsize=5.8, color="#444444")
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.87, bottom=0.57, wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.42, bottom=0.1, wspace=0.55, width_ratios=[1, 0.9, 1.1])
    a = fig.add_subplot(r0[0]); panel_schematic(a, spec, DL, None); letter(a, "a", dx=-0.02, dy=1.0)
    b = fig.add_subplot(r0[1]); panel_timecourse(b, DL, spec, rng, "learning", rtc, key); letter(b, "b", dx=-0.26)
    b.set_title("whole brain, learning", fontsize=6.5)
    c = fig.add_subplot(r0[2]); panel_diff(c, DL, spec, rng, key); letter(c, "c", dx=-0.26)
    c.set_title("whole brain, learning: R+ minus R-", fontsize=6.5)
    d = fig.add_subplot(r1[0]); panel_windows(d, DL, spec, key); letter(d, "d", dx=-0.3)
    d.set_title("windows, learning\nR+ vs R- (MW | Welch)", fontsize=5.8)
    e = fig.add_subplot(r1[1]); panel_real_vs_null(e, DL, spec, key); letter(e, "e", dx=-0.35)
    e.set_title("real vs per-session null", fontsize=5.8)
    return fig


def captions():
    S = pd.DataFrame(STATS)
    lines = ["# 110 publication figures: panel guide\n",
             "Unit = session. Group curves: mean ± SEM over sessions. Cluster tests: group mouse-block sign-flip null "
             f"(real vs one per-session surrogate, {N_PERM} draws), z > 1.96, cluster-mass corrected. R+ vs R−: Mann–Whitney "
             "and Welch (both reported). Area tests: BH-FDR across areas. Latency: first bin at 50% of the group-mean peak of "
             f"(accuracy − null) within the latency range; 95% CI from {N_BOOT} mouse bootstraps.\n"]
    for key, spec in ANALYSES.items():
        s = S[S.analysis == key]
        if not len(s):
            continue
        lines.append(f"## {spec['title']} (`110_{key}.pdf`)\n")
        wb = s[(s.level == "whole_brain") & (s.stage == "learning") & (s.cohort == "R+ vs R-")]
        for r in wb.itertuples():
            lines.append(f"- whole brain, learning, {r.window}: R+ {r.mean_a:.3f} vs R− {r.mean_b:.3f} (accuracy − null), "
                         f"MW p={r.p_nonparam:.3g}, Welch p={r.p_param:.3g}")
        ag = s[(s.level == "area_group") & (s.cohort == "R+ vs R-") & (s.test.str.startswith("Mann"))]
        for stg, g in ag.groupby("stage"):
            sig = g[(g.q_nonparam < 0.05) & (g.q_param < 0.05)].area.tolist()
            lines.append(f"- area group, {stg}: R+ vs R− differs (both FDR q<0.05) in {len(sig)}/{len(g)} areas"
                         + (f": {', '.join(sig)}" if sig else ""))
        lines.append("")
    (FIGDIR / "110_captions.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global ROOT, ST, TT
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    ROOT = resolve_dataset_dir("ssl_ephys")
    ST, TT = pd.read_parquet(ROOT / "metadata" / "sessions.parquet"), pd.read_parquet(ROOT / "metadata" / "trials.parquet")
    style()
    keys = sys.argv[1:] or list(ANALYSES)
    rtc: dict = {}
    vals = [make_figure(k, rtc) for k in keys]
    pd.concat(vals, ignore_index=True).to_parquet(OUT / "110_publication_values.parquet", index=False)
    S = pd.DataFrame(STATS)
    S["seed"], S["n_perm"], S["n_boot"] = SEED, N_PERM, N_BOOT
    S.to_csv(OUT / "110_publication_stats.csv", index=False)
    captions()
    print(f"[110] stats rows: {len(S)}", flush=True)


if __name__ == "__main__":
    main()
