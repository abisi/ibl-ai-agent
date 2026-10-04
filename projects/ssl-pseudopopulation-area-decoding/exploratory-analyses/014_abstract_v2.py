"""014 -- COSYNE abstract figure v2 (user 2026-09-30). Two rows x 5 square panels, compact, one-line area legend, R+ / R-
text in cohort colours, every panel labelled and annotated. Learning day.
Row 1, hit vs miss (stimulus-aligned):
  a  area overlay, pseudo-populations R+ (real - paired shift null, mean over iterations) + R+ reaction-time band;
  b  onset latency per area (R+ pseudo-populations): dot = onset (first above-chance bin with >= 4 of 5 above, after the
     stimulus), bar = 95% interval over 500 bootstrap resamples of the iterations; omnibus test of equal onsets = bootstrap
     test (statistic = variance of onsets across areas; null = each area's bootstrap distribution centred on the grand
     mean; the ANOVA analogue for pseudo-populations, which have no per-mouse replicates); post hoc = pairwise bootstrap
     tests (two-sided), BH across pairs, summarised as compact letters (areas sharing a letter do not differ);
  c  whole brain, single-session decoding with a SEPARATE decoder per session half (024 "half" condition, ssl-whisker-
     hitmiss-timeresolved-decoding): 1st half light, 2nd half dark cohort shade (mean +- SEM), window shaded, RT per cohort;
     (the single-decoder 114 run has no 5-100 ms area window and no lick-aligned modality, user 2026-09-30)
  d  window 5-100 ms (mean of real - null over the window bins, 024): each session (dots, paired line) and mean +- SEM; within cohort paired
     Wilcoxon | paired t; cohort x half interaction = mixed ANOVA (between cohort, within half; with 2 halves the
     interaction F = t^2 of the pooled-variance t test on the change) + Mann-Whitney on the change;
  e  per area (same window): mean 1st -> 2nd half per cohort (light -> dark area shade; R+ circles, R- squares);
     linear mixed model (random intercept per session) Wald tests of half x area and cohort x half x area; a coloured
     star = within-cohort paired Wilcoxon and paired t both p < 0.05 (uncorrected).
Row 2, whisker vs auditory:
  f  stimulus-aligned, R+ and R- averaged (pseudo-populations), inset = first 50 ms;
  g  first-lick-aligned, R+ only (pseudo-populations) + R+ band of stimulus times relative to the lick;
  h-j as c-e for first-lick-aligned whisker vs auditory decoding (024), pre-lick window -100..0 ms.
Pseudo-population counts are the pilot 100 iterations x 10 shifts (1000 iterations running).
Outputs: figures/014_abstract_v2.pdf/.png, figures/014_panels/*, ../artifacts/014_stats.csv
Run (haas, repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/014_abstract_v2.py [n_iter]
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"
HALVES = ROOT_REPO / "projects" / "ssl-whisker-hitmiss-timeresolved-decoding" / "exploratory-analyses"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S003 = _load("s003", "003_summarize_pseudopop.py")
S005 = _load("s005", "005_overlay_areas.py")
S011 = _load("s011", "011_overlay_latency.py")
S012 = _load("s012", "012_rt_band.py")
COL = {"R+": "#00B400", "R-": "#C800C8"}
CLAB = {"R+": "R+", "R-": "R−"}
N_BOOT = 500
STATS = []
REC = {"on": True}
ROW = {"hitmiss": dict(win=(5, 100), wkey="sensory100", wlab="5-100 ms", ref="stimulus", view=(-200, 600),
                       label="Hit vs miss"),
       "modality_lick": dict(win=(-100, 0), wkey="prelick", wlab="-100-0 ms", ref="first lick", view=(-600, 200),
                             label="Whisker vs auditory")}


def rec(**kw):
    if REC["on"]:
        STATS.append(kw)


def shade(c, f):
    """f < 1: lighter (mix with white), f = 1: colour itself."""
    r = np.array(to_rgb(c))
    return tuple(1 - f * (1 - r))


def sem(a, axis=0):
    a = np.asarray(a, float)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(a), axis=axis))


def bh(p):
    return S005.bh(p)


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def cohort_text(ax, x, y, parts, **kw):
    """Text made of (string, colour) parts on one line (R+ / R- in cohort colours)."""
    t = ax.transAxes if kw.pop("axes", True) else ax.transData
    fs = kw.pop("fontsize", 5.5)
    ha = kw.pop("ha", "left")
    fig = ax.figure
    fig.canvas.draw()
    txt = None
    xoff = 0
    objs = []
    for s, c in parts:
        txt = ax.text(x, y, s, transform=t, color=c, fontsize=fs, ha="left", va=kw.get("va", "bottom"))
        objs.append(txt)
    # lay out left to right in display space
    r = fig.canvas.get_renderer()
    widths = [o.get_window_extent(r).width for o in objs]
    total = sum(widths)
    x0d, y0d = t.transform((x, y))
    if ha == "center":
        x0d -= total / 2
    elif ha == "right":
        x0d -= total
    for o, w in zip(objs, widths):
        o.set_transform(matplotlib.transforms.IdentityTransform())
        o.set_position((x0d + xoff, y0d))
        xoff += w
    return objs


# ------------------------------------------------------------------------------------------------ data
TAG024 = {"hitmiss": "hitmiss_stim", "modality_lick": "modality_lick"}


def halves024(target):
    """Session halves from the SEPARATE-decoder-per-half single-session decoding (024 "half" condition; user
    2026-09-30: the single-decoder run lacks the 5-100 ms area window and lick-aligned modality). Unified frame:
    area_col, area_value, session_id, reward_group, condition_value, real_curve, null_mean_curve (mean of the
    linear-shift null curves) and corr_<window key> = mean over the window's bins of real - null."""
    R = ROW[target]
    out, t = [], None
    for scope in ("whole_brain", "area_group"):
        d = pd.read_parquet(HALVES / f"024_master_results_{TAG024[target]}_{scope}.parquet",
                            columns=["session_id", "subject_id", "reward_group", "area_value", "condition_type",
                                     "condition_value", "skipped_reason", "n_units", "n_trials", "real_curve",
                                     "shift_null_curves"])
        d = d[d.skipped_reason.isna() & (d.condition_type == "half")].copy()
        t = np.array([b[1] for b in json.load(open(HALVES / f"024_bin_edges_{TAG024[target]}_{scope}.json"))]) * 1000
        d["null_mean_curve"] = [np.nanmean(np.stack([np.asarray(x, float) for x in L]), 0) for L in d.shift_null_curves]
        m = (t >= R["win"][0]) & (t <= R["win"][1])
        d[f"corr_{R['wkey']}"] = [float(np.nanmean((np.asarray(r, float) - n)[m])) for r, n in zip(d.real_curve, d.null_mean_curve)]
        d["area_col"] = scope
        out.append(d.drop(columns=["shift_null_curves"]))
    return pd.concat(out, ignore_index=True), t


def wb_pairs(d, rg, key):
    g = d[(d.area_col == "whole_brain") & (d.reward_group == rg)]
    f = g[g.condition_value == "first"].set_index("session_id")[key]
    s = g[g.condition_value == "second"].set_index("session_id")[key]
    ids = sorted(set(f.index) & set(s.index))
    return f.loc[ids], s.loc[ids], ids


# ------------------------------------------------------------------------------------------------ panels
def p_overlay(ax, X, colors, view, ref, ylabel, rt, title):
    S011.tc(ax, X, colors, view, ref, ylabel, rt)
    ax.set_title(title, fontsize=6.2, pad=3)


def onset_boot(D, t, search):
    rng = np.random.default_rng(0)
    on = S003.onset((np.nanpercentile(D, 5, 0) > 0)[search], t[search]) * 1000
    bo = np.array([S003.onset((np.nanpercentile(D[rng.integers(0, len(D), len(D))], 5, 0) > 0)[search], t[search]) * 1000
                   for _ in range(N_BOOT)])
    return on, bo


def cld(names, sig):
    """Compact letter display (insert-absorb): sig = set of frozenset pairs that differ."""
    cols = [set(names)]
    for pair in sig:
        a, b = tuple(pair)
        new = []
        for c in cols:
            if a in c and b in c:
                new += [c - {a}, c - {b}]
            else:
                new.append(c)
        new = [c for c in new if c]
        cols = [c for c in new if not any(c < o for o in new)]
        cols = [c for i, c in enumerate(cols) if c not in cols[:i]]
    letters = {n: "" for n in names}
    order = sorted(cols, key=lambda c: min(names.index(x) for x in c))
    for k, c in enumerate(order):
        for n in c:
            letters[n] += "abcdefghijklmnopqrstuvwxyz"[k]
    return letters


def p_onset(ax, target, cohort, n_iter, order, colors, T):
    align = "lick" if target == "modality_lick" else "stim"
    t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
    d = pd.read_parquet(ART / f"002_pseudo_{target}_{cohort}.parquet")
    d = d[d.skipped_reason.isna()]
    search = t > 0 if align == "stim" else np.ones(len(t), bool)
    res = {}
    for a in [x for x in order if x not in S005.EXCLUDE]:
        g = d[d.area == a].sort_values("rep").head(n_iter)
        if len(g) < n_iter:
            continue
        D = np.stack(g.curve.map(np.asarray).to_numpy()) - np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
        res[a] = onset_boot(D, t, search)
    areas = sorted([a for a in res if np.isfinite(res[a][0])], key=lambda a: res[a][0])
    # omnibus bootstrap test of equal onsets
    obs = np.array([res[a][0] for a in areas])
    B = np.stack([res[a][1] for a in areas], 1)                  # boot x areas
    V0 = np.var(obs)
    Bc = B - obs[None, :] + obs.mean()
    Vb = np.nanvar(Bc, 1)
    p_omni = float((1 + np.sum(Vb >= V0)) / (1 + len(Vb)))
    pairs, ps = [], []
    for a, b in itertools.combinations(areas, 2):
        df = res[a][1] - res[b][1]
        df = df[np.isfinite(df)]
        p = min(1.0, 2 * min(np.mean(df <= 0), np.mean(df >= 0))) if len(df) else 1.0
        pairs.append((a, b))
        ps.append(max(p, 1 / N_BOOT))
    q = bh(ps)
    sig = {frozenset(pr) for pr, qq in zip(pairs, q) if qq < 0.05}
    letters = cld(areas, sig)
    for k, a in enumerate(areas):
        on, bo = res[a]
        lo, hi = np.nanpercentile(bo, [2.5, 97.5]) if np.isfinite(bo).any() else (np.nan, np.nan)
        ax.plot([k, k], [lo, hi], color=colors.get(a, "0.5"), lw=0.9, zorder=1)
        ax.scatter(k, on, s=12, color=colors.get(a, "0.5"), zorder=2, lw=0)
        ax.text(k, hi + 6, letters[a], ha="center", va="bottom", fontsize=4.2, color="0.25")
        rec(row=target, panel="b", test="onset latency (pseudo-pop bootstrap)", cohort=cohort, area=a, onset_ms=on,
            ci_lo=lo, ci_hi=hi, letters=letters[a])
    for (a, b), p, qq in zip(pairs, ps, q):
        rec(row=target, panel="b", test="post hoc pairwise onset (bootstrap, BH)", cohort=cohort, area_a=a, area_b=b, p=p, q_bh=qq)
    rec(row=target, panel="b", test="omnibus equal onsets (centred-bootstrap variance test)", cohort=cohort,
        n_areas=len(areas), p=p_omni, n_sig_pairs_bh=len(sig))
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels([S005.ABBR.get(a, a) for a in areas], rotation=90, fontsize=4.3)
    for lab, a in zip(ax.get_xticklabels(), areas):
        lab.set_color(colors.get(a, "0.3"))
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylabel(f"onset from {ROW[target]['ref']} (ms)")
    ax.set_xlim(-0.7, len(areas) - 0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.text(0.02, 0.98, f"omnibus {pfmt(p_omni)}\npost hoc: letters (BH)", transform=ax.transAxes, fontsize=4.8,
            va="top", ha="left")
    ax.set_title(f"onset latency, {CLAB[cohort]}", fontsize=6.2, pad=3, color=COL[cohort])


def p_halves_curves(ax, d, t, target):
    """Square cell split in two (user 2026-09-30: the halves difference was hard to see): top = 1st (light) vs 2nd
    (dark) half per cohort with the RT histograms; bottom = change (2nd - 1st), mean +- SEM per cohort."""
    R = ROW[target]
    view = R["view"]
    z = (t >= view[0]) & (t <= view[1])
    ax.set_axis_off()
    top = ax.inset_axes([0, 0.40, 1, 0.60])
    bot = ax.inset_axes([0, 0.0, 1, 0.33], sharex=top)
    ns = {}
    for rg in ("R+", "R-"):
        g = d[(d.area_col == "whole_brain") & (d.reward_group == rg)].copy()
        g["dc"] = [np.asarray(r, float) - np.asarray(n, float) for r, n in zip(g.real_curve, g.null_mean_curve)]
        F = g[g.condition_value == "first"].set_index("session_id").dc
        S = g[g.condition_value == "second"].set_index("session_id").dc
        ids = sorted(set(F.index) & set(S.index))
        ns[rg] = len(ids)
        Fm, Sm = np.stack([F[i] for i in ids]), np.stack([S[i] for i in ids])
        for M, f in ((Fm, 0.35), (Sm, 1.0)):
            mu, se = np.nanmean(M, 0), sem(M)
            top.fill_between(t[z], (mu - se)[z], (mu + se)[z], color=shade(COL[rg], f), alpha=0.25, lw=0)
            top.plot(t[z], mu[z], color=shade(COL[rg], f), lw=0.9)
        D = Sm - Fm
        mu, se = np.nanmean(D, 0), sem(D)
        bot.fill_between(t[z], (mu - se)[z], (mu + se)[z], color=COL[rg], alpha=0.22, lw=0)
        bot.plot(t[z], mu[z], color=COL[rg], lw=0.8)
    for a_ in (top, bot):
        a_.axvspan(*R["win"], color="0.9", lw=0, zorder=0)
        a_.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
        a_.axvline(0, color="k", lw=0.5)
        a_.set_xlim(*view)
        a_.tick_params(labelsize=5)
    top.tick_params(labelbottom=False)
    top.set_ylabel("whole brain\nacc. − null", fontsize=5.5)
    bot.set_ylabel("2nd − 1st", fontsize=5.5)
    bot.set_xlabel(f"time from {R['ref']} (ms)")
    S012.add_rt_band(top, [(S012.rt_values(target, c), COL[c]) for c in ("R+", "R-")], view)
    hs = [Line2D([], [], color=shade(COL[c], f), lw=1.2) for c in ("R+", "R-") for f in (0.35, 1.0)]
    labs = [f"{CLAB[c]} {h}" + (f" (n={ns[c]})" if h == "1st" else "") for c in ("R+", "R-") for h in ("1st", "2nd")]
    leg = top.legend(hs, labs, loc="upper left", fontsize=4.2, handlelength=1.0, ncol=2, columnspacing=0.6,
                     borderaxespad=0.1, labelspacing=0.15)
    for tx, c in zip(leg.get_texts(), ["R+", "R+", "R-", "R-"]):
        tx.set_color(COL[c])
    ax.set_title(f"session halves (window {R['wlab']})", fontsize=5.8, pad=3)


def p_halves_stats(ax, d, target):
    """Group means +- SEM only (user 2026-09-30), 1st (light) -> 2nd (dark) per cohort; within-cohort paired tests,
    mixed-ANOVA cohort x half interaction."""
    R = ROW[target]
    key = f"corr_{R['wkey']}"
    chg, xs = {}, {"R+": (0, 1), "R-": (2.4, 3.4)}
    tops = []
    for rg in ("R+", "R-"):
        f, s, ids = wb_pairs(d, rg, key)
        ok = np.isfinite(f.to_numpy()) & np.isfinite(s.to_numpy())
        f, s = f.to_numpy()[ok], s.to_numpy()[ok]
        chg[rg] = s - f
        x0, x1 = xs[rg]
        ax.plot([x0, x1], [f.mean(), s.mean()], color=COL[rg], lw=0.8, zorder=1)
        for x, v, fsh in ((x0, f, 0.35), (x1, s, 1.0)):
            ax.errorbar(x, v.mean(), yerr=sem(v), fmt="o", ms=4, color=shade(COL[rg], fsh), elinewidth=0.9, capsize=1.5,
                        mew=0, zorder=2)
        pw, pt = stats.wilcoxon(s, f).pvalue, stats.ttest_rel(s, f).pvalue
        tops.append((x0, x1, max(f.mean() + sem(f), s.mean() + sem(s)), pw, pt, rg, len(f)))
        rec(row=target, panel="d", test=f"1st vs 2nd half, whole brain {R['wlab']} (mean over window bins)", cohort=rg,
            n=len(f), mean_first=f.mean(), sem_first=sem(f), mean_second=s.mean(), sem_second=sem(s),
            p_wilcoxon=pw, p_paired_t=pt)
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for x0, x1, tp, pw, pt, rg, n in tops:
        ax.plot([x0, x1], [tp + 0.06 * span] * 2, color=COL[rg], lw=0.6)
        ax.text((x0 + x1) / 2, tp + 0.08 * span, f"W {pfmt(pw)}\nt {pfmt(pt)}", ha="center", va="bottom", fontsize=4.2,
                color=COL[rg])
        ax.text((x0 + x1) / 2, y0 + 0.02 * span, f"n = {n}", ha="center", va="bottom", fontsize=4.2, color=COL[rg])
    tt = stats.ttest_ind(chg["R+"], chg["R-"], equal_var=True)
    df_ = len(chg["R+"]) + len(chg["R-"]) - 2
    pm = stats.mannwhitneyu(chg["R+"], chg["R-"]).pvalue
    rec(row=target, panel="d", test="mixed ANOVA cohort x half interaction (= pooled t on change) + MW on change",
        n_rplus=len(chg["R+"]), n_rminus=len(chg["R-"]), F=tt.statistic ** 2, df1=1, df2=df_, p_interaction=tt.pvalue,
        p_mannwhitney_change=pm, mean_change_rplus=chg["R+"].mean(), mean_change_rminus=chg["R-"].mean())
    ax.set_ylim(y0, y1 + 0.45 * span)
    ax.set_xticks([0, 1, 2.4, 3.4], ["1st", "2nd", "1st", "2nd"])
    for lab, c in zip(ax.get_xticklabels(), ["R+", "R+", "R-", "R-"]):
        lab.set_color(COL[c])
    ax.set_xlim(-0.6, 4.0)
    ax.set_ylabel(f"acc. − null, {R['wlab']}")
    ax.set_xlabel(f"cohort × half: F(1,{df_}) = {tt.statistic ** 2:.2f}, {pfmt(tt.pvalue)}\nchange MW {pfmt(pm)}",
                  fontsize=5)
    ax.set_title("whole brain, mean ± SEM", fontsize=5.8, pad=3)


def p_halves_areas(ax, d, target, colors, order):
    """Square cell split in two, one sub-panel per cohort (user 2026-09-30): per area mean 1st (light) -> 2nd (dark
    area shade); star = within-cohort paired Wilcoxon and paired t both p < 0.05 (uncorrected); LMM Wald tests."""
    R = ROW[target]
    key = f"corr_{R['wkey']}"
    a = d[(d.area_col == "area_group") & ~d.area_value.isin(S005.EXCLUDE)]
    P = a.pivot_table(index=["session_id", "reward_group", "area_value"], columns="condition_value", values=key).dropna()
    P = P.reset_index()
    areas = [x for x in order if x in set(P.area_value)]
    ax.set_axis_off()
    sub = {"R+": ax.inset_axes([0, 0.53, 1, 0.43]), "R-": ax.inset_axes([0, 0.06, 1, 0.43])}
    sub["R-"].sharex(sub["R+"])
    sub["R-"].sharey(sub["R+"])
    for rg, sa in sub.items():
        for k, ar in enumerate(areas):
            g = P[(P.area_value == ar) & (P.reward_group == rg)]
            if len(g) < 2:
                continue
            f, s = g["first"].mean(), g["second"].mean()
            c = colors.get(ar, "0.5")
            sa.plot([k, k], [f, s], color=shade(c, 0.6), lw=0.8, zorder=1)
            sa.scatter([k], [f], s=8, color=shade(c, 0.35), lw=0, zorder=2)
            sa.scatter([k], [s], s=8, color=c, lw=0, zorder=3)
            if len(g) >= 3:
                pw = stats.wilcoxon(g["second"], g["first"]).pvalue
                pt = stats.ttest_rel(g["second"], g["first"]).pvalue
                rec(row=target, panel="e", test=f"1st vs 2nd half per area {R['wlab']}", cohort=rg, area=ar, n=len(g),
                    mean_first=f, mean_second=s, p_wilcoxon=pw, p_paired_t=pt)
                if pw < 0.05 and pt < 0.05:
                    sa.text(k, max(f, s), "*", ha="center", va="bottom", fontsize=6, color="k")
        sa.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
        sa.text(0.99, 0.97, CLAB[rg], transform=sa.transAxes, ha="right", va="top", fontsize=5.5, color=COL[rg],
                fontweight="bold")
        sa.tick_params(labelsize=4.8, axis="y")
        sa.set_ylabel("acc. − null", fontsize=5)
        sa.set_xticks(range(len(areas)))
        sa.set_xlim(-0.7, len(areas) - 0.3)
    sub["R+"].tick_params(labelbottom=False, length=0, axis="x")
    sub["R-"].set_xticklabels([S005.ABBR.get(x, x) for x in areas], rotation=90, fontsize=4.3)
    for lab, x in zip(sub["R-"].get_xticklabels(), areas):
        lab.set_color(colors.get(x, "0.3"))
    sub["R-"].tick_params(axis="x", length=0, pad=1)
    import statsmodels.formula.api as smf
    L = P.melt(id_vars=["session_id", "reward_group", "area_value"], value_vars=["first", "second"], var_name="half",
               value_name="y").rename(columns={"reward_group": "cohort", "area_value": "area"})
    m = smf.mixedlm("y ~ C(cohort, Sum) * C(half, Sum) * C(area, Sum)", L, groups=L["session_id"]).fit(reml=True)
    names = list(m.params.index)
    var_of = lambda piece: piece.split("(", 1)[1].split(",")[0].split(")")[0]
    out = {}
    for lab, want in (("half x area", ["area", "half"]), ("cohort x half x area", ["area", "cohort", "half"]),
                      ("cohort x half", ["cohort", "half"])):
        idx = [i for i, n in enumerate(names) if n.startswith("C(") and sorted(var_of(p) for p in n.split(":")) == want]
        Rm = np.zeros((len(idx), len(names)))
        Rm[np.arange(len(idx)), idx] = 1
        w = m.wald_test(Rm, scalar=True)
        out[lab] = (float(w.statistic), len(idx), float(w.pvalue))
        rec(row=target, panel="e", test=f"LMM (session random intercept) Wald, {R['wlab']}", term=lab,
            chi2=float(w.statistic), df=len(idx), p=float(w.pvalue), n_rows=len(L), n_sessions=L.session_id.nunique())
    short = {"cohort x half": "cohort×half", "half x area": "half×area", "cohort x half x area": "3-way"}
    lines = [f"{short[k]} {pfmt(out[k][2])}" for k in ("cohort x half", "half x area", "cohort x half x area")]
    y0, y1 = sub["R+"].get_ylim()
    sub["R+"].set_ylim(y0, y1 + 0.75 * (y1 - y0))
    sub["R+"].text(0.02, 0.98, "LMM\n" + "\n".join(lines), transform=sub["R+"].transAxes, fontsize=3.7, va="top",
                   ha="left", linespacing=1.05)
    ax.set_title("per area, 1st → 2nd half", fontsize=5.8, pad=3)


def p_stim_avg_inset(ax, X, colors):
    S011.tc(ax, X, colors, (-200, 600), "stimulus", "Whisker vs auditory\nacc. \u2212 null",
            S012.rt_values("modality_stim", "avg"))
    ax.set_title("stim-aligned, R+ & R\u2212 averaged", fontsize=5.8, pad=3)
    ins = inset_axes(ax, width="40%", height="42%", loc="upper left", bbox_to_anchor=(0.07, 0.0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0.3)
    x = X["x"]
    z = (x >= -10) & (x <= 50)
    for a in X["areas"]:
        ins.plot(x[z], X["res"][a]["curve"][z], color=colors.get(a, "0.6"), lw=0.5)
    ins.axvline(0, color="k", lw=0.4)
    ins.set_xlim(-10, 50)
    ins.set_xticks([0, 25, 50])
    ins.tick_params(labelsize=4, length=1.5, pad=1)
    ins.set_xlabel("first 50 ms", fontsize=4.2, labelpad=1)
    ins.patch.set_alpha(0.9)


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    colors, order = S005.allen()
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 3.95))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.885, bottom=0.16, hspace=0.72, wspace=0.8)
    for ax in axes.flat:
        ax.set_box_aspect(1)
    hm_p = S011.compute("R+", n_iter, order, T)["hitmiss"]
    ml_p = S011.compute("R+", n_iter, order, T)["modality_lick"]
    ms_avg = S011.compute("avg", n_iter, order, T)["modality_stim"]
    dh, th = halves024("hitmiss")
    dl, tl = halves024("modality_lick")
    a, b, c, d, e = axes[0]
    f, g, h, i, j = axes[1]
    p_overlay(a, hm_p, colors, (-200, 600), "stimulus", "Hit vs miss\nacc. \u2212 null",
              S012.rt_values("hitmiss", "R+"), "areas, R+ (pseudo-pop.)")
    a.title.set_color(COL["R+"])
    p_onset(b, "hitmiss", "R+", n_iter, order, colors, T)
    p_halves_curves(c, dh, th, "hitmiss")
    p_halves_stats(d, dh, "hitmiss")
    p_halves_areas(e, dh, "hitmiss", colors, order)
    p_stim_avg_inset(f, ms_avg, colors)
    p_overlay(g, ml_p, colors, (-600, 200), "first lick", "acc. \u2212 null",
              S012.rt_values("modality_lick", "R+"), "lick-aligned, R+ (pseudo-pop.)")
    g.title.set_color(COL["R+"])
    p_halves_curves(h, dl, tl, "modality_lick")
    p_halves_stats(i, dl, "modality_lick")
    p_halves_areas(j, dl, "modality_lick", colors, order)
    for ax, L in zip(axes.flat, "abcdefghij"):
        ax.text(-0.42, 1.13, L, transform=ax.transAxes, fontweight="bold", fontsize=8.5)
    names = [x for x in order if x not in S005.EXCLUDE]
    handles = [Patch(color=colors.get(x, "0.6"), label=S005.ABBR.get(x, x)) for x in names]
    handles += [Patch(color="#b0b0b0", label="reaction times")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=4.6, handlelength=0.7, handleheight=0.6,
               handletextpad=0.25, columnspacing=0.55, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Learning day. a, b, f, g: pseudo-populations (20 mice x 10 neurons, {n_iter} iterations x 10 shifts"
                 f"{', PILOT' if n_iter < 1000 else ''});\nc-e, h-j: single sessions, separate decoder per "
                 "session half (mean \u00b1 SEM); acc. \u2212 null = balanced accuracy \u2212 linear-shift null", fontsize=5.2, y=0.998)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"014_abstract_v2.{ext}", dpi=300)
    plt.close(fig)
    # single panels (no extra stats rows)
    REC["on"] = False
    pan = FIG / "014_panels"
    pan.mkdir(exist_ok=True)
    jobs = {"a_hitmiss_overlay": lambda ax: p_overlay(ax, hm_p, colors, (-200, 600), "stimulus",
                                                      "Hit vs miss\nbalanced accuracy - null", S012.rt_values("hitmiss", "R+"), "areas, R+"),
            "b_hitmiss_onset": lambda ax: p_onset(ax, "hitmiss", "R+", n_iter, order, colors, T),
            "c_hitmiss_halves": lambda ax: p_halves_curves(ax, dh, th, "hitmiss"),
            "d_hitmiss_halves_stats": lambda ax: p_halves_stats(ax, dh, "hitmiss"),
            "e_hitmiss_halves_areas": lambda ax: p_halves_areas(ax, dh, "hitmiss", colors, order),
            "f_stim_avg_inset": lambda ax: p_stim_avg_inset(ax, ms_avg, colors),
            "g_lick_overlay_rplus": lambda ax: p_overlay(ax, ml_p, colors, (-600, 200), "first lick",
                                                         "Whisker vs auditory, lick-aligned\nbalanced accuracy - null",
                                                         S012.rt_values("modality_lick", "R+"), "lick-aligned, R+"),
            "h_lick_halves": lambda ax: p_halves_curves(ax, dl, tl, "modality_lick"),
            "i_lick_halves_stats": lambda ax: p_halves_stats(ax, dl, "modality_lick"),
            "j_lick_halves_areas": lambda ax: p_halves_areas(ax, dl, "modality_lick", colors, order)}
    for name, fn in jobs.items():
        f1, ax1 = plt.subplots(figsize=(2.4, 2.4))
        ax1.set_box_aspect(1)
        fn(ax1)
        f1.tight_layout()
        for ext in ("pdf", "png"):
            f1.savefig(pan / f"014_{name}.{ext}", dpi=300)
        plt.close(f1)
    S = pd.DataFrame(STATS)
    S.to_csv(ART / "014_stats.csv", index=False)
    pd.set_option("display.width", 250)
    keep = S[S.test.str.contains("omnibus|interaction|LMM|1st vs 2nd half, whole")]
    print(keep.dropna(axis=1, how="all").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
