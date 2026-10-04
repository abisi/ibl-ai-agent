"""113 -- Publication figures, whole brain and area group merged on single figures (user request 2026-09-28,
superseding the 112 layout). Learning stage only; two figures per decoding analysis.

SET 1, session-wide decoding (condition "whole"), 113_<analysis>_set1_session_wide:
  a schematic | b,c example session per cohort (the whole-brain session closest to the cohort median in the main
  window): real curve vs its own null (mean +- 2 SD of the null curves)
  d whole brain R+ / R- (mean +- SEM; bars = clusters vs group mouse-block null) | e R+ minus R- (two-sided cluster,
  mouse-level cohort permutation) | f window values (R+ vs R-: MW | Welch) | g decoding (main window, accuracy - null)
  vs behavioural d' = z(whisker lick rate) - z(no-stim lick rate), log-linear corrected, per cohort (scatter + OLS
  + 95% CI; line solid only if p < 0.05; Pearson | Spearman)
  h,i area overlays R+ and R-: group-mean accuracy - null per area, no error bars (latency comparison)
  j onset latency per area (first bin reaching 50% of the peak of the group-mean curve; areas with a significant
  cluster only; point estimates, no spread), R+ and R-; y labels = area colour legend for h,i
  k,l area x time heatmaps R+ / R-, areas ordered by pooled (R+ & R-) latency, same order in every area panel
  m per-area R+ vs R- (main window; mean +- SEM; filled = above null, Wilcoxon BH-FDR); two-way ANOVA cohort x area
  (OLS, type II; session-area rows) in the title; * = post-hoc R+ vs R- with both MW and Welch BH-FDR q < 0.05
SET 2, within-session change (session halves only; no performance states), 113_<analysis>_set2_within_session:
  a schematic | b,c whole brain 1st vs 2nd half, R+ and R- (bars: paired-difference clusters, mouse sign-flip)
  d whole brain main window, 1st vs 2nd half per cohort (mean +- SEM, no individual sessions; paired Wilcoxon |
  paired t; R+ vs R- on the change: MW | Welch) | e change (2nd - 1st) over time, both cohorts (bars as b,c)
  | f cross-half generalisation (cross - within accuracy; vs 0 and R+ vs R-)
  g,h area x time heatmaps of the change, R+ / R- (latency order; pale = outside significant paired clusters)
  i,j per area, 1st vs 2nd half, R+ and R- (two-way ANOVA half x area; * = post-hoc paired Wilcoxon and paired t,
  BH-FDR q < 0.05) | k change per area, R+ vs R- (two-way ANOVA cohort x area; * = MW and Welch BH-FDR q < 0.05)
Values are accuracy minus each session/area/condition's own null. Unit = session (area-level rows = session x area).
Performance-state decoding (whole brain only, no halves) gets Set 1 rows a-g only.
Outputs: figures/publication/113_<analysis>_set{1,2}_*.pdf/.png, 113_stats.csv (every test, full provenance),
         113_behaviour_dprime.csv (per-session hit / FA rates and d')
Run (haas): python 113_publication_merged.py [analysis ...]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, OUT / fname)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


P = _load("p110", "110_publication_figures.py")
Q = _load("p112", "112_publication_sets.py")
COL, COHORTS = P.COL, P.COHORTS
FIG = OUT / "figures" / "publication"
SETNAME = {"v": ""}
Q.CONDS = {"half": Q.CONDS["half"]}      # halves only (no performance states in Set 2)


def rec(**k):
    P.STATS.append(dict(set=SETNAME["v"], **k))


Q.rec = rec


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def xlab(spec):
    return f"time from {'stimulus' if spec['align'] == 'stim' else 'first lick'} (ms)"


def anova2(df, value, f1, f2):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = df[[value, f1, f2]].dropna()
    a = anova_lm(smf.ols(f"{value} ~ C({f1}) * C({f2})", data=d).fit(), typ=2)
    return a, d


def anova_text(a, f1, f2, l1, l2):
    rdf = int(a.loc["Residual", "df"])
    out = []
    for key, lab in ((f"C({f1})", l1), (f"C({f2})", l2), (f"C({f1}):C({f2})", f"{l1}×{l2}")):
        out.append(f"{lab} F({int(a.loc[key, 'df'])},{rdf})={a.loc[key, 'F']:.1f}, {P.fmt(a.loc[key, 'PR(>F)'])}")
    return "ANOVA: " + "; ".join(out)


def rec_anova(a, analysis, tag, what, f1, f2):
    for key in a.index:
        if key == "Residual":
            continue
        rec(analysis=analysis, tag=tag, stage="learning", level="area_group", area="all", window=what, condition=key,
            cohort="", test=f"two-way ANOVA (OLS, type II) {f1} x {f2}", F=a.loc[key, "F"], df=a.loc[key, "df"],
            df_resid=a.loc["Residual", "df"], p=a.loc[key, "PR(>F)"])


# ------------------------------------------------------------------------------------------------ set 1
def session_dprime(sids):
    from ssl_timeresolved_decoding import _active_trials_from_whisker_onset_for_curve
    rows = []
    for sid in sids:
        tr = _active_trials_from_whisker_onset_for_curve(sid, P.TT)
        w, ns = tr[tr.trial_type == "whisker_trial"], tr[tr.trial_type == "no_stim_trial"]
        if len(w) < 5 or len(ns) < 5:
            continue
        hr = (w.lick_flag.sum() + 0.5) / (len(w) + 1)
        fa = (ns.lick_flag.sum() + 0.5) / (len(ns) + 1)
        rows.append(dict(session_id=sid, n_whisker=len(w), n_nostim=len(ns), whisker_lick_rate=w.lick_flag.mean(),
                         fa_rate=ns.lick_flag.mean(), dprime=norm.ppf(hr) - norm.ppf(fa)))
    return pd.DataFrame(rows)


def panel_example(ax, D, rg, spec):
    w = spec["main"]
    s = P.whole(D)
    s = s[s.reward_group == rg]
    r = s.iloc[int(np.argmin(np.abs(s[f"corr_{w}"] - s[f"corr_{w}"].median())))]
    t = D["t"] * 1000
    P.shade_windows(ax, spec)
    ax.fill_between(t, r.nullmean - 2 * r.nullsd, r.nullmean + 2 * r.nullsd, color="#bbbbbb", alpha=0.5, lw=0, label="null ± 2 SD")
    ax.plot(t, r.nullmean, color="#888888", lw=0.7)
    ax.plot(t, r.real, color=COL[rg], lw=1.0, label="real")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0.5, color="#999999", lw=0.5, ls="--")
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(t[0] - 50, t[-1])
    ax.set_xlabel(xlab(spec))
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"example {rg} session {r.session_id[:5]}\n({int(r.n_units)} units, {int(r.n_trials)} trials)", fontsize=5.8)
    ax.legend(loc="upper left", fontsize=5.2, handlelength=1.0)
    return r.session_id


def panel_corr(ax, D, beh, spec, analysis):
    w = spec["main"]
    d = P.whole(D).merge(beh, on="session_id")
    lines = []
    for rg in COHORTS:
        s = d[d.reward_group == rg][["dprime", f"corr_{w}", "subject_id"]].dropna()
        x, y = s.dprime.to_numpy(), s[f"corr_{w}"].to_numpy()
        ax.scatter(x, y, s=8, color=COL[rg], alpha=0.75, lw=0)
        if len(s) < 4:
            continue
        pr, pp_ = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        s2 = res @ res / (len(x) - 2)
        xx = np.linspace(x.min(), x.max(), 100)
        XX = np.column_stack([np.ones(100), xx])
        se = np.sqrt(np.einsum("ij,jk,ik->i", XX, s2 * np.linalg.inv(X.T @ X), XX))
        from scipy.stats import t as tdist
        q = tdist.ppf(0.975, len(x) - 2)
        ax.fill_between(xx, XX @ beta - q * se, XX @ beta + q * se, color=COL[rg], alpha=0.15, lw=0)
        ax.plot(xx, XX @ beta, color=COL[rg], lw=1.0, ls="-" if pp_ < 0.05 else "--")
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition="whole",
            cohort=rg, test="correlation with behavioural d' (Pearson | Spearman)", n=len(s), n_mice=s.subject_id.nunique(),
            r_pearson=pr, p_nonparam=sp, p_param=pp_, rho_spearman=sr, slope=beta[1])
        lines.append(f"{rg}: r={pr:.2f} ({P.fmt_s(pp_)}) | ρ={sr:.2f} ({P.fmt_s(sp)})")
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xlabel("behavioural d′")
    ax.set_ylabel(f"accuracy − null ({w})")
    ax.set_title("\n".join(lines), fontsize=5.2)


def latency_order(D, areas, lat_range):
    t = D["t"]
    m = (t >= lat_range[0]) & (t <= lat_range[1])
    lat = {}
    for a in areas:
        c = np.nanmean(np.stack(P.whole(D, a)["corr"].to_numpy())[:, m], 0)
        k = int(np.nanargmax(c))
        lat[a] = t[m][np.where(c[: k + 1] >= 0.5 * c[k])[0][0]] if c[k] > 0 else np.inf
    return sorted(areas, key=lambda a: lat[a]), lat


def panel_overlay(ax, D, areas, rg, spec, colors, ylim):
    t = D["t"] * 1000
    P.shade_windows(ax, spec)
    for a in areas:
        s = P.whole(D, a)
        s = s[s.reward_group == rg]
        ax.plot(t, np.nanmean(np.stack(s["corr"].to_numpy()), 0), color=colors.get(a, "#888"), lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_ylim(*ylim)
    ax.set_xlim(t[0] - 50, t[-1])
    ax.set_xlabel(xlab(spec))
    ax.set_ylabel("accuracy − null")
    ax.set_title(f"{rg}: all areas (group means)", fontsize=6.3, color=COL[rg])


def panel_latency_points(ax, maps, areas, colors):
    for k, rg in enumerate(COHORTS):
        for i, (e, _, _) in enumerate(maps[rg][2]):
            if np.isfinite(e):
                ax.plot(e * 1000, i + (-0.15 if k == 0 else 0.15), "o", ms=3.2, color=COL[rg])
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels(areas, fontsize=5.6)
    for tl, a in zip(ax.get_yticklabels(), areas):
        tl.set_color(colors.get(a, "#333"))
    ax.grid(axis="y", color="#eeeeee", lw=0.5)
    ax.set_xlabel("onset latency (ms, 50% of peak)")
    ax.set_title("onset latency (sig. areas)", fontsize=6.3)


def panel_area_cohort(ax, D, areas, spec, analysis):
    """Per-area R+ vs R- in cohort colours; two-way ANOVA cohort x area + post-hoc MW | Welch (FDR)."""
    w = spec["main"]
    d = P.whole(D)
    d = d[d.area_value.isin(areas)].rename(columns={f"corr_{w}": "value"})
    a, _ = anova2(d, "value", "reward_group", "area_value")
    rec_anova(a, analysis, D["tag"], w, "cohort", "area")
    rows = []
    for ar in areas:
        va = d[(d.area_value == ar) & (d.reward_group == "R+")].value.dropna()
        vb = d[(d.area_value == ar) & (d.reward_group == "R-")].value.dropna()
        pm, pw_ = P.p2(va, vb)
        rows.append(dict(area=ar, pm=pm, pw=pw_, pa=P.p1(va)[0], pb=P.p1(vb)[0], va=va, vb=vb))
    R = pd.DataFrame(rows)
    R["qm"] = multipletests(R.pm.fillna(1), method="fdr_bh")[1]
    R["qw"] = multipletests(R.pw.fillna(1), method="fdr_bh")[1]
    R["qa"] = multipletests(R.pa.fillna(1), method="fdr_bh")[1]
    R["qb"] = multipletests(R.pb.fillna(1), method="fdr_bh")[1]
    top = []
    for i, r in R.iterrows():
        for k, (rg, v, q) in enumerate((("R+", r.va, r.qa), ("R-", r.vb, r.qb))):
            m, se = v.mean(), v.std() / np.sqrt(len(v))
            ax.errorbar(i + (-0.16 if k == 0 else 0.16), m, yerr=se, fmt="o", ms=3.2, mfc=COL[rg] if q < 0.05 else "white",
                        mec=COL[rg], mew=0.9, ecolor=COL[rg], elinewidth=0.8, capsize=1.5)
            top.append(m + se)
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=r.area, window=w, condition="whole",
            cohort="R+ vs R-", test="post-hoc Mann-Whitney | Welch, BH-FDR across areas", n=len(r.va) + len(r.vb),
            mean_a=r.va.mean(), mean_b=r.vb.mean(), p_nonparam=r.pm, p_param=r.pw, q_nonparam=r.qm, q_param=r.qw)
    y1 = np.nanmax(top) + 0.02
    for i, r in R.iterrows():
        if r.qm < 0.05 and r.qw < 0.05:
            ax.text(i, y1, "*", ha="center", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=40, ha="right", fontsize=5.8)
    ax.set_xlim(-0.6, len(areas) - 0.4)
    ax.set_ylim(top=y1 + 0.04)
    ax.set_ylabel(f"accuracy − null ({w})")
    ax.set_title(anova_text(a, "reward_group", "area_value", "cohort", "area")
                 + "\nfilled: > null (FDR); *: R+ vs R− post-hoc (MW & Welch, FDR)", fontsize=5.6)


def set1(key, spec, DL, AL, beh, rng, rtc):
    SETNAME["v"] = "set1_session_wide"
    has_area = AL is not None
    fig = plt.figure(figsize=(7.2, 13.0 if has_area else 5.6))
    H = fig.get_figheight()
    Q.header(fig, spec, DL, "Set 1, session-wide decoding, learning stage: whole brain and area groups")
    y = lambda v: 1 - (1 - v) * 13.0 / H  # noqa: E731  (keeps the top rows' absolute layout in the short figure)
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=y(0.935), bottom=y(0.815), wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 4, left=0.07, right=0.98, top=y(0.76), bottom=y(0.64), wspace=0.62)
    a = fig.add_subplot(r0[0]); P.panel_schematic(a, spec, DL, None); P.letter(a, "a", dx=-0.02, dy=1.0)
    for j, rg in enumerate(COHORTS):
        ax = fig.add_subplot(r0[1 + j]); sid = panel_example(ax, DL, rg, spec); P.letter(ax, "bc"[j], dx=-0.26)
        rec(analysis=key, tag=DL["tag"], stage="learning", level="whole_brain", area="All units", window=spec["main"], condition="whole",
            cohort=rg, test="example session (closest to cohort median)", session_id=sid)
    d = fig.add_subplot(r1[0]); P.panel_timecourse(d, DL, spec, rng, "learning", rtc, key); P.letter(d, "d", dx=-0.4)
    d.set_title("whole brain", fontsize=6.3)
    d.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=5.0, handlelength=1.0)
    e = fig.add_subplot(r1[1]); P.panel_diff(e, DL, spec, rng, key); P.letter(e, "e", dx=-0.4)
    e.set_title("R+ minus R−", fontsize=6.3)
    f = fig.add_subplot(r1[2]); P.panel_windows(f, DL, spec, key); P.letter(f, "f", dx=-0.4)
    f.set_title("windows (MW | Welch)", fontsize=6.0)
    g = fig.add_subplot(r1[3]); panel_corr(g, DL, beh, spec, key); P.letter(g, "g", dx=-0.4)
    if has_area:
        areas0 = P.area_list(AL)
        areas, _ = latency_order(AL, areas0, spec["lat_range"])
        colors = P.m034.get_area_color_map(areas)
        maps = P.area_maps(AL, areas, spec, rng, key, "learning")
        r2a = fig.add_gridspec(1, 2, left=0.07, right=0.66, top=0.585, bottom=0.46, wspace=0.28)
        r2b = fig.add_gridspec(1, 1, left=0.8, right=0.98, top=0.585, bottom=0.46)
        means = [np.nanmean(np.stack(P.whole(AL, a_)[P.whole(AL, a_).reward_group == rg]["corr"].to_numpy()), 0)
                 for a_ in areas for rg in COHORTS]
        ylim = (min(-0.03, np.nanmin(means) * 1.1), np.nanmax(means) * 1.1)
        for j, rg in enumerate(COHORTS):
            ax = fig.add_subplot(r2a[j]); panel_overlay(ax, AL, areas, rg, spec, colors, ylim); P.letter(ax, "hi"[j], dx=-0.25)
        jx = fig.add_subplot(r2b[0]); panel_latency_points(jx, maps, areas, colors); P.letter(jx, "j", dx=-0.95)
        r3 = fig.add_gridspec(1, 3, left=0.2, right=0.8, top=0.4, bottom=0.265, width_ratios=[1, 1, 0.04], wspace=0.1)
        vmax = max(0.05, np.nanpercentile(np.concatenate([maps[rg][0].ravel() for rg in COHORTS]), 99))
        norm_ = TwoSlopeNorm(vmin=-vmax / 3, vcenter=0, vmax=vmax)
        k = fig.add_subplot(r3[0]); im = P.panel_heatmap(k, maps, "R+", areas, AL, spec, colors, True, norm_); P.letter(k, "k", dx=-0.62)
        l_ = fig.add_subplot(r3[1]); P.panel_heatmap(l_, maps, "R-", areas, AL, spec, colors, False, norm_); P.letter(l_, "l", dx=-0.06)
        cb = fig.colorbar(im, cax=fig.add_subplot(r3[2]))
        cb.ax.tick_params(labelsize=5.5)
        fig.text(0.2, 0.428, "areas ordered by decoding latency (pooled cohorts); pale = outside significant clusters", fontsize=6.0)
        r4 = fig.add_gridspec(1, 1, left=0.08, right=0.98, top=0.2, bottom=0.085)
        m = fig.add_subplot(r4[0]); panel_area_cohort(m, AL, areas, spec, key); P.letter(m, "m", dx=-0.06)
    save(fig, f"113_{key}_set1_session_wide")


# ------------------------------------------------------------------------------------------------ set 2
def panel_wb_halves(ax, D, spec, analysis):
    w = spec["main"]
    deltas, brk = {}, []
    for k, rg in enumerate(COHORTS):
        A, B = Q.paired(D, "half", rg)
        a_, b_ = A[f"corr_{w}"].to_numpy(), B[f"corr_{w}"].to_numpy()
        x0 = k * 2.2
        for j, (v, c) in enumerate(((a_, P.m034.lighten(COL[rg], 0.5)), (b_, COL[rg]))):
            ax.errorbar(x0 + j, np.nanmean(v), yerr=np.nanstd(v) / np.sqrt(len(v)), fmt="o", ms=4, color=c, capsize=2, lw=1.0)
        ax.plot([x0, x0 + 1], [np.nanmean(a_), np.nanmean(b_)], color=COL[rg], lw=0.8)
        pw, pt = P.pp(a_, b_)
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition="half 2nd vs 1st",
            cohort=rg, test="paired Wilcoxon | paired t", n=len(a_), n_mice=A.subject_id.nunique(), mean_a=np.nanmean(a_),
            mean_b=np.nanmean(b_), p_nonparam=pw, p_param=pt)
        brk.append((x0, x0 + 1, f"{P.fmt_s(pw)} | {P.fmt_s(pt)}"))
        deltas[rg] = b_ - a_
    pm, pw_ = P.p2(deltas["R+"], deltas["R-"])
    rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition="half change",
        cohort="R+ vs R-", test="Mann-Whitney | Welch on 2nd - 1st", n=len(deltas["R+"]) + len(deltas["R-"]),
        mean_a=np.nanmean(deltas["R+"]), mean_b=np.nanmean(deltas["R-"]), p_nonparam=pm, p_param=pw_)
    ax.set_xticks([0, 1, 2.2, 3.2], ["1st", "2nd", "1st", "2nd"])
    ax.set_xlim(-0.6, 3.8)
    ax.set_xlabel("R+ halves      R− halves")
    ax.set_ylabel(f"accuracy − null ({w})")
    P.draw_brackets(ax, brk)
    ax.set_title(f"whole brain (paired W | t)\nchange R+ vs R−: {P.fmt_s(pm)} | {P.fmt_s(pw_)}", fontsize=5.8)


def panel_delta_curves(ax, D, spec, rng, analysis):
    t = D["t"] * 1000
    P.shade_windows(ax, spec)
    for k, rg in enumerate(COHORTS):
        A, B = Q.paired(D, "half", rg)
        Dm = np.stack(B["corr"].to_numpy()) - np.stack(A["corr"].to_numpy())
        mu, se = np.nanmean(Dm, 0), np.nanstd(Dm, 0) / np.sqrt(len(Dm))
        ax.fill_between(t, mu - se, mu + se, color=COL[rg], alpha=0.25, lw=0)
        ax.plot(t, mu, color=COL[rg], lw=1.0, label=rg)
        _, cl = Q.signflip_clusters(Dm, A.subject_id.to_numpy(), D["t"], rng)
        for c in cl:
            if c["p"] < 0.05:
                ax.plot([c["t0"] * 1000, c["t1"] * 1000], [0.9 - 0.08 * k] * 2, transform=ax.get_xaxis_transform(),
                        color=COL[rg], lw=2.0, solid_capstyle="butt")
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(t[0] - 50, t[-1])
    lim = np.nanmax(np.abs(ax.get_ylim()))
    ax.set_ylim(-lim, lim * 1.3)
    ax.set_xlabel(xlab(spec))
    ax.set_ylabel("2nd − 1st (accuracy − null)")
    ax.legend(loc="lower left", fontsize=5.2, handlelength=1.0)
    ax.set_title("change over time (bars: paired clusters)", fontsize=6.0)


def panel_area_halves(ax, D, areas, rg, spec, analysis):
    w = spec["main"]
    rows, long = [], []
    for ar in areas:
        A, B = Q.paired(D, "half", rg, ar)
        a_, b_ = A[f"corr_{w}"].to_numpy(), B[f"corr_{w}"].to_numpy()
        pw, pt = P.pp(a_, b_) if len(a_) >= 3 else (np.nan, np.nan)
        rows.append(dict(area=ar, a=a_, b=b_, pw=pw, pt=pt))
        long += [dict(value=v, half="first", area_value=ar) for v in a_] + [dict(value=v, half="second", area_value=ar) for v in b_]
    R = pd.DataFrame(rows)
    R["qw"] = multipletests(R.pw.fillna(1), method="fdr_bh")[1]
    R["qt"] = multipletests(R.pt.fillna(1), method="fdr_bh")[1]
    an, _ = anova2(pd.DataFrame(long), "value", "half", "area_value")
    rec_anova(an, analysis, D["tag"], f"{w}, {rg}, halves", "half", "area")
    top = []
    for i, r in R.iterrows():
        for j, (v, c) in enumerate(((r.a, P.m034.lighten(COL[rg], 0.5)), (r.b, COL[rg]))):
            if len(v):
                m_, se = np.nanmean(v), np.nanstd(v) / np.sqrt(len(v))
                ax.errorbar(i + (-0.15 if j == 0 else 0.15), m_, yerr=se, fmt="o", ms=3.0, color=c, capsize=1.2, lw=0.8)
                top.append(m_ + se)
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=r.area, window=w, condition="half 2nd vs 1st",
            cohort=rg, test="post-hoc paired Wilcoxon | paired t, BH-FDR across areas", n=len(r.a), mean_a=np.nanmean(r.a) if len(r.a) else np.nan,
            mean_b=np.nanmean(r.b) if len(r.b) else np.nan, p_nonparam=r.pw, p_param=r.pt, q_nonparam=r.qw, q_param=r.qt)
    y1 = np.nanmax(top) + 0.015
    for i, r in R.iterrows():
        if r.qw < 0.05 and r.qt < 0.05:
            ax.text(i, y1, "*", ha="center", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=5.4)
    ax.set_xlim(-0.6, len(areas) - 0.4)
    ax.set_ylim(top=y1 + 0.03)
    ax.set_ylabel(f"accuracy − null ({w})")
    ax.set_title(f"{rg}: 1st (light) vs 2nd half per area\n" + anova_text(an, "half", "area_value", "half", "area"), fontsize=5.3)


def panel_area_change_cohorts(ax, D, areas, spec, analysis):
    w = spec["main"]
    rows, long = [], []
    for ar in areas:
        v = {}
        for rg in COHORTS:
            A, B = Q.paired(D, "half", rg, ar)
            v[rg] = B[f"corr_{w}"].to_numpy() - A[f"corr_{w}"].to_numpy()
            long += [dict(value=x, reward_group=rg, area_value=ar) for x in v[rg]]
        pm, pw_ = P.p2(v["R+"], v["R-"])
        rows.append(dict(area=ar, vp=v["R+"], vm=v["R-"], pm=pm, pw=pw_))
    R = pd.DataFrame(rows)
    R["qm"] = multipletests(R.pm.fillna(1), method="fdr_bh")[1]
    R["qw"] = multipletests(R.pw.fillna(1), method="fdr_bh")[1]
    an, _ = anova2(pd.DataFrame(long), "value", "reward_group", "area_value")
    rec_anova(an, analysis, D["tag"], f"{w}, half change", "cohort", "area")
    top = []
    for i, r in R.iterrows():
        for k, (rg, v) in enumerate((("R+", r.vp), ("R-", r.vm))):
            if len(v):
                m_, se = np.nanmean(v), np.nanstd(v) / np.sqrt(len(v))
                ax.errorbar(i + (-0.16 if k == 0 else 0.16), m_, yerr=se, fmt="o", ms=3.2, color=COL[rg], capsize=1.5, lw=0.8)
                top.append(m_ + se)
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=r.area, window=w, condition="half change",
            cohort="R+ vs R-", test="post-hoc Mann-Whitney | Welch on 2nd - 1st, BH-FDR across areas", n=len(r.vp) + len(r.vm),
            mean_a=np.nanmean(r.vp) if len(r.vp) else np.nan, mean_b=np.nanmean(r.vm) if len(r.vm) else np.nan,
            p_nonparam=r.pm, p_param=r.pw, q_nonparam=r.qm, q_param=r.qw)
    y1 = np.nanmax(top) + 0.015
    for i, r in R.iterrows():
        if r.qm < 0.05 and r.qw < 0.05:
            ax.text(i, y1, "*", ha="center", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=40, ha="right", fontsize=5.8)
    ax.set_xlim(-0.6, len(areas) - 0.4)
    ax.set_ylim(top=y1 + 0.03)
    ax.set_ylabel(f"2nd − 1st half ({w})")
    ax.set_title("change per area, R+ vs R−\n" + anova_text(an, "reward_group", "area_value", "cohort", "area")
                 + "; *: post-hoc MW & Welch (FDR)", fontsize=5.6)


def schematic_halves(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    rng = np.random.default_rng(3)
    n = 60
    xs = np.linspace(0.04, 0.96, n)
    p = 1 / (1 + np.exp(-(np.arange(n) - 25) / 5))
    lick = rng.random(n) < p * 0.85 + 0.05
    ax.plot(xs, 0.6 + 0.22 * p, color="#555555", lw=1.0)
    ax.scatter(xs, np.where(lick, 0.88, 0.54), s=3, color="k", marker="|", lw=0.6)
    ax.text(0.0, 0.93, "whisker trials (ticks: lick / no lick)", fontsize=5.0, va="bottom")
    ax.add_patch(Rectangle((0.04, 0.4), 0.46, 0.07, color=P.m034.lighten("#555555", 0.5), lw=0))
    ax.add_patch(Rectangle((0.5, 0.4), 0.46, 0.07, color="#555555", lw=0))
    ax.text(0.27, 0.435, "1st half", ha="center", va="center", fontsize=5.2)
    ax.text(0.73, 0.435, "2nd half", ha="center", va="center", fontsize=5.2, color="white")
    ax.text(0.0, 0.3, "each half decoded separately (own C, own null);\nchange = (accuracy − null) 2nd − 1st;\n"
                      "cross-half: train on one half, test on the other,\n  minus within-half accuracy", fontsize=5.0, va="top",
            linespacing=1.35)


def set2(key, spec, DL, AL, rng):
    SETNAME["v"] = "set2_within_session"
    fig = plt.figure(figsize=(7.2, 13.0))
    Q.header(fig, spec, DL, "Set 2, within-session change (1st vs 2nd half), learning stage: whole brain and area groups")
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.935, bottom=0.815, wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.755, bottom=0.635, wspace=0.5)
    a = fig.add_subplot(r0[0]); schematic_halves(a); P.letter(a, "a", dx=-0.02, dy=1.0)
    for j, rg in enumerate(COHORTS):
        ax = fig.add_subplot(r0[1 + j]); Q.panel_cond_timecourse(ax, DL, "half", rg, spec, rng, key); P.letter(ax, "bc"[j], dx=-0.26)
        ax.set_title(f"whole brain, {rg}: 1st vs 2nd half", fontsize=6.3)
    d = fig.add_subplot(r1[0]); panel_wb_halves(d, DL, spec, key); P.letter(d, "d", dx=-0.32)
    e = fig.add_subplot(r1[1]); panel_delta_curves(e, DL, spec, rng, key); P.letter(e, "e", dx=-0.32)
    f = fig.add_subplot(r1[2]); Q.panel_crossgen(f, DL, spec, key); P.letter(f, "f", dx=-0.32)
    f.set_title("cross-half generalisation\nR+ vs R− (MW | Welch)", fontsize=5.8)
    if AL is not None:
        areas, _ = latency_order(AL, P.area_list(AL), spec["lat_range"])
        colors = P.m034.get_area_color_map(areas)
        maps = Q.delta_maps(AL, areas, rng, key)
        r2 = fig.add_gridspec(1, 3, left=0.2, right=0.8, top=0.555, bottom=0.42, width_ratios=[1, 1, 0.04], wspace=0.1)
        vmax = max(0.03, np.nanpercentile(np.abs(np.concatenate([maps[rg][0].ravel() for rg in COHORTS])), 99))
        norm_ = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        g = fig.add_subplot(r2[0]); im = P.panel_heatmap(g, maps, "R+", areas, AL, spec, colors, True, norm_); P.letter(g, "g", dx=-0.62)
        h = fig.add_subplot(r2[1]); P.panel_heatmap(h, maps, "R-", areas, AL, spec, colors, False, norm_); P.letter(h, "h", dx=-0.06)
        cb = fig.colorbar(im, cax=fig.add_subplot(r2[2]))
        cb.ax.tick_params(labelsize=5.5)
        fig.text(0.2, 0.585, "area group: 2nd − 1st half, accuracy − null (latency order; pale = outside significant paired clusters)",
                 fontsize=6.0)
        r3 = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.36, bottom=0.265, wspace=0.22)
        for j, rg in enumerate(COHORTS):
            ax = fig.add_subplot(r3[j]); panel_area_halves(ax, AL, areas, rg, spec, key); P.letter(ax, "ij"[j], dx=-0.14)
        r4 = fig.add_gridspec(1, 1, left=0.08, right=0.98, top=0.165, bottom=0.085)
        k = fig.add_subplot(r4[0]); panel_area_change_cohorts(k, AL, areas, spec, key); P.letter(k, "k", dx=-0.06)
    save(fig, f"113_{key}_set2_within_session")


def load114():
    """114 results (single whole-session decoder scored per half) as 110-style D dicts (whole brain, area group)."""
    import json
    p = OUT / "114_halves_single_decoder.parquet"
    if not p.exists():
        return None, None
    d = pd.read_parquet(p)
    d = d[d.skipped_reason.isna()].copy()
    t = np.array([b[1] for b in json.load(open(OUT / "114_bin_edges.json"))])
    wb = d[d.area_col == "whole_brain"].copy()
    wb["real"] = wb.real_curve.map(lambda c: np.asarray(c, float))
    wb["nullmean"] = wb.null_mean_curve.map(lambda c: np.asarray(c, float))
    wb["corr"] = wb.real - wb.nullmean
    ag = d[d.area_col == "area_group"].copy()
    return (dict(df=wb, t=t, null_kind="linear-shift", tag="114_hitmiss_single_decoder_whole_brain"),
            dict(df=ag, t=t, null_kind="linear-shift", tag="114_hitmiss_single_decoder_area_group"))


def panel_counts_methods(ax, DL, DW):
    """Sessions with both halves available: separate decoding per half (024, >= 3 per class per half) vs one
    whole-session decoder scored per half (114, >= 3 per class per half for scoring)."""
    for k, rg in enumerate(COHORTS):
        n_sep = len(Q.paired(DL, "half", rg)[0])
        n_one = len(Q.paired(DW, "half", rg)[0])
        n_all = P.whole(DL)[P.whole(DL).reward_group == rg].session_id.nunique()
        for j, (v, lab) in enumerate(((n_sep, "separate"), (n_one, "single"))):
            ax.bar(k * 2.6 + j, v, color=COL[rg], alpha=0.45 if j == 0 else 0.9, width=0.8, lw=0)
            ax.text(k * 2.6 + j, v + 0.5, str(v), ha="center", fontsize=5.5)
        ax.plot([k * 2.6 - 0.5, k * 2.6 + 1.5], [n_all] * 2, color="k", lw=0.6, ls=":")
    ax.set_xticks([0, 1, 2.6, 3.6], ["sep.", "single", "sep.", "single"])
    ax.set_xlabel("R+            R−")
    ax.set_ylabel("sessions with both halves")
    ax.set_title("sessions per method\n(dotted: whole-session decoded)", fontsize=5.8)


def set2_single(key, spec, DL, AL, DW, DA, rng):
    SETNAME["v"] = "set2b_single_decoder"
    fig = plt.figure(figsize=(7.2, 11.0))
    Q.header(fig, spec, DL, "Set 2b, within-session change with ONE whole-session decoder scored per half, learning stage")
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.93, bottom=0.8, wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.735, bottom=0.6, wspace=0.5)
    a = fig.add_subplot(r0[0])
    a.set_axis_off()
    a.text(0.0, 0.95, "one decoder per session and area, trained on ALL trials\n(pooled stratified CV, folds = min(5, minority));\n"
                      "held-out predictions scored separately in the\n1st and 2nd half (median split of the session);\n"
                      "a half is scored if it has >= 3 hits and >= 3 misses;\nchance: linear-shift null scored the same way\n"
                      f"({P.N_PERM} mouse sign-flips for clusters); time course:\n50-ms bins, 20-ms steps",
           fontsize=5.3, va="top", linespacing=1.4, transform=a.transAxes)
    P.letter(a, "a", dx=-0.02, dy=1.0)
    for j, rg in enumerate(COHORTS):
        ax = fig.add_subplot(r0[1 + j]); Q.panel_cond_timecourse(ax, DW, "half", rg, spec, rng, key); P.letter(ax, "bc"[j], dx=-0.26)
        ax.set_title(f"whole brain, {rg}: 1st vs 2nd half", fontsize=6.3)
    d = fig.add_subplot(r1[0]); panel_wb_halves(d, DW, spec, key); P.letter(d, "d", dx=-0.32)
    e = fig.add_subplot(r1[1]); panel_delta_curves(e, DW, spec, rng, key); P.letter(e, "e", dx=-0.32)
    f = fig.add_subplot(r1[2]); panel_counts_methods(f, DL, DW); P.letter(f, "f", dx=-0.32)
    if DA is not None and len(DA["df"]):
        areas, _ = latency_order(AL, P.area_list(AL), spec["lat_range"])
        areas = [x for x in areas if x in set(DA["df"].area_value)]
        r3 = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.505, bottom=0.39, wspace=0.22)
        for j, rg in enumerate(COHORTS):
            ax = fig.add_subplot(r3[j]); panel_area_halves(ax, DA, areas, rg, spec, key); P.letter(ax, "gh"[j], dx=-0.14)
        r4 = fig.add_gridspec(1, 1, left=0.08, right=0.98, top=0.24, bottom=0.14)
        k = fig.add_subplot(r4[0]); panel_area_change_cohorts(k, DA, areas, spec, key); P.letter(k, "i", dx=-0.06)
    save(fig, f"113_{key}_set2b_single_decoder")


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    P.ROOT = resolve_dataset_dir("ssl_ephys")
    P.ST, P.TT = pd.read_parquet(P.ROOT / "metadata" / "sessions.parquet"), pd.read_parquet(P.ROOT / "metadata" / "trials.parquet")
    P.style()
    keys = sys.argv[1:] or list(P.ANALYSES)
    rtc: dict = {}
    beh_all = []
    for key in keys:
        spec = P.ANALYSES[key]
        pre = spec["prefix"]
        rng = np.random.default_rng(P.SEED)
        DL, AL = P.load_tag(f"{pre}_whole_brain"), P.load_tag(f"{pre}_area_group")
        for D in (DL, AL):
            if D is not None:
                P.add_windows(D, spec["windows"])
        P.D_NULL[pre] = DL["null_kind"]
        beh = session_dprime(P.whole(DL).session_id.unique())
        beh_all.append(beh.assign(analysis=key))
        set1(key, spec, DL, AL, beh, rng, rtc)
        if (DL["df"].condition_type == "half").any():
            set2(key, spec, DL, AL, rng)
        if key == "hitmiss":
            DW, DA = load114()
            if DW is not None:   # 114 rows already carry the real_/null_/corr_ window columns
                set2_single(key, spec, DL, AL, DW, DA, rng)
        print(f"[113] {key} done", flush=True)
    S = pd.DataFrame(P.STATS)
    S["seed"], S["n_perm"] = P.SEED, P.N_PERM
    S.to_csv(OUT / "113_stats.csv", index=False)
    pd.concat(beh_all).drop_duplicates("session_id").to_csv(OUT / "113_behaviour_dprime.csv", index=False)
    print(f"[113] stats rows: {len(S)}", flush=True)


if __name__ == "__main__":
    main()
