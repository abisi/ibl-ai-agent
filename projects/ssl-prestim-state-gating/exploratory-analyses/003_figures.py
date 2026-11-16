"""003 -- Publication figures of the pre-stimulus state / gating tests (user 2026-11-12: "Visualize this with publication ready
figures, also do it per area and area group"). Inputs: 001 (session x area), 002 stats, 004 trial-level tests.

Figure 003_main: whole brain, day-0 learners (R+ #00B400, R- #C800C8), session = unit, mean +- s.e.m. over sessions.
  Row 1, the state:  A state decodability (bal. acc. - linear-shift null), day 0 -> expert; B P(lick) per state quintile;
                     C specificity (whisker / auditory hit vs miss AUC of the whisker-trained state, gain over movements);
                     D state within hits / misses aligned to the learning trial.
  Row 2, gating:     E gating rho (state, early whisker response | lick, time) variants; F early whisker response per state
                     quintile within hits / misses; G gating by pre-stimulus movements and pupil (positive control);
                     H early whisker response within hits / misses aligned to the learning trial.
  Row 3, passive:    I passive state pre -> post (whisker, auditory trials); J session state shift vs passive sensory readout
                     shift; K passive readout change raw vs controlled for state (day 0, expert); L passive response
                     amplitude change raw vs controlled.
Figures 003_areas{_fine}: per area group (or fine area), day 0: state decodability, gating (excess over shift null), passive
  gating, whole-brain state -> area response, passive state shift, passive readout change raw and controlled; areas coloured
  by the allen_utils palette (fine areas: shades of their group's colour), * BH q < 0.05 (Wilcoxon vs 0 within cohort).
Style: skills/ssl-figure-style (7.4 in, Arial 5-7.5 pt, bold capital letters, edge-free shading, editable text).
Output: figures/003_main.{png,pdf,svg}, 003_areas{_fine}.{png,pdf,svg}; tables/003_main_stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/003_figures.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m61 = importlib.import_module("061_roc_prelick_learners")
ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-prestim-state-gating"
TAB, FIG = ROOT / "tables", ROOT / "figures"
COH = {"R+": "#00B400", "R-": "#C800C8"}
CL = {"R+": "R+", "R-": "R−"}
WB = "whole_brain"
MIN_SESS = 5
STATS = []

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 6,
                     "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "lines.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def stars(p):
    return "" if not np.isfinite(p) or p >= 0.05 else "*" if p >= 0.01 else "**" if p >= 0.001 else "***"


def wil(v):
    v = pd.Series(v).dropna()
    return stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan


def two(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def log(panel, what, cohort, v, p, test="Wilcoxon vs 0", **kw):
    v = pd.Series(v).dropna()
    STATS.append(dict(panel=panel, measure=what, cohort=cohort, n=len(v), mean=v.mean() if len(v) else np.nan,
                      sem=v.sem() if len(v) > 1 else np.nan, p=p, test=test, **kw))


def msem(ax, x, v, c, filled=True, ms=3.5, star=True, panel="", what=""):
    v = pd.Series(v).dropna()
    if not len(v):
        return np.nan
    ax.errorbar(x, v.mean(), v.sem(), fmt="o", color=c, mfc=c if filled else "white", ms=ms, mew=0.7, elinewidth=0.7,
                capsize=0, zorder=3)
    p = wil(v)
    if panel:
        log(panel, what, [k for k, cc in COH.items() if cc == c][0] if c in COH.values() else "", v, p)
    if star and stars(p):
        ax.annotate(stars(p), (x, v.mean() + v.sem()), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                    color=c, fontsize=6)
    return v.mean()


def zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color="0.6", lw=0.4, ls=":", zorder=0)


def letter(fig, ax, s, dx=-0.32, dy=0.12):
    bb = ax.get_position()
    fig.text(bb.x0 + dx * bb.width / 1.0 * 0 - 0.035, bb.y1 + 0.012, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def regress(ax, x, y, color, label):
    """ssl-figure-style rule 5: scatter + OLS + analytic 95 % CI band; solid if p < 0.05."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ax.scatter(x, y, s=5, color=color, alpha=0.6, lw=0)
    if len(x) < 4:
        return np.nan, np.nan
    r, p = stats.spearmanr(x, y)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    res = y - (b * x + a)
    s_err = np.sqrt(res @ res / (len(x) - 2))
    ci = stats.t.ppf(0.975, len(x) - 2) * s_err * np.sqrt(1 / len(x) + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    ax.fill_between(xs, b * xs + a - ci, b * xs + a + ci, color=color, alpha=0.12, lw=0, edgecolor="none")
    ax.plot(xs, b * xs + a, color=color, lw=0.8, ls="-" if p < 0.05 else "--")
    return r, p


def load():
    learners, _ = m61.learner_mice()
    D0 = pd.read_parquet(TAB / "001_state_gating.parquet")
    D0 = D0[D0.skipped_reason.isna() & D0.mouse_id.isin(learners)].assign(stage="day 0")
    DE = pd.read_parquet(TAB / "001_state_gating_expert.parquet")
    DE = DE[DE.skipped_reason.isna()].assign(stage="expert")
    return pd.concat([D0, DE], ignore_index=True)


def main_figure(D):
    W = D[D.area == WB]
    d0 = W[W.stage == "day 0"]
    dose = pd.read_parquet(TAB / "004_dose.parquet").query("area == @WB")
    mov = pd.read_parquet(TAB / "004_movement.parquet").query("area == @WB")
    aud = pd.read_parquet(TAB / "004_auditory.parquet").query("area == @WB")
    lrn = pd.read_parquet(TAB / "004_learning.parquet").query("area == @WB")

    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(3, 4, left=0.085, right=0.985, top=0.94, bottom=0.075, wspace=0.55, hspace=0.75)
    ax = {k: fig.add_subplot(gs[i // 4, i % 4]) for i, k in enumerate("ABCDEFGHIJKL")}
    off = {"R+": -0.12, "R-": 0.12}

    # A state decodability day 0 -> expert
    a = ax["A"]
    for c in COH:
        m = [msem(a, j + off[c], W[(W.stage == s) & (W.reward_group == c)].state_bal_acc_excess, COH[c], filled=s == "expert",
                  panel="A", what=f"state_bal_acc_excess {s}") for j, s in enumerate(("day 0", "expert"))]
        a.plot(np.arange(2) + off[c], m, color=COH[c], lw=0.7, zorder=2)
        mw, we = two(W[(W.stage == "day 0") & (W.reward_group == c)].state_bal_acc_excess,
                     W[(W.stage == "expert") & (W.reward_group == c)].state_bal_acc_excess)
        STATS.append(dict(panel="A", measure="day 0 vs expert", cohort=c, p=mw, p_welch=we, test="Mann-Whitney / Welch"))
        a.text(0.03, 0.97 - 0.09 * (c == "R-"), f"{CL[c]} day 0 vs expert p = {mw:.3f}", transform=a.transAxes, ha="left",
               va="top", color=COH[c], fontsize=4.5)
    zero(a)
    a.set_xticks([0, 1], ["Day 0", "Expert"]); a.set_xlim(-0.5, 1.5)
    a.set_ylabel("Hit vs miss decoding from\npre-stimulus activity\n(bal. accuracy − shift null)")
    a.set_title("Pre-stimulus state predicts choice")

    # B P(lick) per state quintile
    a = ax["B"]
    for c in COH:
        g = dose[dose.reward_group == c].groupby("quintile").p_lick
        a.errorbar(g.mean().index + off[c] / 2, g.mean(), g.sem(), color=COH[c], marker="o", ms=2.5, lw=0.8, elinewidth=0.6,
                   label=f"{CL[c]} (n = {dose[dose.reward_group == c].session_id.nunique()})")
        rs = dose[dose.reward_group == c].groupby("session_id").apply(lambda d: stats.spearmanr(d.quintile, d.p_lick)[0])
        log("B", "rho(quintile, P(lick)) per session", c, rs, wil(rs))
    a.set_xticks(range(1, 6), ["1\nmiss-like", "2", "3", "4", "5\nhit-like"])
    a.set_xlabel("Pre-stimulus state quintile"); a.set_ylabel("P(lick | whisker)")
    a.set_ylim(0, 1.02); a.legend(frameon=False, loc="upper left", handlelength=1)
    a.set_title("Held-out state score vs behaviour")

    # C specificity
    a = ax["C"]
    cats = [("Whisker\nhit vs miss", aud, lambda t: t.auc_whisker - 0.5), ("Auditory\nhit vs miss", aud, lambda t: t.auc_aud - 0.5),
            ("Beyond pre-stim.\nmovements", mov, lambda t: t.auc_mov_state - t.auc_mov)]
    for j, (lab, t, f) in enumerate(cats):
        for c in COH:
            v = f(t[t.reward_group == c])
            msem(a, j + off[c], v, COH[c], panel="C", what=lab.replace("\n", " "))
    zero(a)
    a.set_xticks(range(3), ["Whisker\nhit/miss", "Auditory\nhit/miss", "Beyond\nmovements"], fontsize=5); a.set_xlim(-0.5, 2.5)
    a.text(0.97, 0.06, "n sessions R+ 28 / 6 / 28\nR− 28 / 9 / 26", transform=a.transAxes, ha="right", va="bottom",
           fontsize=4.5, color="0.3")
    a.set_ylabel("Δ AUC of the whisker-trained\nstate score (vs 0.5 / vs movements)")
    a.set_title("General engagement, not movement")

    # D / H learning trial alignment (within hits / misses)
    for key, pre, ttl, yl in (("D", "s", "State around the learning trial", "Pre-stimulus state (z)"),
                              ("H", "g", "Early whisker response around\nthe learning trial", "Early whisker response (z)")):
        a = ax[key]
        for c in COH:
            for lab, ls in (("hits", "-"), ("misses", "--")):
                t = lrn[lrn.reward_group == c]
                g = t[t.n >= 2].groupby("bin")[f"{pre}_{lab}"]
                cnt = g.count()
                mm, se = g.mean()[cnt >= MIN_SESS], g.sem()[cnt >= MIN_SESS]
                x = mm.index + 2.5
                a.plot(x, mm, color=COH[c], ls=ls, lw=0.8, label=f"{CL[c]} {lab}")
                a.fill_between(x, mm - se, mm + se, color=COH[c], alpha=0.15, lw=0, edgecolor="none")
            for lab in ("hits", "misses"):
                # pre vs post learning trial within outcome, per session
                t = lrn[lrn.reward_group == c]
                pp = t.assign(post=t.bin >= 0).groupby(["session_id", "post"])[f"{pre}_{lab}"].mean().unstack().dropna()
                if len(pp) >= MIN_SESS:
                    STATS.append(dict(panel=key, measure=f"{pre}_{lab} post - pre learning trial", cohort=c, n=len(pp),
                                      mean=(pp[True] - pp[False]).mean(), p=stats.wilcoxon(pp[True], pp[False]).pvalue,
                                      test="Wilcoxon paired"))
        a.axvline(0, color="0.4", lw=0.5, ls="--")
        zero(a)
        a.set_xlabel("Whisker trials from the learning trial"); a.set_ylabel(yl); a.set_title(ttl)
        if key == "D":
            a.legend(frameon=False, fontsize=4.5, loc="upper left", handlelength=1.6, ncol=1)

    # E gating variants
    a = ax["E"]
    gv = [("All (− null)", d0, "gate_rho_excess"), ("Hits", d0, "gate_rho_hits"), ("Misses", d0, "gate_rho_misses"),
          ("Passive", d0, "gate_rho_passive"), ("Mov. ctrl", mov, "gate_rho_movctrl"), ("Auditory", d0, "gate_rho_aud")]
    for j, (lab, t, m) in enumerate(gv):
        for c in COH:
            msem(a, j + off[c], t[t.reward_group == c][m], COH[c], ms=3, panel="E", what=m)
    zero(a)
    a.set_xticks(range(len(gv)), [g[0] for g in gv], fontsize=5, rotation=40, ha="right", rotation_mode="anchor"); a.set_xlim(-0.5, len(gv) - 0.5)
    a.set_ylabel("Gating: partial ρ(state, early\nwhisker response | lick, time)")
    a.set_title("No trial-by-trial gating by the state")

    # F response per quintile within hits / misses
    a = ax["F"]
    for c in COH:
        for lab, ls in (("hits", "-"), ("misses", "--")):
            g = dose[dose.reward_group == c].groupby("quintile")[f"g_{lab}"]
            a.errorbar(g.mean().index + off[c] / 2, g.mean(), g.sem(), color=COH[c], ls=ls, marker="o", ms=2,
                       mfc=COH[c] if lab == "hits" else "white", lw=0.8, elinewidth=0.5, label=f"{CL[c]} {lab}")
    zero(a)
    a.set_xticks(range(1, 6), ["1\nmiss-like", "2", "3", "4", "5\nhit-like"])
    a.set_xlabel("Pre-stimulus state quintile"); a.set_ylabel("Early whisker response (z)")
    a.legend(frameon=False, fontsize=4.5, ncol=2, loc="upper left", handlelength=1.6)
    a.set_title("Response does not scale with state")

    # G movement / pupil gating
    a = ax["G"]
    mv = [("Jaw", "gate_jaw_motion"), ("Nose", "gate_nose_motion"), ("Whisker", "gate_whisker_motion"), ("Pupil", "gate_pupil_area"),
          ("Neural\nstate", "gate_rho")]
    for j, (lab, m) in enumerate(mv):
        for c in COH:
            msem(a, j + off[c], mov[mov.reward_group == c][m], COH[c], ms=3, panel="G", what=m)
    zero(a)
    a.set_xticks(range(len(mv)), [m[0] for m in mv], fontsize=5); a.set_xlim(-0.5, len(mv) - 0.5)
    a.set_ylabel("Partial ρ(pre-stimulus variable,\nearly whisker response | lick, time)")
    a.set_title("Movement gates the response,\nthe choice state does not")

    # I passive state pre -> post
    a = ax["I"]
    for j, (lab, pre, post) in enumerate((("Whisker", "state_pre_W", "state_post_W"), ("Auditory", "state_pre_A", "state_post_A"))):
        for c in COH:
            t = d0[d0.reward_group == c]
            x0 = j * 2.6 + (0 if c == "R+" else 1.0)
            mp = [msem(a, x0 + k * 0.55, t[col], COH[c], filled=k == 1, ms=3, star=False) for k, col in enumerate((pre, post))]
            a.plot([x0, x0 + 0.55], mp, color=COH[c], lw=0.7)
            p = wil(t[post] - t[pre])
            log("I", f"state post - pre {lab}", c, t[post] - t[pre], p, test="Wilcoxon paired")
            a.text(x0 + 0.27, max(mp) + 0.06, stars(p), ha="center", color=COH[c], fontsize=6)
    zero(a)
    a.set_xticks([0.27, 1.27, 2.87, 3.87], ["pre  post\n" + CL["R+"], "pre  post\n" + CL["R-"]] * 2, fontsize=4.5)
    a.text(0.77, -0.32, "Whisker trials", transform=a.get_xaxis_transform(), ha="center", fontsize=5)
    a.text(3.37, -0.32, "Auditory trials", transform=a.get_xaxis_transform(), ha="center", fontsize=5)
    a.set_ylabel("Passive pre-stimulus state\n(< 0 miss-like)")
    a.set_title("After the task: miss-like state")

    # J state shift vs readout shift
    a = ax["J"]
    for i, c in enumerate(COH):
        t = d0[d0.reward_group == c]
        r, p = regress(a, t.state_dW.to_numpy(), t.ro_dW.to_numpy(), COH[c], CL[c])
        STATS.append(dict(panel="J", measure="spearman(state_dW, ro_dW)", cohort=c, n=t[["state_dW", "ro_dW"]].dropna().shape[0],
                          mean=r, p=p, test="Spearman"))
        a.text(0.03, 0.97 - 0.1 * i, f"{CL[c]}: ρ = {r:.2f}, p = {p:.3f}", transform=a.transAxes, color=COH[c], fontsize=5,
               va="top")
    zero(a); zero(a, horizontal=False)
    a.set_xlabel("Passive state shift, post − pre"); a.set_ylabel("Passive whisker sensory\nreadout shift, post − pre")
    a.set_title("State shift tracks sensory shift")

    # K / L raw vs controlled for state
    for key, raw, adj, yl, ttl in (("K", "ro_dW", "ro_dW_adj", "Passive whisker sensory readout\nchange post − pre (hit-like > 0)",
                                    "Part of the passive change is state"),
                                   ("L", "g_dW", "g_dW_adj", "Passive whisker response\namplitude change post − pre",
                                    "Response amplitude drop is not")):
        a = ax[key]
        for j, (s, c) in enumerate((("day 0", "R+"), ("day 0", "R-"), ("expert", "R+"), ("expert", "R-"))):
            t = W[(W.stage == s) & (W.reward_group == c)]
            m = [msem(a, j + k * 0.38 - 0.19, t[col], COH[c], filled=k == 1, ms=3, star=True, panel=key,
                      what=f"{col} {s}") for k, col in enumerate((raw, adj))]
            a.plot([j - 0.19, j + 0.19], m, color=COH[c], lw=0.7)
            pr = t[[raw, adj]].dropna()
            if len(pr) >= MIN_SESS:
                p = stats.wilcoxon(pr[raw], pr[adj]).pvalue
                STATS.append(dict(panel=key, measure=f"{raw} raw vs controlled {s}", cohort=c, n=len(pr),
                                  mean=(pr[raw] - pr[adj]).mean(), p=p, test="Wilcoxon paired"))
                if p < 0.05:
                    a.annotate(f"p={p:.3f}", (j, 1), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=4.5,
                               color=COH[c])
        zero(a)
        a.set_xticks(range(4), [f"{CL[c]}\n{s}" for s, c in (("day 0", "R+"), ("day 0", "R-"), ("expert", "R+"), ("expert", "R-"))],
                     fontsize=4.5)
        a.set_xlim(-0.5, 3.5)
        a.set_ylabel(yl); a.set_title(ttl)
        if key == "K":
            a.text(0.02, 0.03, "open: raw   filled: controlled for state", transform=a.transAxes, fontsize=4.5, color="0.3")

    for k, a in ax.items():
        bb = a.get_position()
        fig.text(bb.x0 - 0.055, bb.y1 + 0.022, k, fontsize=9, fontweight="bold", va="bottom")
    for ext, dpi in (("png", 300), ("pdf", 300), ("svg", 300)):
        fig.savefig(FIG / f"003_main.{ext}", dpi=dpi)
    plt.close(fig)


def area_palette(fine, areas):
    import ephys_utilities.allen_utils.allen_utils as au
    colors, order = au.get_custom_area_groups_colors(), list(au.get_area_group_custom_order())
    if not fine:
        missing = [a for a in areas if a not in colors]
        if missing:
            raise KeyError(f"area groups missing from the palette: {missing}")
        return {a: colors[a] for a in areas}, sorted(areas, key=lambda a: order.index(a) if a in order else 99)
    L = pd.read_parquet(PROJECTS.parent / "reports" / "ssl_analysis" / "derived" / "unit_area_labels.parquet",
                        columns=["area_acronym_custom", "area_group"]).dropna()
    parent = L.groupby("area_acronym_custom").area_group.agg(lambda v: v.value_counts().index[0])
    srt = sorted(areas, key=lambda a: (order.index(parent[a]) if parent.get(a) in order else 99, a))
    pal = {}
    for grp, members in pd.Series(srt).groupby(pd.Series([parent.get(a) for a in srt])):
        base = np.array(mcolors.to_rgb(colors[grp]))
        for i, a in enumerate(members):
            f = 0.3 * i / max(1, len(members) - 1) if len(members) > 1 else 0
            pal[a] = tuple(base + (1 - base) * f)  # shades of the group colour
    return pal, srt


def area_figure(D, fine):
    sfx = "_fine" if fine else ""
    if fine:
        learners, _ = m61.learner_mice()
        D = pd.read_parquet(TAB / "001_state_gating_fine.parquet")
        D = D[D.skipped_reason.isna() & D.mouse_id.isin(learners)].assign(stage="day 0")
    d0 = D[(D.stage == "day 0") & (D.area != WB)]
    cross = pd.read_parquet(TAB / f"004_crossarea{sfx}.parquet")
    cross = cross[cross.kind.str.startswith("whole") & (cross.gain_area != WB)].rename(columns={"gain_area": "area"})
    panels = [("state_bal_acc_excess", d0, "State decodability\n(bal. acc. − shift null)"),
              ("gate_rho_excess", d0, "Gating by own state\n(ρ − shift null)"),
              ("gate_rho_passive", d0, "Gating, passive\nwhisker trials (ρ)"),
              ("excess", cross, "Gating by whole-brain\nstate (ρ − shift null)"),
              ("state_dW", d0, "Passive state shift\npost − pre"),
              ("ro_dW", d0, "Passive sensory readout\nchange (raw)"),
              ("ro_dW_adj", d0, "… controlled\nfor state")]
    cnt = d0.groupby(["area", "reward_group"]).session_id.nunique().unstack().fillna(0)
    keep = cnt[(cnt >= MIN_SESS).any(axis=1)].index.tolist()
    pal, areas = area_palette(fine, keep)
    rows = []
    for m, t, _ in panels:
        for c in COH:
            for a in areas:
                v = t[(t.area == a) & (t.reward_group == c)][m].dropna()
                if len(v) >= MIN_SESS:
                    rows.append(dict(measure=m, cohort=c, area=a, n=len(v), mean=v.mean(), sem=v.sem(), p=wil(v)))
    R = pd.DataFrame(rows)
    R["q_bh"] = np.nan
    for _, g in R[R.p.notna()].groupby(["measure", "cohort"]):
        R.loc[g.index, "q_bh"] = multipletests(g.p, method="fdr_bh")[1]
    R.to_csv(TAB / f"003_areas{sfx}_stats.csv", index=False)

    h = 0.14 * len(areas) + 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, h), sharey=True)
    fig.subplots_adjust(left=0.165 if not fine else 0.08, right=0.99, top=1 - 0.55 / h, bottom=0.35 / h, wspace=0.25)
    y = np.arange(len(areas))
    for ax, (m, _, lab) in zip(axes, panels):
        for k, c in enumerate(COH):
            t = R[(R.measure == m) & (R.cohort == c)].set_index("area").reindex(areas)
            yy = y + (k - 0.5) * 0.32
            ax.errorbar(t["mean"], yy, xerr=t["sem"], fmt="o", ms=2.2, color=COH[c], elinewidth=0.5, mew=0)
            for y0, x0, e0, q in zip(yy, t["mean"], t["sem"], t["q_bh"]):
                if np.isfinite(q) and q < 0.05:
                    ax.text(x0 + e0, y0, " *", color=COH[c], fontsize=6, va="center", ha="left")
        ax.axvline(0, color="0.6", lw=0.4, ls=":")
        for i in range(0, len(areas), 2):
            ax.axhspan(i - 0.5, i + 0.5, color="0.95", lw=0, edgecolor="none", zorder=0)
        ax.set_title(lab, fontsize=5.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y, [a.replace("Somatosensory-", "Ss-") for a in areas], fontsize=5 if not fine else 4.5)
    for t, a in zip(axes[0].get_yticklabels(), areas):
        t.set_color(pal[a])
    axes[0].set_ylim(len(areas) - 0.5, -0.5)
    axes[-1].text(1.0, -0.35 / h * 0.5, "mean ± s.e.m. over sessions; * BH q < 0.05 (Wilcoxon vs 0)", transform=fig.transFigure,
                  ha="right", fontsize=4.5, color="0.3")
    for i, ax in enumerate(axes):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.01, bb.y1 + 0.32 / h, "ABCDEFG"[i], fontsize=9, fontweight="bold")
    h0 = axes[0].plot([], [], "o", color=COH["R+"], ms=3, label="R+")[0], axes[0].plot([], [], "o", color=COH["R-"], ms=3, label="R−")[0]
    fig.legend(handles=h0, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=5, ncol=2, handletextpad=0.1)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"003_areas{sfx}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    D = load()
    main_figure(D)
    pd.DataFrame(STATS).to_csv(TAB / "003_main_stats.csv", index=False)
    area_figure(D, fine=False)
    area_figure(D, fine=True)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(pd.DataFrame(STATS).round(4).to_string())
