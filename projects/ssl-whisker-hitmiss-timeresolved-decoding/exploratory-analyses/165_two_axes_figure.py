"""165 -- Two-row figure, one row per within-session axis, same quantities in both rows (user 2026-10-08, revised).
All values: projection on a unit-norm axis, per neuron (/ sqrt(N)); learners, learning day; session = unit (one session per mouse).
"Shifted" = axes rebuilt from labels shifted by 10-50 % of the time-ordered defining trials (non-wrapping, 20 shifts); the shifted value
of a session is the mean over its shifts.
Row 1, hit - miss axis (stimulus onset 5-35 ms; 135 SSL_135_HARM=1 and 166):
  b  example sessions (R+, R-) in state space (166): x = hit - miss axis, y = passive-pre whisker pattern orthogonal to x (leave-one-out
     for the pre trials); dots: passive whisker trials, pre (grey) and post (coloured); large symbols: means; black arrow: pre -> post
     on the real axis; grey dashed arrow: mean over the shifted-label axes (x on the shifted axis, same y)
  c  passive whisker projection in passive pre and passive post, each as real axis - mean over shifted axes; mean +- s.e.m.
  d  quantity: (post - pre on the real axis) - (mean post - pre on the shifted axes), per cohort (x), whisker and auditory (lines); per
     line R+ vs R- Mann-Whitney; cohort x modality interaction = difference of the whisker - auditory values between cohorts, cohort labels
     permuted across mice (10,000)
  e  per area group: the whisker quantity of d (axis from the group's units); see "Areas" below
Row 2, reward-lick axis (pre-lick 100 ms; ssl-prelick-convergence within_day 026):
  b  example sessions: whisker-hit (yellow), auditory-hit (blue) and spontaneous-lick (black) projections over normalised session time
     with OLS lines (slopes beta written next to them); dotted horizontal lines: no change from each line's start (zero slope); grey
     dashed: whisker-hit line on the shifted CDs
  c  whisker-hit projection and whisker-hit minus spontaneous-lick distance in five bins of session time, relative to the first bin:
     real CD (solid) vs shifted CDs (grey dashed)
  d  quantity: slope over the session on the real CD - mean slope on the shifted CDs, per cohort (x), whisker hits and spontaneous licks
     (lines); interaction test as row 1 (= the whisker-hit - spontaneous-lick distance)
  e  per area group: the whisker-hit - spontaneous-lick distance quantity
Areas: hit - miss axis: area groups with >= 20 tracked units in the session (needed for the axis and its shifts, 135); reward-lick axis:
>= 5 units and a held-out AH-vs-SL d' >= 0.3 on the group's own axis (as 015). Both: >= 3 sessions in each cohort. Two-way ANOVA
(type II sums of squares) value ~ cohort x area on session x area values; p of the cohort and cohort x area F by permuting cohort labels
across mice (2,000; keeps all of a mouse's areas together); per area R+ vs R- Mann-Whitney with Benjamini-Hochberg q over area groups.
x axis clipped to the means +- 1.5 median s.e.m. (error bars may extend beyond).
Supplementary figure: the hit - miss row for the auditory responses (b, c, e).
Variant SSL_165_PSTH=1 (2026-10-08): an extra column after the schematic with the time course of the projection on each axis, mean
+- s.e.m. over sessions, one subpanel per cohort (held out: no trial is projected on an axis it helped build):
  row 1 (168): active whisker hits / misses (held-out halves) and passive whisker pre / post, -100 to +200 ms from stimulus onset; unit
  group SSL_165_PSTH_AREA (default whole brain); shaded: 5-35 ms; gap: artefact window
  row 2 (027 of ssl-prelick-convergence): AH, SL (held-out folds) and WH, -400 to +200 ms from the first lick; shaded: [-100, 0) ms
  outputs 165_two_axes_figure_psth[_<area>].{svg,pdf,png}
Compact variant SSL_165_COMPACT=1 (2026-10-08): built at print size (180 mm wide); same panels as the main figure (b-e), the time
courses small under the schematic (R+ and R- side by side), no spacer column; per-line p values in the legend of d; short area names,
q < 0.05 marked * (exact p / q in the _stats.csv); ring / marker conventions go in the caption. Output 165_two_axes_compact[_<area>].{svg,pdf,png}
Outputs: figures/publication/165_two_axes_figure.{svg,pdf,png}, 165_two_axes_auditory.{svg,pdf,png}; 165_stats.csv
Run (haas, repo root): python .../165_two_axes_figure.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
PL = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence/within_day/sl_stable/cd_shift_null")
EV = {"W": "#f7b519", "A": "#2c2cdb", "WH": "#f7b519", "AH": "#2c2cdb", "SL": "#262626", "D": "#b07e00"}
N_PERM_A, N_PERM_I = 2000, 10000
import os  # noqa: E402
COMPACT = bool(os.environ.get("SSL_165_COMPACT"))
PSTH = bool(os.environ.get("SSL_165_PSTH")) or COMPACT
SHORT = {"Motor areas": "Motor", "Somatosensory-whisker": "S1 whisker", "Somatosensory-orofacial": "S1 orofacial", "Frontal areas": "Frontal",
         "Retrosplenial areas": "Retrosplenial", "Auditory areas": "Auditory", "Lateral septal complex": "Lat. septum",
         "Posterior parietal areas": "Post. parietal", "Visual areas": "Visual", "Insular areas": "Insular", "Olfactory areas": "Olfactory",
         "Hippocampus": "Hippocampus"}
PSTH_AREA = os.environ.get("SSL_165_PSTH_AREA", "All units")
PSFX = "_psth" + ("" if PSTH_AREA == "All units" else "_" + PSTH_AREA.split()[0].lower())
PL_PSTH = PL.parent / "cd_psth" / "cd_psth_projection.parquet"

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
Q1 = "real − shifted axes:\nΔ(post − pre) per neuron"
Q2 = "real − shifted CDs:\nslope over the session per neuron"


def area_colors():
    try:
        sys.path.insert(0, str(Path.home() / "code" / "ephys_utilities"))
        from ephys_utilities.allen_utils import allen_utils
        return allen_utils.get_custom_area_groups_colors()
    except Exception:  # noqa: BLE001
        return {}


def wil(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan


def mwu(a, b):
    a, b = (np.asarray(v, float) for v in (a, b)); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue if min(len(a), len(b)) >= 3 else np.nan


def pn(p):
    return "n.s." if not np.isfinite(p) or p >= 0.05 else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def anova_perm(D, val, rng):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    D = D.dropna(subset=[val]).copy()

    def fstats(d):
        a = sm.stats.anova_lm(smf.ols(f"{val} ~ C(cohort) * C(area)", d).fit(), typ=2)
        return a.loc["C(cohort)", "F"], a.loc["C(cohort):C(area)", "F"]
    f0 = fstats(D)
    mice = D.mouse_id.unique(); mc = D.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    idx = pd.Index(mice).get_indexer(D.mouse_id)
    null = np.asarray([fstats(D.assign(cohort=rng.permutation(mc)[idx])) for _ in range(N_PERM_A)])
    return [(1 + np.sum(null[:, j] >= f0[j])) / (1 + len(null)) for j in range(2)]


def interaction_perm(D, c1, c2, rng):
    """(mean c1 - mean c2)(R+) - (...)(R-), cohort labels permuted across mice (one session per mouse)"""
    d = (D[c1] - D[c2]).to_numpy(float); ok = np.isfinite(d); d = d[ok]; coh = D.cohort.to_numpy()[ok]
    obs = d[coh == "R+"].mean() - d[coh == "R-"].mean()
    null = np.empty(N_PERM_I)
    for i in range(N_PERM_I):
        c = rng.permutation(coh); null[i] = d[c == "R+"].mean() - d[c == "R-"].mean()
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + N_PERM_I)


# ---------------------------------------------------------------- panels
def example_hm(axs, Ex, Sh, s_):
    for ax, c in zip(axs, COH):
        e = Ex[(Ex.cohort == c) & (Ex.stim == s_)]; sh = Sh[(Sh.cohort == c) & (Sh.stim == s_)]
        a = sh.groupby("epoch")[["x", "y"]].mean()                          # mean over shifted-label axes
        m = {}
        for ep in ("passive_pre", "passive_post"):
            d = e[e.epoch == ep]
            fc = "0.6" if ep == "passive_pre" else EV[s_]
            ax.plot(d.real, d.y, "o", ms=2.2, mfc=fc, mec=fc, mew=0.3, alpha=0.6)
            m[ep] = (d.real.mean(), d.y.mean())
            ax.plot(*m[ep], "o", ms=5.5, mfc=fc, mec="black", mew=0.8, zorder=5)
        ax.annotate("", (a.loc["passive_post", "x"], a.loc["passive_post", "y"]), (a.loc["passive_pre", "x"], a.loc["passive_pre", "y"]),
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.2, ls=(0, (2, 1.2)), shrinkA=3, shrinkB=3, mutation_scale=8), zorder=4)
        ax.annotate("", m["passive_post"], m["passive_pre"], arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=3, shrinkB=3), zorder=6)
        ax.axvline(0, color="0.85", lw=0.5); ax.axhline(0, color="0.85", lw=0.5)
        ax.set_xlabel("hit − miss axis (→ hit)", fontsize=5)
        ax.set_title(f"{c.replace('-', '−')} {e.session_id.iloc[0][:5]}" if COMPACT else f"{c.replace('-', '−')} example\n{e.session_id.iloc[0]}",
                     color=COL[c], fontsize=5.2)
    axs[0].set_ylabel("pre whisker pattern\n(orthogonal; per neuron)")
    if not COMPACT:
        axs[1].text(0.98, 0.02, "grey: pre, coloured: post\nblack arrow: real axis\ngrey dashed: mean of shifted axes", transform=axs[1].transAxes, ha="right", fontsize=4.2, color="0.4")


def example_pl(axs, Ex):
    for ax, c in zip(axs, COH):
        e = Ex[Ex.cohort == c]
        txt = []
        for cl in ("SL", "AH", "WH"):
            d = e[e.cls == cl].dropna(subset=["real"])
            ax.plot(d.tau, d.real, "o", ms=1.4 if cl == "SL" else 2.0, color=EV.get(cl, "#2c2cdb"), alpha=0.3 if cl == "SL" else 0.7, mew=0)
            b = np.polyfit(d.tau, d.real, 1); xx = np.r_[0, 1]
            ax.plot(xx, np.polyval(b, xx), color=EV.get(cl, "#2c2cdb"), lw=1.4)
            ax.plot(xx, [b[1], b[1]], color=EV.get(cl, "#2c2cdb"), lw=0.6, ls=(0, (1, 1.5)))
            txt.append((cl, b[0]))
            if cl == "WH":
                dn = d.dropna(subset=["null"]); bn = np.polyfit(dn.tau, dn.null, 1)
                ax.plot(xx, np.polyval(bn, xx), color="0.6", lw=1.0, ls=(0, (2.5, 1.5)))
                txt.append(("WH shifted", bn[0]))
        ax.text(0.02, 0.98, "\n".join(f"β {k}: {v:+.2f}" for k, v in txt), transform=ax.transAxes, va="top", fontsize=4.3)
        ax.set_xlabel("time in session τ", fontsize=5)
        ax.set_title(f"{c.replace('-', '−')} {e.session_id.iloc[0][:5]}" if COMPACT else f"{c.replace('-', '−')} example\n{e.session_id.iloc[0]}",
                     color=COL[c], fontsize=5.2)
    axs[0].set_ylabel("trial projection on CD\n(per neuron)")


def change_panel(ax, items, ylab, xt):
    nx = len(xt)
    for j, (lab, col_ev, dd) in enumerate(items):
        off = j * (nx + 0.8)
        for c in COH:
            real, null = dd[c]
            for arr, ls, mfc, lc in ((real, "-", COL[c], COL[c]), (null, (0, (2.5, 1.5)), "white", "0.6")):
                if arr is None:
                    continue
                m = np.nanmean(arr, 0); s = np.array([H.sem(arr[:, i]) for i in range(arr.shape[1])])
                ax.errorbar(off + np.arange(nx), m, yerr=s, fmt="o", ls=ls, ms=2.8, color=lc, mfc=mfc, mec=COL[c], lw=1, capsize=0)
        ax.text(off + (nx - 1) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5, color=col_ev)
    ax.axhline(0, color="0.7", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([j * (nx + 0.8) + i for j in range(len(items)) for i in range(nx)]); ax.set_xticklabels(xt * len(items), fontsize=4.8)
    ax.set_ylabel(ylab)


def cohort_lines(ax, D, cols, labs, colors, ylab, R, row, rng):
    for k, (col, lab, cc) in enumerate(zip(cols, labs, colors)):
        v = {c: D[D.cohort == c][col].to_numpy(float) for c in COH}
        m = [np.nanmean(v[c]) for c in COH]; s = [H.sem(v[c]) for c in COH]
        x = np.arange(2) + (k - 0.5) * 0.08
        pm = mwu(v["R+"], v["R-"])
        ax.errorbar(x, m, yerr=s, fmt="-o", ms=3.4, color=cc, lw=1.3, capsize=0, label=f"{lab}: {pn(pm)}" if COMPACT else lab)
        for i, c in enumerate(COH):
            p0 = wil(v[c])
            if p0 < 0.05:
                ax.plot(x[i], m[i], "o", ms=5.5, mfc="none", mec="0.2", mew=0.6)
            R.append(dict(row=row, panel="d", measure=col, cohort=c, n=int(np.isfinite(v[c]).sum()), mean=m[i], p_vs0=p0))
        R.append(dict(row=row, panel="d", measure=col, cohort="R+ vs R-", p_nonparam=pm))
        if not COMPACT:
            ax.text(1.04, m[1], f"{lab}: {pn(pm)}", color=cc if cc != "#f7b519" else "#b07e00", fontsize=4.8, va="center",
                transform=ax.get_yaxis_transform())
    obs, pi = interaction_perm(D, cols[0], cols[1], rng)
    R.append(dict(row=row, panel="d", measure=f"interaction {cols[0]} - {cols[1]}", cohort="R+ vs R-", mean=obs, p_perm=pi))
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R−"]); ax.set_xlim(-0.4, 1.4)
    for t, c in zip(ax.get_xticklabels(), COH):
        t.set_color(COL[c])
    ax.set_ylabel(ylab, fontsize=5.5)
    ttl = f"cohort × {'modality' if row == 'hit-miss' else 'lick type'}: p = {pi:.3g}"
    if COMPACT:                                                    # interaction p as the legend title, above the axes
        lg = ax.legend(frameon=False, fontsize=4.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, borderaxespad=0.1,
                       title=ttl, title_fontsize=5.3)
        lg._legend_box.align = "left"
    else:
        ax.set_title(ttl + "\n(mouse-level permutation)", fontsize=5.5, pad=6)
        ax.legend(frameon=False, fontsize=4.8, loc="best", handlelength=1.2)
    if not COMPACT:
        ax.text(0.02, 0.02, "ring: ≠ 0 within cohort (Wilcoxon p < 0.05)", transform=ax.transAxes, fontsize=4.2, color="0.4")


def area_panel(ax, D, val, R, row, title, rng, AC, xlab):
    n = D.dropna(subset=[val]).groupby(["area", "cohort"]).size().unstack().fillna(0)
    areas = sorted([a for a in n.index if min(n.loc[a].get("R+", 0), n.loc[a].get("R-", 0)) >= 3], key=lambda a: -n.loc[a].sum())
    Dq = D[D.area.isin(areas)]
    p_coh, p_int = anova_perm(Dq, val, rng)
    ps, ms, ss = [], [], []
    for i, a in enumerate(areas):
        v = {c: Dq[(Dq.area == a) & (Dq.cohort == c)][val].dropna().to_numpy(float) for c in COH}
        for k, c in enumerate(COH):
            p0 = wil(v[c]); m, s = np.nanmean(v[c]), H.sem(v[c]); ms.append(m); ss.append(s)
            ax.errorbar(m, i + (k - 0.5) * 0.3, xerr=s, fmt="o", ms=2.8, color=COL[c], mfc=COL[c] if p0 < 0.05 else "white", lw=0.9, capsize=0)
        ps.append(mwu(v["R+"], v["R-"]))
    q = bh(ps)
    for i, (a, p, qq) in enumerate(zip(areas, ps, q)):
        if not COMPACT:
            ax.text(1.02, i, f"{H.pnum(p)} (q {H.pnum(qq)})", transform=ax.get_yaxis_transform(), va="center", fontsize=4.4,
                weight="bold" if qq < 0.05 else "normal")
        R.append(dict(row=row, panel="e", measure=val, area=a, n_Rp=int(n.loc[a]["R+"]), n_Rm=int(n.loc[a]["R-"]), p_nonparam=p, q_BH=qq))
    R.append(dict(row=row, panel="e", measure=val, area="ANOVA", p_cohort_perm=p_coh, p_cohort_x_area_perm=p_int))
    pad = 1.5 * np.nanmedian(ss)
    ax.set_xlim(np.nanmin(ms) - pad, np.nanmax(ms) + pad)
    ax.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    if COMPACT:
        ax.set_yticks(range(len(areas))); ax.set_yticklabels([SHORT.get(a, a) + (" *" if qq < 0.05 else "") for a, qq in zip(areas, q)], fontsize=4.8)
    else:
        ax.set_yticks(range(len(areas))); ax.set_yticklabels([f"{a} ({int(n.loc[a]['R+'])}/{int(n.loc[a]['R-'])})" for a in areas], fontsize=4.8)
    for t, a in zip(ax.get_yticklabels(), areas):
        t.set_color(AC.get(a, "black"))
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_xlabel("real − shifted (per neuron)" if COMPACT else xlab, fontsize=5)
    ax.set_title(f"ANOVA cohort {pn(p_coh)}\ncohort × area {pn(p_int)}" if COMPACT else
                 f"{title}\nANOVA (permutation): cohort {pn(p_coh)}, cohort × area {pn(p_int)}", fontsize=5.6)


def savefig(fig, path):
    """save; if the file is locked (open in a viewer on the share), write <name>_new.<ext> instead"""
    try:
        fig.savefig(path, dpi=300)
    except OSError:
        alt = path.with_name(path.stem + "_new" + path.suffix); fig.savefig(alt, dpi=300); print("locked, wrote", alt, flush=True)


# ---------------------------------------------------------------- data
def load_hm():
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked_harm.parquet")
    A = A[A.skipped_reason.isna()]; A = A[H.mouse_of(A).isin(H.learners())].copy()
    A["cohort"] = A.reward_group
    for s_ in ("W", "A"):
        A[f"q_{s_}"] = A[f"shift_excess_d{s_}P"]
    return A, pd.read_csv(EA / "166_example_trials.csv")


def load_pl():
    S = pd.read_csv(PL / "sessions_areas.csv"); B = pd.read_csv(PL / "bins.csv"); Ex = pd.read_csv(PL / "example_trials.csv")
    L = H.learners()
    S = S[(S.stage == "learning") & S.mouse_id.isin(L)].copy(); B = B[(B.stage == "learning") & B.mouse_id.isin(L)].copy()
    return S, B, Ex


def frame(fig, spec, txt):
    ax = fig.add_subplot(spec); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.85"); sp.set_visible(True)
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=6, color="0.6", transform=ax.transAxes)


def psth_panels(fig, spec, D, classes, win, xlab, lim):
    """time course of the projection per cohort: D long (session_id, cohort, cls, t, value); classes: (cls, colour, ls, label)"""
    sub = spec.subgridspec(1, 2, wspace=0.12) if COMPACT else spec.subgridspec(2, 1, hspace=0.45)
    axs = []
    for i, c in enumerate(COH):
        ax = fig.add_subplot(sub[0, i] if COMPACT else sub[i, 0], sharey=axs[0] if axs else None); axs.append(ax)
        ax.axvspan(*win, color="0.92", lw=0, zorder=0); ax.axvline(0, color="0.6", lw=0.5, ls=":")
        ax.axhline(0, color="0.85", lw=0.5)
        d = D[D.cohort == c]
        for cl, col, ls, lab in classes:
            piv = d[d.cls == cl].pivot_table(index="session_id", columns="t", values="value")
            m, se = piv.mean(0), piv.sem(0)
            ax.plot(piv.columns, m, color=col, ls=ls, lw=1.0, label=lab)
            ax.fill_between(piv.columns, m - se, m + se, color=col, alpha=0.2, lw=0)
        n = d.session_id.nunique()
        if COMPACT:                                                # cohort label inside; legend goes above the panels
            ax.text(0.04, 0.97, f"{c.replace('-', '−')} ({n})", color=COL[c], fontsize=4.6, ha="left", va="top", transform=ax.transAxes)
        else:
            ax.set_title(f"{c.replace('-', '−')} (n = {n} sessions)", color=COL[c], fontsize=5.4, pad=2)
        ax.set_xlim(*lim); ax.tick_params(labelsize=4.5 if COMPACT else 5)
        if COMPACT:
            ax.set_xticks([lim[0], 0] if i == 0 else [0, lim[1]])
            if i == 1:
                ax.tick_params(labelleft=False)
        elif i == 0:
            ax.tick_params(labelbottom=False)
    if COMPACT:
        fig.text(0.5 * (axs[0].get_position().x0 + axs[1].get_position().x1), axs[0].get_position().y0 - 0.045, xlab,
                 ha="center", va="top", fontsize=4.8)
        axs[0].legend(frameon=False, fontsize=4.2, loc="lower left", bbox_to_anchor=(-0.05, 1.0), handlelength=1.2, ncol=2,
                      columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)
    else:
        axs[-1].set_xlabel(xlab, fontsize=5)
    if COMPACT:
        pass
    else:
        axs[-1].legend(frameon=False, fontsize=4.2, loc="upper left", handlelength=1.4, ncol=1, borderaxespad=0.2)   # R- panel: empty top left
    return axs


def hm_psth(fig, spec):
    D = pd.read_parquet(EA / "168_cd_psth_projection.parquet")
    D = D[(D.area == PSTH_AREA) & H.mouse_of(D).isin(H.learners())].rename(columns={"reward_group": "cohort"})
    axs = psth_panels(fig, spec, D, [("hit", "0.1", "-", "active hit"), ("miss", "0.6", "-", "active miss"),
                                     ("pre_W", EV["W"], (0, (2, 1.2)), "passive pre"), ("post_W", EV["W"], "-", "passive post")],
                      (0.005, 0.035), "time from stimulus (s)", (-0.1, 0.2))
    for ax in axs[:1] if COMPACT else axs:
        ax.set_ylabel(("projection" + (f"\n({PSTH_AREA})" if PSTH_AREA != "All units" else "")) if COMPACT else "projection on\nhit − miss axis", fontsize=5)
    if not COMPACT:
        axs[0].text(1.0, 1.25, "whole brain" if PSTH_AREA == "All units" else PSTH_AREA, transform=axs[0].transAxes, ha="right", fontsize=5, color="0.4")


def pl_psth(fig, spec):
    D = pd.read_parquet(PL_PSTH)
    D = D[(D.stage == "learning") & D.mouse_id.isin(H.learners())]
    axs = psth_panels(fig, spec, D, [("AH", EV["AH"], "-", "auditory hit"), ("SL", EV["SL"], "-", "spont. lick"), ("WH", EV["WH"], "-", "whisker hit")],
                      (-0.1, 0.0), "time from first lick (s)", (-0.4, 0.2))
    for ax in axs[:1] if COMPACT else axs:
        ax.set_ylabel("projection" if COMPACT else "projection on\nreward-lick CD", fontsize=5)


def hm_row(fig, gs, r, A, Ex, Sh, s_, R, rng, AC, with_d=True, o=0):
    W = A[A.area == "All units"].dropna(subset=[f"shift_null_post_{s_}_mean"])
    sub = gs[r, 1 + o].subgridspec(1, 2, wspace=0.4 if COMPACT else 0.15)
    example_hm([fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])], Ex, Sh, s_)
    dd = {}
    for c in COH:
        d = W[W.cohort == c]
        dd[c] = (np.c_[(d[f"shift_real_pre_{s_}"] - d[f"shift_null_pre_{s_}_mean"]).to_numpy(),
                       (d[f"shift_real_post_{s_}"] - d[f"shift_null_post_{s_}_mean"]).to_numpy()], None)
    lab = "passive whisker" if s_ == "W" else "passive auditory"
    ax = fig.add_subplot(gs[r, 2 + o])
    change_panel(ax, [(lab, EV[s_], dd)], "projection: real − shifted axes\n(per neuron)", ["pre", "post"])
    ax.set_title("passive pre and post,\nreal − shifted axes" if COMPACT else "passive pre and post,\nreal axis − mean of shifted axes",
                 fontsize=5.6, pad=12)
    if with_d:
        cohort_lines(fig.add_subplot(gs[r, 3 + o]), W, ["q_W", "q_A"], ["whisker", "auditory"], [EV["W"], EV["A"]], Q1, R, "hit-miss", rng)
    area_panel(fig.add_subplot(gs[r, (4 if COMPACT else 5) + o]), A[A.area != "All units"], f"q_{s_}", R, "hit-miss" if s_ == "W" else "hit-miss auditory",
               f"areas: {lab}", rng, AC, Q1.replace("\n", " "))


def main():
    H.setup()
    rng = np.random.default_rng(0)
    AC = area_colors()
    R = []
    A, ExH = load_hm(); S, B, ExP = load_pl(); ShH = pd.read_csv(EA / "166_example_shifts.csv")
    S["q_WH"], S["q_SL"], S["q_D"] = S.excess_pu, S.excessSL_pu, S.excessC_pu
    from matplotlib.lines import Line2D
    leg = ([Line2D([], [], color=COL["R+"], marker="o", ms=3), Line2D([], [], color=COL["R-"], marker="o", ms=3),
            Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.6", ls=(0, (2.5, 1.5)))],
           ["R+", "R−", "real axis", "shifted-label axes (mean over shifts; grey)"])
    # ------------------------------------------------ main figure
    o = 1 if PSTH and not COMPACT else 0
    if COMPACT:
        plt.rcParams.update({"font.size": 5.5, "axes.labelsize": 5.5, "xtick.labelsize": 5, "ytick.labelsize": 5, "axes.linewidth": 0.6,
                             "xtick.major.size": 2, "ytick.major.size": 2})
        fig = plt.figure(figsize=(7.1, 4.8))
        gs = fig.add_gridspec(2, 5, width_ratios=[0.95, 1.4, 0.62, 0.68, 0.78], wspace=0.6, hspace=0.62, left=0.075, right=0.99, top=0.87, bottom=0.08)
        for r, t in enumerate(("hit − miss axis", "reward-lick axis")):
            sc = gs[r, 0].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.55)
            frame(fig, sc[0, 0], "a  schematic")
            (hm_psth if r == 0 else pl_psth)(fig, sc[1, 0])
            fig.text(0.004, 0.68 - r * 0.45, t, fontsize=7, weight="bold", rotation=90, va="center")
    else:
        fig = plt.figure(figsize=(12.4 + 1.6 * o, 5.2))
        wr = [0.9] + ([1.0] if PSTH else []) + [1.45, 0.85, 0.95, 0.25, 0.9]
        gs = fig.add_gridspec(2, 6 + o, width_ratios=wr, wspace=0.6, hspace=0.85, left=0.03, right=0.93, top=0.88, bottom=0.09)
        for r, t in enumerate(("hit − miss axis (stimulus onset, 5–35 ms; passive whisker responses)", "reward-lick axis (100 ms before the first lick; whisker hits)")):
            frame(fig, gs[r, 0], "a  schematic"); fig.text(0.005, 0.94 - r * 0.47, t, fontsize=7.5, weight="bold")
    hm_row(fig, gs, 0, A, ExH, ShH, "W", R, rng, AC, o=o)
    if PSTH and not COMPACT:
        hm_psth(fig, gs[0, 1]); pl_psth(fig, gs[1, 1])
    sub = gs[1, 1 + o].subgridspec(1, 2, wspace=0.4 if COMPACT else 0.25)
    example_pl([fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])], ExP)
    Sw = S[S.area == "whole_brain"]
    items = []
    for lab, col_ev, f in (("WH" if COMPACT else "whisker hits", EV["WH"], lambda b: b["WH"]),
                           ("WH − SL" if COMPACT else "WH − SL distance", EV["D"], lambda b: b["WH"] - b["SL"])):
        dd = {}
        for c in COH:
            arr = {}
            for kind in ("real", "null"):
                piv = {cl: B[(B.cls == cl) & (B.cohort == c)].pivot_table(index="session_id", columns="bin", values=kind) for cl in ("WH", "SL")}
                idx = piv["WH"].index.intersection(piv["SL"].index)
                v = f({cl: piv[cl].loc[idx].to_numpy() for cl in piv})
                arr[kind] = v - v[:, :1]
            dd[c] = (arr["real"], arr["null"])
        items.append((lab, col_ev, dd))
    ax = fig.add_subplot(gs[1, 2 + o])
    change_panel(ax, items, "projection on reward-lick CD\n(per neuron, relative to first fifth)", ["1", "2", "3", "4", "5"])
    ax.set_xlabel("fifth of the session", fontsize=5)
    ax.set_title("across the session: real (solid)\nvs shifted CDs (grey dashed)" if COMPACT else
                 "across the session: real CD (solid) vs\nshifted-label CDs (grey dashed)", fontsize=5.6, pad=12)
    cohort_lines(fig.add_subplot(gs[1, 3 + o]), Sw, ["q_WH", "q_SL"], ["whisker hits", "spont. licks"], [EV["WH"], EV["SL"]], Q2, R, "reward-lick", rng)
    area_panel(fig.add_subplot(gs[1, (4 if COMPACT else 5) + o]), S[S.area != "whole_brain"], "q_D", R, "reward-lick",
               "areas: distance whisker hit − spontaneous lick", rng, AC, Q2.replace("\n", " ") + " (WH − SL)")
    fig.legend(*leg, loc="upper right", ncol=4, frameon=False, fontsize=5 if COMPACT else 6)
    name = ("165_two_axes_compact" + PSFX[5:]) if COMPACT else f"165_two_axes_figure{PSFX if PSTH else ''}"
    for ext in ("svg", "pdf", "png"):
        savefig(fig, FIGDIR / f"{name}.{ext}")
    if PSTH:                                                       # the variants change only the main figure
        pd.DataFrame(R).to_csv(EA / f"{name}_stats.csv", index=False)
        print("ALL DONE", FIGDIR / f"{name}.png"); return
    # ------------------------------------------------ auditory (supplementary)
    fig = plt.figure(figsize=(12.4, 2.7))
    gs = fig.add_gridspec(1, 6, width_ratios=[0.9, 1.45, 0.85, 0.95, 0.25, 0.9], wspace=0.6, left=0.03, right=0.93, top=0.78, bottom=0.17)
    frame(fig, gs[0, 0], "")
    fig.text(0.005, 0.93, "hit − miss axis: passive auditory responses (control)", fontsize=7.5, weight="bold")
    hm_row(fig, gs, 0, A, ExH, ShH, "A", R, rng, AC, with_d=False)
    fig.legend(*leg, loc="upper right", ncol=4, frameon=False, fontsize=6)
    for ext in ("svg", "pdf", "png"):
        savefig(fig, FIGDIR / f"165_two_axes_auditory.{ext}")
    pd.DataFrame(R).to_csv(EA / "165_stats.csv", index=False)
    print(pd.DataFrame(R).round(4).to_string())
    print("ALL DONE", FIGDIR / "165_two_axes_figure.png")


if __name__ == "__main__":
    main()
