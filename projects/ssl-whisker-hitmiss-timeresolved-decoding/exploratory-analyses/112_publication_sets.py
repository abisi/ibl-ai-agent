"""112 -- Publication figure sets for the decoding analyses (user request 2026-09-28): two sets per analysis.

SET 1, session-wide decoding (condition "whole"): WHEN and WHERE decoding occurs, compared across cohorts.
LEARNING STAGE ONLY (user 2026-09-28: no expert results in these sets).
  <analysis>_set1_wholebrain:  a schematic | b whole-brain time course R+ / R- (cluster bars vs group mouse-block null)
                               | c R+ minus R- (two-sided cluster, mouse-level cohort-label permutation)
                               | d window values (vs 0; R+ vs R- MW | Welch) | e real vs per-session null

  <analysis>_set1_areas:       a per-area time courses (accuracy - area-specific session null), R+ / R-, cluster bars
                               | b,c area x time heatmaps R+ / R- | d onset latency (50% of peak, mouse-bootstrap 95% CI;
                               * = R+ vs R- latency difference, bootstrap p < 0.05) | e R+ minus R- latency per area
                               | f main-window value per area, learning (filled > null FDR; * R+ vs R- MW & Welch FDR)
                               | g pooled-cohort area ranking (main window; * > null FDR) | h sessions per area
  "Area-specific null": every area/session has its own linear-shift (or label-shuffle) null curve; values are
  accuracy minus that null.
SET 2, within-session changes (conditions "half" first vs second, "perfstate" low vs high; latency not a focus):
  <analysis>_set2_within_session:
     a schematic of the splits | b,c whole brain, 1st vs 2nd half, R+ and R- (bars: paired difference clusters,
     mouse sign-flip null, two-sided) | d,e whole brain, low vs high performance state, R+ and R- | f window change
     (2nd - 1st, high - low) per cohort (* vs 0: Wilcoxon and t p < 0.05; brackets R+ vs R- MW | Welch)
     | g,h area x time heatmaps of the half change (pale: outside significant paired clusters) | i cross-condition
     generalisation (cross minus within accuracy, main window) | j,k per-area half and state changes (filled:
     change != 0, Wilcoxon FDR; * R+ vs R- MW & Welch FDR)
Each condition is decoded separately (own C, own null); changes use accuracy minus the condition's own null.
Unit = session. The performance-state DECODING analysis has no within-session conditions -> set 1 only (compact).
Reuses the panels/statistics of 110_publication_figures.py (same styling, same tests).
Outputs: figures/publication/set1_session_wide/, figures/publication/set2_within_session/ (PDF + PNG),
         112_set_stats.csv (all tests, full provenance; `set` column)
Run (haas): python 112_publication_sets.py [analysis ...]
TODO (project TODO.md): pooled decoding approach equalising trial and neuron counts across sessions/areas.
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
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("p110", OUT / "110_publication_figures.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)
COL, COHORTS = P.COL, P.COHORTS
F1 = OUT / "figures" / "publication" / "set1_session_wide"
F2 = OUT / "figures" / "publication" / "set2_within_session"
CONDS = {"half": ("first", "second", "1st half", "2nd half"), "perfstate": ("low", "high", "low state", "high state")}
SET = {"name": ""}


def rec(**k):
    P.STATS.append(dict(set=SET["name"], **k))


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def header(fig, spec, DL, extra, y=0.99):
    fig.suptitle(spec["title"], fontsize=9, fontweight="bold", x=0.02, ha="left", y=y)
    fig.text(0.02, y - 0.26 / fig.get_figheight(), f"{extra}; chance = per-session {DL['null_kind']} null"
             + ("; aligned to the corrected first lick" if spec["align"] == "lick" else "")
             + "; brackets: non-parametric | parametric p", fontsize=5.8, color="#444444")


def xlab(spec):
    return f"time from {'stimulus' if spec['align'] == 'stim' else 'first lick'} (ms)"


# ================================================================================ SET 1: session-wide
def lat_boot(sub, t, lat_range, rng):
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
    boots = np.array([lat(X[np.concatenate([np.where(subj == s)[0] for s in rng.choice(us, len(us))])]) for _ in range(P.N_BOOT)])
    return lat(X), boots


def panel_area_curves(fig, gs, D, areas, spec, colors, maps):
    t = D["t"] * 1000
    ncol = 5
    nrow = int(np.ceil(len(areas) / ncol))
    sub_gs = gs.subgridspec(nrow, ncol, hspace=0.7, wspace=0.28)
    allv = []
    for rg in COHORTS:
        for a in areas:
            s = P.whole(D, a)
            s = s[s.reward_group == rg]
            X = np.stack(s["corr"].to_numpy())
            allv.append(np.nanmean(X, 0) + np.nanstd(X, 0) / np.sqrt(len(X)))
    top = np.nanmax(allv) * 1.2
    axes = []
    for i, a in enumerate(areas):
        ax = fig.add_subplot(sub_gs[i // ncol, i % ncol])
        axes.append(ax)
        P.shade_windows(ax, spec)
        for k, rg in enumerate(COHORTS):
            s = P.whole(D, a)
            s = s[s.reward_group == rg]
            X = np.stack(s["corr"].to_numpy())
            mu, se = np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(len(X))
            ax.fill_between(t, mu - se, mu + se, color=COL[rg], alpha=0.25, lw=0)
            ax.plot(t, mu, color=COL[rg], lw=0.8)
            sig = maps[rg][1][i]
            ys = top * (0.97 - 0.07 * k)
            ax.plot(np.where(sig, t, np.nan), np.full(len(t), ys), color=COL[rg], lw=1.6, solid_capstyle="butt")
        ax.axhline(0, color="#999999", lw=0.4, ls="--")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_ylim(-0.3 * top / 1.2 if top > 0 else -0.05, top)
        ax.set_title(a, fontsize=5.8, color=colors.get(a, "#333333"), pad=2)
        ax.tick_params(labelsize=5)
        if i % ncol:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("acc. − null", fontsize=5.5)
        if i < len(areas) - ncol:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(xlab(spec), fontsize=5.5)
    return axes


def panel_latency_diff(ax, D, areas, spec, rng, analysis, jax):
    """R+ minus R- latency per area (bootstrap CI); marks significant areas on the latency panel too."""
    t = D["t"]
    for i, a in enumerate(areas):
        est, boots = {}, {}
        for rg in COHORTS:
            s = P.whole(D, a)
            est[rg], boots[rg] = lat_boot(s[s.reward_group == rg], t, spec["lat_range"], rng)
        dif = (boots["R+"] - boots["R-"]) * 1000
        dif = dif[np.isfinite(dif)]
        e = (est["R+"] - est["R-"]) * 1000
        if len(dif) < 50 or not np.isfinite(e):
            continue
        lo, hi = np.percentile(dif, [2.5, 97.5])
        p = float(min(1.0, 2 * min((dif <= 0).mean(), (dif >= 0).mean())))
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=a, window=f"latency {spec['lat_range']}",
            condition="whole", cohort="R+ - R-", test="latency difference, mouse bootstrap", n=int(P.whole(D, a).session_id.nunique()),
            latency_diff_ms=e, ci_lo_ms=lo, ci_hi_ms=hi, p=p)
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], color="k" if p < 0.05 else "#999999", marker="o", ms=2.6, capsize=1.5, lw=0.7)
        if p < 0.05:
            jax.text(1.02, i, "*", transform=jax.get_yaxis_transform(), fontsize=8, va="center")
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_yticks(range(len(areas)), [])
    ax.set_xlabel("R+ − R− latency (ms)")
    ax.grid(axis="y", color="#eeeeee", lw=0.5)


def panel_pooled_rank(ax, D, areas, spec, colors, analysis):
    w = spec["main"]
    d = P.whole(D)
    rows = []
    for a in areas:
        v = d[d.area_value == a][f"corr_{w}"].dropna()
        pw, pt = P.p1(v)
        rows.append(dict(area=a, mean=v.mean(), sem=v.std() / np.sqrt(len(v)), n=len(v), pw=pw, pt=pt))
    R = pd.DataFrame(rows)
    R["qw"] = multipletests(R.pw.fillna(1), method="fdr_bh")[1]
    R["qt"] = multipletests(R.pt.fillna(1), method="fdr_bh")[1]
    R = R.sort_values("mean", ascending=False).reset_index(drop=True)
    for i, r in R.iterrows():
        ax.bar(i, r["mean"], yerr=r["sem"], color=colors.get(r.area, "#888"), width=0.7, lw=0, error_kw=dict(lw=0.6, capsize=1.2))
        if r.qw < 0.05 and r.qt < 0.05:
            ax.text(i, r["mean"] + r["sem"] + 0.004, "*", ha="center", fontsize=8)
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=r.area, window=w, condition="whole",
            cohort="R+ & R- pooled", test="accuracy - null vs 0 (Wilcoxon | one-sample t), BH-FDR across areas", n=int(r.n),
            mean=r["mean"], p_nonparam=r.pw, p_param=r.pt, q_nonparam=r.qw, q_param=r.qt)
    ax.set_xticks(range(len(R)))
    ax.set_xticklabels(R.area, rotation=55, ha="right", fontsize=5.6)
    for tl, a in zip(ax.get_xticklabels(), R.area):
        tl.set_color(colors.get(a, "#333"))
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_ylabel(f"accuracy − null ({w})")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.12)


def set1(key, spec, DL, DE, AL, AE, rng, rtc):
    SET["name"] = "set1_session_wide"
    # --- whole brain
    fig = plt.figure(figsize=(7.2, 5.4))
    header(fig, spec, DL, "Set 1, session-wide decoding: whole brain, learning stage")
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.86, bottom=0.58, wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 2, left=0.08, right=0.8, top=0.42, bottom=0.1, wspace=0.45, width_ratios=[1.3, 1])
    a = fig.add_subplot(r0[0]); P.panel_schematic(a, spec, DL, None); P.letter(a, "a", dx=-0.02, dy=1.0)
    b = fig.add_subplot(r0[1]); P.panel_timecourse(b, DL, spec, rng, "learning", rtc, key); P.letter(b, "b", dx=-0.26)
    b.set_title("whole brain, learning", fontsize=6.5)
    c = fig.add_subplot(r0[2]); P.panel_diff(c, DL, spec, rng, key); P.letter(c, "c", dx=-0.26)
    c.set_title("R+ minus R− (accuracy − null)", fontsize=6.5)
    d = fig.add_subplot(r1[0]); P.panel_windows(d, DL, spec, key); P.letter(d, "d", dx=-0.2)
    d.set_title("windows, learning\nR+ vs R− (MW | Welch)", fontsize=5.8)
    e = fig.add_subplot(r1[1]); P.panel_real_vs_null(e, DL, spec, key); P.letter(e, "e", dx=-0.35)
    e.set_title("real vs per-session null", fontsize=5.8)
    save(fig, F1, f"112_{key}_set1_wholebrain")
    if AL is None:
        return
    # --- areas
    areas = P.area_list(AL)
    colors = P.m034.get_area_color_map(areas)
    maps = P.area_maps(AL, areas, spec, rng, key, "learning")
    fig = plt.figure(figsize=(7.2, 10.0))
    header(fig, spec, DL, "Set 1, per area group, learning (accuracy − area-specific null)",
           y=0.992)
    ra = fig.add_gridspec(1, 1, left=0.07, right=0.98, top=0.925, bottom=0.66)
    rb = fig.add_gridspec(1, 3, left=0.2, right=0.66, top=0.59, bottom=0.39, width_ratios=[1, 1, 0.05], wspace=0.1)
    rd = fig.add_gridspec(1, 2, left=0.72, right=0.98, top=0.59, bottom=0.39, wspace=0.12)
    rf = fig.add_gridspec(1, 2, left=0.075, right=0.75, top=0.3, bottom=0.12, wspace=0.28)
    ri = fig.add_gridspec(1, 1, left=0.89, right=0.98, top=0.3, bottom=0.12)
    axs = panel_area_curves(fig, ra[0], AL, areas, spec, colors, maps)
    P.letter(axs[0], "a", dx=-0.45, dy=1.15)
    fig.text(0.07, 0.94, "per-area time courses, learning (mean ± SEM; bars: significant clusters vs group mouse-block null)",
             fontsize=6.2)
    vmax = max(0.05, np.nanpercentile(np.concatenate([maps[rg][0].ravel() for rg in COHORTS]), 99))
    norm = TwoSlopeNorm(vmin=-vmax / 3, vcenter=0, vmax=vmax)
    h = fig.add_subplot(rb[0]); im = P.panel_heatmap(h, maps, "R+", areas, AL, spec, colors, True, norm); P.letter(h, "b", dx=-0.62)
    i_ = fig.add_subplot(rb[1]); P.panel_heatmap(i_, maps, "R-", areas, AL, spec, colors, False, norm); P.letter(i_, "c", dx=-0.06)
    cb = fig.colorbar(im, cax=fig.add_subplot(rb[2]))
    cb.ax.tick_params(labelsize=5.5)
    j = fig.add_subplot(rd[0]); P.panel_latency(j, maps, areas, spec); P.letter(j, "d", dx=-0.12)
    j.set_title("onset latency", fontsize=6.3)
    j.set_xlabel("latency (ms)\n50% of peak", fontsize=5.8)
    o = fig.add_subplot(rd[1]); panel_latency_diff(o, AL, areas, spec, rng, key, j); P.letter(o, "e", dx=-0.12)
    o.set_title("R+ − R− latency", fontsize=6.3)
    k = fig.add_subplot(rf[0]); P.panel_area_window(k, AL, areas, spec, colors, key, "learning"); P.letter(k, "f", dx=-0.16)
    k.set_title(f"per area, learning ({spec['main']})", fontsize=6.3)
    g = fig.add_subplot(rf[1]); panel_pooled_rank(g, AL, areas, spec, colors, key); P.letter(g, "g", dx=-0.16)
    g.set_title(f"area ranking, cohorts pooled ({spec['main']})", fontsize=6.3)
    m = fig.add_subplot(ri[0]); P.panel_counts(m, AL, None, areas); P.letter(m, "h", dx=-1.9)
    m.set_title("sessions", fontsize=6.3)
    m.set_xlabel("n")
    save(fig, F1, f"112_{key}_set1_areas")


# ================================================================================ SET 2: within session
def paired(D, ctype, rg, area=None, col="corr"):
    va, vb = CONDS[ctype][:2]
    d = D["df"][(D["df"].condition_type == ctype) & (D["df"].reward_group == rg)]
    if area is not None:
        d = d[d.area_value == area]
    A = d[d.condition_value == va].drop_duplicates("session_id").set_index("session_id")
    B = d[d.condition_value == vb].drop_duplicates("session_id").set_index("session_id")
    s = A.index.intersection(B.index)
    return A.loc[s], B.loc[s]


def signflip_clusters(Dm, mice, t, rng):
    obs = np.nanmean(Dm, 0)
    um = np.unique(mice)
    idx = np.searchsorted(um, mice)
    null = np.array([np.nanmean(Dm * rng.choice([-1.0, 1.0], len(um))[idx][:, None], 0) for _ in range(P.N_PERM)])
    return obs, P.clusters_from(obs, null, t, two_sided=True)


def panel_cond_timecourse(ax, D, ctype, rg, spec, rng, analysis, stage="learning"):
    t = D["t"] * 1000
    A, B = paired(D, ctype, rg)
    lab_a, lab_b = CONDS[ctype][2:]
    P.shade_windows(ax, spec)
    if len(A) < 3:
        ax.text(0.5, 0.5, "too few sessions", ha="center", va="center", transform=ax.transAxes, color="#777777")
        ax.set_axis_off()
        return
    for S_, c, lab, ls in ((A, P.m034.lighten(COL[rg], 0.5), lab_a, "-"), (B, COL[rg], lab_b, "-")):
        X = np.stack(S_.real.to_numpy())
        mu, se = np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(len(X))
        ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.3, lw=0)
        ax.plot(t, mu, color=c, lw=1.0, ls=ls, label=lab)
        ax.plot(t, np.nanmean(np.stack(S_.nullmean.to_numpy()), 0), color=c, lw=0.6, ls=":")
    Dm = np.stack(B["corr"].to_numpy()) - np.stack(A["corr"].to_numpy())
    obs, cl = signflip_clusters(Dm, A.subject_id.to_numpy(), D["t"], rng)
    for c in cl:
        rec(analysis=analysis, tag=D["tag"], stage=stage, level="whole_brain", area="All units", window="time-resolved",
            condition=f"{ctype}: {CONDS[ctype][1]} - {CONDS[ctype][0]}", cohort=rg,
            test="paired difference cluster (accuracy - null), mouse sign-flip, two-sided", n=len(A), n_mice=A.subject_id.nunique(),
            t_start_ms=c["t0"] * 1000, t_end_ms=c["t1"] * 1000, cluster_mass=c["mass"], sign=c["sign"], p=c["p"])
        if c["p"] < 0.05:
            ax.plot([c["t0"] * 1000, c["t1"] * 1000], [1.035] * 2, color="k" if c["sign"] > 0 else "#888888", lw=2.2,
                    solid_capstyle="butt")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0.5, color="#999999", lw=0.5, ls="--")
    ax.set_ylim(0.4, 1.06)
    ax.set_xlim(t[0] - 50, t[-1])
    ax.set_xlabel(xlab(spec))
    ax.set_ylabel("balanced accuracy")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95), handlelength=1.2, fontsize=5.4, title=f"{rg} (n={len(A)})",
              title_fontsize=5.6)


def panel_delta_windows(ax, D, spec, analysis, stage="learning"):
    w = spec["main"]
    brk = []
    ticks, labs = [], []
    for i, ctype in enumerate(CONDS):
        vals = {}
        for k, rg in enumerate(COHORTS):
            A, B = paired(D, ctype, rg)
            v = (B[f"corr_{w}"] - A[f"corr_{w}"]).to_numpy() if len(A) else np.array([])
            vals[rg] = v
            x = i * 2.6 + k
            ticks.append(x)
            labs.append(rg)
            if len(v) < 3:
                continue
            P.strip(ax, x, v, COL[rg])
            pw, pt = P.p1(v)
            rec(analysis=analysis, tag=D["tag"], stage=stage, level="whole_brain", area="All units", window=w,
                condition=f"{ctype}: {CONDS[ctype][1]} - {CONDS[ctype][0]}", cohort=rg, test="paired change vs 0 (Wilcoxon | one-sample t)",
                n=len(v), n_mice=A.subject_id.nunique(), mean=np.nanmean(v), p_nonparam=pw, p_param=pt)
            if pw < 0.05 and pt < 0.05:
                ax.text(x + 0.3, np.nanmax(v), "*", ha="center", fontsize=8, color=COL[rg])
        if min(len(vals["R+"]), len(vals["R-"])) >= 3:
            pm, pw_ = P.p2(vals["R+"], vals["R-"])
            rec(analysis=analysis, tag=D["tag"], stage=stage, level="whole_brain", area="All units", window=w,
                condition=f"{ctype}: {CONDS[ctype][1]} - {CONDS[ctype][0]}", cohort="R+ vs R-", test="Mann-Whitney | Welch on change",
                n=len(vals["R+"]) + len(vals["R-"]), mean_a=np.nanmean(vals["R+"]), mean_b=np.nanmean(vals["R-"]), p_nonparam=pm, p_param=pw_)
            brk.append((i * 2.6 + 0.15, i * 2.6 + 1.15, f"{P.fmt_s(pm)} | {P.fmt_s(pw_)}"))
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(ticks, labs)
    for i, ctype in enumerate(CONDS):
        ax.text(i * 2.6 + 0.5, -0.2, f"{CONDS[ctype][3]} − {CONDS[ctype][2]}", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.8)
    ax.set_ylabel(f"change in accuracy − null ({w})")
    P.draw_brackets(ax, brk)


def delta_maps(D, areas, rng, analysis, ctype="half"):
    out = {}
    for rg in COHORTS:
        M, S = [], []
        for a in areas:
            A, B = paired(D, ctype, rg, a)
            if len(A) < 3:
                M.append(np.full(len(D["t"]), np.nan))
                S.append(np.zeros(len(D["t"]), bool))
                continue
            Dm = np.stack(B["corr"].to_numpy()) - np.stack(A["corr"].to_numpy())
            obs, cl = signflip_clusters(Dm, A.subject_id.to_numpy(), D["t"], rng)
            sig = np.zeros(len(D["t"]), bool)
            for c in cl:
                rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=a, window="time-resolved",
                    condition=f"{ctype}: {CONDS[ctype][1]} - {CONDS[ctype][0]}", cohort=rg,
                    test="paired difference cluster, mouse sign-flip, two-sided", n=len(A), n_mice=A.subject_id.nunique(),
                    t_start_ms=c["t0"] * 1000, t_end_ms=c["t1"] * 1000, cluster_mass=c["mass"], sign=c["sign"], p=c["p"])
                if c["p"] < 0.05:
                    sig[c["i0"]:c["i1"] + 1] = True
            M.append(obs)
            S.append(sig)
        out[rg] = (np.array(M), np.array(S), None)
    return out


def panel_area_delta(ax, D, areas, ctype, spec, colors, analysis):
    w = spec["main"]
    rows = []
    for a in areas:
        for rg in COHORTS:
            A, B = paired(D, ctype, rg, a)
            v = (B[f"corr_{w}"] - A[f"corr_{w}"]).dropna() if len(A) else pd.Series(dtype=float)
            pw, pt = P.p1(v)
            rows.append(dict(area=a, rg=rg, v=v, mean=v.mean() if len(v) else np.nan,
                             sem=v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan, n=len(v), pw=pw, pt=pt))
    R = pd.DataFrame(rows)
    for rg in COHORTS:
        m = R.rg == rg
        R.loc[m, "qw"] = multipletests(R.loc[m, "pw"].fillna(1), method="fdr_bh")[1]
        R.loc[m, "qt"] = multipletests(R.loc[m, "pt"].fillna(1), method="fdr_bh")[1]
    comp = []
    for a in areas:
        va, vb = R[(R.area == a) & (R.rg == "R+")].v.iloc[0], R[(R.area == a) & (R.rg == "R-")].v.iloc[0]
        pm, pw_ = P.p2(va, vb)
        comp.append(dict(area=a, pm=pm, pwe=pw_))
    Cc = pd.DataFrame(comp)
    Cc["qm"] = multipletests(Cc.pm.fillna(1), method="fdr_bh")[1]
    Cc["qwe"] = multipletests(Cc.pwe.fillna(1), method="fdr_bh")[1]
    cond = f"{ctype}: {CONDS[ctype][1]} - {CONDS[ctype][0]}"
    for i, a in enumerate(areas):
        for k, rg in enumerate(COHORTS):
            r = R[(R.area == a) & (R.rg == rg)].iloc[0]
            if r["n"] < 3:
                continue
            ax.errorbar(i + (-0.17 if k == 0 else 0.17), r["mean"], yerr=r["sem"], fmt="o", ms=3.0,
                        mfc=colors.get(a, "#888") if r.qw < 0.05 else "white", mec=COL[rg], mew=0.9, ecolor=COL[rg], elinewidth=0.7)
            rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=a, window=w, condition=cond, cohort=rg,
                test="paired change vs 0 (Wilcoxon | one-sample t), BH-FDR across areas", n=int(r["n"]), mean=r["mean"],
                p_nonparam=r.pw, p_param=r.pt, q_nonparam=r.qw, q_param=r.qt)
        c = Cc[Cc.area == a].iloc[0]
        rec(analysis=analysis, tag=D["tag"], stage="learning", level="area_group", area=a, window=w, condition=cond, cohort="R+ vs R-",
            test="Mann-Whitney | Welch on change, BH-FDR across areas", n=int(R[R.area == a]["n"].sum()), p_nonparam=c.pm, p_param=c.pwe,
            q_nonparam=c.qm, q_param=c.qwe)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=55, ha="right", fontsize=5.5)
    for tl, a in zip(ax.get_xticklabels(), areas):
        tl.set_color(colors.get(a, "#333"))
    y0, y1 = ax.get_ylim()
    top = y1 + 0.01
    for i, a in enumerate(areas):
        c = Cc[Cc.area == a].iloc[0]
        if c.qm < 0.05 and c.qwe < 0.05:
            ax.text(i, top, "*", ha="center", fontsize=9)
    ax.set_ylim(y0 - 0.01, top + 0.04)
    ax.set_xlim(-0.6, len(areas) - 0.4)
    ax.set_ylabel(f"change ({w})")


def panel_crossgen(ax, D, spec, analysis):
    w = spec["main"]
    m = (D["t"] >= spec["windows"][w][0]) & (D["t"] <= spec["windows"][w][1])
    brk, ticks, labs = [], [], []
    for i, ctype in enumerate(CONDS):
        vals = {}
        for k, rg in enumerate(COHORTS):
            A, B = paired(D, ctype, rg)
            v = []
            for s in A.index:
                xa, xb = A.loc[s, "xgen"], B.loc[s, "xgen"]
                if xa is None or xb is None:
                    continue
                cross = np.nanmean([np.nanmean(xa[m]), np.nanmean(xb[m])])
                within = np.nanmean([A.loc[s, f"real_{w}"], B.loc[s, f"real_{w}"]])
                v.append(cross - within)
            v = np.array(v)
            vals[rg] = v
            x = i * 2.6 + k
            ticks.append(x)
            labs.append(rg)
            if len(v) < 3:
                continue
            P.strip(ax, x, v, COL[rg])
            pw, pt = P.p1(v)
            rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition=f"{ctype} cross-generalisation",
                cohort=rg, test="cross minus within accuracy vs 0 (Wilcoxon | one-sample t)", n=len(v), mean=v.mean(), p_nonparam=pw, p_param=pt)
            if pw < 0.05 and pt < 0.05:
                ax.text(x + 0.3, np.nanmin(v) - 0.02, "*", ha="center", fontsize=8, color=COL[rg])
        if min(len(vals["R+"]), len(vals["R-"])) >= 3:
            pm, pw_ = P.p2(vals["R+"], vals["R-"])
            rec(analysis=analysis, tag=D["tag"], stage="learning", level="whole_brain", area="All units", window=w, condition=f"{ctype} cross-generalisation",
                cohort="R+ vs R-", test="Mann-Whitney | Welch", n=len(vals["R+"]) + len(vals["R-"]), mean_a=vals["R+"].mean(),
                mean_b=vals["R-"].mean(), p_nonparam=pm, p_param=pw_)
            brk.append((i * 2.6 + 0.15, i * 2.6 + 1.15, f"{P.fmt_s(pm)} | {P.fmt_s(pw_)}"))
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(ticks, labs)
    for i, ctype in enumerate(CONDS):
        ax.text(i * 2.6 + 0.5, -0.2, "halves" if ctype == "half" else "states", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.8)
    ax.set_ylabel(f"cross − within accuracy ({w})")
    P.draw_brackets(ax, brk)


def panel_split_schematic(ax, spec):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    rng = np.random.default_rng(3)
    n = 60
    xs = np.linspace(0.04, 0.96, n)
    p = 1 / (1 + np.exp(-(np.arange(n) - 25) / 5))
    lick = rng.random(n) < p * 0.85 + 0.05
    ax.plot(xs, 0.62 + 0.22 * p, color="#555555", lw=1.0)
    ax.scatter(xs, np.where(lick, 0.9, 0.56), s=3, color="k", marker="|", lw=0.6)
    ax.text(0.0, 0.95, "whisker trials (ticks: lick / no lick), learning curve", fontsize=5.0, va="bottom")
    ax.add_patch(Rectangle((0.04, 0.42), 0.46, 0.06, color=P.m034.lighten("#555555", 0.5), lw=0))
    ax.add_patch(Rectangle((0.5, 0.42), 0.46, 0.06, color="#555555", lw=0))
    ax.text(0.27, 0.45, "1st half", ha="center", va="center", fontsize=5.2, color="k")
    ax.text(0.73, 0.45, "2nd half", ha="center", va="center", fontsize=5.2, color="white")
    blocks = (p > 0.5)
    for i in range(n):
        ax.add_patch(Rectangle((xs[i] - 0.008, 0.33), 0.0155, 0.05, color="#555555" if blocks[i] else P.m034.lighten("#555555", 0.5), lw=0))
    ax.text(0.5, 0.29, "performance state: low | high (learning-curve based)", ha="center", va="top", fontsize=5.0)
    txt = ["each condition decoded separately (own C, own null);",
           "change = (accuracy − null) in 2nd − 1st, high − low;",
           "cross-generalisation: train on one condition,",
           "  test on the other, minus within-condition accuracy"]
    ax.text(0.0, 0.2, "\n".join(txt), fontsize=5.0, va="top", linespacing=1.35)


def set2(key, spec, DL, DE, AL, rng):
    SET["name"] = "set2_within_session"
    fig = plt.figure(figsize=(7.2, 11.0))
    header(fig, spec, DL, "Set 2, within-session changes (learning stage): session halves and performance states")
    r0 = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.925, bottom=0.775, wspace=0.42, width_ratios=[1.2, 1, 1])
    r1 = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.69, bottom=0.54, wspace=0.42)
    r2a = fig.add_gridspec(1, 3, left=0.2, right=0.705, top=0.455, bottom=0.285, width_ratios=[1, 1, 0.05], wspace=0.1)
    r2b = fig.add_gridspec(1, 1, left=0.845, right=0.98, top=0.455, bottom=0.285)
    r3 = fig.add_gridspec(1, 2, left=0.075, right=0.98, top=0.205, bottom=0.085, wspace=0.2)
    a = fig.add_subplot(r0[0]); panel_split_schematic(a, spec); P.letter(a, "a", dx=-0.02, dy=1.0)
    for j, rg in enumerate(COHORTS):
        ax = fig.add_subplot(r0[1 + j]); panel_cond_timecourse(ax, DL, "half", rg, spec, rng, key); P.letter(ax, "bc"[j], dx=-0.26)
        ax.set_title(f"whole brain, {rg}: 1st vs 2nd half", fontsize=6.3)
        ax2 = fig.add_subplot(r1[j]); panel_cond_timecourse(ax2, DL, "perfstate", rg, spec, rng, key); P.letter(ax2, "de"[j], dx=-0.26)
        ax2.set_title(f"whole brain, {rg}: low vs high state", fontsize=6.3)
    f = fig.add_subplot(r1[2]); panel_delta_windows(f, DL, spec, key); P.letter(f, "f", dx=-0.3)
    f.set_title("window change, learning\nR+ vs R− (MW | Welch)", fontsize=5.8)
    if AL is not None:
        areas = P.area_list(AL)
        colors = P.m034.get_area_color_map(areas)
        maps = delta_maps(AL, areas, rng, key)
        vmax = max(0.03, np.nanpercentile(np.abs(np.concatenate([maps[rg][0].ravel() for rg in COHORTS])), 99))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        g = fig.add_subplot(r2a[0]); im = P.panel_heatmap(g, maps, "R+", areas, AL, spec, colors, True, norm); P.letter(g, "g", dx=-0.62)
        h = fig.add_subplot(r2a[1]); P.panel_heatmap(h, maps, "R-", areas, AL, spec, colors, False, norm); P.letter(h, "h", dx=-0.06)
        cb = fig.colorbar(im, cax=fig.add_subplot(r2a[2]))
        cb.ax.tick_params(labelsize=5.5)
        fig.text(0.2, 0.488, "area group: change 2nd − 1st half in accuracy − null (pale: outside significant paired clusters)", fontsize=6.2)
        j_ = fig.add_subplot(r3[0]); panel_area_delta(j_, AL, areas, "half", spec, colors, key); P.letter(j_, "j", dx=-0.2)
        j_.set_title(f"per area: 2nd − 1st half ({spec['main']})", fontsize=6.2)
        k = fig.add_subplot(r3[1]); panel_area_delta(k, AL, areas, "perfstate", spec, colors, key); P.letter(k, "k", dx=-0.2)
        k.set_title(f"per area: high − low state ({spec['main']})", fontsize=6.2)
    i_ = fig.add_subplot(r2b[0]); panel_crossgen(i_, DL, spec, key); P.letter(i_, "i", dx=-0.35)
    i_.set_title("cross-condition generalisation\nR+ vs R− (MW | Welch)", fontsize=5.8)
    save(fig, F2, f"112_{key}_set2_within_session")


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    P.ROOT = resolve_dataset_dir("ssl_ephys")
    P.ST, P.TT = pd.read_parquet(P.ROOT / "metadata" / "sessions.parquet"), pd.read_parquet(P.ROOT / "metadata" / "trials.parquet")
    P.style()
    keys = sys.argv[1:] or list(P.ANALYSES)
    rtc: dict = {}
    for key in keys:
        spec = P.ANALYSES[key]
        pre = spec["prefix"]
        rng = np.random.default_rng(P.SEED)
        # learning stage only (user 2026-09-28: "Do not show expert results in those")
        DL, AL = P.load_tag(f"{pre}_whole_brain"), P.load_tag(f"{pre}_area_group")
        DE = AE = None
        for D in (DL, AL):
            if D is not None:
                P.add_windows(D, spec["windows"])
        P.D_NULL[pre] = DL["null_kind"]
        if AL is None and DE is None and not (DL["df"].condition_type == "half").any():
            SET["name"] = "set1_session_wide"
            save(P.compact_figure(key, spec, DL, rng, rtc), F1, f"112_{key}_set1_wholebrain")
            print(f"[112] {key}: set 1 (compact) only", flush=True)
            continue
        set1(key, spec, DL, DE, AL, AE, rng, rtc)
        set2(key, spec, DL, DE, AL, rng)
        print(f"[112] {key}: sets 1 and 2 done", flush=True)
    S = pd.DataFrame(P.STATS)
    S["seed"], S["n_perm"], S["n_boot"] = P.SEED, P.N_PERM, P.N_BOOT
    S.to_csv(OUT / "112_set_stats.csv", index=False)
    print(f"[112] stats rows: {len(S)}", flush=True)


if __name__ == "__main__":
    main()
