"""Exploration (user 2026-11-09): does reaction time change with learning, trial by trial? Learners, whisker day 0.

Per session (shared behaviour table, early licks dropped; exact curves and learning trial of Figure 1, session_curves):
  a. whisker-hit reaction time (computed_reaction_time, 0-1 s) and jaw onset (behaviour_analysis dlc_jaw_onset_times.pkl,
     via explore_rt_session.jaw_table) against whisker trials relative to the learning trial (bins of 3 trials, -30..+60);
     sessions without a learning trial are left out of this view
  b. the same against hit number (1st, 2nd, ... whisker hit), with auditory hits as control
  c. within each mouse, Spearman correlation across whisker hits between reaction time and the whisker curve (p_mean at
     that trial), and the partial correlation controlling for normalised session time (satiety drift); auditory hits
     against the same curve as control. One value per mouse; Wilcoxon against 0, R+ vs R- Mann-Whitney + Welch.
Output: Figures/_explore/rt_learning_<date>.{pdf,png} and the CSVs.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_learning.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, two_group_tests  # noqa: E402
from explore_movement_corr import corr_with_null  # noqa: E402
from explore_rt_session import jaw_table  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
BIN, LO, HI, MIN_MICE = 5, -30, 60, 8


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z (ranks, residuals of linear fits)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)[0]


def main():
    from compute import day0_table, session_curves
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    d0 = d0[d0.early_lick == 0]
    rows, corr, ctx, jaws = [], [], {}, {}
    for sid, st in d0.groupby("session_id"):
        st = st.sort_values("start_time")
        mouse, cohort = st.mouse_id.iloc[0], st.cohort.iloc[0]
        curves, lt, _ = session_curves(st, ctx)
        t = st.start_time.to_numpy()
        tn = (t - t.min()) / max(t.max() - t.min(), 1e-9)
        w = st[st.trial_type == "whisker_trial"].copy()
        if len(w) != len(curves):
            continue
        w["w_index"] = np.arange(len(w))
        w["p_mean"] = curves.p_mean.to_numpy()
        w["time_norm"] = tn[st.trial_type.to_numpy() == "whisker_trial"]
        if mouse not in jaws:
            jaws[mouse] = jaw_table(mouse)
        j = jaws[mouse]
        st = st.assign(jaw_ms=np.nan)
        if j is not None and len(j[j.session_id == sid]):
            jj = j[j.session_id == sid].sort_values("start_time")
            k = np.clip(np.searchsorted(jj.start_time.to_numpy(), t), 0, len(jj) - 1)
            v = jj.jaw_dlc_onset.to_numpy(float)[k]
            ok = (np.abs(jj.start_time.to_numpy()[k] - t) < 0.005) & (v >= 0) & (v <= 1)
            st["jaw_ms"] = np.where(ok, v * 1000, np.nan)
        st["rt_ms"] = (st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000
                       ).where(st.lick_flag == 1)
        st["time_norm"] = tn
        # whisker index of every trial (most recent whisker trial) for the curve value of auditory hits
        widx = np.cumsum(st.trial_type.to_numpy() == "whisker_trial") - 1
        cw = np.clip(widx, 0, len(curves) - 1)
        CURVES = {"whisker": curves.p_mean.to_numpy(), "false_alarm": curves.p_chance.to_numpy(),
                  "difference": (curves.p_mean - curves.p_chance).to_numpy()}
        for name, v in CURVES.items():  # value at the most recent whisker trial (itself for whisker trials)
            st[f"curve_{name}"] = np.where(widx >= 0, v[cw], np.nan)
        st["p_mean"] = st["curve_whisker"]
        w = w.join(st[["rt_ms", "jaw_ms"]])
        for tt in ("whisker_trial", "auditory_trial", "no_stim_trial"):
            h = st[(st.trial_type == tt) & (st.lick_flag == 1)]
            for k, r in enumerate(h.itertuples(), start=1):
                rows.append({"mouse_id": mouse, "session_id": sid, "cohort": cohort, "trial_type": tt, "hit_number": k,
                             "rt_ms": r.rt_ms, "jaw_ms": r.jaw_ms,
                             "rel_learning": (int(widx[st.index.get_loc(r.Index)]) - lt) if np.isfinite(lt) else np.nan})
            for y in ("rt_ms", "jaw_ms"):
                for name in CURVES:  # user 2026-11-09: whisker, false-alarm and difference curves
                    cc = f"curve_{name}"
                    d = h[[cc, y, "time_norm"]].dropna()
                    if len(d) >= 10 and d[cc].std() > 0:
                        corr.append({"mouse_id": mouse, "cohort": cohort, "trial_type": tt, "latency": y, "curve": name,
                                     "rho": stats.spearmanr(d[cc], d[y])[0],
                                     "rho_partial_time": partial_spearman(d[cc].to_numpy(), d[y].to_numpy(),
                                                                           d.time_norm.to_numpy()),
                                     "rho_time": stats.spearmanr(d.time_norm, d[y])[0], "n_hits": len(d),
                                     # Spearman minus the median of linear shifts (hits in time order; user 2026-11-09)
                                     "rho_vs_shift": (lambda r: r[0] - r[1])(corr_with_null(
                                         stats.rankdata(d[cc]).astype(float), stats.rankdata(d[y]).astype(float)))})
    hits, corr = pd.DataFrame(rows), pd.DataFrame(corr)
    stem = OUT / f"rt_learning_{datetime.date.today()}"
    hits.to_csv(f"{stem}_hits.csv", index=False)
    corr.to_csv(f"{stem}_corr.csv", index=False)
    summary = []
    for (tt, y, cu), g in corr.groupby(["trial_type", "latency", "curve"]):
        for col in ("rho", "rho_vs_shift", "rho_partial_time", "rho_time"):
            row = {"trial_type": tt, "latency": y, "curve": cu, "measure": col}
            for c in ("R+", "R-"):
                v = g[g.cohort == c][col].dropna()
                row[f"{c} mean"] = round(v.mean(), 3)
                row[f"{c} wilcoxon_p"] = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
                row[f"{c} n"] = len(v)
            row["MW_p"] = two_group_tests(g[g.cohort == "R+"][col], g[g.cohort == "R-"][col])["mannwhitney_p"]
            summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{stem}_summary.csv", index=False)
    print(summary[summary.measure != "rho_time"].to_string())
    plot(hits[hits.trial_type != "no_stim_trial"], corr[(corr.curve == "whisker") & (corr.trial_type != "no_stim_trial")], stem)
    plot_curve_corr(corr, OUT / f"rt_curve_corr_{datetime.date.today()}")


def med_band(values_by_x, n_boot=1000, seed=0):
    """Median over mice per x, and the bootstrap 68% interval of that median (the median's equivalent of +- s.e.m.)."""
    rng = np.random.default_rng(seed)
    xs, med, lo, hi = [], [], [], []
    for x, v in values_by_x:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < MIN_MICE:
            continue
        boots = np.median(rng.choice(v, (n_boot, len(v))), axis=1)
        xs.append(x)
        med.append(np.median(v))
        lo.append(np.percentile(boots, 16))
        hi.append(np.percentile(boots, 84))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def plot(hits, corr, stem):
    """Top: per-mouse median latency, then median over mice +- bootstrap 68% interval (user 2026-11-09: medians);
    against whisker trials from the learning trial (bins of 3) and against hit number (each hit). Bottom: Spearman rho per
    mouse; median and the Wilcoxon p per group printed on the panel."""
    set_nature_rc()
    A = {"R+": COLORS["auditory"], "R-": "#9a9aee"}
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 105 * MM))
    w = hits[(hits.trial_type == "whisker_trial") & hits.rel_learning.between(LO, HI)].copy()
    w["bin"] = np.floor(w.rel_learning / 3) * 3 + 1.5
    for c, y in enumerate(("rt_ms", "jaw_ms")):
        a = axs[0, c]
        for co in ("R+", "R-"):
            m = w[w.cohort == co].groupby(["mouse_id", "bin"])[y].median().reset_index()
            x, md, lo, hi = med_band((b, g[y]) for b, g in m.groupby("bin"))
            a.plot(x, md, color=COLORS[co], lw=0.8, label=f"{co} (n={w[w.cohort == co].mouse_id.nunique()})")
            a.fill_between(x, lo, hi, color=COLORS[co], alpha=0.2, lw=0)
        a.axvline(0, color="0.5", lw=0.4, ls=":")
        a.set_xlabel("Whisker trials from learning trial (bins of 3)", labelpad=1)
        a.set_ylabel(("Reaction time" if y == "rt_ms" else "Jaw onset") + " on W hits (ms, median)", labelpad=1)
        a.legend(fontsize=4.5, frameon=False)
    a = axs[0, 2]
    h = hits[hits.hit_number <= 30]
    for tt, cols, ls in (("whisker_trial", COLORS, "-"), ("auditory_trial", A, "--")):
        for co in ("R+", "R-"):
            m = h[(h.trial_type == tt) & (h.cohort == co)]
            x, md, lo, hi = med_band((k, g.rt_ms) for k, g in m.groupby("hit_number"))
            a.plot(x, md, color=cols[co], ls=ls, lw=0.8, label=f"{'W' if tt == 'whisker_trial' else 'A'} {co}")
            a.fill_between(x, lo, hi, color=cols[co], alpha=0.2, lw=0)
    a.set_xlabel("Hit number", labelpad=1)
    a.set_ylabel("Reaction time (ms, median)", labelpad=1)
    a.legend(fontsize=4.5, frameon=False, ncol=2)
    groups = [(t, c2) for t in ("whisker_trial", "auditory_trial") for c2 in ("R+", "R-")]
    for c, (y, lab) in enumerate((("rt_ms", "RT"), ("jaw_ms", "Jaw onset"))):
        a = axs[1, c]
        for k, col in enumerate(("rho", "rho_partial_time", "rho_time")):
            for i, (tt, co) in enumerate(groups):
                v = corr[(corr.trial_type == tt) & (corr.cohort == co) & (corr.latency == y)][col].dropna()
                x = k * 5 + i
                color = (COLORS if tt == "whisker_trial" else A)[co]
                a.scatter(x + np.random.default_rng(i + 10 * k).uniform(-0.2, 0.2, len(v)), v, s=1.2, color="0.75", lw=0)
                _, md, lo, hi = med_band([(0, v)])
                if len(md):
                    a.errorbar(x, md[0], [[md[0] - lo[0]], [hi[0] - md[0]]], color=color, marker="o", ms=2.5, capsize=0,
                               elinewidth=0.8)
                p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                a.text(x, 1.01, f"{np.median(v):.2f} {star}", ha="center", va="bottom", fontsize=3.8, color=color,
                       transform=a.get_xaxis_transform(), rotation=90)
        a.axhline(0, color="0.5", lw=0.4)
        a.set_xticks([1.5, 6.5, 11.5], [f"{lab} vs curve", f"{lab} vs curve | time", f"{lab} vs time"], fontsize=4.5)
        a.set_ylabel("Spearman ρ per mouse", labelpad=1)
    axs[1, 2].axis("off")
    axs[1, 2].text(0, 0.5, "Learners, whisker day 0, hits only\nper-mouse median, then median over mice\n"
                   "band / bars: bootstrap 68% interval of the median\ncurve = whisker learning curve at the trial\n"
                   "(auditory hits: at the preceding whisker trial)\n| time = partial, controlling session time\n"
                   "order per group: W R+, W R-, A R+, A R-\nnumbers: median ρ; stars, Wilcoxon vs 0 (exact P: summary CSV)", fontsize=5, va="center")
    fig.tight_layout(h_pad=1.6, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


def plot_curve_corr(corr, stem):
    """Rows: reaction time, jaw onset; columns: whisker, false-alarm and difference curves. In each panel, for whisker
    hits, auditory hits and false alarms per cohort: Spearman rho (left) and partial rho controlling session time
    (right) per mouse; median +- bootstrap 68% interval; median rho and Wilcoxon stars printed."""
    from plot import group_color
    set_nature_rc()
    groups = [(t, c) for t in ("whisker_trial", "auditory_trial", "no_stim_trial") for c in ("R+", "R-")]
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 100 * MM), sharey=True)
    for r, (y, ylab) in enumerate((("rt_ms", "Reaction time"), ("jaw_ms", "Jaw onset"))):
        for c, cu in enumerate(("whisker", "false_alarm", "difference")):
            a = axs[r, c]
            for k, col in enumerate(("rho_vs_shift", "rho_partial_time")):
                for i, (tt, co) in enumerate(groups):
                    v = corr[(corr.trial_type == tt) & (corr.cohort == co) & (corr.latency == y) & (corr.curve == cu)][col].dropna()
                    x = k * 7 + i
                    color = group_color(tt, co)
                    a.scatter(x + np.random.default_rng(i + 10 * k).uniform(-0.2, 0.2, len(v)), v, s=1, color="0.8", lw=0)
                    _, md, lo, hi = med_band([(0, v)])
                    if len(md):
                        a.errorbar(x, md[0], [[md[0] - lo[0]], [hi[0] - md[0]]], color=color, marker="o", ms=2.3,
                                   capsize=0, elinewidth=0.8)
                    p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
                    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    a.text(x, 1.01, f"{np.median(v):.2f} {star}", ha="center", va="bottom", fontsize=3.6, color=color,
                           transform=a.get_xaxis_transform(), rotation=90)
            a.axhline(0, color="0.5", lw=0.4)
            a.set_xticks([2.5, 9.5], ["ρ − shift median", "ρ | session time"], fontsize=5)
            if r == 0:
                a.set_title(f"vs {cu.replace('_', '-')} curve", fontsize=6.5, pad=22)
            if c == 0:
                a.set_ylabel(f"{ylab}: Spearman ρ per mouse", labelpad=1)
    fig.text(0.01, 0.005, "Learners, whisker day 0, licked trials only (min. 10 per mouse). Order in each group: W hits R+, R-; "
             "A hits R+, R-; false alarms R+, R-. Curves at the trial (non-whisker trials: at the preceding whisker trial). "
             "Median ± bootstrap 68% interval; stars, Wilcoxon vs 0.", fontsize=4.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.0, w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
