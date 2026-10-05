"""154 -- COSYNE figure, positive results only (user 2026-10-05: "for cosyne, only positive results ... link it to the within-day-sl
results"): within the learning session, the reward contingency remaps the whisker sensorimotor chain in opposite directions.
  a  schematic: session (passive pre -> active -> passive post), the two windows and the two axes
  b  pre-lick (ssl-within-day-remapping 002 with SSL_MIN_WH=4, SL reference): whisker-hit (WH) minus spontaneous-lick (SL) projection on the session's
     reward-lick coding direction (SL = 0, auditory hit AH = 1), five bins of day-0 session time; expert sessions at the right
  c  pre-lick: per-session day-0 drift of WH relative to SL along that direction (slope WH - slope SL)
  d  stimulus onset (135, this project): raw cosine of the passive / active whisker-evoked pattern (5-35 ms) with the active lick axis
     (hit - miss) in passive pre, active halves, passive post; auditory-evoked pattern dashed (control)
  e  stimulus onset beyond session time: passive pre -> post change minus the change produced by lick axes rebuilt from
     time-shifted labels (linear-shift null): whisker raw cosine and whisker - auditory projection
  f  state space (151): passive whisker response pre -> post displacement, passive pre at the origin; x = choice axis, y = passive-pre
     whisker pattern orthogonalised to x; dashed = active miss level on x (relative to passive pre), per cohort
Tests: within cohort Wilcoxon | t vs 0; R+ vs R- Mann-Whitney | Welch (and cohort-label permutation across mice for c). Session =
unit (one learning session per mouse). Scope: all (default) | learners. Units: b, c KS4 good + mua (pre-lick rate >= 0.1 Hz, the
within-day project); d-f the shared tracked stable units (137b).
Outputs: figures/publication/154_cosyne_remapping_<scope>.{png,pdf,svg}, 154_stats_<scope>.csv, 154_caption_<scope>.md
Run (haas, repo root): python .../154_cosyne_remapping_figure.py [all|learners]
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
from axel_bisi_paths import axel_bisi_root  # noqa: E402

COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
WC, AC = "#f7b519", "#2c2cdb"
WD = axel_bisi_root() / "combined_results_ks4" / "_within_day_sl"
# within-day trial-level results with >= 4 whisker hits per session (user 2026-10-05; AH and SL keep >= 8; env override)
WD_SLOPES = __import__("os").environ.get("SSL_154_WD_SLOPES", "slopes_wh4")
EPL = ["passive\npre", "active\n1st", "active\n2nd", "passive\npost"]


def P(a, b):
    return f"{H.pnum(a)} | {H.pnum(b)}"


def perm_diff(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed); z = np.r_[x, y]; k = len(x); d0 = np.mean(x) - np.mean(y)
    null = np.array([np.mean(p[:k]) - np.mean(p[k:]) for p in (rng.permutation(z) for _ in range(n))])
    return (np.sum(np.abs(null) >= abs(d0)) + 1) / (n + 1)


def scope_filter(d, scope, mouse_col="mouse_id"):
    if scope != "learners":
        return d
    m = d[mouse_col] if mouse_col in d else H.mouse_of(d)
    return d[m.isin(H.learners())]


def arrow(ax, p0, p1, color, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, linestyle=ls))


def panel_a(ax):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for x0, x1, t, c in ((0.0, 0.2, "passive\npre", "0.88"), (0.23, 0.77, "active (whisker, auditory, catch)", "#d6ece6"), (0.8, 1.0, "passive\npost", "0.88")):
        ax.add_patch(FancyBboxPatch((x0, 0.78), x1 - x0, 0.17, boxstyle="round,pad=0.005,rounding_size=0.02", color=c, lw=0))
        ax.text((x0 + x1) / 2, 0.865, t, ha="center", va="center", fontsize=5)
    ax.text(0.5, 0.72, "100 ms before the first lick (b, c)", ha="center", fontsize=4.6, color="0.25")
    ax.text(0.1, 0.72, "5-35 ms after\nstimulus (d-f)", ha="center", fontsize=4.6, color="0.25", va="top")
    ax.text(0.9, 0.72, "5-35 ms after\nstimulus (d-f)", ha="center", fontsize=4.6, color="0.25", va="top")
    # left: reward-lick direction (pre-lick)
    o = (0.06, 0.12)
    arrow(ax, o, (0.42, 0.12), "0.2"); ax.text(0.06, 0.03, "SL", fontsize=5, ha="center"); ax.text(0.42, 0.03, "AH", fontsize=5, ha="center")
    ax.text(0.24, 0.48, "reward-lick direction\n(pre-lick WH)", ha="center", fontsize=4.8)
    arrow(ax, (0.22, 0.3), (0.34, 0.3), COL["R+"]); ax.text(0.36, 0.3, "R+", color=COL["R+"], fontsize=5, va="center")
    arrow(ax, (0.22, 0.22), (0.1, 0.22), COL["R-"]); ax.text(0.05, 0.22, "R-", color=COL["R-"], fontsize=5, va="center", ha="right")
    # right: lick axis (stimulus onset)
    o = (0.58, 0.08)
    arrow(ax, o, (0.97, 0.08), "0.2"); ax.text(0.93, 0.0, "lick axis (hit - miss)", fontsize=4.6, ha="right")
    a0 = np.deg2rad(35); arrow(ax, o, (o[0] + 0.25 * np.cos(a0), o[1] + 0.4 * np.sin(a0)), WC, lw=1.4)
    a1 = np.deg2rad(80); arrow(ax, o, (o[0] + 0.2 * np.cos(a1), o[1] + 0.42 * np.sin(a1)), COL["R-"], lw=1.2, ls=(0, (2, 1.5)))
    ax.text(0.86, 0.36, "whisker\npattern", color=WC, fontsize=4.6, ha="center")
    ax.text(0.66, 0.55, "R- post", color=COL["R-"], fontsize=4.6, ha="center")


def panel_b(ax, TR, R):
    md = TR[TR.axis == "md"].pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "bin"], columns="cls", values="score").reset_index()
    md["rel"] = md["WH"] - md["FA"]
    xb = (np.arange(5) + 0.5) / 5
    for c in COH:
        g = md[(md.cohort == c) & (md.stage == "learning")]
        q = g.groupby("bin").rel.agg(["mean", "sem"]).reindex(range(5))
        ax.fill_between(xb, q["mean"] - q["sem"], q["mean"] + q["sem"], color=COL[c], alpha=0.15, lw=0)
        ax.plot(xb, q["mean"], color=COL[c], lw=1.1, marker="o", ms=2.4, mfc="white", label=f"{c} ({g.session_id.nunique()})")
        e = md[(md.cohort == c) & (md.stage == "expert")].groupby("session_id").rel.mean()
        ax.errorbar(1.14, e.mean(), e.sem(), fmt="o", ms=3.2, color=COL[c], capsize=0, lw=0.9, clip_on=False)
        R.append(dict(panel="b", measure="WH - SL on reward-lick CD", cohort=c, n=g.session_id.nunique(), mean_a=q["mean"].iloc[0],
                      mean_b=q["mean"].iloc[-1], p_nonparam=np.nan, p_param=np.nan, note=f"expert {e.mean():.3f} (n = {len(e)})"))
    ax.text(1.14, 1.0, "expert", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.6)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(0, 1.22); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("time in the learning session")
    ax.set_ylabel("WH - SL on reward-lick\ndirection (SL = 0, AH = 1)"); ax.legend(frameon=False, fontsize=4.6, loc="lower left")
    ax.set_title("pre-lick: whisker hits move toward\nrewarded licks in R+, away in R-")


def panel_c(ax, D2, R):
    col = "md_WH-FA_slope"
    d0 = D2[D2.stage == "learning"].dropna(subset=[col])
    v = {c: d0[d0.cohort == c][col].to_numpy(float) for c in COH}
    for k, c in enumerate(COH):
        ax.plot(k + np.random.default_rng(k).uniform(-0.13, 0.13, len(v[c])), v[c], "o", ms=2, color=COL[c], alpha=0.45, mew=0)
        ax.errorbar(k, v[c].mean(), H.sem(v[c]), fmt="o", ms=3.6, color=COL[c], lw=1, capsize=0, zorder=5)
        pw, pt, n = H.one_sample(v[c])
        R.append(dict(panel="c", measure="day-0 WH drift (slope WH - slope SL)", cohort=c, n=n, mean_a=v[c].mean(), mean_b=np.nan,
                      p_nonparam=pw, p_param=pt, note=""))
        ax.text(k, 1.01, P(pw, pt), color=COL[c], ha="center", fontsize=4.3, transform=ax.get_xaxis_transform())
    mw, we = H.unpaired(v["R+"], v["R-"]); pm = perm_diff(v["R+"], v["R-"])
    R.append(dict(panel="c", measure="day-0 WH drift (slope WH - slope SL)", cohort="R+ vs R-", n=np.nan, mean_a=v["R+"].mean(),
                  mean_b=v["R-"].mean(), p_nonparam=mw, p_param=we, note=f"cohort-label permutation p = {pm:.4f}"))
    ax.text(0.5, 1.1, f"R+ vs R- {P(mw, we)}", ha="center", fontsize=4.5, transform=ax.get_xaxis_transform())
    lo, hi = np.nanpercentile(np.r_[v["R+"], v["R-"]], [2, 98]); ax.set_ylim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R-"]); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("WH drift toward AH per session"); ax.set_title("pre-lick: day-0 drift", pad=24)


def panel_d(ax, A, R):
    ep = ["passive_pre", "active_1", "active_2", "passive_post"]
    x = np.arange(4)
    for c in COH:
        g = A[A.reward_group == c]
        for s, ls, lw, a_ in (("W", "-", 1.2, 1.0), ("A", (0, (2, 1.5)), 0.8, 0.6)):
            M = g[[f"evoked{s}_cos_{e}" for e in ep]].to_numpy(float)
            ax.errorbar(x + (0.05 if c == "R-" else -0.05), np.nanmean(M, 0), [H.sem(M[:, j]) for j in range(4)], color=COL[c], lw=lw, ls=ls,
                        marker="o", ms=2.6 if s == "W" else 1.8, mfc="white" if s == "A" else COL[c], capsize=0, alpha=a_,
                        label=f"{c} ({len(g)})" if s == "W" else None)
    ch = {c: (A[A.reward_group == c].evokedW_cos_passive_post - A[A.reward_group == c].evokedW_cos_passive_pre).to_numpy(float) for c in COH}
    mw, we = H.unpaired(ch["R+"], ch["R-"])
    R.append(dict(panel="d", measure="whisker-evoked cos with lick axis, post - pre (raw)", cohort="R+ vs R-", n=np.nan,
                  mean_a=np.nanmean(ch["R+"]), mean_b=np.nanmean(ch["R-"]), p_nonparam=mw, p_param=we, note=""))
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks(x); ax.set_xticklabels(EPL, fontsize=4.6); ax.set_ylabel(r"cos(pattern, lick axis)")
    ax.legend(frameon=False, fontsize=4.6, loc="upper right")
    ax.set_title(f"stimulus onset: R- whisker response\nleaves the lick axis (post - pre {P(mw, we)})")


def panel_e(ax, A, R):
    for j, (col, lab) in enumerate((("shift_excess_dWR", "whisker\ncosine"), ("shift_excess_dWAP", "whisker -\nauditory proj."))):
        v = {c: A[A.reward_group == c][col].dropna().to_numpy(float) for c in COH}
        for k, c in enumerate(COH):
            xx = j * 2.6 + k
            ax.plot(xx + np.random.default_rng(k).uniform(-0.13, 0.13, len(v[c])), v[c], "o", ms=1.8, color=COL[c], alpha=0.4, mew=0)
            ax.errorbar(xx, v[c].mean(), H.sem(v[c]), fmt="o", ms=3.4, color=COL[c], lw=1, capsize=0, zorder=5)
            pw, pt, n = H.one_sample(v[c])
            R.append(dict(panel="e", measure=col, cohort=c, n=n, mean_a=v[c].mean(), mean_b=np.nan, p_nonparam=pw, p_param=pt, note=""))
        mw, we = H.unpaired(v["R+"], v["R-"])
        R.append(dict(panel="e", measure=col, cohort="R+ vs R-", n=np.nan, mean_a=v["R+"].mean(), mean_b=v["R-"].mean(), p_nonparam=mw,
                      p_param=we, note=""))
        ax.text(j * 2.6 + 0.5, 1.02, P(mw, we), ha="center", fontsize=4.5, transform=ax.get_xaxis_transform())
        ax.text(j * 2.6 + 0.5, -0.2, lab, ha="center", va="top", fontsize=4.8, transform=ax.get_xaxis_transform())
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(["R+", "R-", "R+", "R-"], fontsize=4.8); ax.set_xlim(-0.6, 4.2)
    ax.set_ylabel("pre -> post change\nbeyond session time")
    ax.set_title("beyond session time\n(linear-shift null)", pad=10)


def panel_f(ax, D, R):
    for c in COH:
        g = D[D.reward_group == c]
        x = (g.ss_post_W_x - g.ss_pre_W_x).to_numpy(float); y = (g.ss_post_W_y - g.ss_pre_W_y).to_numpy(float)
        ax.plot(x, y, "o", ms=1.8, color=COL[c], alpha=0.3, mew=0)
        mx, my = np.nanmean(x), np.nanmean(y)
        ax.add_patch(FancyArrowPatch((0, 0), (mx, my), arrowstyle="-|>", mutation_scale=7, color=COL[c], lw=1.3))
        ax.errorbar(mx, my, xerr=H.sem(x), yerr=H.sem(y), color=COL[c], lw=0.8, capsize=0)
        mis = np.nanmean((((g.ss_act1_miss_x + g.ss_act2_miss_x) / 2) - g.ss_pre_W_x).to_numpy(float))
        ax.axvline(mis, color=COL[c], lw=0.7, ls=(0, (3, 2)))
    dx = {c: (D[D.reward_group == c].ss_post_W_x - D[D.reward_group == c].ss_pre_W_x).to_numpy(float) for c in COH}
    mw, we = H.unpaired(dx["R+"], dx["R-"])
    R.append(dict(panel="f", measure="whisker displacement along the choice axis", cohort="R+ vs R-", n=np.nan, mean_a=np.nanmean(dx["R+"]),
                  mean_b=np.nanmean(dx["R-"]), p_nonparam=mw, p_param=we, note=""))
    ax.plot(0, 0, "o", ms=3.2, mfc="white", mec="k", mew=0.7, zorder=6)
    ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
    ax.text(0.02, 0.03, "dashed: active miss level", transform=ax.transAxes, fontsize=4.3, color="0.3")
    ax.set_xlabel("choice axis (post - pre)"); ax.set_ylabel("whisker-pattern axis (post - pre)")
    ax.set_title(f"state space: R- whisker response slides\nto the miss level (x: {P(mw, we)})")


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    H.setup()
    TR = scope_filter(pd.read_csv(WD / WD_SLOPES / "trial_slopes_trajectories.csv"), scope)
    D2 = scope_filter(pd.read_csv(WD / WD_SLOPES / "trial_slopes_sessions.csv"), scope)
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet"); A = scope_filter(A[(A.area == "All units") & A.skipped_reason.isna()], scope)
    D = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    D = scope_filter(D[D.skipped_reason.isna() & (D.area == "whole_brain") & (D.unit_set == "stable") & (D.response == "epochbase")], scope)
    R = []
    fig = plt.figure(figsize=(7.2, 4.4))
    W_, H_ = fig.get_size_inches()
    rects = {"a": [0.02, 0.55, 0.3, 0.36], "b": [0.42, 0.56, 0.22, 0.31], "c": [0.75, 0.56, 0.12, 0.31],
             "d": [0.07, 0.1, 0.24, 0.3], "e": [0.42, 0.1, 0.2, 0.3], "f": [0.73, 0.1, 0.22, 0.3]}
    ax = {k: fig.add_axes(r) for k, r in rects.items()}
    panel_a(ax["a"]); panel_b(ax["b"], TR, R); panel_c(ax["c"], D2, R); panel_d(ax["d"], A, R); panel_e(ax["e"], A, R); panel_f(ax["f"], D, R)
    for k, a in ax.items():
        x0 = a.get_position().x0 - (0.06 if k != "a" else 0.0)
        fig.text(max(x0, 0.005), a.get_position().y1 + 0.03, k, fontsize=9, weight="bold")
    fig.suptitle("Within one learning session, the reward contingency remaps the whisker sensorimotor chain in opposite directions",
                 fontsize=7.2, weight="bold", x=0.02, ha="left", y=0.995)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"154_cosyne_remapping_{scope}.{ext}", dpi=300)
    S = pd.DataFrame(R); S.insert(0, "scope", scope); S.to_csv(EA / f"154_stats_{scope}.csv", index=False)
    g = lambda pnl, coh, m=None: S[(S.panel == pnl) & (S.cohort == coh) & ((S.measure == m) if m else True)].iloc[0]
    c_ = g("c", "R+ vs R-"); cp, cm = g("c", "R+"), g("c", "R-")
    e1, e2 = g("e", "R+ vs R-", "shift_excess_dWR"), g("e", "R+ vs R-", "shift_excess_dWAP")
    d_, f_ = g("d", "R+ vs R-"), g("f", "R+ vs R-")
    cap = (f"**Within one learning session, the reward contingency remaps the whisker sensorimotor chain in opposite directions.** "
           f"(a) Mice learn to lick to a whisker stimulus within one session; licks after whisker stimuli are rewarded (R+) or not (R-), "
           f"licks after a tone always. (b) Pre-lick activity (100 ms before the first lick) of whisker hits (WH), projected on the "
           f"session's reward-lick coding direction (spontaneous licks SL = 0, auditory hits AH = 1), relative to SL, across the "
           f"learning session; dots right: expert sessions. (c) Per-session drift of WH toward AH (slope WH - slope SL): R+ "
           f"{cp.mean_a:+.2f} (p = {P(cp.p_nonparam, cp.p_param)}), R- {cm.mean_a:+.2f} (p = {P(cm.p_nonparam, cm.p_param)}); R+ vs R- "
           f"p = {P(c_.p_nonparam, c_.p_param)}, {c_.note}. (d) Stimulus onset (5-35 ms after the whisker stimulus, before any lick): "
           f"cosine of the whisker-evoked population pattern with the active lick axis (hits - misses), in passive presentations before "
           f"and after the task and in the two active halves (dashed: auditory-evoked, control); post - pre R+ vs R- p = "
           f"{P(d_.p_nonparam, d_.p_param)}. (e) The passive pre -> post change beyond session time (lick axes rebuilt from labels "
           f"shifted against the trial sequence): whisker cosine p = {P(e1.p_nonparam, e1.p_param)}, whisker - auditory projection "
           f"p = {P(e2.p_nonparam, e2.p_param)}. (f) State space with passive pre at the origin: the R- whisker response slides along "
           f"the choice axis to the level of active misses (dashed), R+ does not (p = {P(f_.p_nonparam, f_.p_param)}); both shrink "
           f"along their own pattern. R- mice receive more whisker and auditory trials in the same proportion; trial counts scale the "
           f"shrinkage but not the choice-axis change, which persists with counts as covariates and is absent for auditory responses. "
           f"Mean +- s.e.m. over sessions (one learning session per mouse); p: Mann-Whitney | Welch (cohorts), Wilcoxon | t (vs 0); "
           f"uncorrected; {'all mice' if scope == 'all' else 'learners only'}. b, c: KS4 good + mua units, sessions with >= 4 whisker hits and >= 8 auditory hits and spontaneous licks; d-f: tracked, drift-checked units.")
    (EA / f"154_caption_{scope}.md").write_text(cap, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(S.round(4).to_string(index=False)); print(cap)


if __name__ == "__main__":
    main()
