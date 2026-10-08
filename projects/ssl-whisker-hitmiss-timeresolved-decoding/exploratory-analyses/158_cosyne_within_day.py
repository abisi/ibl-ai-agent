"""158 -- COSYNE figure v2, within-day change on the hit / miss axis only (user 2026-10-08: hide expert data, focus on the within-day
change of the hit / miss coding direction; state space first, projections per epoch per modality comparing cohorts, pre -> post change
beyond session time as dots with s.e.m. and non-parametric tests, state-space summary per cohort; one line).
Learners only; whole brain; shared tracked stable units (137b); evoked 5-35 ms response minus the epoch's baseline.
  a  state space before the task (passive pre): x = unit active hit - miss coding direction (146, all active whisker trials), y = the
     passive-pre whisker pattern orthogonalised to x; active hits / misses (both halves), passive-pre whisker and auditory responses;
     mean over sessions of both cohorts (they do not differ before the task); arrows: the two hypotheses
  b  projection of the whisker-evoked pattern on the active lick axis (135: unit hit - miss axis, projection / sqrt(n units)) in
     passive pre, active 1st / 2nd half, passive post; R+ vs R- mean +- s.e.m.; top: post - pre within cohort (Wilcoxon) and R+ vs R-
     on post - pre (Mann-Whitney)
  c  same for the auditory-evoked pattern (control: rewarded in both cohorts)
  d  pre -> post change beyond session time: excess over lick axes rebuilt from linearly shifted labels (135 shift null), whisker,
     auditory and whisker - auditory projection; mean +- s.e.m.; filled: Wilcoxon vs 0 p < 0.05; top: R+ vs R- Mann-Whitney
  e1, e2  state-space summary per cohort (146): passive pre -> post displacement of the whisker and auditory responses, active hit /
     miss positions for reference
Statistics non-parametric only (session = unit, one learning session per mouse); uncorrected.
Outputs: figures/publication/158_cosyne_within_day_learners.{png,pdf,svg}; 158_stats_learners.csv
Run (haas, repo root): python .../158_cosyne_within_day.py
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
WC, AC = "#f7b519", "#2c2cdb"
EPOCHS = [("passive_pre", "passive\npre"), ("active_1", "active\n1st"), ("active_2", "active\n2nd"), ("passive_post", "passive\npost")]


def pn(p):
    return "n.s." if not np.isfinite(p) or p >= 0.05 else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def wil(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan


def mwu(x, y):
    x, y = (np.asarray(v, float) for v in (x, y)); x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    return stats.mannwhitneyu(x, y).pvalue if min(len(x), len(y)) >= 3 else np.nan


def load():
    L = H.learners()
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    A = A[(A.area == "All units") & A.skipped_reason.isna()]; A = A[H.mouse_of(A).isin(L)]
    D = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    D = D[D.skipped_reason.isna() & (D.area == "whole_brain") & (D.unit_set == "stable") & (D.response == "epochbase")]
    D = D[H.mouse_of(D).isin(L)]
    for k in ("x", "y"):
        D[f"ss_hit_{k}"] = D[[f"ss_act1_hit_{k}", f"ss_act2_hit_{k}"]].mean(1)
        D[f"ss_miss_{k}"] = D[[f"ss_act1_miss_{k}", f"ss_act2_miss_{k}"]].mean(1)
    return A, D


def state_points(ax, D, conds, hyp=False):
    """conds: list of (column stem, marker, colour, filled, label)"""
    P = {}
    for stem, mk, colr, filled, lab in conds:
        x, y = D[f"{stem}_x"].astype(float), D[f"{stem}_y"].astype(float)
        P[stem] = (x.mean(), y.mean())
        ax.errorbar(x.mean(), y.mean(), xerr=H.sem(x), yerr=H.sem(y), fmt=mk, ms=4.2, color=colr, mfc=colr if filled else "white",
                    mec=colr, lw=0.6, capsize=0, zorder=4, label=lab)
    ax.axhline(0, color="0.88", lw=0.5, zorder=0); ax.axvline(0, color="0.88", lw=0.5, zorder=0)
    ax.set_xlabel("hit − miss axis"); ax.set_ylabel("whisker pattern\n(orthogonal to hit − miss)")
    return P


def panel_a(ax, D, R):
    P = state_points(ax, D, [("ss_hit", "^", "0.15", True, "active hit"), ("ss_miss", "v", "0.6", True, "active miss"),
                             ("ss_pre_W", "o", WC, False, "whisker, pre"), ("ss_pre_A", "D", AC, False, "auditory, pre")])
    w = P["ss_pre_W"]
    for tgt, c in (("ss_hit", "R+"), ("ss_miss", "R-")):
        t = P[tgt]; d = np.array(t) - np.array(w)
        ax.annotate("", (w[0] + 0.45 * d[0], w[1] + 0.45 * d[1]), w,
                    arrowprops=dict(arrowstyle="-|>", color=COL[c], lw=1.1, ls=(0, (2.5, 1.5)), shrinkA=4, shrinkB=0))
        ax.text(w[0] + 0.5 * d[0], w[1] + 0.5 * d[1] + 0.35, f"{c.replace('-', '−')}?", color=COL[c], fontsize=5.5, ha="center")
    ax.legend(frameon=False, fontsize=4.6, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    ax.set_title(f"Before the task: whisker response\nbetween hit and miss (n = {len(D)})")
    R.append(dict(panel="a", measure="passive-pre whisker x", cohort="both", n=len(D), mean_a=w[0], mean_b=P["ss_hit"][0], note="mean b = hit x"))


def panel_epochs(ax, A, s, title, R, key):
    x = np.arange(len(EPOCHS))
    for k, c in enumerate(COH):
        d = A[A.reward_group == c]
        m = [d[f"evoked{s}_proj_{e}"].mean() for e, _ in EPOCHS]; se = [H.sem(d[f"evoked{s}_proj_{e}"]) for e, _ in EPOCHS]
        ax.errorbar(x + (k - 0.5) * 0.12, m, yerr=se, fmt="-o", ms=3, color=COL[c], lw=1.1, capsize=0,
                    label=f"{c.replace('-', '−')} (n = {d[f'evoked{s}_proj_passive_pre'].notna().sum()})")
    ch = {c: (A[A.reward_group == c][f"evoked{s}_proj_passive_post"] - A[A.reward_group == c][f"evoked{s}_proj_passive_pre"]).to_numpy(float)
          for c in COH}
    pc = {c: wil(ch[c]) for c in COH}; pm = mwu(ch["R+"], ch["R-"])
    ax.text(0.5, 1.0, f"post − pre: R+ {pn(pc['R+'])}, R− {pn(pc['R-'])}\nR+ vs R− {pn(pm)}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=4.8)
    ax.axhline(0, color="0.7", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in EPOCHS], fontsize=5); ax.set_xlim(-0.4, len(EPOCHS) - 0.6)
    ax.set_ylabel("projection on hit − miss axis\n(per unit)" if s == "W" else "")
    ax.set_title(title, pad=17)
    if s == "W":
        ax.legend(frameon=False, fontsize=4.8, loc="lower left", handlelength=1.2)
    for c in COH:
        R.append(dict(panel=key, measure=f"{s} projection post - pre", cohort=c, n=int(np.isfinite(ch[c]).sum()), mean_a=np.nanmean(ch[c]),
                      p_nonparam=pc[c]))
    R.append(dict(panel=key, measure=f"{s} projection post - pre", cohort="R+ vs R-", mean_a=np.nanmean(ch["R+"]), mean_b=np.nanmean(ch["R-"]),
                  p_nonparam=pm))


def panel_d(ax, A, R):
    items = [("shift_excess_dWP", "whisker"), ("shift_excess_dAP", "auditory"), ("shift_excess_dWAP", "whisker −\nauditory")]
    for i, (col, lab) in enumerate(items):
        v = {c: A[A.reward_group == c][col].to_numpy(float) for c in COH}
        for k, c in enumerate(COH):
            p = wil(v[c]); m = np.nanmean(v[c])
            ax.errorbar(i + (k - 0.5) * 0.3, m, H.sem(v[c]), fmt="o", ms=3.6, color=COL[c], mfc=COL[c] if p < 0.05 else "white", lw=1.1, capsize=0)
            R.append(dict(panel="d", measure=col, cohort=c, n=int(np.isfinite(v[c]).sum()), mean_a=m, p_nonparam=p))
        pm = mwu(v["R+"], v["R-"])
        ax.text(i, 1.0, pn(pm), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.8)
        R.append(dict(panel="d", measure=col, cohort="R+ vs R-", mean_a=np.nanmean(v["R+"]), mean_b=np.nanmean(v["R-"]), p_nonparam=pm))
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for _, l in items], fontsize=5); ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylabel("pre → post change\nbeyond session time")
    ax.set_title("Beyond session time\n(R+ vs R−; filled: ≠ 0)", pad=9)


def panel_e(ax, D, c, lims, R):
    d = D[D.reward_group == c]
    P = state_points(ax, d, [("ss_hit", "^", "0.15", True, "hit"), ("ss_miss", "v", "0.6", True, "miss"),
                             ("ss_pre_W", "o", WC, False, "W pre"), ("ss_post_W", "s", WC, True, "W post"),
                             ("ss_pre_A", "D", AC, False, "A pre"), ("ss_post_A", "s", AC, True, "A post")])
    for s, colr in (("W", WC), ("A", AC)):
        ax.annotate("", P[f"ss_post_{s}"], P[f"ss_pre_{s}"], arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.2, shrinkA=3, shrinkB=3))
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    dx = (d.ss_post_W_x - d.ss_pre_W_x).to_numpy(float)
    ax.set_title(f"{c.replace('-', '−')} (n = {len(d)})\nwhisker shift on hit − miss: {np.nanmean(dx):+.1f}", color=COL[c])
    if c == "R-":
        ax.set_ylabel("")
    R.append(dict(panel="e", measure="whisker x displacement", cohort=c, n=len(d), mean_a=np.nanmean(dx), p_nonparam=wil(dx)))


def main():
    H.setup()
    A, D = load()
    R = []
    fig = plt.figure(figsize=(11.6, 2.4))
    gs = fig.add_gridspec(1, 6, wspace=0.68, left=0.05, right=0.985, top=0.74, bottom=0.2)
    axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
    panel_a(axs[0], D, R)
    panel_epochs(axs[1], A, "W", "Whisker response\non the hit − miss axis", R, "b")
    panel_epochs(axs[2], A, "A", "Auditory response\n(control)", R, "c")
    yl = (min(axs[1].get_ylim()[0], axs[2].get_ylim()[0]), max(axs[1].get_ylim()[1], axs[2].get_ylim()[1]))
    for a in axs[1:3]:
        a.set_ylim(*yl)
    panel_d(axs[3], A, R)
    cols = ["ss_hit", "ss_miss", "ss_pre_W", "ss_post_W", "ss_pre_A", "ss_post_A"]
    xs = np.concatenate([D[f"{c}_x"].groupby(D.reward_group).mean().to_numpy() for c in cols])
    ys = np.concatenate([D[f"{c}_y"].groupby(D.reward_group).mean().to_numpy() for c in cols])
    pad = lambda v: (np.nanmin(v) - 0.12 * np.ptp(v) - 0.3, np.nanmax(v) + 0.12 * np.ptp(v) + 0.3)
    lims = (pad(np.r_[xs, 0]), pad(np.r_[ys, 0]))
    for ax, c in zip(axs[4:], COH):
        panel_e(ax, D, c, lims, R)
    dx = {c: (D[D.reward_group == c].ss_post_W_x - D[D.reward_group == c].ss_pre_W_x).to_numpy(float) for c in COH}
    R.append(dict(panel="e", measure="whisker x displacement", cohort="R+ vs R-", mean_a=np.nanmean(dx["R+"]), mean_b=np.nanmean(dx["R-"]),
                  p_nonparam=mwu(dx["R+"], dx["R-"])))
    axs[0].set_xlim(*lims[0]); axs[0].set_ylim(*lims[1])
    for a in axs:
        a.set_box_aspect(1)
    for a, l in zip(axs, "abcdef"):
        bb = a.get_position(); fig.text(bb.x0 - 0.035, 0.965, l if l != "f" else "", fontsize=9, weight="bold")
    fig.text(axs[4].get_position().x0 - 0.035, 0.965, "e", fontsize=9, weight="bold")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"158_cosyne_within_day_learners.{ext}", dpi=300)
    S = pd.DataFrame(R); S.to_csv(EA / "158_stats_learners.csv", index=False)
    print(S.round(4).to_string())
    print("ALL DONE", FIGDIR / "158_cosyne_within_day_learners.png")


if __name__ == "__main__":
    main()
