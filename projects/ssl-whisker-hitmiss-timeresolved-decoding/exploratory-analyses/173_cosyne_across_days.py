"""173 -- COSYNE-style across-days figure for the two coding directions with the harmonised (within-day) metric definitions
(user 2026-10-08: "port the metric standardisation to the across-days analysis, for both coding directions, then create new,
updated COSYNE panels like ssl-prelick-convergence"; nothing deleted, 083 unchanged). Same compact inch layout as 165 (row height =
side of the square panels). Population: learners on day 0 (learning), all mice at expert (learner rule); unit = session.
Row 1, hit - miss axis (passive whisker, within each session, as 165 / 169; 135 SSL_135_HARM=1, learning and expert):
  b  angle theta between the passive whisker response and the hit - miss axis, pre and post, day 0 and expert (mean +- s.e.m.)
  c  change of angle within the session, real - shifted axes (post - pre), day 0 vs expert
  d  change of projection on the axis within the session, real - shifted, day 0 vs expert
  expert sessions need >= 10 licked and >= 10 unlicked whisker trials to build the axis: few at expert (R+ lick almost always, R-
  almost never), so the expert row is underpowered (n in the panels)
Row 2, reward-lick CD (087: 026 folds / scaling, per neuron):
  b  position of whisker hits between spontaneous licks (0) and auditory hits (1) on the CD, lambda, day 0 vs expert
  c  angle theta between the whisker hit - spontaneous lick difference and the CD
  d  length rho of the whisker hit - spontaneous lick difference (per neuron, cross-validated)
  e  per area group: change of the native whisker-hit position d_WS (expert - day 0) per cohort, mean +- combined s.e.m.; * = BH q < 0.05
     of the per-area cohort x stage interaction (mouse permutation); area groups with >= 3 sessions per cohort at both stages
Stage lines: per cohort, expert vs day 0 Mann-Whitney U (also Welch t in the _stats.csv); title: cohort x stage interaction
[E - L](R+) - [E - L](R-), cohort labels permuted across mice (10,000; a mouse keeps all its sessions).
Output: figures/publication/173_cosyne_across_days.{svg,pdf,png} + 173_cosyne_across_days_stats.csv
Run (haas, repo root): python .../173_cosyne_across_days.py
"""
from __future__ import annotations

import importlib
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
H = importlib.import_module("143_lt_split_windows_figures")
M165 = importlib.import_module("165_two_axes_figure")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
PLH = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence/across_days/sl_stable/harmonised/sessions_areas.csv")
STG = ("learning", "expert"); STL = {"learning": "day 0", "expert": "expert"}
N_PERM = 10000
R = []
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def two(a, b):
    a, b = (np.asarray(v, float) for v in (a, b)); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def interaction(D, q, key):
    d = D.dropna(subset=[q]).sort_values("session_id")
    rng = np.random.default_rng(zlib.crc32(f"173|{key}|{q}".encode()))
    mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy(); idx = pd.Index(mice).get_indexer(d.mouse_id)
    st, v = d.stage.to_numpy(), d[q].to_numpy(float)

    def stat(coh):
        f = lambda c, s_: np.nanmean(v[(coh == c) & (st == s_)]) if ((coh == c) & (st == s_)).any() else np.nan
        return (f("R+", "expert") - f("R+", "learning")) - (f("R-", "expert") - f("R-", "learning"))
    obs = stat(d.cohort.to_numpy())
    null = np.array([stat(rng.permutation(mc)[idx]) for _ in range(N_PERM)]); null = null[np.isfinite(null)]
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))


def stage_lines(ax, D, q, ylab, key, ref=None):
    for j, c in enumerate(COH):
        v = [D[(D.cohort == c) & (D.stage == s_)][q].dropna().to_numpy(float) for s_ in STG]
        m = [x.mean() if len(x) else np.nan for x in v]; se = [H.sem(x) for x in v]
        xx = np.arange(2) + (j - 0.5) * 0.08
        pm, pt = two(v[1], v[0])
        ax.errorbar(xx, m, se, fmt="-o", ms=3.4, color=COL[c], lw=1.3, capsize=0,
                    label=f"{c.replace('-', '−')} (n = {len(v[0])}/{len(v[1])}): {M165.pn(pm)}")
        R.append(dict(row=key, measure=q, cohort=c, n_day0=len(v[0]), n_expert=len(v[1]), mean_day0=m[0], mean_expert=m[1], p_mwu=pm, p_welch=pt))
    obs, pi = interaction(D, q, key)
    R.append(dict(row=key, measure=q, cohort="cohort x stage", diff=obs, p_perm=pi))
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels([STL[s_] for s_ in STG]); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylab, fontsize=5.5)
    lg = ax.legend(frameon=False, fontsize=4.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, borderaxespad=0.1,
                   title=f"cohort × stage: p = {pi:.3g}", title_fontsize=5.3)
    lg._legend_box.align = "left"


def angle_four(ax, D):
    """angle pre / post at day 0 and expert, per cohort"""
    xs = [0, 1, 2.4, 3.4]
    for j, c in enumerate(COH):
        for k, s_ in enumerate(STG):
            d = D[(D.cohort == c) & (D.stage == s_)]
            m = d[["th_pre", "th_post"]].mean().to_numpy(); se = d[["th_pre", "th_post"]].sem().to_numpy()
            ax.errorbar(np.array(xs[2 * k: 2 * k + 2]) + (j - 0.5) * 0.12, m, se, fmt="-o", ms=3, color=COL[c], lw=1.2, capsize=0,
                        mfc=COL[c] if s_ == "expert" else "white")
            ax.text(xs[2 * k] + 0.5, 0.02 + 0.07 * j, f"n = {len(d)}", color=COL[c], fontsize=4.3, ha="center", transform=ax.get_xaxis_transform())
    ax.axhline(90, color="0.5", lw=0.6, ls=":"); ax.text(1.0, 90, " 90°", fontsize=4.5, color="0.4", va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(xs); ax.set_xticklabels(["pre", "post", "pre", "post"], fontsize=5)
    for x0, s_ in ((0.5, "learning"), (2.9, "expert")):
        ax.text(x0, -0.2, STL[s_], ha="center", fontsize=5.2, transform=ax.get_xaxis_transform())
    ax.set_ylabel("angle θ between whisker response\nand hit − miss axis (°)", fontsize=5.5)
    ax.set_title("passive whisker, within session", fontsize=5.5, pad=4)


def load_hm():
    th = lambda c: np.degrees(np.arccos(np.clip(c, -1, 1)))
    rows = []
    for st_, f169, f135 in (("learning", "169_along_orthogonal.parquet", "135_alignment_epochs_tracked_harm.parquet"),
                            ("expert", "169_along_orthogonal_expert.parquet", "135_alignment_epochs_tracked_harm_expert.parquet")):
        G = pd.read_parquet(EA / f169); G = G[G.stim == "W"].copy()
        for k in ("pre", "post"):
            G[f"th_{k}"], G[f"thn_{k}"] = th(G[f"cos_{k}"]), th(G[f"cos_{k}_null"])
        G["q_th"] = (G.th_post - G.th_pre) - (G.thn_post - G.thn_pre)
        A = pd.read_parquet(EA / f135); A = A[(A.area == "All units") & A.skipped_reason.isna()][["session_id", "shift_excess_dWP"]]
        d = G.merge(A, on="session_id", how="inner").dropna(subset=["q_th", "shift_excess_dWP"])
        d["stage"] = st_; rows.append(d)
    D = pd.concat(rows).rename(columns={"reward_group": "cohort"})
    return D[(D.stage == "expert") | H.mouse_of(D).isin(H.learners())]


def load_pl():
    P = pd.read_csv(PLH)
    return P[(P.stage == "expert") | P.mouse_id.isin(H.learners())]


def area_change(ax, P, AC, q="d_WS"):
    d = P[P.area != "whole_brain"].dropna(subset=[q])
    n = d.groupby(["area", "cohort", "stage"]).size().unstack([1, 2]).fillna(0)
    areas = [a for a in n.index if all(n.loc[a].get((c, s_), 0) >= 3 for c in COH for s_ in STG)]
    areas = sorted(areas, key=lambda a: -n.loc[a].sum())
    ps = []
    for i, a in enumerate(areas):
        da = d[d.area == a]
        for k, c in enumerate(COH):
            L_, E_ = (da[(da.cohort == c) & (da.stage == s_)][q].to_numpy(float) for s_ in STG)
            ch, se = E_.mean() - L_.mean(), np.sqrt(H.sem(E_) ** 2 + H.sem(L_) ** 2)
            ax.errorbar(ch, i + (k - 0.5) * 0.3, xerr=se, fmt="o", ms=2.8, color=COL[c], lw=0.9, capsize=0)
        obs, pi = interaction(da, q, f"area {a}"); ps.append(pi)
    q_ = M165.bh(ps)
    for a, p_, qq in zip(areas, ps, q_):
        R.append(dict(row="reward-lick areas", measure=q, area=a, p_perm=p_, q_BH=qq))
    ax.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels([f"{M165.SHORT.get(a, a)}{' *' if qq < 0.05 else ''}" for a, qq in zip(areas, q_)], fontsize=4.6)
    for t, a in zip(ax.get_yticklabels(), areas):
        t.set_color(AC.get(a, "black"))
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_xlabel("Δ WH position on CD, d_WS\n(expert − day 0, per neuron)", fontsize=5)
    ax.set_title("per area: change across days\n* cohort × stage q < 0.05", fontsize=5.5)
    ax._keep_aspect = True


def main():
    H.setup()
    plt.rcParams.update({"font.size": 5.5, "axes.labelsize": 5.5, "xtick.labelsize": 5, "ytick.labelsize": 5, "axes.linewidth": 0.6,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    AC = M165.area_colors()
    D = load_hm(); P = load_pl(); Pw = P[P.area == "whole_brain"]
    h = 0.95                                                       # schematic | b | c | d | areas, b-d squares of side h
    fig, gs, (top_, h_, gap_, Hf_) = M165.compact_grid(fig_w_extra=0.4, h=h, widths=[0.65, 0.65, h, 0.7, h, 0.7, h, 0.85, 0.65])
    for r, t in enumerate(("hit − miss axis", "reward-lick axis")):
        M165.frame(fig, gs[r, 0], "a  schematic")
        yc = 1 - (top_ + h_ / 2 + r * (h_ + gap_)) / Hf_
        fig.text(0.004, yc, t, fontsize=7, weight="bold", rotation=90, va="center")
    angle_four(fig.add_subplot(gs[0, 1]), D)
    stage_lines(fig.add_subplot(gs[0, 2]), D, "q_th", "Δ angle θ to hit − miss axis\n(post − pre, °; real − shifted)", "hit-miss", ref=0)
    stage_lines(fig.add_subplot(gs[0, 3]), D, "shift_excess_dWP", "Δ projection on hit − miss axis\n(post − pre, per neuron; real − shifted)", "hit-miss", ref=0)
    stage_lines(fig.add_subplot(gs[1, 1]), Pw, "lam", "WH position on reward-lick CD, λ\n(0 = spont. lick, 1 = auditory hit)", "reward-lick", ref=0)
    stage_lines(fig.add_subplot(gs[1, 2]), Pw, "theta", "angle θ between WH − SL\nand reward-lick CD (°)", "reward-lick")
    stage_lines(fig.add_subplot(gs[1, 3]), Pw, "rho_v", "length of WH − SL difference ρ\n(per neuron, cross-validated)", "reward-lick")
    area_change(fig.add_subplot(gs[1, 4]), P, AC)
    for a_ in fig.axes:
        if not getattr(a_, "_keep_aspect", False):
            a_.set_box_aspect(1)
    for ext in ("svg", "pdf", "png"):
        M165.savefig(fig, FIGDIR / f"173_cosyne_across_days.{ext}")
    S = pd.DataFrame(R); S.to_csv(EA / "173_cosyne_across_days_stats.csv", index=False)
    pd.set_option("display.width", 220); print(S.round(4).to_string())
    print("ALL DONE", FIGDIR / "173_cosyne_across_days.png")


if __name__ == "__main__":
    main()
