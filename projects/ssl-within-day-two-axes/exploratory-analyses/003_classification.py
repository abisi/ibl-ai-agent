"""003 -- Learning-specific vs stage-independent within-session changes on the two axes, whole brain and per area group (user 2026-10-15:
"the goal is to see which effects are specific to learning (either whole-brain or area-wise), and which effects are more pronounced with
learning"; expert sessions as control).
Measures (per session; real minus linear-shift null unless 'raw'):
  reward-lick CD (ssl-prelick-convergence within_day 026; per neuron):
    rl_WH_raw   OLS slope of whisker-hit projections over the session (raw)
    rl_WH       the same beyond the shift null (excess_pu)
    rl_WHSL     whisker-hit minus spontaneous-lick slope beyond the null (excessC_pu)
  hit - miss axis (ssl-hitmiss-axis-passive 135, >= 4 hits and misses at both stages, harmonised scaling; per neuron):
    hm_W        passive whisker post - pre projection change beyond the null (shift_excess_dWP)
    hm_A        passive auditory change (shift_excess_dAP)
    hm_WA       whisker - auditory change (shift_excess_dWAP)
Population: learners rule on day 0, all expert sessions; session = unit. Areas: an area group is analysed for a cohort when the cohort has
>= MIN_SESS sessions at both stages (as ssl-prelick-convergence); allen_utils default order.
Statistics and classes: common.measure_stats (Wilcoxon vs 0 per cohort and stage; day 0 vs expert MWU / Welch; R+ vs R- per stage;
cohort x stage mouse-level permutation; class per cohort). Per area: BH over areas of the day-0 p (q_d0) within measure x cohort; a cell is
called learning-specific / stronger with learning only on the uncorrected tests and flagged when q_d0 >= 0.05.
Output: tables/003_sessions.csv, 003_stats.csv; figures/003_whole_brain.{png,pdf,svg}, 003_areas.{png,pdf,svg}
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C

MIN_SESS = 3
MEAS = [("rl_WH_raw", "Reward-lick CD: WH drift (raw)"), ("rl_WH", "Reward-lick CD: WH drift beyond null"),
        ("rl_WHSL", "Reward-lick CD: WH − SL drift beyond null"),
        ("hm_W", "Hit − miss axis: passive W post − pre"), ("hm_A", "Hit − miss axis: passive A post − pre"),
        ("hm_WA", "Hit − miss axis: W − A post − pre")]
CLS_COL = {"learning-specific": "#D73027", "stronger with learning": "#FC8D59", "both stages": "#4575B4", "both stages, differ": "#91BFDB",
           "day 0 only (stage n.s.)": "#FEE090", "expert only": "#BDBDBD", "none": "#FFFFFF", "n/a": "#F0F0F0"}


def load():
    rl = C.population(pd.read_csv(C.m51.WITHIN / "cd_shift_null" / "sessions_areas.csv"))
    rl = rl.rename(columns={"beta_pu": "rl_WH_raw", "excess_pu": "rl_WH", "excessC_pu": "rl_WHSL"})
    rl = rl[["session_id", "mouse_id", "cohort", "stage", "area", "rl_WH_raw", "rl_WH", "rl_WHSL"]]
    H = []
    for st_, sfx in (("learning", ""), ("expert", "_expert")):
        A = pd.read_parquet(C.HM / f"135_alignment_epochs_min2_tracked_harm{sfx}.parquet")
        A = A[A.skipped_reason.isna() & A.area.notna()].copy(); A["stage"] = st_; H.append(A)
    H = pd.concat(H).rename(columns={"reward_group": "cohort", "shift_excess_dWP": "hm_W", "shift_excess_dAP": "hm_A", "shift_excess_dWAP": "hm_WA"})
    H["area"] = H.area.replace({"All units": "whole_brain"})
    H = H[(H.stage == "expert") | H.session_id.isin(C.population(H.assign(stage="learning")).session_id)]
    H = H[["session_id", "mouse_id", "cohort", "stage", "area", "hm_W", "hm_A", "hm_WA"]]
    return rl, H


def run(rl, H, n_perm):
    rows = []
    import ephys_utilities.allen_utils.allen_utils as au
    order = ["whole_brain"] + list(au.get_area_group_custom_order())
    for D, ms in ((rl, MEAS[:3]), (H, MEAS[3:])):
        for area in [a for a in order if a in set(D.area)]:
            d = D[D.area == area]
            for m, _ in ms:
                n = d.dropna(subset=[m]).groupby(["cohort", "stage"]).size()
                ok = {c: all(n.get((c, s_), 0) >= MIN_SESS for s_ in ("learning", "expert")) for c in ("R+", "R-")}
                if area != "whole_brain" and not any(ok.values()):
                    continue
                for r in C.measure_stats(d, m, area, m, n_perm=n_perm if area == "whole_brain" else 2000):
                    if r["cohort"] in ok and not ok[r["cohort"]] and area != "whole_brain":
                        continue
                    if r["cohort"] == "R+ vs R-" and area != "whole_brain" and not all(ok.values()):
                        continue
                    rows.append(r)
    R = pd.DataFrame(rows)
    R["q_d0"] = np.nan
    for (m, c), g in R[R.level != "whole_brain"].groupby(["measure", "cohort"]):
        R.loc[g.index, "q_d0"] = C.bh(g.p_d0)
    R["flag"] = np.where((R.level != "whole_brain") & R.cls.isin(["learning-specific", "stronger with learning"]) & (R.q_d0 >= 0.05),
                         "uncorrected", "")
    return R


def short(a):
    return (a.replace(" areas", "").replace("Somatosensory-", "Ss-").replace("Lateral septal complex", "LSX").replace("Posterior parietal", "PPC")
            .replace("Retrosplenial", "RSP").replace("Hippocampus", "HPF").replace("Amygdala and hypothalamus", "Amyg.").replace("whole_brain", "Whole brain"))


def fig_whole(S, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.2), gridspec_kw=dict(wspace=0.55, hspace=1.35))
    for ax, (m, lab), l in zip(axs.ravel(), MEAS, "abcdef"):
        D = S[m][S[m].area == "whole_brain"]; Rm = R[(R.level == "whole_brain") & (R.measure == m)]
        C.dot_panel(ax, D, m, Rm, lab, l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle("Within-session changes on the two axes, day 0 vs expert (whole brain; green R+, magenta R−; open day 0, filled expert; "
                 "p: Wilcoxon vs 0, Mann–Whitney day 0 vs expert)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"003_whole_brain.{ext}", dpi=300, bbox_inches="tight")


def fig_areas(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    meas = [m for m, _ in MEAS if m != "rl_WH_raw"]
    cols = [(m, c) for m in meas for c in ("R+", "R-")]
    areas = [a for a in ["whole_brain"] + list(au.get_area_group_custom_order()) if a in set(R.level)]
    fig, ax = plt.subplots(figsize=(7.0, 0.22 * len(areas) + 1.4))
    for i, a in enumerate(areas):
        for j, (m, c) in enumerate(cols):
            r = R[(R.level == a) & (R.measure == m) & (R.cohort == c)]
            if not len(r):
                ax.add_patch(Rectangle((j, i), 1, 1, color="#F7F7F7", lw=0)); continue
            r = r.iloc[0]
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CLS_COL.get(r.cls, "w"), edgecolor="0.85", lw=0.3))
            txt = f"{r.mean_d0:+.2g}" + ("†" if r.flag == "uncorrected" else "")
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=3.8)
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(areas), 0)
    ax.set_yticks(np.arange(len(areas)) + 0.5, [short(a) for a in areas], fontsize=5)
    for tl, a in zip(ax.get_yticklabels(), areas):
        tl.set_color(gcol.get(a, "k"))
    SH = {"rl_WH": "WH drift", "rl_WHSL": "WH − SL drift", "hm_W": "whisker", "hm_A": "auditory", "hm_WA": "W − A"}
    ax.set_xticks(np.arange(len(cols)) + 0.5, [f"{SH[m]}\n{c.replace('-', '−')}" for m, c in cols], fontsize=4.6, rotation=0)
    ax.xaxis.tick_top()
    for k in (2, 6, 8):
        ax.axvline(k, color="0.3", lw=0.6)
    ax.axvline(4, color="k", lw=1.4)
    ax.text(2, -1.9, "reward-lick CD (drift beyond null)", ha="center", fontsize=6, weight="bold")
    ax.text(7, -1.9, "hit − miss axis (passive post − pre beyond null)", ha="center", fontsize=6, weight="bold")
    ax.legend(handles=[Patch(facecolor=v, edgecolor="0.6", label=k) for k, v in CLS_COL.items() if k != "n/a"], ncol=4, fontsize=4.6,
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Class of the within-session change per area group and cohort (number: day-0 mean; †: day-0 effect not significant after "
                 "BH over areas; grey: < 3 sessions at a stage)", fontsize=5.6, pad=44)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"003_areas.{ext}", dpi=300, bbox_inches="tight")


def fp(p):
    return "n/a" if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.2f}".replace("0.", "."))


def main(a):
    rl, H = load()
    S = {m: (rl if m.startswith("rl") else H) for m, _ in MEAS}
    pd.concat([rl.melt(id_vars=["session_id", "mouse_id", "cohort", "stage", "area"], var_name="measure"),
               H.melt(id_vars=["session_id", "mouse_id", "cohort", "stage", "area"], var_name="measure")]).to_csv(C.TAB / "003_sessions.csv", index=False)
    R = run(rl, H, a.n_perm)
    R.to_csv(C.TAB / "003_stats.csv", index=False)
    fig_whole(S, R); fig_areas(R)
    pd.set_option("display.width", 250)
    print(R[R.level == "whole_brain"][["measure", "cohort", "n_d0", "n_e", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(4).to_string())
    print(R[(R.level != "whole_brain")].groupby(["measure", "cohort"]).cls.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-perm", type=int, default=10000)
    main(ap.parse_args())
