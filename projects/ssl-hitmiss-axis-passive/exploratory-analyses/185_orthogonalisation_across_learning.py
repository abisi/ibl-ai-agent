"""185 -- Orthogonalisation across learning (user 2026-11-24, "5. do this"; Kingsbury et al. 2025: coding directions decorrelate
with experience). Does the alignment of the pre-task passive response with the hit - miss axis, and of the hit - miss axis with
the stimulus (whisker - auditory) axis, change from the learning day to expert days, differently in R+ and R-?
Input: 179_cd_clean / 179_cd_clean_expert (whole brain; learners on day 0; every expert session; session = unit).
Measures: angle of the passive pre response to the hit - miss axis (whisker; auditory as control) and its shifted-axis value;
  same after the task (post); cos(hit - miss axis, stimulus axis); held-out hit / miss d'.
Stats: day 0 vs expert per cohort (Mann-Whitney + Welch; different sessions), R+ vs R- per stage, cohort x stage OLS
  interaction; uncorrected.
Outputs: 185_orthogonalisation_stats.csv, figures/185_orthogonalisation_across_learning.png.
Run (haas, repo root): .venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/185_orthogonalisation_across_learning.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA.parents[1] / "ssl-paper-figures"))
import shared_passive_cd as S  # noqa: E402
from paperstyle import COLORS  # noqa: E402

MEAS = {"thW_pre": "Whisker pre, angle to axis (°)", "thW_post": "Whisker post, angle to axis (°)",
        "thA_pre": "Auditory pre, angle to axis (°)", "thW_pre_shift": "Whisker pre, shifted axes (°)",
        "cos_u_s": "cos(hit − miss axis, stimulus axis)", "dprime_loo": "Held-out hit / miss d′"}


def table(stage):
    w, a = S.whole("W", stage), S.whole("A", stage).set_index("session_id")
    return pd.DataFrame(dict(session_id=w.session_id, mouse_id=w.mouse_id, cohort=w.cohort,
                             stage="day 0" if stage == "learning" else "expert",
                             thW_pre=S.theta(w.cos_pre), thW_post=S.theta(w.cos_post), thW_pre_shift=S.theta(w.cos_pre_null),
                             thA_pre=S.theta(a.cos_pre.reindex(w.session_id).to_numpy()), cos_u_s=w.cos_u_s,
                             dprime_loo=w.dprime_loo))


def p2(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def main():
    d = pd.concat([table("learning"), table("expert")], ignore_index=True)
    rows = []
    for m in MEAS:
        for c in S.COH:
            x, y = (d[(d.cohort == c) & (d.stage == s)][m] for s in ("day 0", "expert"))
            rows.append(dict(measure=m, test=f"{c} day 0 vs expert", a=x.mean(), b=y.mean(), n_a=x.count(), n_b=y.count(), p=p2(x, y)))
        for s in ("day 0", "expert"):
            x, y = (d[(d.cohort == c) & (d.stage == s)][m] for c in S.COH)
            rows.append(dict(measure=m, test=f"{s} R+ vs R-", a=x.mean(), b=y.mean(), n_a=x.count(), n_b=y.count(), p=p2(x, y)))
        f = smf.ols(f"{m} ~ C(cohort) * C(stage)", data=d).fit()
        rows.append(dict(measure=m, test="cohort x stage (OLS)", p=(np.nan, float(f.pvalues.filter(like=":").iloc[0]))))
    st = pd.DataFrame(rows)
    st["p_nonparam"], st["p_param"] = st.p.str[0], st.p.str[1]
    st = st.drop(columns="p")
    st.to_csv(EA / "185_orthogonalisation_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(st.round(4).to_string(index=False))

    fig, axs = plt.subplots(1, len(MEAS), figsize=(13, 2.8))
    for ax, (m, lab) in zip(axs, MEAS.items()):
        for c, off in (("R+", -0.05), ("R-", 0.05)):
            g = d[d.cohort == c].groupby("stage")[m].agg(["mean", "sem"]).reindex(["day 0", "expert"])
            ax.errorbar(np.arange(2) + off, g["mean"], g["sem"], color=COLORS[c], marker="o", ms=3, lw=1, capsize=0, label=c)
        ax.set_xticks([0, 1], ["Learning\n(day 0)", "Expert"], fontsize=6)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(lab, fontsize=7)
        ax.tick_params(labelsize=6)
        r = st[st.measure == m].set_index("test")
        txt = [f"{t}: {max(r.loc[t, 'p_nonparam'], r.loc[t, 'p_param']):.3f}" for t in
               ("R+ day 0 vs expert", "R- day 0 vs expert", "day 0 R+ vs R-", "expert R+ vs R-")]
        txt.append(f"cohort x stage: {r.loc['cohort x stage (OLS)', 'p_param']:.3f}")
        ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, fontsize=4.5, va="bottom")
    axs[0].legend(fontsize=6, loc="upper right")
    fig.suptitle("Alignment with the hit − miss axis, learning day vs expert days (larger P of Mann-Whitney / Welch)", fontsize=8)
    fig.tight_layout()
    (EA / "figures").mkdir(exist_ok=True)
    fig.savefig(EA / "figures" / "185_orthogonalisation_across_learning.png", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
