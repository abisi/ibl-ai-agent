"""184 -- Results of 183 (subspace alignment of passive responses) for learners on the learning day and for expert sessions.
Per stimulus (W, A) and subspace (task, decision [data - shifted], stimulus): passive pre / post cross-validated energy fraction;
pre vs post per cohort (Wilcoxon + paired t), R+ vs R- on the change (Mann-Whitney + Welch); within-epoch floor (halves:
within = h2 - h1, across = post h1 - pre h2); day 0 vs expert pre-task values per cohort. Uncorrected (exploratory).
Outputs: 184_subspace_summary_stats.csv, figures/184_subspace_summary.png.
Run (haas, repo root): .venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/184_subspace_summary.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA.parents[1] / "ssl-paper-figures"))
import shared_passive_cd as S  # noqa: E402
from paperstyle import COLORS  # noqa: E402

SUB = {"task": "Task subspace (hit, miss; K=3)", "dec": "Decision subspace (hit − miss; K=2), data − shifted",
       "stim": "Stimulus subspace (W − A; K=2)"}


def load(stage):
    d = pd.read_parquet(S.TAB / f"183_subspace_alignment{'' if stage == 'learning' else '_' + stage}.parquet")
    d = d[d.skipped_reason.isna()].rename(columns={"reward_group": "cohort"})
    d = d[d.session_id.isin(S.whole("W", stage).session_id)]         # same sessions as 179 / the figure (learners on day 0)
    # a fraction is unstable when the passive response energy (cross-validated denominator) is ~0 (e.g. tiny auditory
    # responses): values outside [-0.5, 1.5] are set to NaN and counted
    fcols = [c for c in d.columns if c.startswith("f_")]
    bad = (d[fcols] < -0.5) | (d[fcols] > 1.5)
    print(f"[{stage}] unstable fractions set to NaN: {int(bad.values.sum())} of {bad.size}")
    d[fcols] = d[fcols].mask(bad)
    for part in ("", "_h1", "_h2"):
        for e in ("pre", "post"):
            d[f"f_decx_{e}{part}"] = d[f"f_dec_{e}{part}"] - d[f"f_dec_{e}{part}_null"]
    return d


def p2(a, b, paired=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        ok = np.isfinite(a) & np.isfinite(b)
        return stats.wilcoxon(a[ok], b[ok]).pvalue, stats.ttest_rel(a[ok], b[ok]).pvalue
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def main():
    L, E = load("learning"), load("expert")
    rows = []
    for stim in ("W", "A"):
        for sub, col in (("task", "f_task"), ("dec", "f_decx"), ("stim", "f_stim")):
            g = L[L.stim == stim]
            for c in S.COH:
                x = g[g.cohort == c]
                rows.append(dict(stim=stim, sub=sub, test=f"{c} pre vs post", a=x[f"{col}_pre"].mean(), b=x[f"{col}_post"].mean(),
                                 n=len(x), p=p2(x[f"{col}_pre"], x[f"{col}_post"], True)))
                within = 0.5 * ((x[f"{col}_pre_h2"] - x[f"{col}_pre_h1"]) + (x[f"{col}_post_h2"] - x[f"{col}_post_h1"]))
                across = x[f"{col}_post_h1"] - x[f"{col}_pre_h2"]
                rows.append(dict(stim=stim, sub=sub, test=f"{c} across vs within (halves)", a=within.mean(), b=across.mean(),
                                 n=len(x), p=p2(across, within, True)))
            ch = g[f"{col}_post"] - g[f"{col}_pre"]
            rows.append(dict(stim=stim, sub=sub, test="R+ vs R- change", a=ch[g.cohort == "R+"].mean(), b=ch[g.cohort == "R-"].mean(),
                             n=len(g), p=p2(ch[g.cohort == "R+"], ch[g.cohort == "R-"])))
            ge = E[E.stim == stim]
            for c in S.COH:
                rows.append(dict(stim=stim, sub=sub, test=f"{c} pre: day 0 vs expert", a=g[g.cohort == c][f"{col}_pre"].mean(),
                                 b=ge[ge.cohort == c][f"{col}_pre"].mean(), n=len(ge[ge.cohort == c]),
                                 p=p2(g[g.cohort == c][f"{col}_pre"], ge[ge.cohort == c][f"{col}_pre"])))
            rows.append(dict(stim=stim, sub=sub, test="expert pre R+ vs R-", a=ge[ge.cohort == "R+"][f"{col}_pre"].mean(),
                             b=ge[ge.cohort == "R-"][f"{col}_pre"].mean(), n=len(ge),
                             p=p2(ge[ge.cohort == "R+"][f"{col}_pre"], ge[ge.cohort == "R-"][f"{col}_pre"])))
    st = pd.DataFrame(rows)
    st["p_nonparam"], st["p_param"] = st.p.str[0], st.p.str[1]
    st = st.drop(columns="p")
    st.to_csv(EA / "184_subspace_summary_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(4).to_string(index=False))

    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    for i, stim in enumerate(("W", "A")):
        for j, (sub, col) in enumerate((("task", "f_task"), ("dec", "f_decx"), ("stim", "f_stim"))):
            ax = axs[i, j]
            for c, off in (("R+", -0.05), ("R-", 0.05)):
                for st_, x0, mk in (("learning", 0, "o"), ("expert", 2.2, "s")):
                    d = (L if st_ == "learning" else E)
                    d = d[(d.stim == stim) & (d.cohort == c)]
                    xs = np.array([x0, x0 + 1]) + off
                    m = [d[f"{col}_pre"].mean(), d[f"{col}_post"].mean()]
                    s = [d[f"{col}_pre"].sem(), d[f"{col}_post"].sem()]
                    ax.errorbar(xs, m, s, color=COLORS[c], marker=mk, ms=3, lw=1, capsize=0,
                                mfc=COLORS[c] if st_ == "learning" else "white", label=f"{c} {st_}" if i == j == 0 else None)
            ax.set_xticks([0, 1, 2.2, 3.2], ["Pre", "Post\nday 0", "Pre", "Post\nexpert"], fontsize=6)
            ax.set_title(f"{'Whisker' if stim == 'W' else 'Auditory'}: {SUB[sub]}", fontsize=6)
            ax.axhline(0, color="0.6", lw=0.5, ls=":")
            if j == 0:
                ax.set_ylabel("Fraction of passive response\nin subspace (cross-validated)", fontsize=6)
            ax.tick_params(labelsize=6)
    axs[0, 0].legend(fontsize=5)
    fig.tight_layout()
    (EA / "figures").mkdir(exist_ok=True)
    fig.savefig(EA / "figures" / "184_subspace_summary.png", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
