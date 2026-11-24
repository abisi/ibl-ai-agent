"""181 -- Within-epoch noise floor for the pre -> post change on the hit - miss axis (user 2026-11-24: "2 can be done"; design as
Kingsbury et al. 2025 bioRxiv, across-context minus within-context split halves).
Input: 179_cd_clean (whole brain, learners as the figure; columns a_/th_{pre,post}_{h1,h2} and _null, temporal halves).
  within = mean of (pre_h2 - pre_h1) and (post_h2 - post_h1)       change between halves of the same passive epoch
  across = post_h1 - pre_h2                                        change across the task, adjacent halves, matched trial counts
  floor-corrected change = across - within; also data - shifted (each term minus its shifted-axis mean).
Stats: per cohort across vs within (Wilcoxon + paired t); R+ vs R- on (across - within) (Mann-Whitney + Welch); uncorrected.
Output: figures/181_split_half_floor.png, tables/181_split_half_floor_stats.csv (via sync_results.sh).
Run (haas, repo root): .venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/181_split_half_floor.py
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

rows, recs = [], []
for stim in ("W", "A"):
    w = S.whole(stim)
    for meas in ("th", "a"):
        for kind, sfx in (("data", ""), ("shifted", "_null")):
            g = lambda k: w[f"{meas}_{k}{sfx}"]
            within = 0.5 * ((g("pre_h2") - g("pre_h1")) + (g("post_h2") - g("post_h1")))
            across = g("post_h1") - g("pre_h2")
            recs.append(pd.DataFrame(dict(session_id=w.session_id, cohort=w.cohort, stim=stim, measure=meas, kind=kind,
                                          within=within, across=across)))
d = pd.concat(recs, ignore_index=True)
k = ["session_id", "cohort", "stim", "measure"]
dd = d[d.kind == "data"].set_index(k)[["within", "across"]] - d[d.kind == "shifted"].set_index(k)[["within", "across"]]
d = pd.concat([d, dd.reset_index().assign(kind="data - shifted")], ignore_index=True)
d["floor_corrected"] = d.across - d.within

for (stim, meas, kind), g in d.groupby(["stim", "measure", "kind"]):
    for c in S.COH:
        x = g[g.cohort == c].dropna(subset=["within", "across"])
        rows.append(dict(stim=stim, measure=meas, kind=kind, test=f"{c} across vs within", n=len(x),
                         mean_within=x.within.mean(), mean_across=x.across.mean(),
                         p_nonparam=stats.wilcoxon(x.across, x.within).pvalue, p_param=stats.ttest_rel(x.across, x.within).pvalue))
    a, b = (g[g.cohort == c].floor_corrected.dropna() for c in S.COH)
    rows.append(dict(stim=stim, measure=meas, kind=kind, test="R+ vs R- (across - within)", n=len(a) + len(b),
                     mean_within=a.mean(), mean_across=b.mean(), p_nonparam=stats.mannwhitneyu(a, b).pvalue,
                     p_param=stats.ttest_ind(a, b, equal_var=False).pvalue))
st = pd.DataFrame(rows)
(EA / "tables").mkdir(exist_ok=True)
st.to_csv(EA / "181_split_half_floor_stats.csv", index=False)
pd.set_option("display.width", 200)
print(st.round(4).to_string(index=False))

fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.6))
for i, stim in enumerate(("W", "A")):
    for j, kind in enumerate(("data", "shifted", "data - shifted")):
        ax = axs[i, j]
        g = d[(d.stim == stim) & (d.measure == "th") & (d.kind == kind)]
        for c, off in (("R+", -0.08), ("R-", 0.08)):
            x = g[g.cohort == c]
            m = [x.within.mean(), x.across.mean()]
            s = [x.within.sem(), x.across.sem()]
            ax.errorbar(np.array([0, 1]) + off, m, s, color=COLORS[c], marker="o", ms=3, lw=1, capsize=0, label=c)
        ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_xticks([0, 1], ["Within epoch\n(h2 − h1)", "Across task\n(post h1 − pre h2)"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{'Whisker' if stim == 'W' else 'Auditory'}, {kind}", fontsize=8)
        if j == 0:
            ax.set_ylabel("Δ angle to hit − miss axis (°)")
        r = st[(st.stim == stim) & (st.measure == "th") & (st.kind == kind)]
        ax.text(0.02, 0.98, "\n".join(f"{t}: p {a_:.3f} / {b_:.3f}" for t, a_, b_ in zip(r.test, r.p_nonparam, r.p_param)),
                transform=ax.transAxes, va="top", fontsize=5)
axs[0, 0].legend(fontsize=6, loc="lower right")
fig.tight_layout()
(EA / "figures").mkdir(exist_ok=True)
fig.savefig(EA / "figures" / "181_split_half_floor.png", dpi=200)
print("saved")
