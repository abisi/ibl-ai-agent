"""Figure 4 / G follow-up (user 2026-11-26: "the correlations with choice coding is good. But also do the neuron-level analysis
you suggest"), plus a movement check for panel D (user: "change in movement after stim pre vs post could explain part of the
changes").

1. Neuron level. 179 units (learners, whisker day 0, stable good + MUA) joined to the ROC table (merged roc_core, 5-35 ms, P < 0.05):
   choice-coding = whisker_choice significant (whisker hit vs whisker miss, active trials), other = not significant.
   Per session and group (>= 5 neurons per group), mean over neurons of the passive evoked change post - pre (5-35 ms,
   spikes/s): whisker dW, auditory dA, and dW - dA. Two neuron sets: all units, and whisker-responsive units only
   (whisker_active significant; controls for choice neurons being more responsive).
   Tests: within cohort, choice vs other (paired over sessions: Wilcoxon + paired t, larger P); between cohorts, the
   choice - other difference and each group (Mann-Whitney + Welch, larger P). Uncorrected.
2. Movement check. Per session: neural dW - dA (all units, as panel F) and the change post - pre of the evoked movement
   (206: DLC PC1 and whisker speed, z within session, each trial minus its -0.2..0 s mean; early 5-35 ms and late 50-300 ms;
   whisker and auditory trials). OLS dW - dA ~ cohort (+ movement changes); cohort coefficient with and without covariates.
Output: Figures/_explore/Figure4_passive_cd_<date>/neuron_choice/neuron_choice.{png,pdf,svg} + csv / txt
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_neuron_choice.py
"""
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT, axel_bisi_root  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

FS = 5
K = ["session_id", "electrode_group", "cid"]
MIN_N = 5
COH = ["R+", "R-"]
COL = {"R+": "#00B400", "R-": "#C800C8"}


def pmax2(a, b):
    return max(a, b)


def units():
    u = S._ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant"])
            r = r[r.analysis_type.isin(["whisker_choice", "whisker_active"])]
            rows.append(r.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type",
                                      values="significant", aggfunc="first").reset_index())
    r = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    r["cid"] = r.cid.astype(int)
    u = u.merge(r, on=K, how="inner")
    u["dW"] = u.ev_post_W - u.ev_pre_W
    u["dA"] = u.ev_post_A - u.ev_pre_A
    u["dWA"] = u.dW - u.dA
    u["grp"] = np.where(u.whisker_choice.astype(bool), "choice", "other")
    return u


def per_session(u):
    d = u.groupby(["session_id", "cohort", "grp"]).agg(n=("cid", "size"), dW=("dW", "mean"), dA=("dA", "mean"),
                                                       dWA=("dWA", "mean")).reset_index()
    d = d[d.n >= MIN_N]
    ok = d.groupby("session_id").grp.nunique()
    return d[d.session_id.isin(ok[ok == 2].index)]


def tests(d, v):
    w = d.pivot_table(index=["session_id", "cohort"], columns="grp", values=v).dropna().reset_index()
    w["diff"] = w.choice - w.other
    out = {}
    for c in COH:
        x = w[w.cohort == c]
        out[f"{c} choice vs other"] = pmax2(stats.wilcoxon(x.choice, x.other).pvalue, stats.ttest_rel(x.choice, x.other).pvalue)
        out[f"{c} n"] = len(x)
    for col in ("diff", "choice", "other"):
        a, b = w[w.cohort == "R+"][col], w[w.cohort == "R-"][col]
        out[f"R+ vs R- {col}"] = pmax2(stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue)
    return w, out


def ptxt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def draw(ax, w, out, ylabel, title):
    for q, c in enumerate(COH):
        x = w[w.cohort == c]
        xo, xc = 2 * q, 2 * q + 1
        for _, r in x.iterrows():
            ax.plot([xo, xc], [r.other, r.choice], color="0.87", lw=0.3, zorder=1)
        for xx, g, mfc in ((xo, "other", "white"), (xc, "choice", COL[c])):
            ax.errorbar(xx, x[g].mean(), x[g].sem(), color=COL[c], marker="o", ms=2.8, mfc=mfc, mew=0.8, lw=0.9,
                        capsize=0, zorder=3)
        ax.plot([xo, xc], [x.other.mean(), x.choice.mean()], color=COL[c], lw=0.9, zorder=2)
        ax.text(2 * q + 0.5, 1.0, f"{ptxt(out[f'{c} choice vs other'])}\nn = {out[f'{c} n']}", color=COL[c], ha="center",
                va="bottom", fontsize=FS - 1, transform=ax.get_xaxis_transform())
    v = w[["other", "choice"]].to_numpy()
    lo, hi = np.nanpercentile(v, [5, 95])
    ms = [w[w.cohort == c][g].mean() for c in COH for g in ("other", "choice")]
    lo, hi = min(lo, min(ms)), max(hi, max(ms))
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks([0, 1, 2, 3], ["Other", "Choice", "Other", "Choice"], fontsize=FS - 0.5)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_title(f"{title}\nR+ vs R− on choice − other: {ptxt(out['R+ vs R- diff'])}", fontsize=FS, pad=12)
    ax.tick_params(labelsize=FS, length=1.5)


def movement_check(u):
    s = pd.read_parquet(S.TAB / "206_passive_movement_summary.parquet")
    s = s[s.mouse_id.isin(S.learners())]
    tt = {t: ("W" if "whisker" in t else "A") for t in s.trial_type.unique()}
    s["stim"] = s.trial_type.map(tt)
    m = s[s.signal.isin(["pc1", "whisker_speed"])].pivot_table(index="session_id", columns=["signal", "stim", "epoch"],
                                                                 values=["early_bc", "late_bc"])
    cov = pd.DataFrame(index=m.index)
    for win in ("early_bc", "late_bc"):
        for sig in ("pc1", "whisker_speed"):
            for st in "WA":
                cov[f"{sig}_{win.split('_')[0]}_{st}"] = m[(win, sig, st, "post")] - m[(win, sig, st, "pre")]
    n = u.groupby(["session_id", "cohort"])[["dW", "dA", "dWA"]].mean().reset_index()
    d = n.merge(cov.reset_index(), on="session_id", how="inner").dropna()
    d["Rm"] = (d.cohort == "R-").astype(int)
    lines = [f"sessions with neural + movement data: {len(d)} (R+ {(d.Rm == 0).sum()}, R- {(d.Rm == 1).sum()})"]
    for c in [x for x in cov.columns]:
        a, b = d[d.Rm == 0][c], d[d.Rm == 1][c]
        lines.append(f"  movement change {c:26s} R+ {a.mean():+.3f}  R- {b.mean():+.3f}  MW {stats.mannwhitneyu(a, b).pvalue:.3f}"
                     f"  Welch {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f}")
    models = {"no covariate": "dWA ~ Rm",
              "+ PC1 early (W, A)": "dWA ~ Rm + pc1_early_W + pc1_early_A",
              "+ PC1 late (W, A)": "dWA ~ Rm + pc1_late_W + pc1_late_A",
              "+ PC1 early + late": "dWA ~ Rm + pc1_early_W + pc1_early_A + pc1_late_W + pc1_late_A",
              "+ whisker speed early + late": "dWA ~ Rm + whisker_speed_early_W + whisker_speed_early_A + whisker_speed_late_W + whisker_speed_late_A"}
    for k, f in models.items():
        r = smf.ols(f, d).fit()
        lines.append(f"  {k:30s} cohort (R- - R+) beta {r.params['Rm']:+.3f} spks/s, p {r.pvalues['Rm']:.4f}"
                     + "".join(f"; {v} p {r.pvalues[v]:.3f}" for v in r.params.index if v not in ("Intercept", "Rm")))
    # same for the whisker change alone
    for k, f in (("dW no covariate", "dW ~ Rm"), ("dW + PC1 early + late (W)", "dW ~ Rm + pc1_early_W + pc1_late_W")):
        r = smf.ols(f, d).fit()
        lines.append(f"  {k:30s} cohort beta {r.params['Rm']:+.3f}, p {r.pvalues['Rm']:.4f}")
    return lines


def main():
    set_nature_rc()
    u = units()
    sets = [("All neurons", u), ("Whisker-responsive neurons", u[u.whisker_active.astype(bool)])]
    fig, axes = plt.subplots(2, 3, figsize=(150 * MM, 95 * MM))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.82, wspace=0.45, hspace=0.75)
    res = []
    for i, (lab, uu) in enumerate(sets):
        d = per_session(uu)
        frac = uu.groupby("cohort").whisker_choice.mean()
        for j, (v, yl) in enumerate((("dW", "Whisker evoked change\n(spks/s, post − pre)"),
                                     ("dA", "Auditory evoked change\n(spks/s, post − pre)"),
                                     ("dWA", "Whisker − auditory\nchange (spks/s)"))):
            w, out = tests(d, v)
            draw(axes[i, j], w, out, yl, f"{lab}" if j == 0 else "")
            res.append(dict(set=lab, measure=v, **out))
        fig.text(0.01, axes[i, 0].get_position().y1 + 0.085, f"{lab}: choice-coding fraction R+ {frac.get('R+', np.nan):.2f}, "
                        f"R− {frac.get('R-', np.nan):.2f}; sessions with ≥ {MIN_N} neurons per group",
                 fontsize=FS + 0.5, ha="left", va="bottom")
    fig.suptitle("Passive evoked change in whisker choice-coding (ROC whisker hit vs miss) vs other neurons",
                 fontsize=FS + 1.5, y=0.995)
    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}" / "neuron_choice"
    out.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, "neuron_choice")
    r = pd.DataFrame(res)
    r.to_csv(out / "neuron_choice_stats.csv", index=False)
    print(r.round(4).to_string())
    lines = movement_check(u)
    (out / "movement_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("saved", out)


if __name__ == "__main__":
    main()
