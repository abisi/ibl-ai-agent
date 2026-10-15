"""170 -- Results of 169 (along vs orthogonal / gain vs rotation on the hit - miss axis) and the links between the two within-session
axes and behaviour (user 2026-10-08: "show me results before updating the figure; connect the two axes"). Diagnostic, not a
publication figure. Learners, learning day; unit = session.
A  passive whisker (and auditory) pre -> post change on the hit - miss axis, exact split
     d_along = (rho_post - rho_pre) cos_pre  +  rho_post (cos_post - cos_pre)
               gain part                         rotation part (change of angle to the axis)
   each on the real axis and as the mean over shifted axes; real - shifted per session; f_along (fraction of the change along the
   axis) real vs shifted; g (cos of the change with the pre pattern; -1 = pure gain down)
   tests: within cohort Wilcoxon vs 0; R+ vs R- Mann-Whitney and Welch t
B  links per session: hit - miss axis quantity (135 harm shift_excess_dWP: real - shifted, post - pre, whisker), reward-lick axis
   quantity (026 excess_pu: real - shifted WH slope; and excessC_pu: WH - SL), behaviour (169 dlick_W: whisker lick probability, last
   third - first third). Spearman and Pearson, pooled and per cohort; OLS y ~ x + cohort (slope p) and y ~ x * cohort (interaction p).
   Scatter + OLS line + 95 % CI band (solid if p < 0.05).
Outputs: figures/170_along_orthogonal_links.{png,pdf}; 170_stats.csv. Run (haas, repo root): python .../170_along_orthogonal_links.py
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
import statsmodels.formula.api as smf
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH = H.COL, H.COH
PL = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence/within_day/sl_stable/cd_shift_null/sessions.csv")
R = []


def wil(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan


def two(a, b):
    a, b = (np.asarray(v, float) for v in (a, b)); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def part_a(D):
    D = D.copy()
    for k in ("", "_null"):
        D[f"gain{k}"] = (D.rho_post - D.rho_pre) * D[f"cos_pre{k}"]
        D[f"rot{k}"] = D.rho_post * (D[f"cos_post{k}"] - D[f"cos_pre{k}"])
    D["drho"] = D.rho_post - D.rho_pre
    D["drho_rel"] = D.rho_post / D.rho_pre - 1
    for q in ("d_along", "gain", "rot", "f_along", "cos_pre", "cos_post"):
        D[f"{q}_exc"] = D[q] - D[f"{q}_null"]
    D["dcos"] = D.cos_post - D.cos_pre; D["dcos_exc"] = D.dcos - (D.cos_post_null - D.cos_pre_null)
    qs = ["d_along", "d_along_null", "d_along_exc", "gain", "gain_null", "gain_exc", "rot", "rot_null", "rot_exc", "d_orth",
          "f_along", "f_along_null", "f_along_exc", "cos_pre", "cos_post", "dcos", "dcos_exc", "rho_pre", "rho_post", "drho", "drho_rel", "g"]
    for s_ in ("W", "A"):
        d = D[D.stim == s_]
        for q in qs:
            v = {c: d[d.reward_group == c][q].to_numpy(float) for c in COH}
            pm, pt = two(v["R+"], v["R-"])
            R.append(dict(part="A", stim=s_, measure=q, **{f"mean_{c}": np.nanmean(v[c]) for c in COH},
                          **{f"sem_{c}": H.sem(v[c]) for c in COH}, **{f"p_vs0_{c}": wil(v[c]) for c in COH},
                          **{f"n_{c}": int(np.isfinite(v[c]).sum()) for c in COH}, p_mwu=pm, p_welch=pt))
    return D


def links(L, pairs):
    out = []
    for x, y, lab in pairs:
        d = L.dropna(subset=[x, y])
        rs, ps = stats.spearmanr(d[x], d[y]); rp, pp = stats.pearsonr(d[x], d[y])
        m1 = smf.ols(f"{y} ~ {x} + C(reward_group)", d).fit(); m2 = smf.ols(f"{y} ~ {x} * C(reward_group)", d).fit()
        pint = m2.pvalues[[k for k in m2.pvalues.index if ":" in k][0]]
        row = dict(part="B", pair=lab, x=x, y=y, n=len(d), spearman_r=rs, spearman_p=ps, pearson_r=rp, pearson_p=pp,
                   slope_given_cohort=m1.params[x], p_slope_given_cohort=m1.pvalues[x], p_x_by_cohort=pint)
        for c in COH:
            dc = d[d.reward_group == c]
            if len(dc) > 4:
                row[f"spearman_r_{c}"], row[f"spearman_p_{c}"] = stats.spearmanr(dc[x], dc[y])
                row[f"pearson_r_{c}"], row[f"pearson_p_{c}"] = stats.pearsonr(dc[x], dc[y])
        out.append(row); R.append(row)
    return out


def scatter(ax, d, x, y, xl, yl, row):
    for c in COH:
        dc = d[d.reward_group == c]
        ax.plot(dc[x], dc[y], "o", ms=3, color=COL[c], alpha=0.8, mew=0)
    dd = d.dropna(subset=[x, y]); m = smf.ols(f"{y} ~ {x}", dd).fit()
    xx = np.linspace(dd[x].min(), dd[x].max(), 50); pr = m.get_prediction(pd.DataFrame({x: xx})).summary_frame(alpha=0.05)
    ls = "-" if row["pearson_p"] < 0.05 else "--"
    ax.plot(xx, pr["mean"], color="0.2", lw=1, ls=ls); ax.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color="0.5", alpha=0.2, lw=0)
    ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
    ax.set_xlabel(xl, fontsize=6); ax.set_ylabel(yl, fontsize=6)
    ax.set_title(f"n = {row['n']}; Spearman r = {row['spearman_r']:.2f} (p = {row['spearman_p']:.3f})\nPearson r = {row['pearson_r']:.2f} "
                 f"(p = {row['pearson_p']:.3f}); given cohort p = {row['p_slope_given_cohort']:.3f}", fontsize=5.5)


def main():
    H.setup()
    D = pd.read_parquet(EA / "169_along_orthogonal.parquet")
    D = D[H.mouse_of(D).isin(H.learners())]
    D = part_a(D)
    # ---------------- links
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked_harm.parquet")
    A = A[(A.area == "All units") & A.skipped_reason.isna()][["session_id", "reward_group", "shift_excess_dWP"]]
    P = pd.read_csv(PL); P = P[(P.stage == "learning") & (P.area == "whole_brain")][["session_id", "excess_pu", "excessC_pu"]]
    Bh = D[D.stim == "W"][["session_id", "dlick_W", "plick_W", "rot_exc", "gain"]]
    L = A.merge(P, on="session_id", how="outer").merge(Bh, on="session_id", how="outer")
    L["mouse_id"] = L.session_id.str.split("_").str[0]
    L = L[L.mouse_id.isin(H.learners())]
    rg = pd.concat([A[["session_id", "reward_group"]], D[["session_id", "reward_group"]]]).drop_duplicates("session_id")
    L = L.drop(columns="reward_group").merge(rg, on="session_id", how="left")
    pairs = [("shift_excess_dWP", "excess_pu", "hit-miss vs reward-lick (WH)"), ("shift_excess_dWP", "excessC_pu", "hit-miss vs reward-lick (WH-SL)"),
             ("dlick_W", "shift_excess_dWP", "behaviour vs hit-miss"), ("dlick_W", "excess_pu", "behaviour vs reward-lick (WH)"),
             ("dlick_W", "rot_exc", "behaviour vs hit-miss rotation part")]
    rows = links(L, pairs)
    S = pd.DataFrame(R); S.to_csv(EA / "170_stats.csv", index=False)
    pd.set_option("display.width", 250)
    a = S[S.part == "A"][["stim", "measure", "mean_R+", "sem_R+", "p_vs0_R+", "mean_R-", "sem_R-", "p_vs0_R-", "n_R+", "n_R-", "p_mwu", "p_welch"]]
    print(a.round(4).to_string(index=False))
    b = S[S.part == "B"].dropna(axis=1, how="all")
    print(b.drop(columns=["part"]).round(3).to_string(index=False))
    # ---------------- figure
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    d = D[D.stim == "W"]
    for ax, cols, ttl in ((axs[0, 0], ["d_along_exc", "gain_exc", "rot_exc"], "whisker: Δ along axis = gain + rotation\n(real − shifted)"),
                          (axs[0, 1], ["drho_rel", "g"], "response size change (relative)\nand cos(change, pre pattern)"),
                          (axs[0, 2], ["f_along", "f_along_null"], "fraction of the change along the axis\nreal vs shifted"),
                          (axs[0, 3], ["cos_pre", "cos_post"], "cos(response, axis): pre, post")):
        for k, q in enumerate(cols):
            for j, c in enumerate(COH):
                v = d[d.reward_group == c][q].to_numpy(float)
                ax.errorbar(k + (j - 0.5) * 0.25, np.nanmean(v), H.sem(v), fmt="o", color=COL[c], ms=3.5, capsize=0)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=5.5, rotation=20); ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_title(ttl, fontsize=6)
    for ax, (x, y, xl, yl), row in zip(axs[1], [("shift_excess_dWP", "excess_pu", "hit − miss: real − shifted Δ(post − pre)", "reward-lick: real − shifted WH slope"),
                                               ("dlick_W", "shift_excess_dWP", "Δ whisker lick prob. (last − first third)", "hit − miss: real − shifted Δ"),
                                               ("dlick_W", "excess_pu", "Δ whisker lick prob. (last − first third)", "reward-lick: real − shifted WH slope"),
                                               ("dlick_W", "rot_exc", "Δ whisker lick prob. (last − first third)", "hit − miss rotation part (real − shifted)")],
                                      [rows[0], rows[2], rows[3], rows[4]]):
        scatter(ax, L, x, y, xl, yl, row)
    fig.tight_layout()
    (EA / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(EA / "figures" / f"170_along_orthogonal_links.{ext}", dpi=200)
    print("ALL DONE")


if __name__ == "__main__":
    main()
