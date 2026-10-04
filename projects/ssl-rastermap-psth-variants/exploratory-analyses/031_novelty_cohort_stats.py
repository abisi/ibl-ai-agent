"""R+ vs R- comparison of the novelty response (learning day), session = unit of analysis (one day-0 session per mouse).

Per session (quality_label == 'good' units; 029 results + 030 per-trial data):
  frac_C1_up / frac_C1b_up / frac_C1b_down   fraction of units with p < .05 in that direction (uncorrected)
  *_excess                                   same minus the session's shuffled-label (chance) fraction
  NI_whisker   population novelty index = mean z(whisker trials 1-5) - mean z(whisker trials 6-20), over the
               session's neurons cue-responsive on whisker T3 (AUC vs baseline > 0.6; independent of trials 1-20)
  NI_auditory  same for auditory trials (familiar stimulus, control); NI_diff = NI_whisker - NI_auditory
  z_trial1     mean z on whisker trial 1 (the only trial before any outcome has been experienced)
Tests: R+ vs R-: Mann-Whitney U (two-sided) AND Welch t-test; within cohort vs 0: Wilcoxon signed-rank AND one-sample t.
Output: <summary>/novelty_cohort_stats.csv, novelty_cohort_sessions.csv, novelty_cohort_comparison.png
"""
import argparse
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from scipy import stats                                                  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m = importlib.import_module("029_roc_rpe_v2")
COH = {"R+": "#00B400", "R-": "#C800C8"}
MIN_RESP = 3                      # min cue-responsive neurons for a session's novelty index


def frac(g, c, sign, null=False):
    s = "_null" if null else ""
    v = g[~g[f"skip_{c}"].astype(bool)]
    return float(((v[f"p{s}_{c}"] < .05) & (np.sign(v[f"auc{s}_{c}"] - .5) == sign)).mean()) if len(v) else np.nan


def main(a):
    summ = m.SUMMARY / a.tag if a.tag else m.SUMMARY
    N = pd.read_parquet(summ / "novelty_explainer_data.parquet")
    rows = []
    for sid, g in N.groupby("session_id"):
        r = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=g.cohort.iloc[0], n_units=len(g))
        for c, sign, nm in [("C1", 1, "C1_up"), ("C1b", 1, "C1b_up"), ("C1b", -1, "C1b_down")]:
            r[f"frac_{nm}"] = frac(g, c, sign)
            r[f"frac_{nm}_excess"] = r[f"frac_{nm}"] - frac(g, c, sign, null=True)
        resp = g[g.popauc_cue_W_all_T3 > 0.6]
        r["n_cue_responsive"] = len(resp)
        if len(resp) >= MIN_RESP:
            zw = resp[[f"zW{t}" for t in range(1, 21)]].to_numpy(float)
            za = resp[[f"zA{t}" for t in range(1, 21)]].to_numpy(float)
            r["NI_whisker"] = np.nanmean(zw[:, :5]) - np.nanmean(zw[:, 5:])
            r["NI_auditory"] = np.nanmean(za[:, :5]) - np.nanmean(za[:, 5:])
            r["NI_diff"] = r["NI_whisker"] - r["NI_auditory"]
            r["z_trial1"] = np.nanmean(zw[:, 0])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(summ / "novelty_cohort_sessions.csv", index=False)
    metrics = ["frac_C1_up_excess", "frac_C1b_up_excess", "frac_C1b_down_excess", "NI_whisker", "NI_auditory", "NI_diff",
               "z_trial1"]
    st = []
    for mt in metrics:
        p_, n_ = S.loc[S.cohort == "R+", mt].dropna(), S.loc[S.cohort == "R-", mt].dropna()
        row = dict(metric=mt, n_Rplus=len(p_), n_Rminus=len(n_), mean_Rplus=p_.mean(), mean_Rminus=n_.mean(),
                   median_Rplus=p_.median(), median_Rminus=n_.median())
        if len(p_) > 2 and len(n_) > 2:
            row["p_mannwhitney"] = stats.mannwhitneyu(p_, n_, alternative="two-sided").pvalue
            row["p_welch"] = stats.ttest_ind(p_, n_, equal_var=False).pvalue
        for coh, v in [("Rplus", p_), ("Rminus", n_)]:
            if len(v) > 2:
                row[f"p_wilcoxon_vs0_{coh}"] = stats.wilcoxon(v).pvalue
                row[f"p_ttest_vs0_{coh}"] = stats.ttest_1samp(v, 0).pvalue
        st.append(row)
    T = pd.DataFrame(st); T.to_csv(summ / "novelty_cohort_stats.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(4).to_string())
    labels = {"frac_C1_up_excess": "C1 up: fraction - chance", "frac_C1b_up_excess": "C1b up (decrement): fraction - chance",
              "frac_C1b_down_excess": "C1b down (increment): fraction - chance",
              "NI_whisker": "novelty index, whisker\nz(trials 1-5) - z(trials 6-20)", "NI_auditory": "novelty index, auditory (control)",
              "NI_diff": "novelty index whisker - auditory", "z_trial1": "cue response on whisker trial 1 (z)"}
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 4.2))
    rng = np.random.default_rng(0)
    for ax, mt in zip(axs, metrics):
        for k, coh in enumerate(COH):
            v = S.loc[S.cohort == coh, mt].dropna()
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=COH[coh], alpha=0.6)
            ax.errorbar(k + 0.3, v.mean(), v.std() / np.sqrt(max(len(v), 1)), color="k", marker="_", ms=14, capsize=3)
        ax.axhline(0, color="k", lw=0.5)
        r = T.set_index("metric").loc[mt]
        ax.set_title(f"{labels[mt]}\nMWU p={r.get('p_mannwhitney', np.nan):.3g}, Welch p={r.get('p_welch', np.nan):.3g}",
                     fontsize=7.5)
        ax.set_xticks([0, 1], [f"R+ (n={int(r.n_Rplus)})", f"R- (n={int(r.n_Rminus)})"], fontsize=8)
    fig.suptitle("Novelty, R+ vs R- (learning day; one point = one session = one mouse; bar = mean +- SEM)", fontsize=10)
    fig.tight_layout(); fig.savefig(summ / "novelty_cohort_comparison.png", dpi=140); plt.close(fig)
    print("saved", summ / "novelty_cohort_comparison.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="learning")
    main(ap.parse_args())
