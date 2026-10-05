"""153 -- Stimulus exposure as a control (user 2026-10-05: "R- also receive more auditory trials ... add to the report ... contrast
that exposure affects shrinkage along sensory axes but not choice axes").
R- sessions are longer, so R- mice receive more active whisker AND auditory trials, in the same proportion. If exposure (e.g.
adaptation) produced the post-task changes, it should (i) affect the auditory response too and (ii) scale with the number of
trials within each cohort. Per session (whole brain, shared tracked stable units, the 146 / 135 sessions):
  exposure      nW, nA = active whisker / auditory trials in the analysed active epoch (132.session_trials), fraction nW / (nW + nA)
  sensory axes  shrinkage of each passive response along its own passive-pre pattern (state space, 146): dyW = whisker displacement
                on the whisker-pattern axis, dyA = auditory displacement on the auditory-pattern axis (y orthogonalised to the choice axis)
  choice axis   dx_W, dx_WA = whisker and whisker - auditory displacement along the choice axis (state space); lick-axis excess over the
                linear-shift null (135): whisker raw cosine, whisker - auditory projection
Tests: counts R+ vs R- (Mann-Whitney | Welch); within cohort, each measure vs nW and nA (Spearman | Pearson); OLS measure ~ R- (+ z(nW)
+ z(nA)) for the cohort coefficient with and without exposure. Session = unit; uncorrected.
Figure: figures/publication/153_exposure_control.{png,pdf,svg}: a trial counts; b shrinkage along the sensory axes vs nW; c choice-axis
measures vs nW (scatter, OLS line + 95% CI band, solid if p < 0.05, per cohort); d R- coefficient +- 95% CI without / with exposure.
Stats: 153_stats.csv. Run (haas, repo root): python .../153_exposure_control.py
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
SENS = [("dyW", "whisker on whisker axis"), ("dyA", "auditory on auditory axis")]
CHOICE = [("dx_W", "whisker along choice axis"), ("dx_WA", "W - A along choice axis"), ("lick_WR", "lick axis, whisker raw cos\n(excess)"),
          ("lick_WAP", "lick axis, W - A projection\n(excess)")]


def load():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    D = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    D = D[D.skipped_reason.isna() & (D.area == "whole_brain") & (D.unit_set == "stable") & (D.response == "epochbase")]
    cnt = []
    for sid in D.session_id.unique():
        a = M132.session_trials(sid, st, tt, T)["active"]
        cnt.append(dict(session_id=sid, nW=int((a.trial_type == "whisker_trial").sum()), nA=int((a.trial_type == "auditory_trial").sum())))
    X = pd.DataFrame(cnt).merge(D[["session_id", "mouse_id", "reward_group", "ss_pre_W_x", "ss_post_W_x", "ss_pre_A_x", "ss_post_A_x",
                                   "ss_pre_W_y", "ss_post_W_y", "ss3_pre_A_y", "ss3_post_A_y"]], on="session_id")
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    A = A[(A.area == "All units") & A.skipped_reason.isna()].rename(columns={"shift_excess_dWR": "lick_WR", "shift_excess_dWAP": "lick_WAP"})
    X = X.merge(A[["session_id", "lick_WR", "lick_WAP"]], on="session_id", how="left")
    X["dx_W"] = X.ss_post_W_x - X.ss_pre_W_x; X["dx_A"] = X.ss_post_A_x - X.ss_pre_A_x; X["dx_WA"] = X.dx_W - X.dx_A
    X["dyW"] = X.ss_post_W_y - X.ss_pre_W_y; X["dyA"] = X.ss3_post_A_y - X.ss3_pre_A_y
    X["fracW"] = X.nW / (X.nW + X.nA); X["Rm"] = (X.reward_group == "R-").astype(int)
    for c in ("nW", "nA"):
        X[c + "z"] = (X[c] - X[c].mean()) / X[c].std()
    return X


def corr_panel(ax, X, y, lab, rows, panel):
    import statsmodels.api as sm
    for c in COH:
        g = X[X.reward_group == c].dropna(subset=[y])
        ax.plot(g.nW, g[y], "o", ms=2.6, color=COL[c], alpha=0.55, mew=0)
        rs, ps = stats.spearmanr(g.nW, g[y]); rp, pp_ = stats.pearsonr(g.nW, g[y])
        fit = sm.OLS(g[y], sm.add_constant(g.nW)).fit()
        xx = np.linspace(g.nW.min(), g.nW.max(), 50)
        pr = fit.get_prediction(sm.add_constant(xx)).summary_frame(alpha=0.05)
        ax.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color=COL[c], alpha=0.15, lw=0)
        ax.plot(xx, pr["mean"], color=COL[c], lw=1, ls="-" if pp_ < 0.05 else (0, (3, 2)))
        rows.append(dict(panel=panel, measure=y, cohort=c, n=len(g), x="nW", spearman=rs, p_nonparam=ps, pearson=rp, p_param=pp_))
        rsA, psA = stats.spearmanr(g.nA, g[y]); rpA, ppA = stats.pearsonr(g.nA, g[y])
        rows.append(dict(panel=panel, measure=y, cohort=c, n=len(g), x="nA", spearman=rsA, p_nonparam=psA, pearson=rpA, p_param=ppA))
    t = [f"{c}: rho {r['spearman']:+.2f} ({H.pnum(r['p_nonparam'])}|{H.pnum(r['p_param'])})" for c in COH for r in rows[-4:]
         if r["cohort"] == c and r["x"] == "nW"]
    ax.set_title(lab.replace("\n", " ") + "\n" + "\n".join(t), fontsize=5.2)
    ax.axhline(0, color="0.8", lw=0.4)
    ax.set_xlabel("active whisker trials", fontsize=5.5)


def main():
    import statsmodels.formula.api as smf
    H.setup()
    X = load()
    rows, ols = [], []
    for c in ("nW", "nA", "fracW"):
        a_, b_ = X[X.Rm == 0][c], X[X.Rm == 1][c]
        mw, we = H.unpaired(a_.to_numpy(float), b_.to_numpy(float))
        rows.append(dict(panel="counts", measure=c, cohort="R+ vs R-", n=len(X), x="", spearman=np.nan, p_nonparam=mw, p_param=we,
                         mean_rp=a_.mean(), mean_rm=b_.mean(), median_rp=a_.median(), median_rm=b_.median()))
    fig = plt.figure(figsize=(7.4, 4.6))
    # a counts
    ax = fig.add_axes([0.06, 0.58, 0.13, 0.27])
    for j, c in enumerate(("nW", "nA")):
        for k, coh in enumerate(COH):
            v = X[X.reward_group == coh][c].to_numpy(float)
            x0 = j * 2.4 + k
            ax.plot(x0 + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, "o", ms=2, color=COL[coh], alpha=0.5, mew=0)
            ax.errorbar(x0, np.mean(v), H.sem(v), fmt="o", ms=3.5, color=COL[coh], lw=1, capsize=0)
    ax.set_xticks([0.5, 2.9]); ax.set_xticklabels(["whisker", "auditory"], fontsize=5.5); ax.set_ylabel("active trials", fontsize=6)
    fr = [r for r in rows if r["measure"] == "fracW"][0]
    ax.set_title(f"R- get more of both\n(whisker fraction R+ {fr['mean_rp']:.2f}, R- {fr['mean_rm']:.2f};\np = {H.pnum(fr['p_nonparam'])}|{H.pnum(fr['p_param'])})", fontsize=5.3)
    fig.text(0.01, 0.93, "a", fontsize=8, weight="bold")
    # b sensory-axis shrinkage
    for j, (y, lab) in enumerate(SENS):
        ax = fig.add_axes([0.28 + j * 0.2, 0.58, 0.14, 0.27])
        corr_panel(ax, X, y, lab, rows, "sensory axes")
        if j == 0:
            ax.set_ylabel("post - pre (z)", fontsize=6)
    fig.text(0.24, 0.93, "b  shrinkage along the sensory axes", fontsize=7, weight="bold")
    # c choice-axis measures
    for j, (y, lab) in enumerate(CHOICE):
        ax = fig.add_axes([0.06 + j * 0.18, 0.1, 0.13, 0.27])
        corr_panel(ax, X, y, lab, rows, "choice axis")
        if j == 0:
            ax.set_ylabel("post - pre (z) / excess", fontsize=6)
    fig.text(0.01, 0.46, "c  choice-axis changes", fontsize=7, weight="bold")
    # d covariate model
    ax = fig.add_axes([0.855, 0.1, 0.12, 0.27])
    meas = ["dyW", "dyA", "dx_W", "dx_WA", "lick_WR", "lick_WAP"]
    for i, y in enumerate(meas):
        g = X.dropna(subset=[y]); sd = g[y].std()
        for k, (form, mk, lab) in enumerate(((f"{y} ~ Rm", "o", "cohort only"), (f"{y} ~ Rm + nWz + nAz", "s", "+ trial counts"))):
            f = smf.ols(form, g).fit(); ci = f.conf_int().loc["Rm"]
            ax.errorbar(f.params.Rm / sd, i + (0.15 if k else -0.15), xerr=[[(f.params.Rm - ci[0]) / sd], [(ci[1] - f.params.Rm) / sd]], fmt=mk,
                        ms=3, color="k" if k else "0.55", lw=0.8, capsize=0, mfc="white" if k else "0.55", label=lab if i == 0 else None)
            ols.append(dict(panel="covariate model", measure=y, cohort="R- coefficient", n=len(g), x="cohort only" if k == 0 else "+ nW, nA",
                            spearman=f.params.Rm, p_nonparam=np.nan, p_param=f.pvalues.Rm,
                            p_nW=f.pvalues.get("nWz", np.nan), p_nA=f.pvalues.get("nAz", np.nan)))
    ax.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(len(meas))); ax.set_yticklabels(["W, W axis", "A, A axis", "W, choice", "W-A, choice", "lick W", "lick W-A"], fontsize=5)
    ax.invert_yaxis(); ax.set_xlabel("R- minus R+ (SD)", fontsize=5.5); ax.legend(frameon=False, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("cohort effect without / with\nexposure covariates (95% CI)", fontsize=5.5)
    fig.text(0.815, 0.46, "d", fontsize=8, weight="bold")
    fig.suptitle("Stimulus exposure: it scales the shrinkage of the sensory responses, not their position on the choice axis "
                 "(whole brain, stable units; dots = sessions)", fontsize=6.5)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"153_exposure_control.{ext}", dpi=300)
    R = pd.DataFrame(rows + ols)
    R.to_csv(EA / "153_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
