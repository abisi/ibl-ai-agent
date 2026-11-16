"""Report of ssl-rl-behaviour-model (v8, 2026-10-24; skills/project-report, report_lib). Rebuilt around the lean final model with
stimulus-specific learning rates (027 lean-abs-stim, mouse-level 025S): whisker learning rates are similar in R+ and R-; the cohorts differ
in how far the whisker value has to move and in satiety, not in learning speed. Supersedes the 2026-10-23 version (fast R- alpha- was a
local optimum of the cohort fit).
Inputs: tables/011_sequences.parquet; fits/023c_fits.csv, 027{,b,c,p,q}_fits.csv; fits/025S_hier_mice.csv, 025S_hier_pop.csv,
tables/025S_recovery_summary.csv; tables/028_025S_mouse_metrics.csv, 028_025S_stats.csv; tables/029_figure_numbers.csv;
figures/pub_v8 (029_pub_figures_v8.py).
Run (haas, after 027*, 025S, 028, 029): cd ~/code/ibl-ai-agent/projects/ssl-rl-behaviour-model/exploratory-analyses && ../../../.venv/bin/python ../report/build_report.py
Render locally: bash projects/ssl-rl-behaviour-model/report/render.sh
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(PROJ / "exploratory-analyses")]
from report_lib import Report  # noqa: E402
import common as C  # noqa: E402
import rl7_model as M  # noqa: E402

OUT = C.ROOT / "report"
PUB = C.FIG / "pub_v8"
TAG = "025S"
FINAL = "lean-abs-stim"
R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = num
num = lambda *a, **k: _num(*a, **k).replace("-", "−")
R.register([
    ("fig-beh", "Figure 1"), ("fig-model", "Figure 2"), ("fig-fit", "Figure 3"), ("fig-ex", "Figure 4"), ("fig-speed", "Figure 5"),
    ("fig-prof", "Figure 6"), ("fig-mouse", "Figure 7"),
    ("eq-learn", "Eq. (1)"), ("eq-fa", "Eq. (2)"), ("eq-thirst", "Eq. (3)"), ("eq-choice", "Eq. (4)"), ("eq-hmm", "Eq. (5)"),
    ("eq-lik", "Eq. (6)"), ("eq-prior", "Eq. (7)"), ("eq-tau", "Eq. (8)"),
    ("tbl-par", "Table 1"), ("tbl-mouse", "Table 2"),
    ("fig-s-start", "Figure S1"), ("fig-s-sess", "Figure S2"), ("fig-s-rec", "Figure S3"),
    ("tbl-s-n", "Table S1"), ("tbl-s-cmp", "Table S2"), ("tbl-registry", "Table S3"),
])
SRC = {"nums": C.TAB / "029_figure_numbers.csv", "f23c": C.FITS / "023c_fits.csv", "f27": C.FITS / "027_fits.csv",
       "f27b": C.FITS / "027b_fits.csv", "f27c": C.FITS / "027c_fits.csv", "f27d": C.FITS / "027d_fits.csv", "f27p": C.FITS / "027p_fits.csv", "f27q": C.FITS / "027q_fits.csv",
       "hm": C.FITS / f"{TAG}_hier_mice.csv", "hp": C.FITS / f"{TAG}_hier_pop.csv", "rec": C.TAB / f"{TAG}_recovery_summary.csv",
       "mm": C.TAB / f"028_{TAG}_mouse_metrics.csv", "ms": C.TAB / f"028_{TAG}_stats.csv"}
SRC = {k: v for k, v in SRC.items() if v.exists()}
FROZEN = dict(zip(SRC, Report.snapshot([SRC[k] for k in SRC], OUT, tag=date.today().isoformat() + "-v8")))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}
NUMS = T["nums"].set_index("key").value
COH = ("R+", "R-")
cs_ = lambda c: c.replace("-", "−")


def nv(k, default=np.nan):
    return float(NUMS[k]) if k in NUMS else default


FITS = pd.concat([T["f23c"].assign(variant=lambda e: "023c-" + e.model.str.split("-").str[2])]
                 + [T[k] for k in ("f27", "f27b", "f27c", "f27d", "f27p", "f27q") if k in T], ignore_index=True)
FITS = FITS.drop_duplicates(["variant", "cohort_set"], keep="last").set_index(["variant", "cohort_set"])
FIN = FITS.loc[FINAL]
MODEL = FIN.model.iloc[0]
D = pd.read_parquet(C.TAB / "011_sequences.parquet")
SIZES = D.groupby("reward_group").agg(mice=("mouse_id", "nunique"), sessions=("session_id", "nunique"), trials=("session_id", "size"))
MM = T["mm"]
MS = T["ms"].set_index("measure")
REC = T["rec"]
RSP = REC[REC.scenario == "map"].dropna(subset=["spearman"]).set_index(["param", "cohort"]).spearman
MIX = REC[REC.scenario == "mixed_rminus"].iloc[0]

PLAB = {"alpha_pos": r"$\alpha_A$", "alpha_neg": r"$\alpha_W$", "beta": r"$\beta$", "c": "$c$", "s_B": "$s_B$", "mu_rho": r"$\mu_\rho$",
        "theta": r"$\theta$", "sd_rho": r"$\sigma_\rho$", "tau_a": r"$\tau_a$", "w_N0": "$w_{N,0}$", "alpha_N": r"$\alpha_N$", "g_N": "$g_N$",
        "h0": "$h_0$", "h1": "$h_1$", "h_back": "$h_b$", "kappa": r"$\kappa$", "sd_u": r"$\sigma_u$", "d_ctx": "$d_{ctx}$", "g": "$g$"}
PDESC = {"alpha_pos": "auditory learning rate", "alpha_neg": "whisker learning rate", "beta": "value gain", "c": "lick cost (negative go bias)",
         "mu_rho": "mean log thirst decline per reward", "sd_rho": "s.d. of log thirst decline across sessions",
         "s_B": "thirst-driven go bias, gain", "theta": "thirst level at which the go bias switches off (R+: fixed, never reached)",
         "tau_a": "water absorption time constant (trials)", "w_N0": "initial no-stimulus value", "alpha_N": "no-stimulus learning rate",
         "g_N": "auditory share of the no-stimulus value", "h0": "disengagement hazard, intercept", "h1": "disengagement hazard, satiety slope",
         "h_back": "re-engagement, logit per trial", "kappa": "thirst exponent of the value term", "sd_u": "s.d. of the session lick offset",
         "d_ctx": "step of the no-stimulus value at whisker onset"}
VLAB = {FINAL: "final: stimulus-specific α (α_A, α_W)", "lean-abs": "valence-specific α (α+, α−)", "lean-inst-stim": "no water-absorption lag",
        "lean-abs-gw": "no-stimulus value follows whisker value (no step)", "lean-abs-hom": "reward scaled by thirst (homeostatic)",
        "lean-abs-gw-hom": "both of the above", "lean-abs-stim-hom": "final + reward scaled by thirst",
        "lean-abs-stim-gw-hom": "final + thirst-scaled reward + FA follows whisker", "min-dis-stim": "minimal: go bias + linear satiety + HMM", "min-stim": "minimal, no engagement HMM",
        "023c-abs": "previous report's fit (local optimum)"}


def param_table():
    rows = []
    mouse = {"alpha_pos", "alpha_neg", "beta", "mu_rho"}
    H = T["hm"]
    for p in [q for q in M.free_params(MODEL) if q != "g"]:
        r = {"parameter": PLAB.get(p, p), "meaning": PDESC.get(p, ""), "level": "mouse" if p in mouse else "cohort"}
        for c in COH:
            v = FIN.loc[c, f"p_{p}"]
            fixed = (c == "R+" and p == "theta")
            r[f"{cs_(c)} cohort fit"] = f"{v:.3g}" + (" (fixed)" if fixed else "")
            r[f"{cs_(c)} mouse median"] = f"{H[H.cohort_set == c][f'p_{p}'].median():.3g}" if p in mouse else ""
        rows.append(r)
    return pd.DataFrame(rows)


def mouse_table():
    lab = {"tau_W": "τ_W, licked whisker trials (Eq. 8)", "p_beta": "β", "rewards_to_half_thirst": "rewards to halve thirst",
           "V_W_end_day0": "V_W end of day 0", "V_W_end_day1": "V_W end of day 1", "V_W_end_day2": "V_W end of day 2",
           "licked_whisker_to_half": "licked whisker trials to |V_W| = 0.5 (censored)"}
    f = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
    n = f"{(MM.reward_group == 'R+').sum()} / {(MM.reward_group == 'R-').sum()}"
    rows = [{"measure": "α_W (whisker learning rate)", "R+ median": f"{nv('fig5_alpha_W_median_R+'):.4f}", "R− median": f"{nv('fig5_alpha_W_median_R-'):.4f}",
             "n R+ / R−": n, "MWU p": f(nv("fig5_alpha_W_p_mwu")), "Welch p": f(nv("fig5_alpha_W_p_welch"))},
            {"measure": "α_A (auditory learning rate)", "R+ median": f"{nv('fig5_alpha_A_median_R+'):.4f}", "R− median": f"{nv('fig5_alpha_A_median_R-'):.4f}",
             "n R+ / R−": n, "MWU p": f(nv("fig5_alpha_A_p_mwu")), "Welch p": f(nv("fig5_alpha_A_p_welch"))}]
    for q, s in MS.iterrows():
        if q in lab:
            rows.append({"measure": lab[q], "R+ median": f"{s.median_rplus:.3g}", "R− median": f"{s.median_rminus:.3g}",
                         "n R+ / R−": f"{int(s.n_rplus)} / {int(s.n_rminus)}", "MWU p": f(s.p_mwu), "Welch p": f(s.p_welch)})
    return pd.DataFrame(rows)


def cmp_table():
    rows = []
    for v, lab in VLAB.items():
        if v not in FITS.index.get_level_values(0):
            continue
        e = FITS.loc[v]
        r = {"variant": lab, "model": e.model.iloc[0].replace("+", " +")}
        for c in COH:
            r[f"k {cs_(c)}"] = int(e.loc[c, "k"]) if c in e.index else ""
            r[f"−LL {cs_(c)}"] = f"{e.loc[c, 'nll']:.0f}" if c in e.index else ""
            r[f"ΔBIC {cs_(c)}"] = f"{e.loc[c, 'bic'] - FIN.loc[c, 'bic']:+.0f}" if c in e.index else ""
        rows.append(r)
    return pd.DataFrame(rows)


def sizes_table():
    t = SIZES.reset_index().rename(columns={"reward_group": "cohort"})
    t["cohort"] = t.cohort.map(cs_)
    t["days"] = f"{int(D.day.min())} to {int(D.day.max())}"
    return t


def dbic(v, c):
    return FITS.loc[(v, c), "bic"] - FIN.loc[c, "bic"] if (v, c) in FITS.index else np.nan


def prof(c, a):
    return nv(f"fig6_profile_{c}_{a:g}")


def fe(kind, c, d, w="start"):
    """whisker - false-alarm lick rate at session start / end (fig 6): kind = data, final, fast (whisker rate fixed at 0.2)"""
    return nv(f"fig6_{kind}_excess_{c}_day{d}_{w}")


def build():
    rp, rm = "R+", "R-"
    ex = {w: {c: {d: nv(f"fig5_excess_{c}_day{d}_{w}") for d in (0, 1, 2)} for c in COH} for w in ("start", "end")}
    fast = {w: {c: {d: nv(f"fig6_fast_excess_{c}_day{d}_{w}") for d in (0, 1, 2)} for c in COH} for w in ("start", "end")}
    old_an = FITS.loc[("023c-abs", rm), "p_alpha_neg"]
    gw = "lean-abs-gw" in FITS.index.get_level_values(0)
    hom = "lean-abs-hom" in FITS.index.get_level_values(0)
    rel = lambda v, c: dbic(v, c) - dbic("lean-abs", c)       # relative to the same (valence-specific) base
    gw_txt = (f"On the valence-specific base, replacing the context step by a no-stimulus value that follows the whisker value changes BIC by "
              f"{num('d_gw_rp', rel('lean-abs-gw', rp), '{:+.0f}')} / {num('d_gw_rm', rel('lean-abs-gw', rm), '{:+.0f}')} (R+ / R−; the R− optimum then has α− = "
              f"{num('gw_an', FITS.loc[('lean-abs-gw', rm), 'p_alpha_neg'], '{:.2f}')}, a worse fit than the slow one). ") if gw else ""
    hom_txt = (f"Scaling the reward by thirst (homeostatic RL, no extra parameter) improves BIC by {num('d_hom_rp', -rel('lean-abs-hom', rp), '{:.0f}')} / "
               f"{num('d_hom_rm', -rel('lean-abs-hom', rm), '{:.0f}')} on the same base, with a slow R− rate (α− = {num('hom_an', FITS.loc[('lean-abs-hom', rm), 'p_alpha_neg'], '{:.4f}')}). ") if hom else ""
    if ("lean-abs-stim-hom", rp) in FITS.index:
        hom_txt += (f"Combined with stimulus-specific rates it changes BIC relative to the final model by {num('d_sh_rp', dbic('lean-abs-stim-hom', rp), '{:+.0f}')} / "
                    f"{num('d_sh_rm', dbic('lean-abs-stim-hom', rm), '{:+.0f}')} (R− α_W = {num('sh_an', FITS.loc[('lean-abs-stim-hom', rm), 'p_alpha_neg'], '{:.4f}')}, "
                    f"R+ α_W = {num('sh_ap', FITS.loc[('lean-abs-stim-hom', rp), 'p_alpha_neg'], '{:.4f}')}). ")
    else:
        hom_txt += "Its combination with stimulus-specific rates is being fitted. "
    md = rf"""---
title: "Whisker learning in R+ and R− mice: a reinforcement-learning model with thirst and engagement"
author: "Axel Bisi (with Claude)"
date: "{date.today().isoformat()}"
toc: true
---

**Data**: {num('n_mice', int(SIZES.mice.sum()))} mice ({num('n_mice_rp', int(SIZES.loc[rp, 'mice']))} R+, {num('n_mice_rm', int(SIZES.loc[rm, 'mice']))} R−),
{num('n_sess', int(SIZES.sessions.sum()))} sessions, {num('n_trials', int(SIZES.trials.sum()))} trials, days {int(D.day.min())} to {int(D.day.max())} relative to
the first whisker day. Final model `{MODEL}` (027, variant {FINAL}), fitted per cohort and per mouse (hierarchical, {TAG}).
This version replaces the report of 2026-10-23: its central result, fast whisker extinction in R−, came from a local optimum of the cohort fit.

# Abstract

Mice trained to lick for water after a tone are exposed to a whisker stimulus that is rewarded (R+) or not (R−). We model each mouse's trial-by-trial licking
as Go/NoGo Q-learning: stimulus values updated by reward prediction errors with a learning rate per stimulus, read out through a thirst-dependent value gain
and go bias, with an engaged/disengaged hidden state and per-session random effects. The motivational part is needed: licking of every trial type falls
two- to four-fold within each session and recovers overnight ({ref('fig-beh')}), and the simpler alternatives cost
{num('d_min_rp', dbic('min-dis-stim', rp), '{:.0f}')} BIC or more ({ref('fig-model')}). The fitted model predicts the within-session and across-day curves and the
session-start rates of all trial types ({ref('fig-fit')}, {ref('fig-ex')}), except a fast change of whisker licking over the first ~30 whisker trials of
day 0 (up in R+, down in R−). Most of the conspicuous day-0 drop of R− whisker licking is shared with false alarms (satiety and engagement): the
whisker-specific excess over false alarms is only {num('ex_rm0', ex['start'][rm][0], '{:.2f}')} at the start of day 0 and is lost largely within the day, whereas R+ mice
build an excess that is still growing on day 2 ({ref('fig-speed')}). The behavioural hypothesis (fast R−, slow R+) therefore holds for whisker-specific licking.
A single learning rate per stimulus does not capture both: the best single-rate fits give similar whisker learning rates in both cohorts (median
{num('aw_rp', nv('fig5_alpha_W_median_R+'), '{:.4f}')} R+ vs {num('aw_rm', nv('fig5_alpha_W_median_R-'), '{:.4f}')} R−, Mann–Whitney p {pv('aw_pm', nv('fig5_alpha_W_p_mwu'))}),
but with this slow rate the model predicts the R− whisker excess to grow across days, opposite to the data; a fast R− rate reproduces the excess but loses
elsewhere in the likelihood ({ref('fig-prof')}). A model with a fast and a slow component is needed before learning speeds can be compared. Per-mouse parameters
are recovered from simulated data and exported for comparison with neural measures ({ref('fig-mouse')}).

# Key results

1. {claim("K1", "Licking of all trial types falls within each session and recovers overnight; a thirst and engagement block is needed to read learning from the curves", "Fig. 1-2", "BIC", n="111 mice")}:
   without the engagement state the model is worse by {num('k1_rp', dbic('min-stim', rp), '{:.0f}')} (R+) and {num('k1_rm', dbic('min-stim', rm), '{:.0f}')} (R−) BIC.
2. {claim("K2", "With one learning rate per stimulus, the fitted whisker rates are similar in R+ and R- (model-dependent, see K3)", "Fig. 5d", "MWU, Welch, mouse = unit", n="67 / 44 mice")}: α_W median
   {num('k2_rp', nv('fig5_alpha_W_median_R+'), '{:.4f}')} (R+) vs {num('k2_rm', nv('fig5_alpha_W_median_R-'), '{:.4f}')} (R−), MWU p {pv('k2_pm', nv('fig5_alpha_W_p_mwu'))},
   Welch p {pv('k2_pw', nv('fig5_alpha_W_p_welch'))}; time constant τ_W ({ref('eq-tau')}) {num('k2_trp', MS.loc['tau_W', 'median_rplus'], '{:.0f}')} vs
   {num('k2_trm', MS.loc['tau_W', 'median_rminus'], '{:.0f}')} licked whisker trials.
3. {claim("K3", "No single R- whisker rate fits both the likelihood and the whisker-specific licking: slow wins the likelihood, fast reproduces the whisker-specific excess", "Fig. 6", "likelihood profile; session-start excess")}:
   fixing the R− rate at 0.2 costs {num('k3_rm', prof(rm, 0.2), '{:.0f}')} nats, yet R− whisker − false-alarm licking at the start of days 0/1/2 is
   {num('k3_d0', fe('data', rm, 0), '{:+.3f}')} / {num('k3_d1', fe('data', rm, 1), '{:+.3f}')} / {num('k3_d2', fe('data', rm, 2), '{:+.3f}')} in the data,
   {num('k3_f0', fe('fast', rm, 0), '{:+.3f}')} / {num('k3_f1', fe('fast', rm, 1), '{:+.3f}')} / {num('k3_f2', fe('fast', rm, 2), '{:+.3f}')} with the fast rate and
   {num('k3_s0', fe('final', rm, 0), '{:+.3f}')} / {num('k3_s1', fe('final', rm, 1), '{:+.3f}')} / {num('k3_s2', fe('final', rm, 2), '{:+.3f}')} with the slow (final) rate.
4. {claim("K4", "Whisker-specific licking is small in R- and lost largely within day 0; in R+ it is built over days", "Fig. 5a-b", "data, mean over mice")}:
   whisker − false-alarm lick rate at session start, days 0/1/2: R+ {num('k4_rp0', ex['start'][rp][0], '{:+.2f}')} / {num('k4_rp1', ex['start'][rp][1], '{:+.2f}')} /
   {num('k4_rp2', ex['start'][rp][2], '{:+.2f}')}; R− {num('k4_rm0', ex['start'][rm][0], '{:+.2f}')} / {num('k4_rm1', ex['start'][rm][1], '{:+.2f}')} /
   {num('k4_rm2', ex['start'][rm][2], '{:+.2f}')}.
5. {claim("K5", "Per-mouse learning rates, gain and satiety are recoverable", "Fig. S3", "Spearman true vs refit")}: ρ = {num('k5_awp', RSP[('alpha_neg', rp)], '{:.2f}')} /
   {num('k5_awm', RSP[('alpha_neg', rm)], '{:.2f}')} for α_W (R+ / R−), {num('k5_bp', RSP[('beta', rp)], '{:.2f}')} / {num('k5_bm', RSP[('beta', rm)], '{:.2f}')} for β; a simulated
   fast-extinguishing R− subgroup is identified with AUC {num('k5_auc', MIX.auc_fast_vs_slow, '{:.2f}')}.

# Introduction

In the whisker-learning task a mouse that already licks for a tone has to learn what a new whisker stimulus means: R+ whisker licks are rewarded, R− whisker
licks are not. The behaviour suggested fast extinction in R− (whisker licking collapses on day 0) and slow acquisition in R+ (whisker licking rises over days),
and the goal of the model is to test this with per-mouse learning rates that can be related to other analyses.

Two features of the data complicate this reading. Licking of every trial type, including the auditory stimulus and false alarms, falls within each session
and recovers at the next session start ({ref('fig-beh')}), so within-session changes mix learning with satiety and disengagement. And the R− whisker response
on day 0 is mostly non-specific licking of a thirsty mouse, shared with false alarms. The model therefore separates learning (stimulus values) from
motivation (thirst and an engagement state), in the form used for Go/NoGo learning: Q-learning of the lick action with a go bias and a lapse
(Guitart-Masip et al. 2012; Huys et al. 2011), thirst-dependent valuation (Keramati & Gutkin 2014; Allen et al. 2019), a satiety term as in a mouse
Go/NoGo model (Berditchevskaia et al. 2016), and discrete engagement states (Ashwood et al. 2022; Pisupati et al. 2021).

# Methods

## Task and data

{R.methods('task')}
All trials of sequence table 011 are used (performance code 6 absent; no disengagement trimming, since the model explains disengagement): {num('n_trials_m', int(SIZES.trials.sum()))}
trials of {num('n_mice_m', int(SIZES.mice.sum()))} mice ({ref('tbl-s-n')}). Trial types: auditory, whisker (from day 0) and no stimulus; the response is lick / no lick.

## Model

**Learning.** Each mouse has an auditory and a whisker value. On a licked stimulus trial the outcome is $o = +1$ (water: auditory, and whisker in R+) or
$o = -1$ (no water: whisker in R−), and

{eq('eq-learn', r"\delta = o - V_s, \qquad V_s \leftarrow V_s + \alpha_s\,\delta, \qquad s \in \{A, W\}")}

with one learning rate per stimulus ($\alpha_A$, $\alpha_W$) whatever the sign of $\delta$. Unlicked trials teach nothing (no outcome is observed). In R−,
$\alpha_W$ is the extinction rate; in R+, the acquisition rate.

**False alarms.** A lick on a no-stimulus trial is an unrewarded outcome for the no-stimulus value,

{eq('eq-fa', r"V_N = w_N + g_N V_A, \qquad w_N \leftarrow w_N + \alpha_N\,(-1 - V_N)")}

and $w_N$ changes by $d_{{ctx}}$ at the mouse's first whisker trial (context step).

**Thirst.** Thirst falls with the water absorbed in the session and resets at each session start,

{eq('eq-thirst', r"T_t = \max(0,\, 1 - \rho_s S_t), \qquad \log \rho_s \sim \mathcal{N}(\mu_\rho, \sigma_\rho^2)")}

where $S_t$ is the water passed through a first-order lag with time constant $\tau_a$ trials and $\rho_s$ a per-session random effect.

**Choice.** In the engaged state

{eq('eq-choice', r"P(\mathrm{lick}_t) = 0.01 + 0.98\,\sigma\!\left(\beta\, T_t^{\kappa}\, V_t - c + s_B\,\mathrm{softplus}(T_t - \theta) + u_s\right), \qquad u_s \sim \mathcal{N}(0, \sigma_u^2)")}

($V = V_N$ on no-stimulus trials). This is the Go/NoGo action weight of Guitart-Masip et al. (2012): value of the lick action minus a lick cost $c$, plus a go
bias, here thirst-dependent ($s_B$, threshold $\theta$), with thirst also scaling the value ($\kappa$), a per-session offset $u_s$ and a 1% lapse on each side.

**Engagement.** A hidden state switches between engaged and disengaged ($P(\mathrm{{lick}}) = 0.01$ when disengaged). Every session starts engaged; after each trial the
engaged probability is updated from the response (Bayes) and propagated,

{eq('eq-hmm', r"\pi_{t+1} = \pi_t^{\mathrm{post}}\,\bigl(1 - \sigma(h_0 + h_1 (1 - T_t))\bigr) + \bigl(1 - \pi_t^{\mathrm{post}}\bigr)\,\sigma(h_b)")}

**Likelihood.** Values, thirst and the engagement filter depend only on observed licks and rewards, so sessions are conditionally independent and the session
random effects are integrated out with Gauss–Hermite quadrature (7 × 5 nodes for $\rho_s$, $u_s$):

{eq('eq-lik', r"\log p(\mathrm{data}) = \sum_s \log \sum_{k} \omega_k \prod_{t \in s} P\!\left(\mathrm{lick}_t \mid \rho^{(k)}, u^{(k)}\right)")}

Predictions are one-step-ahead: the nodes are weighted by the likelihood of the earlier trials of the session. Learning ({ref('eq-learn')}, {ref('eq-fa')}),
thirst ({ref('eq-thirst')}), choice ({ref('eq-choice')}) and engagement ({ref('eq-hmm')}) together define each trial's term in {ref('eq-lik')}; the mouse-level prior
({ref('eq-prior')}) is below.

## Simplification and model comparison

Starting from the model of the previous report, parameters that cannot act or cannot be identified were removed: the auditory → whisker transfer (fitted 0 in
both cohorts); $\theta$ in R+ (fitted below 0, so thirst never reaches it and it only trades off with $c$); and the valence-specific rates $\alpha_\pm$ were replaced
by stimulus-specific rates (in R+, $\delta < 0$ never occurs, so $\alpha_-$ cannot act; in R−, $\alpha_-$ is the whisker rate). Each cohort was fitted on all its trials
by L-BFGS-B from the default, five random starts and the optima of the earlier fits (027). Alternatives tested ({ref('tbl-s-cmp')}): no absorption lag, the
no-stimulus value following the whisker value instead of the context step, thirst-scaled reward (homeostatic RL), and a minimal model (go bias + linear satiety,
Berditchevskaia et al. 2016) with and without the engagement state. The likelihood profile of the whisker learning rate fixes it on a grid (0.0001 to 0.5)
and refits all other parameters.

## Mouse level

$\alpha_A$, $\alpha_W$, $\beta$ and $\mu_\rho$ are per mouse; the other parameters are fixed at the cohort fit. Parameters are mapped to the real line $y$ (scaled logit,
log first for rates) with a population prior per cohort,

{eq('eq-prior', r"y_m \sim \mathcal{N}(\mu, \mathrm{diag}\,\sigma^2)")}

fitted by expectation–maximisation with a Laplace approximation (Huys et al. 2011) until $\mu$ moves less than 0.02. Learning speed is also reported as the
number of licked whisker trials needed to move the value 63% of the way to the outcome,

{eq('eq-tau', r"\tau_W = -1 / \ln(1 - \alpha_W)")}

which is in trial units (it does not remove the trade-off between $\beta$ and $\alpha$). **Recovery**: every mouse is simulated from its MAP parameters and refitted; a
mixed R− cohort (a quarter of the mice $\alpha_W = 0.5$, the rest 0.001) tests whether fast extinguishers would be found.

## Statistics

Mouse = unit. R+ vs R− comparisons use Mann–Whitney U and Welch's t (log scale for rates, gains and counts), both reported. Correlations: Pearson and
Spearman, OLS line with 95% band (solid when p < 0.05). Model comparison: likelihood and BIC on the cohort fits.

# Results

## Licking falls within every session and recovers overnight

Auditory, whisker and no-stimulus licking fall within each session on every day ({ref('fig-beh')}a, b). Sessions end after very different numbers of rewards
(10th–90th percentile {num('rw_p10', nv('rewards_per_session_p10'), '{:.0f}')}–{num('rw_p90', nv('rewards_per_session_p90'), '{:.0f}')}; {ref('fig-beh')}c), and the decline
is aligned to the session end: false alarms fall from {num('fa_80', nv('fa_80_160_before_end'), '{:.2f}')} 80–160 trials before the end to
{num('fa_10', nv('fa_last10'), '{:.2f}')} in the last 10 trials, auditory licking from {num('aud_80', nv('aud_80_160_before_end'), '{:.2f}')} to
{num('aud_10', nv('aud_last10'), '{:.2f}')} ({ref('fig-beh')}d–f).

## The model and what it needs

{ref('fig-model')} gives the model and the cost of each simplification ({ref('tbl-s-cmp')}). Stimulus-specific learning rates fit R+ better than valence-specific
ones (ΔBIC {num('d_val_rp', dbic('lean-abs', rp), '{:+.0f}')}; identical in R−). The minimal model with a single satiety term is worse by
{num('d_min_rp2', dbic('min-dis-stim', rp), '{:+.0f}')} / {num('d_min_rm2', dbic('min-dis-stim', rm), '{:+.0f}')} BIC (R+ / R−) and without the engagement state by
{num('d_nohmm_rp2', dbic('min-stim', rp), '{:+.0f}')} / {num('d_nohmm_rm2', dbic('min-stim', rm), '{:+.0f}')}; removing the absorption lag costs
{num('d_lag_rp', dbic('lean-inst-stim', rp), '{:+.0f}')} / {num('d_lag_rm', dbic('lean-inst-stim', rm), '{:+.0f}')}. {gw_txt}{hom_txt}
The previous report's R− fit (α− = {num('old_an', old_an, '{:.2f}')}) is a local optimum: the same model refitted from more starts is better by
{num('d_old_rm', dbic('023c-abs', rm), '{:.0f}')} BIC with α− ≈ {num('new_an', FIN.loc[rm, 'p_alpha_neg'], '{:.4f}')}. Cohort-level parameters are in {ref('tbl-par')}.

## The model predicts the curves and the individual mice

One-step-ahead predictions follow the within-session decline and the across-day changes of all three trial types in both cohorts, at the cohort and the
mouse level ({ref('fig-fit')}; mean absolute error per mouse × day × session fifth × trial type {num('mae_c', nv('fig3_mae_cohort-level fit'), '{:.3f}')} cohort-level,
{num('mae_m', nv('fig3_mae_mouse-level fit'), '{:.3f}')} mouse-level). Session-start false-alarm rates are predicted per session with r =
{num('r_fa_rp', nv('fig3b_fa_start_r_R+_mouse-level fit'), '{:.2f}')} (R+) and {num('r_fa_rm', nv('fig3b_fa_start_r_R-_mouse-level fit'), '{:.2f}')} (R−) ({ref('fig-s-start')}).
Example mice ({ref('fig-ex')}, {ref('fig-s-sess')}) show the per-trial prediction, the learned values and the engaged-state probability.

## Whisker-specific licking: lost within day 0 in R−, built over days in R+

What differs between the cohorts is the whisker-specific licking they start from ({ref('fig-speed')}a, b). R+ mice lick
{num('s_rp0', ex['start'][rp][0], '{:+.2f}')} more to the whisker than on no-stimulus trials at the start of day 0, {num('s_rp1', ex['start'][rp][1], '{:+.2f}')} on day 1 and
{num('s_rp2', ex['start'][rp][2], '{:+.2f}')} on day 2. R− mice start at {num('s_rm0', ex['start'][rm][0], '{:+.2f}')}, end day 0 at {num('s_rm0e', ex['end'][rm][0], '{:+.2f}')},
start day 1 at {num('s_rm1', ex['start'][rm][1], '{:+.2f}')} and day 2 at {num('s_rm2', ex['start'][rm][2], '{:+.2f}')}: the large day-0 drop of R− whisker licking is mostly
shared with false alarms (satiety and engagement), and the whisker-specific part, small to begin with, is lost largely within day 0, partly returns overnight
and is gone by day 2. In the single-rate model the learned whisker value moves away from zero at a similar pace in both cohorts, towards +1 in R+ and −1 in R−
({ref('fig-speed')}c).

Per mouse ({ref('tbl-mouse')}), the fitted whisker learning rate of the single-rate model does not differ (α_W median {num('s_awp', nv('fig5_alpha_W_median_R+'), '{:.4f}')} vs
{num('s_awm', nv('fig5_alpha_W_median_R-'), '{:.4f}')}, MWU p {pv('s_awpm', nv('fig5_alpha_W_p_mwu'))}, Welch p {pv('s_awpw', nv('fig5_alpha_W_p_welch'))}; {ref('fig-speed')}d), nor
does τ_W ({ref('fig-speed')}f). The auditory learning rate is slightly higher in R+ ({num('s_aap', nv('fig5_alpha_A_median_R+'), '{:.4f}')} vs
{num('s_aam', nv('fig5_alpha_A_median_R-'), '{:.4f}')}; MWU p {pv('s_aapm', nv('fig5_alpha_A_p_mwu'))}, Welch p {pv('s_aapw', nv('fig5_alpha_A_p_welch'))}). The whisker value at the
end of days 0, 1 and 2 is {num('vw_rp0', MS.loc['V_W_end_day0', 'median_rplus'], '{:+.2f}')}, {num('vw_rp1', MS.loc['V_W_end_day1', 'median_rplus'], '{:+.2f}')},
{num('vw_rp2', MS.loc['V_W_end_day2', 'median_rplus'], '{:+.2f}')} in R+ and {num('vw_rm0', MS.loc['V_W_end_day0', 'median_rminus'], '{:+.2f}')},
{num('vw_rm1', MS.loc['V_W_end_day1', 'median_rminus'], '{:+.2f}')}, {num('vw_rm2', MS.loc['V_W_end_day2', 'median_rminus'], '{:+.2f}')} in R− (medians; {ref('fig-speed')}g):
a small negative value suffices to remove the small R− whisker excess, because the value gain is large (β median {num('b_rm', MS.loc['p_beta', 'median_rminus'], '{:.1f}')}).

The single learning rate misses one feature of day 0 in both cohorts: whisker licking changes quickly over the first ~30 whisker trials, upward in R+
({num('e_rp0', nv('fig5_early_R+_data_b0'), '{:.2f}')} → {num('e_rp2', nv('fig5_early_R+_data_b2'), '{:.2f}')} over blocks 1–3 of 10 trials; model
{num('m_rp0', nv('fig5_early_R+_model_b0'), '{:.2f}')} → {num('m_rp2', nv('fig5_early_R+_model_b2'), '{:.2f}')}) and downward in R− ({num('e_rm0', nv('fig5_early_R-_data_b0'), '{:.2f}')} →
{num('e_rm2', nv('fig5_early_R-_data_b2'), '{:.2f}')}; model {num('m_rm0', nv('fig5_early_R-_model_b0'), '{:.2f}')} → {num('m_rm2', nv('fig5_early_R-_model_b2'), '{:.2f}')}).
This fast component, of opposite sign in the two cohorts, is not captured; it may be a second, fast learning process (two-rate models) or a response to the novel
stimulus. The similar learning rates above are those of a model with one rate per stimulus, which this fast component does not fit.

## One learning rate cannot fit R− whisker-specific licking and the rest of the data

The likelihood profile of the whisker learning rate ({ref('fig-prof')}a) peaks near the fitted values ({num('pk_rm', FIN.loc[rm, 'p_alpha_neg'], '{:.4f}')} in R−,
{num('pk_rp', FIN.loc[rp, 'p_alpha_neg'], '{:.4f}')} in R+); fixing it at 0.2 costs {num('pr_rm', prof(rm, 0.2), '{:.0f}')} nats in R− and {num('pr_rp', prof(rp, 0.2), '{:.0f}')} in R+
(other parameters refitted). The profiles are not smooth: each point is the best of about 11 starts, and the non-monotone shape (e.g. R− cost
{num('pr_rm01', prof(rm, 0.01), '{:.0f}')} nats at 0.01 but {num('pr_rm05', prof(rm, 0.5), '{:.0f}')} at 0.5) shows that the likelihood has several local optima, so
cohort fits are not guaranteed to be global: with the R− rate fixed at 0.001 the fit is {num('pr_gain', FIN.loc[rm, 'nll'] - FITS.loc[('prof-R--alpha_neg-0.001', rm), 'nll'], '{:.0f}')} nats
better than the final R− fit (also in the slow region), and no fast value (≥ 0.1) comes within {num('pr_minfast', min(prof(rm, a) for a in (0.1, 0.2, 0.3, 0.5)), '{:.0f}')} nats of the best point.
The likelihood and the whisker-specific licking disagree in R− ({ref('fig-prof')}b, c). With the slow rate the model predicts almost no whisker-specific excess at
the start of day 0 ({num('ps0', fe('final', rm, 0), '{:+.3f}')} vs {num('pd0', fe('data', rm, 0), '{:+.3f}')} in the data) and an excess that grows at the start of days 1 and 2
({num('ps1', fe('final', rm, 1), '{:+.3f}')}, {num('ps2', fe('final', rm, 2), '{:+.3f}')} vs {num('pd1', fe('data', rm, 1), '{:+.3f}')}, {num('pd2', fe('data', rm, 2), '{:+.3f}')}),
because the no-stimulus value keeps falling while the whisker value barely moves. With the rate fixed at 0.2 the excess is lost within day 0 and is small
afterwards ({num('pf0', fe('fast', rm, 0), '{:+.3f}')}, {num('pf1', fe('fast', rm, 1), '{:+.3f}')}, {num('pf2', fe('fast', rm, 2), '{:+.3f}')}), close to the data. The
whisker-specific excess is a small part of R− licking, so the likelihood is dominated by other features that the slow fit captures better. In R+ the fast rate is
wrong on both counts (end of day 0: {num('pfr', fe('fast', rp, 0, 'end'), '{:+.2f}')} vs {num('pdr', fe('data', rp, 0, 'end'), '{:+.2f}')} in the data). A model with a fast
and a slow component of whisker learning is therefore needed in R−. Recovery shows that the per-mouse fit would identify fast mice under the single-rate model
(fast quarter refitted at α_W ≈ {num('mix_f', MIX.median_refit_fast, '{:.2f}')} vs {num('mix_s', MIX.median_refit_slow, '{:.4f}')}, AUC {num('mix_auc', MIX.auc_fast_vs_slow, '{:.2f}')});
the largest fitted R− α_W is {num('max_aw', MM[MM.reward_group == rm].p_alpha_neg.max(), '{:.2f}')}.

## Per-mouse parameters for other analyses

Per mouse, $\alpha_A$, $\alpha_W$, $\beta$ and $\mu_\rho$ are recovered from simulated data (Spearman ρ, R+ / R−: α_A {num('r_aa_p', RSP[('alpha_pos', rp)], '{:.2f}')} /
{num('r_aa_m', RSP[('alpha_pos', rm)], '{:.2f}')}, α_W {num('r_aw_p', RSP[('alpha_neg', rp)], '{:.2f}')} / {num('r_aw_m', RSP[('alpha_neg', rm)], '{:.2f}')}, β
{num('r_b_p', RSP[('beta', rp)], '{:.2f}')} / {num('r_b_m', RSP[('beta', rm)], '{:.2f}')}, μ_ρ {num('r_mr_p', RSP[('mu_rho', rp)], '{:.2f}')} / {num('r_mr_m', RSP[('mu_rho', rm)], '{:.2f}')};
{ref('fig-s-rec')}). R− mice sate faster per reward ({num('h_rp', MS.loc['rewards_to_half_thirst', 'median_rplus'], '{:.0f}')} vs
{num('h_rm', MS.loc['rewards_to_half_thirst', 'median_rminus'], '{:.0f}')} rewards to halve thirst; MWU p {pv('h_pm', MS.loc['rewards_to_half_thirst', 'p_mwu'])}, Welch p
{pv('h_pw', MS.loc['rewards_to_half_thirst', 'p_welch'])}; {ref('fig-mouse')}c) and have a lower value gain (β; {ref('fig-mouse')}a). The per-mouse whisker learning rate
relates to model-free learning: in R+, log α_W vs the day-0 → day-2 growth of whisker-specific licking, r = {num('c_rp', nv('fig7_rplus_alphaW_mf_r'), '{:.2f}')}
(p {pv('c_rpp', nv('fig7_rplus_alphaW_mf_p'))}); in R−, vs the day-0 → day-1 loss, r = {num('c_rm', nv('fig7_rminus_alphaW_mf_r'), '{:.2f}')} (p {pv('c_rmp', nv('fig7_rminus_alphaW_mf_p'))})
({ref('fig-mouse')}e, f). Satiety vs the rewards taken per session: r = {num('c_th', nv('fig7_thirst_rewards_r'), '{:.2f}')}, p {pv('c_thp', nv('fig7_thirst_rewards_p'))}
({ref('fig-mouse')}g). Exported: a per-mouse table (tables/029_mouse_table.csv: parameters, posterior s.d., τ_W, V_W per day, satiety) and per-trial latents keyed by
mouse_id, session_id, trial_idx (tables/028_{TAG}_trial_latents.parquet: V_A, V_W, V_N, prediction error δ, thirst, P(engaged), predicted lick probability).

# Discussion

The model separates what the mouse has learned about the whisker from how much it wants to lick. Two conclusions follow. First, most of the conspicuous
day-0 collapse of R− whisker licking is the satiety- and engagement-driven decline shared with false alarms, not whisker-specific learning; this resembles the
acquisition/expression distinction of Kuchibhotla et al. (2019). Second, the whisker-specific part that remains is small in R− and lost within a day, while
in R+ it is built over days, as hypothesised; but a single learning rate per stimulus cannot express both a fast loss of a small R− excess and the slow changes
that dominate the likelihood, and its fitted rates (similar in both cohorts) should not be read as equal learning speeds. A fast and a slow learning component
(or a novelty response to the new stimulus) is the next model, and the per-mouse fast component would then be the learning-speed read-out.

The framing is standard: Go/NoGo Q-learning with a go bias and lapse (Guitart-Masip et al. 2012), fitted hierarchically (Huys et al. 2011), with a
thirst-dependent value and go bias (Keramati & Gutkin 2014; Allen et al. 2019) and engagement states (Ashwood et al. 2022). The motivational block is larger
than in most RL fits of rodent behaviour, but each part is needed by the likelihood ({ref('tbl-s-cmp')}), and the parameters that were not identifiable are fixed at
the cohort level. Following Wilson & Collins (2019), parameters are recovered from simulated data and the competing single-rate fits are compared on the
data features they should reproduce, which is what exposed the limitation above.

# Caveats and limitations

- BIC treats trials as independent; model comparison on held-out mice and a model-recovery matrix (Wilson & Collins 2019) are not done.
- R+ $c$ is near its bound (10) and trades off with $s_B$; cohort-level gain and bias parameters differ between cohorts (e.g. κ, {ref('tbl-par')}), so per-mouse
  β is not strictly comparable across cohorts. Learning rates, τ_W and value trajectories are the comparable read-outs.
- μ_ρ is less well recovered in R− (ρ = {num('cav_mr', RSP[('mu_rho', rm)], '{:.2f}')}). The faster R− satiety per reward may partly reflect fewer rewards per session in R−.
- Whisker trials to |V_W| = 0.5 is censored for most mice (the value does not reach 0.5 within days 0–2); τ_W is the uncensored speed measure.
- The first ~30 whisker trials of day 0 change faster than the model allows in both cohorts (up in R+, down in R−), and the slow single-rate fit predicts
  the R− whisker-specific excess to grow across days; per-mouse α_W of the single-rate model is therefore not a validated learning-speed measure yet.
- The context step $d_{{ctx}}$ is descriptive.

# Next steps

- Two-rate (fast + slow) whisker learning or a novelty term; does it reproduce the R− whisker-specific excess, and is the fast component faster in R−?
- Held-out-mouse model comparison of the final, minimal and valence-specific models; model-recovery matrix.
- Relate per-mouse α_W, β and satiety, and per-trial δ and V_W, to the neural analyses (exported tables).

# Figures
"""
    md += figure("fig-beh", PUB / "fig1_behaviour.png",
                 "Behaviour. (a, b) Lick probability per trial type (blue auditory, yellow whisker, black no stimulus) on days −4 to 2, each day in session fifths; "
                 "mean ± s.e.m. over mice (R+ n = 67, R− n = 44). (c) Rewards per session (R+ green, R− magenta). (d–f) Auditory (blue) and no-stimulus (black) licking "
                 "aligned to the session end (d), to the rewards received (e) and to the fraction of the session's rewards (f); R+ solid, R− dashed.",
                 script="029_pub_figures_v8.py")
    md += figure("fig-model", PUB / "fig2_model.png",
                 "Model. (a) Equations (Eqs. 1–5). (b) BIC of alternative models relative to the final model, cohort-level fits on all trials (R+ green, R− magenta; "
                 "symmetric-log axis; values in Table S2).", script="029_pub_figures_v8.py")
    md += figure("fig-fit", PUB / "fig3_fit_vs_data.png",
                 "Fit to the data. Rows: final model with cohort-level and with mouse-level parameters. (a, b) Data (bands: mean ± s.e.m. over mice) and one-step-ahead "
                 "prediction (lines) per session fifth, days −4 to 2. (c) Aligned to the session end (log scale; R+ solid, R− dashed).", script="029_pub_figures_v8.py")
    md += figure("fig-ex", PUB / "fig4_example_mice.png",
                 "Example mice (chosen programmatically: R+ strongest and weakest late whisker licking, R− largest and median day-0 whisker excess). Top: data (thick, "
                 "running mean of 15 trials of each type) and one-step-ahead prediction of the mouse-level fit (thin), P(engaged) (dotted brown); bottom: learned values "
                 "V_A, V_W, V_N. Vertical lines: session starts.", script="029_pub_figures_v8.py")
    md += figure("fig-speed", PUB / "fig5_learning_speed.png",
                 "Learning speed. (a, b) Whisker minus false-alarm lick rate in the first and last 20 trials of each type per session, days 0–2; data mean ± s.e.m. over "
                 "mice (dots), mouse-level model (dashed). (c) Learned whisker value vs whisker trials since onset, mean ± s.e.m. over mice (bins with ≥ 20 mice). "
                 "(d–f) Per-mouse α_W, α_A and τ_W (Eq. 8), geometric mean (large dot); Mann–Whitney and Welch (log scale). (g) V_W at the last whisker trial of each "
                 "day, mean ± s.e.m. over mice.", script="029_pub_figures_v8.py")
    md += figure("fig-prof", PUB / "fig6_profiles.png",
                 "Single learning rate: likelihood vs whisker-specific licking. (a) Likelihood profile of the whisker learning rate (cohort-level fits with the rate fixed, all else refitted; R+ α_W under the "
                 "final model, R− α−, which is the whisker rate; dotted: R+ shared α+ of the valence-specific model); dashed line: 1.92 nats. (b, c) Whisker-specific "
                 "licking at session start / end (as Figure 5a, b): data (dots), final model (dashed) and the model with the whisker rate fixed at 0.2 (red dotted), "
                 "cohort-level parameters.", script="029_pub_figures_v8.py")
    md += figure("fig-mouse", PUB / "fig7_mouse_parameters.png",
                 "Per-mouse parameters. (a–d) β, μ_ρ, rewards to halve thirst (0.5 / exp μ_ρ) and posterior s.d. of α_W; mean ± s.e.m.; Mann–Whitney and Welch. "
                 "(e, f) log α_W vs model-free whisker learning (R+: day-0 → day-2 growth of the session-start whisker − false-alarm rate; R−: day-0 → day-1 loss). "
                 "(g) Rewards to halve thirst vs median rewards per session. (h) α_A vs α_W. OLS line with 95% band, solid when p < 0.05.",
                 script="029_pub_figures_v8.py")
    md += "\n# Tables\n"
    md += table("tbl-par", param_table(), "Parameters of the final model: cohort-level fit (027) and median of the per-mouse MAP estimates for the mouse-level "
                "parameters (025S). Eqs. 1–5.")
    md += table("tbl-mouse", mouse_table(), "Per-mouse learning measures, R+ vs R− (mouse = unit; Mann–Whitney U, Welch's t on the log scale for rates and counts).")
    md += "\n# Supplementary figures\n"
    md += figure("fig-s-start", PUB / "fig3b_start_errors.png",
                 "Session-start errors (cohort-level and mouse-level final model). (a, b, e, f) Predicted minus observed lick probability in the first fifth of each "
                 "session, per day; mean ± s.e.m. over mice. (c, g) Per session: observed vs predicted session-start false-alarm rate. (d, h) Spread across sessions of "
                 "the predicted start false-alarm rate (bars) vs data (dashed).", script="029_pub_figures_v8.py", kind="supplementary")
    md += figure("fig-s-sess", PUB / "fig4b_example_sessions.png",
                 "Day-0 sessions of the example mice: licks (ticks) and one-step-ahead lick probability per trial type (top); thirst and P(engaged) (bottom).",
                 script="029_pub_figures_v8.py", kind="supplementary")
    md += figure("fig-s-rec", PUB / "figS2_hier_recovery.png",
                 "Recovery of the mouse-level parameters: true (MAP of each mouse) vs refitted from data simulated with the final model, on the unbounded scale (α+ = α_A, "
                 "α− = α_W), Spearman ρ; last panel: mixed R− cohort (a quarter of the mice α_W = 0.5, the rest 0.001), refitted α_W and AUC.",
                 script="029_pub_figures_v8.py", kind="supplementary")
    md += "\n# Supplementary tables\n"
    md += table("tbl-s-n", sizes_table(), "Sample sizes per cohort (011_sequences).")
    md += table("tbl-s-cmp", cmp_table(), "Model comparison: cohort-level negative log-likelihood, number of free parameters k and BIC relative to the final model "
                "(023c, 027). Shown in Figure 2b.")
    claims()
    md += "\n# Appendix\n" + R.registry_table("tbl-registry") + APPENDIX
    md += f"\nThe claims registry is {ref('tbl-registry')}.\n"
    return md


def claims():
    claim("C1", "Previous fast R- alpha- (0.23) was a local optimum", "Table S2", "BIC", status="holds")
    claim("C2", "Engagement state and thirst scaling are needed", "Fig. 2b", "BIC", status="holds")
    claim("C3", "R- day-0 whisker drop is mostly shared with false alarms", "Fig. 5b", "data", status="holds")
    claim("C4", "R- satiety per reward faster than R+", "Fig. 7c", "MWU, Welch", status="fragile", note="may reflect fewer rewards per session")
    claim("C5", "R+ c near bound, trades off with s_B", "Table 1", "cohort fit", status="fragile")


APPENDIX = """
### Scripts and results

| step | script | output (combined_results_ks4/ssl-rl-behaviour-model/) |
|:--|:--|:--|
| trial sequences | 011 | tables/011_sequences.parquet |
| previous cohort fits | 023_v6_prediction_check.py (tag c) | fits/023c_fits.csv |
| lean cohort fits, variants, profiles | 027_lean_model.py | fits/027{,b,c,p,q}_fits.csv, tables/027_latents |
| mouse-level fit, recovery | 025_fit_hier.py --lean (rl8_hier.py) | fits/025S_hier_*.csv, 025S_recovery_mice.csv, tables/025S_* |
| per-mouse measures, per-trial latents | 028_mouse_metrics.py | tables/028_025S_* |
| figures | 029_pub_figures_v8.py | figures/pub_v8, tables/029_* |

Model code: rl7_model.py (likelihood, simulation; options +stim, +gw, +hom added 2026-10-24), rl8_hier.py (EM-Laplace).

### References

Allen WE et al. (2019) Thirst regulates motivated behavior through modulation of brainwide neural population dynamics. Science 364:253.
Ashwood ZC et al. (2022) Mice alternate between discrete strategies during perceptual decision-making. Nat Neurosci 25:201–212.
Berditchevskaia A, Cazé RD, Schultz SR (2016) Performance in a GO/NOGO perceptual task reflects a balance between impulsive and instrumental components of behaviour. Sci Rep 6:27389.
Guitart-Masip M et al. (2012) Go and no-go learning in reward and punishment: interactions between affect and effect. NeuroImage 62:154–166.
Huys QJM et al. (2011) Disentangling the roles of approach, activation and valence in instrumental and Pavlovian responding. PLoS Comput Biol 7:e1002028.
Keramati M, Gutkin B (2014) Homeostatic reinforcement learning for integrating reward collection and physiological stability. eLife 3:e04811.
Kuchibhotla KV et al. (2019) Dissociating task acquisition from expression during learning reveals latent knowledge. Nat Commun 10:2151.
Pisupati S et al. (2021) Lapses in perceptual decisions reflect exploration. eLife 10:e55490.
Wilson RC, Collins AGE (2019) Ten simple rules for the computational modeling of behavioral data. eLife 8:e49547.

### Version history

v1–v5 (2026-10-16 to 21): learning core with exploration and engagement terms, flat within sessions. v6 (023): thirst, lick bias, engaged state, session
offset, context step, absorption lag. v7 (025, report of 2026-10-23): mouse-level α+, α−, β, c, s_B, μ_ρ, θ; reported fast R− extinction, which came from a
local optimum. v8 (this report): lean model with stimulus-specific rates, refitted from more starts, four mouse-level parameters, learning-rate profiles.
"""


def summary():
    return f"""# ssl-rl-behaviour-model: summary, {date.today().isoformat()}

Go/NoGo Q-learning with stimulus-specific learning rates, thirst-dependent value and go bias, engagement state and per-session random effects; final
`{MODEL}`, fitted per cohort and per mouse (α_A, α_W, β, μ_ρ).

1. Licking of all trial types falls within sessions and recovers overnight; the thirst and engagement block is needed (Fig. 1–2).
2. Whisker-specific licking (whisker − false alarms) is small in R− and lost largely within day 0; R+ builds it over days. Most of the R− day-0 drop is shared
   with false alarms (satiety, engagement).
3. With one learning rate per stimulus, fitted whisker rates are similar (median α_W {nv('fig5_alpha_W_median_R+'):.4f} vs {nv('fig5_alpha_W_median_R-'):.4f}, MWU p = {nv('fig5_alpha_W_p_mwu'):.2f}),
   but the slow rate mispredicts R− whisker-specific licking and a fast rate loses in likelihood: a fast + slow learning model is needed.
4. The previous report's fast R− α− = 0.23 was a local optimum of the cohort fit.
   A fast change over the first ~30 whisker trials of day 0 (up in R+, down in R−) is not captured either.
5. Per-mouse parameters are recovered (α_W ρ ≈ {RSP[('alpha_neg', 'R+')]:.2f} / {RSP[('alpha_neg', 'R-')]:.2f}) and exported with per-trial latents.

![Learning speed](figures/fig-speed.png)

Full report: report.pdf / report.html; claims registry: registry.csv.
"""


def main():
    md = build()
    R.write(OUT, md, summary=summary())


if __name__ == "__main__":
    main()
