"""Report of ssl-prelick-convergence, across days (day 0 vs expert), rebuilt 2026-10-15 (skills/project-report, report_lib).
Storyline (user 2026-10-15): task and data -> single neurons -> population coding direction -> decoding; at every level the
cohort (R+ / R-), stage (day 0 / expert), auditory comparison (AH) and area groups are shown inside the level's figure; controls
(reaction time, pre-lick kinematics, jaw-onset alignment, passive behaviour) as supplementary figures. Report figures are
re-arranged from the latest COSYNE panels (130_report_figures); within-day analyses move to the combined within-day report.
The previous merged report (Parts I + II, Fig2-Fig5) is superseded; its generator is in git history.
Every quoted number comes from the frozen statistics tables (report_lib snapshot); summary.md, registry.csv and figures.json
are written next to report.md.
Run (haas, after 087, 088, 090, 104, 112, 116, 118-120 jaw runs, 128, 130, 131):
  cd ~/code/unit_spikes_analysis && PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
    ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-prelick-convergence/report/build_report.py
Render locally: REPORT_DIR=report_stable bash projects/ssl-prelick-convergence/report/render.sh
"""
from __future__ import annotations

import importlib
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
sys.path.insert(0, str(EA))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
from report_lib import Report  # noqa: E402

m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
BASE = m51.OUTROOT                                    # across_days/sl_stable
LRN = BASE / "publication" / "learners"
JAW = {v: BASE.parent / f"{BASE.name}_{v}" for v in ("jaw", "jawc")}
OUT = m51.HOME / ("report" if m51.UNITS != "stable" else f"report_{m51.UNITS}")
EXCLUDED_R_MINUS = "AB094, AB152, AB157 and MH036"

R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = num
num = lambda *a, **k: _num(*a, **k).replace("-", "−")            # typographic minus for quoted values
R.register([
    ("fig-task", "Figure 1"), ("fig-neurons", "Figure 2"), ("fig-cd", "Figure 3"), ("fig-ccgp", "Figure 4"),
    ("eq-si", "Eq. (1)"), ("eq-cd", "Eq. (2)"), ("eq-proj", "Eq. (3)"), ("eq-theta", "Eq. (4)"), ("eq-ccgp", "Eq. (5)"),
    ("fig-s-signed", "Figure S1"), ("fig-s-rt", "Figure S2"), ("fig-s-kin", "Figure S3"), ("fig-s-jaw", "Figure S4"),
    ("fig-s-dlc", "Figure S5"),
    ("tbl-sizes", "Table S1"), ("tbl-robust", "Table S2"), ("tbl-registry", "Table S3"),
])

# ------------------------------------------------------------------------------------------------ frozen statistics
SRC = {
    "fig": LRN / "report_figures_stats.csv",                  # 130 (SI, d_WS, theta, rho, class levels, area ANOVAs)
    "rt": LRN / "rt_stats.csv",                               # 131 (Fig 1d)
    "harm": BASE / "harmonised" / "tests.csv",                # 087
    "sess": BASE / "harmonised" / "sessions_areas.csv",
    "rtadj": BASE / "rt_adjusted" / "stats.csv",              # 090
    "ccgp": BASE / "ccgp_pub2" / "stats_learners.csv",        # 112
    "ccgp_eq": BASE / "ccgp_sessions_eq" / "stats_learners_n40.csv",   # 114 (= 112 m-q)
    "kin": BASE / "prelick_kinematics" / "stats_learners.csv",          # 116
    "jaw": JAW["jaw"] / "harmonised" / "tests.csv",           # 087 on jaw-onset events (118, pre)
    "jawc": JAW["jawc"] / "harmonised" / "tests.csv",         # 087 on jaw-onset events (118, centred)
    "signed": BASE / "unified_single_neuron_all" / "fig_unified_shared_signed_learners_stats.csv",   # 127 / 128
}
FROZEN = dict(zip(SRC, Report.snapshot([SRC[k] for k in SRC], OUT, tag=f"{date.today().isoformat()}_across_days")))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}


def row(k, panel):
    d = T[k][T[k].panel == panel]
    if len(d) != 1:
        raise KeyError(f"{k}: {len(d)} rows for panel {panel}")
    return d.iloc[0]


def ch(key, r, fmt="{:.2f}", unit=""):
    """'R+ a -> b (p), R- c -> d (p)' with the cohort x stage and expert R+ vs R- p values (group_stats rows)"""
    s = {}
    s["rp"] = (f"R+ {num(key + '_rp_d0', r['mean_R+ learning'], fmt)} → {num(key + '_rp_e', r['mean_R+ expert'], fmt)}{unit} "
               f"(p {pv(key + '_rp_p', r['R+ L vs E p_MWU'])})")
    s["rm"] = (f"R− {num(key + '_rm_d0', r['mean_R− learning'], fmt)} → {num(key + '_rm_e', r['mean_R− expert'], fmt)}{unit} "
               f"(p {pv(key + '_rm_p', r['R- L vs E p_MWU'])})")
    s["int"] = f"cohort × stage p {pv(key + '_int', r['interaction_p_perm'])}"
    s["ex"] = f"expert R+ vs R− p {pv(key + '_ex', r['expert R+ vs R- p_MWU'])}"
    s["d0"] = f"day 0 R+ vs R− p {pv(key + '_d0', r['learning R+ vs R- p_MWU'])}"
    s["n"] = (f"{int(r['n_R+ learning'])} / {int(r['n_R+ expert'])} / {int(r['n_R− learning'])} / {int(r['n_R− expert'])}")
    return s


def harm(k, measure, key, fmt="{:.2f}"):
    """087-style tests.csv (one row per comparison) -> the same strings as ch()"""
    d = T[k][T[k].measure == measure].set_index("comparison")
    rp, rm = d.loc["R+ expert vs learning"], d.loc["R- expert vs learning"]
    return dict(
        rp=f"R+ {num(key + '_rp_d0', rp.mean_learning, fmt)} → {num(key + '_rp_e', rp.mean_expert, fmt)} (p {pv(key + '_rp_p', rp.p_mwu)})",
        rm=f"R− {num(key + '_rm_d0', rm.mean_learning, fmt)} → {num(key + '_rm_e', rm.mean_expert, fmt)} (p {pv(key + '_rm_p', rm.p_mwu)})",
        int=f"cohort × stage p {pv(key + '_int', d.loc['cohort x stage (mouse permutation)'].p_perm)}",
        ex=f"expert R+ vs R− p {pv(key + '_ex', d.loc['R+ vs R- at expert'].p_mwu)}",
        n=f"{int(rp.n_learning)} / {int(rp.n_expert)} / {int(rm.n_learning)} / {int(rm.n_expert)}")


def anova(panel, key):
    a = row("fig", f"{panel} ANOVA")
    q = T["fig"][T["fig"].panel == f"{panel} q"]
    return dict(cs=f"cohort × stage p {pv(key + '_cs', a['cohort x stage p_perm'])}",
                csa=f"cohort × stage × area p {pv(key + '_csa', a['cohort x stage x area p_perm'])}",
                n_areas=num(key + "_n_areas", int(a["n_areas"])), qmin=num(key + "_qmin", q.q_BH.min(), "{:.2f}"),
                n_sig=int((q.q_BH < 0.05).sum()))


# ------------------------------------------------------------------------------------------------ numbers
RTW, RTA = ch("rt_wh", row("rt", "131 RT WH"), "{:.0f}", " ms"), ch("rt_ah", row("rt", "131 RT AH"), "{:.0f}", " ms")
SI = ch("si", row("fig", "130 si"))
DWS = ch("dws", row("fig", "130 dWS"))
TH = ch("theta", row("fig", "088 theta"), "{:.0f}", "°")
RHO = ch("rho", row("fig", "088 rho_v"))
A_SI, A_D = anova("f2d", "area_si"), anova("f3e", "area_dws")
LV = T["fig"][T["fig"].panel == "l"].set_index(["cohort", "cls"])
CC = ch("ccgp", row("ccgp", "112 CCGP"))
EQ = T["ccgp_eq"][T["ccgp_eq"].area == "Whole brain"].iloc[0]
EQA = T["ccgp_eq"][(T["ccgp_eq"].area != "Whole brain") & (T["ccgp_eq"].measure == "ex")]
SG = ch("swa", row("signed", "125 d sWA"), "{:+.3f}")
SGW = ch("sws", row("signed", "125 d sWS"), "{:+.3f}")
JAWK = ch("jaw", row("kin", "116 jaw opening WH-AH"), "{:+.2f}", " mm")
J1, J2 = harm("jaw", "d_WS", "jaw_dws"), harm("jawc", "d_WS", "jawc_dws")
J1T = harm("jaw", "theta", "jaw_theta", "{:.0f}")


def level(c, cls, key):
    r = LV.loc[(c, cls)]
    return (f"{num(key + '_d0', r.mean_day0)} → {num(key + '_e', r.mean_expert)} (p {pv(key + '_p', r.p_MWU)})")


# ------------------------------------------------------------------------------------------------ tables
def sizes():
    S = m61.learner_filter(T["sess"])
    w = S[S.area == "whole_brain"]
    g = w.groupby(["cohort", "stage"]).agg(sessions=("session_id", "nunique"), mice=("mouse_id", "nunique"), units=("n_units", "sum"))
    lab = {("R+", "learning"): "R+ day 0", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− day 0", ("R-", "expert"): "R− expert"}
    nfrom = lambda r, c, s: int(r[f"n_{c} {s}"])
    rows = []
    for (c, s), nm in lab.items():
        cc = c.replace("-", "−")
        rows.append({"group": nm, "sessions (CD)": int(g.loc[(c, s), "sessions"]), "mice (CD)": int(g.loc[(c, s), "mice"]),
                     "units (CD)": f"{int(g.loc[(c, s), 'units']):,}",
                     "RT (Fig. 1d)": nfrom(row("rt", "131 RT WH"), cc, s),
                     "SI (Fig. 2c)": nfrom(row("fig", "130 si"), cc, s),
                     "CCGP (Fig. 4g)": nfrom(row("ccgp", "112 CCGP"), cc, s),
                     "CCGP 40 units (Fig. 4m)": int(EQ[f"n {c} {s}"])})
    return pd.DataFrame(rows)


def robust():
    rows = []
    for nm, key, src in (("Lick-aligned, all events (Fig. 3c)", "all", None), ("RT ≥ 150 ms", "long", None),
                         ("WH / AH RT-matched", "match", None), ("Jaw onset, CD 100 ms before", "jaw", "jaw"),
                         ("Jaw onset, CD centred (± 50 ms)", "jawc", "jawc")):
        if src is None:
            r = row("rtadj", f"090 {key} d_WS")
            d = dict(n=f"{int(r['n_R+ learning'])}/{int(r['n_R+ expert'])}/{int(r['n_R− learning'])}/{int(r['n_R− expert'])}",
                     a=r["mean_R+ learning"], b=r["mean_R+ expert"], p=r["R+ L vs E p_MWU"], pm=r["R- L vs E p_MWU"],
                     i=r["interaction_p_perm"], e=r["expert R+ vs R- p_MWU"])
        else:
            t = T[src][T[src].measure == "d_WS"].set_index("comparison")
            rp, rm = t.loc["R+ expert vs learning"], t.loc["R- expert vs learning"]
            d = dict(n=f"{int(rp.n_learning)}/{int(rp.n_expert)}/{int(rm.n_learning)}/{int(rm.n_expert)}", a=rp.mean_learning,
                     b=rp.mean_expert, p=rp.p_mwu, pm=rm.p_mwu, i=t.loc["cohort x stage (mouse permutation)"].p_perm,
                     e=t.loc["R+ vs R- at expert"].p_mwu)
        f = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
        rows.append({"variant": nm, "n (R+ D0/E, R− D0/E)": d["n"], "R+ d_WS day 0 → expert": f"{d['a']:.2f} → {d['b']:.2f}",
                     "p R+": f(d["p"]), "p R−": f(d["pm"]), "cohort × stage p": f(d["i"]), "expert R+ vs R− p": f(d["e"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ text
def build():
    md = rf"""---
title: "Whisker-cued licks converge on the auditory reward-lick code with learning (across days)"
subtitle: "ssl-prelick-convergence · day 0 vs expert · learners · stable units · Kilosort 4"
date: "{date.today().isoformat()}"
---

## Abstract

Mice licked for water after auditory tones and were then exposed to a whisker stimulus that was rewarded for one cohort (R+)
and not for the other (R−). We asked whether learning makes the neural activity just before a whisker-cued lick resemble the
activity before a rewarded auditory-cued lick rather than the activity before a spontaneous, unrewarded lick. Across day-0 and
expert sessions of Neuropixels recordings over many brain areas, whisker reaction times became auditory-like in R+ mice only.
Non-sensory neurons that distinguished auditory-cued from spontaneous licks moved from a spontaneous-lick-like to an
auditory-lick-like whisker response in R+ but not in R− mice, and the cohort difference of this change was significant. At
the population level, whisker hits moved along the reward-lick coding direction in R+ mice, and a decoder of cued versus
spontaneous licks generalised better across modalities in R+ experts than in R− experts; for these two measures the cohort ×
stage interaction did not reach significance. The changes were distributed across area groups, without an area-specific
effect after correction, and they held after controlling for reaction time, pre-lick jaw kinematics and alignment to jaw-opening onset.

**Key results** (sessions are the unit; R+ / R− day 0 → expert; Mann–Whitney U unless stated).

1. Whisker reaction times shorten in R+ only: {RTW['rp']}; {RTW['rm']} ({ref('fig-task')}d).
2. Single neurons: the similarity index of non-sensory reward-lick neurons rises in R+ only: {SI['rp']}; {SI['rm']}; {SI['int']} ({ref('fig-neurons')}c).
3. Population: whisker hits move toward auditory hits on the reward-lick coding direction in R+: {DWS['rp']}; {DWS['rm']}; {DWS['int']} ({ref('fig-cd')}c).
4. Decoding: cross-modal generalisation of the cued-lick code rises in R+ and exceeds R− in experts: {CC['rp']}; {CC['rm']}; {CC['ex']}; {CC['int']} ({ref('fig-ccgp')}g).
5. Areas: the omnibus cohort × stage × area terms are not significant for the single-neuron index ({A_SI['csa']}) or the population projection ({A_D['csa']}); no area survives the post hoc correction ({ref('fig-neurons')}d, {ref('fig-cd')}e).

## Introduction

In this task a lick can follow an auditory stimulus (auditory hit, AH: rewarded in both cohorts), a whisker stimulus (whisker
hit, WH: rewarded in R+ only) or no stimulus (spontaneous lick, SL: a false alarm, never rewarded). The three events share the
motor act, so differences in the 100 ms before the first lick reflect the cue, the expectation of reward, or both. If R+ mice
learn that the whisker stimulus predicts reward, pre-lick activity on whisker hits should become more similar to auditory
hits (the rewarded cued lick) and less similar to spontaneous licks. R− mice, which still lick to the whisker stimulus at
first but are never rewarded for it, are the control for exposure and licking.

We test this prediction across days, comparing the first whisker-training session (day 0) with expert sessions, at three
levels. The levels are single neurons ({ref('fig-neurons')}), a population coding direction ({ref('fig-cd')}) and decoders
({ref('fig-ccgp')}). At each level the figure shows both cohorts, both stages, the auditory events as the reference and the
breakdown by area group. Within-session changes on day 0, with expert sessions as a control, are the subject of the combined
within-day report. Every claim of this report, with its figure, statistic and status, is listed in {ref('tbl-registry')}.

## Methods

### Data and inclusion

{R.methods('task', 'learners', 'trials', 'units', 'lick_alignment')}
Pre-lick activity is the spike count in the 100 ms before the corrected first lick, [−100, 0) ms, minus the per-trial
baseline [−1, −0.015] s before trial start (spontaneous licks: [−1, −0.5] s before the lick); units with a mean pre-lick rate
below 0.1 Hz are excluded. A session enters the coding-direction analyses with at least 4 WH, 8 AH and 8 SL events and
the single-neuron and decoding analyses with at least 8 events per lick type. This event minimum excludes four R−
day-0 learner sessions with 2–6 whisker hits ({EXCLUDED_R_MINUS}); the decision is kept (user, 2026-10-15) and the
remaining sessions per analysis are in {ref('tbl-sizes')}.

### Single neurons

Reward-lick neurons are units whose pre-lick rate separates auditory hits from half of the spontaneous licks (SL-A; ROC
label permutation p < 0.05, larger before auditory hits). Non-sensory neurons are reward-lick neurons with no stimulus response
to either modality: no significant change 5–35 ms after the stimulus relative to [−50, −15) ms on active hit and miss trials,
for whisker and for auditory stimuli. For each such neuron the whisker-hit response is compared, on the other half of the
spontaneous licks (SL-B), with the spontaneous-lick and auditory-hit responses through a similarity index:
{eq('eq-si', r"SI = \frac{|s_{WS}| - |s_{WA}|}{|s_{WS}| + |s_{WA}|}, \qquad s = 2\,\mathrm{AUC} - 1")}
where $s_{{WS}}$ is the ROC selectivity of WH vs SL-B and $s_{{WA}}$ of WH vs AH. The magnitudes are cross-validated (the
geometric mean of the selectivities of random halves of the events, 50 splits), so noise alone gives magnitudes near zero.
SI = −1 means that whisker hits look like spontaneous licks and SI = +1 that they look like auditory hits. A session
contributes the mean SI of its neurons (≥ 3 with a defined SI). A signed variant without neuron selection (all non-sensory
neurons, count-matched draws of 4 WH / 4 AH / 8 SL, baseline [−1.5, −1.0) s before the lick) is the robustness check of
{ref('fig-s-signed')}.

### Population coding direction

Per session, units are z-scored over the AH and SL events, and the reward-lick coding direction (CD) is estimated by 5-fold
cross-validation over these events:
{eq('eq-cd', r"\mathbf{u} = \frac{\bar{\mathbf{r}}_{AH} - \bar{\mathbf{r}}_{SL}}{\lVert \bar{\mathbf{r}}_{AH} - \bar{\mathbf{r}}_{SL} \rVert}")}
where $\bar{{\mathbf r}}$ are the class means of the population vectors of the training folds. Held-out AH and SL events and
all WH events (averaged over folds) are projected on $\mathbf u$, per neuron:
{eq('eq-proj', r"p_c = \frac{1}{\sqrt{N}} \left\langle \mathbf{u}^\top \mathbf{r} \right\rangle_{c}, \qquad d_{WS} = p_{WH} - p_{SL}")}
with $N$ the number of units and $c$ the event class. The quantity $d_{{WS}}$ is the position of whisker hits relative to
spontaneous licks along the CD. The full whisker-hit displacement has a cross-validated length $\rho$, which is
$\lVert \bar{{\mathbf r}}_{{WH}} - \bar{{\mathbf r}}_{{SL}} \rVert / \sqrt N$ estimated from the product of random halves
(50 splits). Its angle to the CD is:
{eq('eq-theta', r"\theta = \arccos\left( d_{WS} / \rho \right)")}
so that a smaller $\theta$ means that the whisker-hit displacement points along the reward-lick direction. Area groups are
analysed with the same steps on their units (≥ 5 units and held-out AH-vs-SL $d' \ge 0.3$).

### Decoding

Cross-condition generalisation (CCGP) of the cued-lick code: a logistic-regression decoder (L2, C = 0.05, balanced classes,
units z-scored on training events) is trained to separate WH from SL-A and tested on AH vs SL-B, and conversely:
{eq('eq-ccgp', r"\mathrm{CCGP} = \tfrac{1}{2}\left[\mathrm{BA}(WH \rightarrow AH) + \mathrm{BA}(AH \rightarrow WH)\right] - \mathrm{null}")}
where BA is the balanced accuracy and the null is the mean over 20 linear shifts. Rates are the mean of the 50-ms bins in
[−100, 0) ms before the lick. To compare sessions and areas with different unit counts, CCGP is also computed on 40 randomly
drawn units (50 draws per session; areas with ≥ 40 units).

{R.methods('shift_null', 'statistics', 'areas')}
For the cohort × stage interaction, the difference of changes [expert − day 0](R+) − [expert − day 0](R−) is compared with its
distribution under 10,000 permutations of the cohort labels across mice. Area forests show the change expert − day 0 per
cohort (mean ± s.e.m. over sessions) with the omnibus terms in the panel title.

## Results

### Behaviour: whisker reaction times become auditory-like in R+

On day 0 whisker hits were slower than auditory hits in both cohorts. With learning, whisker reaction times shortened in R+
mice only: {RTW['rp']}; {RTW['rm']}; {RTW['int']} ({ref('fig-task')}d). Auditory reaction times changed less: {RTA['rp']}; {RTA['rm']}.
The dataset is summarised in {ref('fig-task')}e–f and {ref('tbl-sizes')}. Whole-brain lick-aligned PSTHs ({ref('fig-task')}g)
show that pre-lick activity before whisker hits lies between spontaneous licks and auditory hits on day 0 and approaches auditory
hits in R+ experts.

### Single neurons: non-sensory reward-lick neurons respond to whisker hits as to auditory hits after learning

The example R+ expert neuron in {ref('fig-neurons')}b has no stimulus response in the first 35 ms, but its rate rises before whisker
and auditory hits and stays low before spontaneous licks. Across sessions, the similarity index ({ref('eq-si')}) of
non-sensory reward-lick neurons started negative (whisker hits SL-like) in both cohorts and rose to about zero in R+ mice only:
{SI['rp']}; {SI['rm']}; {SI['int']}; {SI['ex']} ({ref('fig-neurons')}c). The cohorts did not differ on day 0 ({SI['d0']}).

Per area, the change in SI was positive for most area groups in R+ and near zero in R− ({ref('fig-neurons')}d). The omnibus model
gave a cohort × stage effect at the edge of significance ({A_SI['cs']}) and no evidence for area dependence ({A_SI['csa']}); no
area survived the post hoc correction (smallest q = {A_SI['qmin']}). The signed selectivities without neuron selection lead
to the same conclusion ({ref('fig-s-signed')}). The whisker-vs-auditory selectivity $s_{{WA}}$ closes toward zero in R+ only:
{SG['rp']}; {SG['rm']}; {SG['int']}. The whisker-vs-spontaneous selectivity $s_{{WS}}$ rises in R+ ({SGW['rp']}), but its
interaction is not significant ({SGW['int']}).

### Population: whisker hits move along the reward-lick coding direction in R+

Projected on the reward-lick CD ({ref('eq-cd')}, {ref('eq-proj')}), auditory hits lie above spontaneous licks before the lick in all
groups by construction, and whisker hits lie in between ({ref('fig-cd')}a). In R+ mice the whisker-hit projection rose with learning
({level('R+', 'WH', 'lvl_rp_wh')}), while the auditory-hit ({level('R+', 'AH', 'lvl_rp_ah')}) and spontaneous-lick projections did not;
in R− mice no class changed (WH {level('R-', 'WH', 'lvl_rm_wh')}) ({ref('fig-cd')}b).

The whisker-hit position relative to spontaneous licks, $d_{{WS}}$, doubled in R+ and did not change in R−: {DWS['rp']}; {DWS['rm']};
{DWS['ex']} ({ref('fig-cd')}c). The cohorts already differed on day 0 ({DWS['d0']}); day-0 sessions include the within-session learning
of R+ mice. The difference of changes was not significant ({DWS['int']}), so the across-days population result is a change in R+
without a significant cohort contrast. The length of the whisker-hit displacement increased in R+ only ({RHO['rp']}; {RHO['rm']};
{RHO['int']}). Its angle to the CD ({ref('eq-theta')}) decreased in R+ ({TH['rp']}; {TH['rm']}; {TH['int']}) ({ref('fig-cd')}d).
Per area, the change of $d_{{WS}}$ was positive in most groups of both cohorts, larger in R+, without an area-specific effect
({A_D['cs']}, {A_D['csa']}; smallest post hoc q = {A_D['qmin']}; {ref('fig-cd')}e).

### Decoding: the cued-lick code generalises across modalities in R+ experts

A decoder trained to separate whisker hits from spontaneous licks also separated auditory hits from other spontaneous licks,
and conversely ({ref('eq-ccgp')}; {ref('fig-ccgp')}a–c). CCGP above the linear-shift null was present on day 0 in both cohorts.
It rose with learning in R+ and was higher in R+ than in R− experts: {CC['rp']}; {CC['rm']}; {CC['ex']}; the interaction was
not significant ({CC['int']}) ({ref('fig-ccgp')}g). The pre-lick time course shows the generalisation building up over the 200 ms
before the lick ({ref('fig-ccgp')}e–f). Both decoding directions and the parallelism of the WH − SL and AH − SL differences follow
the same pattern ({ref('fig-ccgp')}h–j). Session CCGP correlates with the CD angle of the population analysis on the same and
on independent trials ({ref('fig-ccgp')}k–l).

With 40 units per session, the whole-brain result held: R+ {num('ccgp_eq_rp_d0', EQ['R+ learning'])} →
{num('ccgp_eq_rp_e', EQ['R+ expert'])} (p {pv('ccgp_eq_rp_p', EQ['p_MWU D0vsE R+'])}); expert R+ vs R− p
{pv('ccgp_eq_ex', EQ['p_MWU RpvsRm expert'])} ({ref('fig-ccgp')}m). Per area, CCGP rose in R+ in most groups and was higher in R+ than in R−
experts in all tested groups, but no area survived the correction across areas (smallest q of the R+ change
{num('ccgp_eq_qmin', EQA['q_MWU D0vsE R+'].min())}; {ref('fig-ccgp')}o–q).

### Controls

*Reaction time.* Pre-lick activity of fast whisker hits could contain the stimulus response. Restricting whisker and auditory hits
to reaction times ≥ 150 ms, or matching their reaction-time distributions, left the R+ increase of $d_{{WS}}$ and the expert cohort
difference unchanged ({ref('fig-s-rt')}, {ref('tbl-robust')}).

*Pre-lick kinematics.* If learning made the licking movement on whisker hits more like that on auditory hits, the CD change could
be motor. The jaw opening before the lick, measured as WH − AH in [−100, 0) ms, did not change with learning in either cohort
({JAWK['rp']}; {JAWK['rm']}; {JAWK['int']}); tongue, whisker and nose speed did not change either ({ref('fig-s-kin')}). Whisker hits
of R− mice opened the jaw less than their auditory hits at both stages.

*Alignment to jaw-opening onset.* The corrected first lick follows jaw-opening onset by about 90–190 ms. Defining the CD in the
100 ms before jaw onset (sessions with video) kept the $d_{{WS}}$ result: {J1['rp']}; {J1['rm']}; {J1['ex']}; {J1['int']}. A CD
centred on jaw onset gave {J2['rp']}; {J2['rm']} ({ref('fig-s-jaw')}, {ref('tbl-robust')}). The angle was less robust to this alignment
({J1T['rp']}).

*Behaviour in passive trials.* In passive whisker trials after the session, R+ experts opened the jaw, whisked, moved the nose and
dilated the pupil; R− experts and day-0 mice of both cohorts did not open the jaw ({ref('fig-s-dlc')}). The whisker stimulus
therefore evokes a learned orofacial response in R+ experts even without a lick.

## Discussion

At every level analysed, learning the whisker-reward contingency made pre-lick activity on whisker hits more similar to the
rewarded auditory-cued lick and less similar to the spontaneous lick, in R+ mice only. The change is clearest in single
non-sensory neurons, where the cohort contrast is significant. The population and decoding measures change in R+ and separate
the cohorts at the expert stage, but their difference of changes does not reach significance, partly because R+ and R− already
differ on day 0. The change is distributed: every area group with enough sessions moved in the same direction in R+, and none
stands out after correction. This fits a reward-expectation or cue-to-action signal that is broadcast widely before the lick,
rather than a change confined to the whisker pathway.

Two alternatives remain. First, the signal may reflect preparatory or motor-timing differences that the kinematic controls do not
capture (the lick follows the jaw, and fine timing is not trial-predictable from jaw onset). Second, the day-0 sessions already
contain within-session learning in R+, which compresses the across-days contrast; the within-day report addresses this directly.

## Caveats and limitations

- The learner definition is under review: the sheet-based learner list and the lick-based classification disagree for some mice
  (Methods). Day-0 results may change when one becomes canonical.
- Expert sessions are few (Table S1), from fewer mice, and are compared with day-0 sessions of other neurons; mouse-level
  permutation protects the interaction tests, but stage tests treat sessions as independent.
- Area tests are underpowered (≥ 3 sessions per group); absence of area specificity is not evidence of uniformity.
- The CD angle $\theta$ is not robust to jaw-onset alignment; $d_{{WS}}$ and the single-neuron index are.
- Spontaneous licks are on the piezo clock (± 25 ms from the trial clock).

## Open questions and next steps

- Settle the learner definition, then rebuild.
- Combined within-day report: both coding directions, day 0 with expert sessions as control, matched-RT and jaw-kinematics controls.
- Pseudo-population decoding with final repetition numbers (current single-session results are final; pseudo-population values are pilots and not shown).

## Figures

{figure('fig-task', LRN / 'Fig1_task_data.png', '''Task, behaviour and dataset (learners). **a**, Event classes and outcomes per cohort.
**b**, Alignment: the pre-lick window is [−100, 0) ms before the corrected first lick. **c**, Reaction-time distributions of all
trials. **d**, Median reaction time per session (mean ± s.e.m. over sessions) for whisker (yellow) and auditory (blue) hits by
cohort and stage (L: day 0, E: expert); tests in the text. **e**, Sessions per group with mice and recorded units (all learner
sessions, before the event minimum). **f**, Recorded units in CCF coordinates (green R+, magenta R−). **g**, First-lick-aligned
population PSTHs of all tested units (baseline-corrected; mean ± s.e.m. over sessions). **h**, Pre-lick rates per session and their
ratios to spontaneous licks; open: day 0, filled: expert; brackets Mann–Whitney U, interaction by mouse-level permutation.''',
                 "062_pub_convergence_figures.py")}
{figure('fig-neurons', LRN / 'report_fig2_single_neurons.png', rf'''Single neurons. **a**, Lick-aligned PSTHs of all tested units
(mean ± s.e.m. over sessions; shading: pre-lick window) for WH (yellow), AH (blue) and SL (black), per cohort and stage. **b**,
Example non-sensory reward-lick neuron (R+ expert): top, stimulus-aligned (whisker miss brown, whisker hit, auditory hit; blue
band: 5–35 ms sensory window, no response); bottom, lick-aligned (WH, AH, SL; event counts in legends). **c**, Similarity index
({ref('eq-si')}) of non-sensory reward-lick neurons, session means (open: day 0, filled: expert; mean ± s.e.m.; n sessions under
the axis: R+ D0 / E / R− D0 / E = {SI['n']}); brackets Mann–Whitney U; title: cohort × stage by mouse-level permutation.
**d**, Change of SI (expert − day 0) per area group (mean ± s.e.m. combined over the two stages; green R+, magenta R−; areas with
≥ 3 sessions per cohort and stage, allen_utils order); title: omnibus cohort × stage × area model (1,000 mouse-level permutations);
* post hoc per-area cohort × stage, BH q < 0.05 (none).''', "130_report_figures.py")}
{figure('fig-cd', LRN / 'report_fig3_population_cd.png', rf'''Population reward-lick coding direction ({ref('eq-cd')}). **a**,
Held-out projections on the CD around the lick (per neuron, mean ± s.e.m. over sessions; WH yellow, AH blue, SL black; n
sessions in titles). **b**, Session-mean projection in the pre-lick window per class, day 0 (open) vs expert (filled); title:
day 0 vs expert Mann–Whitney p per class. **c**, $d_{{WS}}$ ({ref('eq-proj')}); n sessions R+ D0 / E / R− D0 / E = {DWS['n']}; brackets
Mann–Whitney U; title cohort × stage permutation. **d**, Angle $\theta$ and length $\rho$ of the WH − SL displacement
({ref('eq-theta')}); n per group under the axis. **e**, Change of $d_{{WS}}$ (expert − day 0) per area group (as {ref('fig-neurons')}d).''',
                 "130_report_figures.py")}
{figure('fig-ccgp', BASE / 'ccgp_pub2' / 'ccgp_cue_pub2_learners.png', rf'''Cross-modal generalisation of the cued-lick code
({ref('eq-ccgp')}). **a**, Schematic: train WH vs SL-A, test AH vs SL-B. **b–c**, Example decoder outputs in an R+ and an R− expert
session. **d**, Within-modality cross-validated accuracy minus shift null (circles WH vs SL, squares AH vs SL). **e–f**, CCGP time
course (50-ms bins) on day 0 (dashed) and expert (solid) sessions, with the linear-shift control (grey). **g**, CCGP − shift null in
[−100, 0) ms; n sessions under the axis; brackets Mann–Whitney U; title cohort × stage permutation. **h–i**, The two decoding
directions. **j**, Parallelism cos(WH − SL, AH − SL) minus null. **k**, Session CCGP vs cos $\theta$ of the CD analysis (same trials).
**l**, CCGP on trial half 2 vs parallelism on half 1. **m–n**, CCGP and parallelism with 40 units × 50 draws per session. **o**,
Per area group, ranked by R+ expert CCGP (40 units). **p**, R+ change expert − day 0. **q**, Expert R+ − R−; filled: BH q < 0.05,
†: p < 0.05 uncorrected. 20 linear shifts per estimate.''', "112_ccgp_cue_pub2.py")}

## Supplementary figures and tables

{figure('fig-s-signed', BASE / 'unified_single_neuron_all' / 'fig_unified_shared_signed_learners.png', rf'''Single-neuron robustness:
signed selectivities of all non-sensory neurons without selection (count-matched draws 4 WH / 4 AH / 8 SL × 200; baseline
[−1.5, −1.0) s before the lick). **a**, Example neurons (R+ day 0 WH SL-like; R+ expert WH AH-like). **b**, Time courses of
$s_{{WS}}$ (WH vs SL) and $s_{{WA}}$ (WH vs AH) per cohort (grey: pre-lick window). **c**, Session means in the $s_{{WS}}$–$s_{{WA}}$ plane,
day 0 → expert (n = {SG['n']}); title p values per measure (Mann–Whitney; cohort × stage permutation). **d**, Per-area change
(expert − day 0) with the cohort ANOVA in the titles.''', "128_unified_figure.py", kind="supplementary")}
{figure('fig-s-rt', BASE / 'rt_adjusted' / 'rt_adjusted.png', '''Reaction-time control of the CD result. Top, RT distributions of WH
and AH per group (dotted: 150 ms). Rows, $d_{WS}$, $\\rho$ and $\\theta$ for all events (left), hits with RT ≥ 150 ms (middle) and WH / AH
matched on RT (20-ms bins, right); n sessions under each axis; tests as Figure 3c.''', "090_rt_adjusted_cd.py", kind="supplementary")}
{figure('fig-s-kin', BASE / 'prelick_kinematics' / 'prelick_kinematics_learners.png', '''Pre-lick kinematics (DLC, mm or °/s;
sessions with video) for the CD events: jaw opening, probability of tongue out, whisker speed and nose speed around the lick for
WH, AH and SL (mean ± s.e.m. over sessions), per group. Right, WH − AH in [−100, 0) ms per session group with Mann–Whitney brackets
and the cohort × stage permutation p.''', "116_prelick_kinematics.py", kind="supplementary")}
{figure('fig-s-jaw', JAW['jaw'] / 'cd_timecourse' / 'cd_jaw_timecourse.png', '''CD defined in the 100 ms before jaw-opening onset
(DLC; onset = first frame above 7 baseline SD), held-out projections around jaw onset per group, and $d_{WS}$ over time (right;
dashed day 0, solid expert). Sessions with video only; n in titles.''', "120_cd_jaw_timecourse.py", kind="supplementary")}
{figure('fig-s-dlc', BASE / 'dlc_passive_pre' / 'dlc_passive_pre_learners.png', '''Orofacial responses to passive stimuli (passive
trials before the session). Whisker speed, nose speed, jaw opening (mm), probability of tongue out and pupil area (% of baseline)
after whisker (top) and auditory (bottom) stimuli; dashed day 0, solid expert, green R+, magenta R−; mean ± s.e.m. over sessions
(n in legend). Pixels converted to mm with the per-setup calibration.''', "113_dlc_passive_pre.py", kind="supplementary")}
{table('tbl-sizes', sizes(), rf'''Sample sizes per group (learners). CD: sessions, mice and units of the whole-brain coding-direction
analysis (stable units, ≥ 0.1 Hz); other columns: sessions per analysis after its event minimum. Four R− day-0 sessions
({EXCLUDED_R_MINUS}) are excluded by the ≥ 8 events per lick type requirement.''')}
{table('tbl-robust', robust(), rf'''Robustness of the whisker-hit position on the CD ($d_{{WS}}$, {ref('eq-proj')}). p values Mann–Whitney U
(day 0 vs expert per cohort; expert R+ vs R−) and mouse-level permutation (cohort × stage).''')}
"""
    R_claims()
    md += "\n## Appendix\n\n" + R.registry_table("tbl-registry") + APPENDIX
    return md


def R_claims():
    claim("C1", "Whisker reaction times shorten to auditory-like values in R+ only", "Fig. 1d", f"{RTW['rp']}; {RTW['rm']}", RTW["n"])
    claim("C2", "Non-sensory reward-lick neurons respond to whisker hits more like auditory hits after learning, R+ only", "Fig. 2c",
          f"{SI['rp']}; {SI['rm']}; {SI['int']}", SI["n"])
    claim("C3", "Without neuron selection, WH vs AH selectivity closes in R+ only", "Fig. S1c", f"{SG['rp']}; {SG['int']}", SG["n"])
    claim("C4", "Whisker hits move along the reward-lick CD in R+, not R−", "Fig. 3c", f"{DWS['rp']}; {DWS['rm']}", DWS["n"])
    claim("C5", "Cohort difference of the d_WS change", "Fig. 3c", DWS["int"], DWS["n"], status="n.s.")
    claim("C6", "WH − SL displacement lengthens (ρ) in R+ only", "Fig. 3d", f"{RHO['rp']}; {RHO['int']}", RHO["n"])
    claim("C7", "WH − SL displacement aligns with the CD (θ) in R+", "Fig. 3d, S4", f"{TH['rp']}; {TH['int']}; jaw: {J1T['rp']}", TH["n"],
          status="fragile", note="not robust to jaw-onset alignment")
    claim("C8", "Cross-modal CCGP rises in R+ and exceeds R− at expert", "Fig. 4g, m", f"{CC['rp']}; {CC['ex']}", CC["n"])
    claim("C9", "Cohort difference of the CCGP change", "Fig. 4g", CC["int"], CC["n"], status="n.s.")
    claim("C10", "No area-specific effect (single neurons, CD)", "Fig. 2d, 3e", f"SI {A_SI['csa']}; d_WS {A_D['csa']}; no post hoc q < 0.05")
    claim("C11", "CD result holds for RT ≥ 150 ms, RT matching and jaw-onset alignment", "Table S2, Fig. S2, S4", f"jaw: {J1['rp']}")
    claim("C12", "Pre-lick jaw opening (WH − AH) does not change with learning", "Fig. S3", f"{JAWK['rp']}; {JAWK['int']}", JAWK["n"])
    claim("C13", "R+ experts open the jaw to passive whisker stimuli", "Fig. S5", "descriptive (no test)", status="fragile",
          note="no statistics computed")


APPENDIX = f"""
### Figure versions and provenance

Figures 2 and 3 are assembled (130_report_figures.py) from the panels of the latest abstract iteration (119, COSYNE_three_rows,
rows 2–3) and replace Fig2_single_neurons, Fig3_population_distance, Fig4_areas and Fig5_decoders of the previous report.
Source files, hashes and dates are in figures.json. COSYNE composites (COSYNE_three_rows, COSYNE_combined, COSYNE_across_days
v2–v6) are not included: their hit/miss-axis panels belong to ssl-hitmiss-axis-passive. Earlier versions superseded here: rate-based
modulation index (replaced by the ROC similarity index), λ / λ_LDA, pseudo-population CCGP (pilot repetitions), within-day Part II
(combined within-day report).

### Result folders (across_days/sl_stable/)

publication/learners (main figures, report_figures_stats.csv, rt_stats.csv); harmonised (087 CD metrics); similarity_index (104);
unified_single_neuron_all (127–128); ccgp_pub2, ccgp_sessions_eq (112, 114); rt_adjusted (090); prelick_kinematics (116);
dlc_passive_pre (113); ../sl_stable_jaw, ../sl_stable_jawc (118–120 jaw-onset runs). Statistics read by this report are frozen
in report_stable/_snapshots/.
"""


def summary(md_main):
    return f"""# ssl-prelick-convergence (across days): summary, {date.today().isoformat()}

Question: does learning make pre-lick activity on whisker-cued licks resemble the rewarded auditory-cued lick rather than the
spontaneous lick, in R+ but not R− mice (day 0 vs expert, learners, stable units)?

1. Behaviour: whisker RTs become auditory-like in R+ only ({RTW['rp']}; {RTW['rm']}).
2. Single neurons (non-sensory reward-lick neurons, ROC similarity index): {SI['rp']}; {SI['rm']}; {SI['int']}. Holds without neuron selection ({SG['int']} for WH vs AH selectivity).
3. Population CD: whisker hits move toward auditory hits in R+ ({DWS['rp']}; {DWS['rm']}); cohort contrast n.s. ({DWS['int']}); expert R+ > R− ({DWS['ex']}).
4. Decoding: cross-modal CCGP {CC['rp']}; {CC['rm']}; {CC['ex']}; {CC['int']}.
5. Areas: distributed, no area-specific effect after correction (SI {A_SI['csa']}; d_WS {A_D['csa']}).
6. Controls: RT ≥ 150 ms, RT matching and jaw-onset alignment keep the d_WS result; pre-lick jaw opening WH − AH unchanged ({JAWK['int']}). The angle θ is fragile.

Open: learner definition under review; within-day analyses go to the combined report.

![Population coding direction](figures/fig-cd.png)

Full report: report.pdf / report.html; claims registry: registry.csv.
"""


def main():
    md = build()
    R.write(OUT, md, summary=summary(md))


if __name__ == "__main__":
    main()
