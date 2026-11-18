"""Report of ssl-spectral-coupling (skills/project-report, report_lib). Version 3, 2026-11-18: exploratory, mouse level,
uncorrected (user decisions 2026-11-18). Replaces v1/v2 (pre-registered BH tests, AP-band envelope; git history).
Inputs: v2_stim / v2_jaw stats + figures (007), v2_jaw_prep (008), simulation (001).
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-spectral-coupling/report/build_report.py
Render locally: bash projects/ssl-spectral-coupling/report/render.sh
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
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(PROJ)]
from report_lib import Report  # noqa: E402
import spec_lib as sl  # noqa: E402

RES = sl.OUT
OUT = RES / "report"
R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
R.register([
    ("eq-msc", "Eq. (1)"), ("eq-corr", "Eq. (2)"), ("eq-excess", "Eq. (3)"),
    ("fig-sim", "Figure 1"), ("fig-spectra", "Figure 2"), ("fig-tr-between", "Figure 3"), ("fig-tr-within", "Figure 4"),
    ("fig-within-areas", "Figure 5"), ("fig-jaw-between", "Figure 6"), ("fig-prep", "Figure 7"),
    ("tbl-rates", "Table 1"), ("tbl-omni", "Table 2"), ("tbl-prep-areas", "Table 3"),
    ("fig-jaw-within", "Figure S1"), ("fig-pairs", "Figure S2"), ("fig-allmice", "Figure S3"),
    ("tbl-latency", "Table S1"), ("tbl-registry", "Table S2"),
])

# ------------------------------------------------------------------------------------------------ frozen inputs
TAG = date.today().isoformat()
FILES = {}
for al in ("stim", "jaw"):
    for pop in ("all", "learners"):
        for f in ("pair_tests", "within_tests", "omnibus_tests", "time_resolved_tests"):
            FILES[f"{al}_{pop}_{f}"] = RES / f"v2_{al}" / "stats" / pop / f"{f}.csv"
for pop in ("all", "learners"):
    for f in ("omnibus", "pairs", "areas"):
        FILES[f"prep_{pop}_{f}"] = RES / "v2_jaw_prep" / "stats" / pop / f"{f}.csv"
FILES["latency"] = RES / "v2_jaw_prep" / "stats" / "jaw_latency_tests.csv"
FILES["sim"] = RES / "simulation" / "001_simulation.csv"
FROZEN = {}
for grp in ("stim", "jaw", "prep", "latency", "sim"):  # one tag per group: report_lib names copies by 2 parent folders
    keys = [k for k in FILES if k.split("_")[0] == grp]
    FROZEN.update(zip(keys, Report.snapshot([FILES[k] for k in keys], OUT, tag=f"{TAG}_v3_{grp}")))
T = {k: pd.read_csv(v, keep_default_na=(k != "sim")) for k, v in FROZEN.items()}
FIG = {al: RES / f"v2_{al}" / "figures" for al in ("stim", "jaw")}


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def rate(t):
    v = t[t.p_mw.notna()]
    return len(v), int(v.both_p05.sum()), float(v.both_p05.mean()), int((v.both_p05 & (v["diff"] > 0)).sum())


def row(t, **kw):
    x = t
    for k, v in kw.items():
        x = x[x[k] == v]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


# ------------------------------------------------------------------------------------------------ numbers
mice = {}
for pop in ("all", "learners"):
    o = T[f"stim_{pop}_omnibus_tests"]
    q = o[(o.scope == "between") & (o.level == "group") & (o.window == "quiet") & (o.condition == "ALL") &
          (o.measure == "corr50")].iloc[0]
    mice[pop] = (int(q.n_Rplus), int(q.n_Rminus))

rate_rows = []
for al, lab in (("stim", "stimulus-aligned"), ("jaw", "jaw-aligned")):
    for pop in ("learners", "all"):
        for f, scope in (("pair_tests", "between-area pairs"), ("within_tests", "within areas"),
                         ("omnibus_tests", "omnibus (mean per mouse)")):
            n, k, fr, kp = rate(T[f"{al}_{pop}_{f}"])
            rate_rows.append({"alignment": lab, "population": pop, "family": scope, "tests": n,
                              "both p < 0.05": k, "fraction": f"{fr:.3f}", "of which R+ > R−": kp})
rates = pd.DataFrame(rate_rows)
rL = {(r["alignment"], r["family"]): r for r in rate_rows if r["population"] == "learners"}
pair_frac = float(rL[("stimulus-aligned", "between-area pairs")]["fraction"])
within_frac = float(rL[("stimulus-aligned", "within areas")]["fraction"])
within_k = rL[("stimulus-aligned", "within areas")]["both p < 0.05"]
within_kp = rL[("stimulus-aligned", "within areas")]["of which R+ > R−"]

oL = T["stim_learners_omnibus_tests"]
w_gl = row(oL, scope="within", level="acronym", window="quiet", condition="ALL", measure="msc", band="gamma_low")
w_nl = row(oL, scope="within", level="group", window="quiet", condition="W_nolick", measure="corr50", band="all")

# jaw-aligned time course: consecutive pre-movement windows with R- > R+ in corr50 (group, between, W_lick)
tj = T["jaw_learners_time_resolved_tests"]
tj = tj[(tj.level == "group") & (~tj.within) & (tj.condition == "W_lick") & (tj.measure == "corr50")].sort_values("t")
pre = tj[(tj.t < 0) & tj.both_p05 & (tj["diff"] < 0)]
pre_lo, pre_hi, pre_n = float(pre.t.min()), float(pre.t.max()), int(len(pre))
pre_pmax = float(pre[["p_mw", "p_welch"]].max(axis=1).max())
tjA = T["jaw_all_time_resolved_tests"]
tjA = tjA[(tjA.level == "group") & (~tjA.within) & (tjA.condition == "W_lick") & (tjA.measure == "corr50")]
preA_n = int((tjA[(tjA.t < 0)].both_p05 & (tjA[(tjA.t < 0)]["diff"] < 0)).sum())

# prep window (008)
po = T["prep_learners_omnibus"]
pw = row(po, level="group", condition="W_lick", measure="corr50", band="all")
pa_ = row(po, level="group", condition="A_hit", measure="corr50", band="all")
pf = row(po, level="group", condition="NS_fa", measure="corr50", band="all")
pwc = row(po, level="acronym", condition="W_lick", measure="corr50", band="all")
poA = T["prep_all_omnibus"]
pwA = row(poA, level="group", condition="W_lick", measure="corr50", band="all")
ar = T["prep_learners_areas"]
ar_tab = ar[(ar.condition == "W_lick") & (ar.measure == "corr50") & ar.p_mw.notna()].copy()
ar_tab = ar_tab.sort_values(["level", "p_welch"])
ar_sig = ar_tab[ar_tab.both_p05]
prep_tab = pd.concat([ar_tab[ar_tab.level == lv].head(8) for lv in ("group", "acronym")])
prep_tab = pd.DataFrame({"level": prep_tab.level, "area": prep_tab.area,
                         "n (R+/R−)": [f"{int(a)}/{int(b)}" for a, b in zip(prep_tab.n_Rplus, prep_tab.n_Rminus)],
                         "R+": prep_tab.mean_Rplus.map("{:.3f}".format), "R−": prep_tab.mean_Rminus.map("{:.3f}".format),
                         "p MW": prep_tab.p_mw.map(fmt_p), "p Welch": prep_tab.p_welch.map(fmt_p)})
n_ar_sig = len(ar_sig)
n_ar_sig_m = int((ar_sig["diff"] < 0).sum())
ar_g = ar_tab[ar_tab.level == "group"]
n_g, n_g_neg = len(ar_g), int((ar_g["diff"] < 0).sum())
ar_sig_names = ", ".join(f"{a} ({lv})" for a, lv in zip(ar_sig.area, ar_sig.level))

lat = T["latency"]
lw = row(lat, population="learners", trial_type="whisker_trial")
lat_tab = pd.DataFrame({"population": lat.population, "trial type": lat.trial_type.str.replace("_trial", ""),
                        "n mice (R+/R−)": [f"{int(a)}/{int(b)}" for a, b in zip(lat.n_Rplus, lat.n_Rminus)],
                        "R+ (s)": lat.mean_Rplus.map("{:.3f}".format), "R− (s)": lat.mean_Rminus.map("{:.3f}".format),
                        "p MW": lat.p_mw.map(fmt_p), "p Welch": lat.p_welch.map(fmt_p)})

# omnibus table: main cells, group level, msc beta + corr50, between and within (learners)
MAIN = [("quiet", "ALL"), ("early", "W_lick"), ("early", "W_nolick"), ("execution", "W_lick"), ("early", "A_hit"),
        ("execution", "A_hit")]
orows = []
for scope in ("between", "within"):
    for w, c in MAIN:
        for m, b in (("msc", "beta"), ("msc", "gamma_low"), ("corr50", "all")):
            r = row(oL, scope=scope, level="group", window=w, condition=c, measure=m, band=b)
            orows.append({"scope": scope, "window": w, "condition": c, "measure": m if b == "all" else f"{m} {b}",
                          "n (R+/R−)": f"{int(r.n_Rplus)}/{int(r.n_Rminus)}", "R+": f"{r.mean_Rplus:.4f}",
                          "R−": f"{r.mean_Rminus:.4f}", "p MW": fmt_p(r.p_mw), "p Welch": fmt_p(r.p_welch),
                          "flag": "*" if r.both_p05 else ""})
omni_tab = pd.DataFrame(orows)
n_omni_flag = int((omni_tab["flag"] == "*").sum())

sim = T["sim"]
sim_s = sim.groupby(["scenario", "measure"]).agg(det=("z", lambda z: np.mean(z > 2))).reset_index()
det_ff = float(sim_s[(sim_s.scenario == "ff") & (sim_s.measure == "msc")].det.iloc[0])
det_null = float(sim_s[(sim_s.scenario == "null") & (sim_s.measure == "msc")].det.iloc[0])

L, A = mice["learners"], mice["all"]

# ------------------------------------------------------------------------------------------------ text
EQ_MSC = r"\mathrm{MSC}_{AB}(f) = \frac{|\langle X_A(f)\,X_B^{*}(f)\rangle|^2}{\langle |X_A(f)|^2\rangle \langle |X_B(f)|^2\rangle}"
EQ_CORR = r"r_{AB} = \operatorname{corr}_{\text{trials} \times \text{50-ms bins}}\!\left(\textstyle\sum_{10\ \text{bins}} a,\ \sum_{10\ \text{bins}} b\right)"
EQ_EX = r"\Delta m = m_{\text{observed}} - \frac{1}{N_{\text{sh}}}\sum_{k=1}^{N_{\text{sh}}} m_{\text{shuffle } k}"

md = f"""---
title: "Coupling between and within brain areas in R+ and R− mice on the first whisker day"
subtitle: "ssl-spectral-coupling, exploratory (uncorrected), {TAG}"
date: "{TAG}"
---

**Data.** NWB_ks4 (Kilosort 4), stable good + mua units, whisker day 0; learners {num('n_l_p', L[0])} R+ /
{num('n_l_m', L[1])} R− mice (all mice {num('n_a_p', A[0])} / {num('n_a_m', A[1])}); cohort, learner status and
exclusions from the reference sheet. **Exploratory:** p values are uncorrected (user decision); a result is flagged
when Mann-Whitney and Welch both give p < 0.05. Unit of analysis: mouse.

# Abstract

We asked whether R+ and R− mice differ in how strongly brain areas co-fluctuate on the first whisker day, between areas
and within an area, before the stimulus, after it and around the first jaw movement. Coupling was measured on
trial-to-trial fluctuations of stable units as coherence (5-ms bins, multitaper) and as the correlation of 50-ms spike
counts, each as excess over a trial shuffle. Coupling is present in both cohorts. Stimulus-aligned between-area tests
are flagged at about the rate expected by chance. Two leads stand out. First, within areas, R+ mice tend to be more
coupled than R− mice in the quiet period. Second, on whisker licks aligned to the jaw onset, R− mice show stronger
between-area 50-ms correlation in the half second before the movement, a time course not explained by a cohort
difference in reaction time. Both are uncorrected and need confirmation.

**Key results**

1. {claim('c-sim', 'Coherence with a trial-shuffle null detects a known coupling in simulated populations', 'Figure 1', f'detection {det_ff:.0%} (coupled), {det_null:.0%} (independent)', '30 simulations')}
   ({ref('fig-sim')}).
2. {claim('c-present', 'Between- and within-area coupling exceed the shuffle null in both cohorts', 'Figure 2, 3, 4', 'mean excess > 0 in nearly all windows and bands', f'{L[0]}/{L[1]} mice')}
   ({ref('fig-spectra')}).
3. {claim('c-between-stim', 'Stimulus-aligned between-area tests are flagged at about chance rate', 'Table 1, Figure 3', f'{pair_frac:.1%} of pair tests both p < 0.05', f'{L[0]}/{L[1]} mice', 'n.s.')}
   ({ref('tbl-rates')}, {ref('fig-tr-between')}).
4. {claim('c-within', 'Within-area coupling tends to be higher in R+ in the quiet period', 'Table 1, 2, Figure 4, 5', f'{within_kp} of {within_k} flagged within-area tests R+ > R−; acronym quiet low-γ coherence p {fmt_p(w_gl.p_mw)} / {fmt_p(w_gl.p_welch)}', f'{L[0]}/{L[1]} mice', 'fragile', 'exploratory, uncorrected')}
   ({ref('fig-tr-within')}, {ref('fig-within-areas')}).
5. {claim('c-jaw', 'On whisker licks, R- has higher between-area 50-ms correlation before the jaw onset', 'Figure 6, 7', f'{pre_n} windows ({pre_lo:+.2f} to {pre_hi:+.2f} s) both p < 0.05; prep window p {fmt_p(pw.p_mw)} / {fmt_p(pw.p_welch)}', f'{int(pw.n_Rplus)}/{int(pw.n_Rminus)} mice', 'fragile', 'exploratory, uncorrected')}
   ({ref('fig-jaw-between')}, {ref('fig-prep')}).
6. {claim('c-latency', 'Jaw-onset latency does not differ between cohorts', 'Table S1', f'whisker licks {lw.mean_Rplus:.3f} vs {lw.mean_Rminus:.3f} s, p {fmt_p(lw.p_mw)} / {fmt_p(lw.p_welch)}', f'{int(lw.n_Rplus)}/{int(lw.n_Rminus)} mice', 'n.s.')}
   ({ref('tbl-latency')}).
7. Every claim with its statistic is listed in {ref('tbl-registry')}.

# Introduction

The communication pilot (ssl-communication-methods-pilot) found no consistent R+ vs R− difference in time-domain
communication between areas. Coupling may instead be confined to a frequency band, to one area, or to the period around
the movement. Here we measure coupling in the frequency domain (coherence) and in 50-ms spike counts, between and within
areas, aligned to the stimulus and to the first jaw movement. The hypothesis (user, 2026-11-17) is that R+ and R− differ
in this coupling in the quiet window or the response window (early sensory or execution). An earlier version of this
report tested pre-registered cells with BH correction and found no surviving difference; this version is exploratory.

# Methods

**Data and inclusion.** Whisker day-0 sessions with ephys; mice with `exclude == 0`, `exclude_ephys == 0`,
`recording == 1`, cohort R+ or R− from the reference sheet; learners = `learning_category` good or moderate; AB068 and
AB077 excluded. Units: stable good + mua (137_stable_units), firing rate ≥ 1 Hz, areas `area_acronym_custom` and coarse
groups. Trials: active context, `perf != 6`, `early_lick == 0`, auditory warm-up cut to one trial before the first
whisker trial, disengaged tail trimmed (rule A1). Conditions: whisker lick (W_lick), whisker no lick (W_nolick),
auditory hit (A_hit), no-stimulus correct rejection (NS_cr) and false alarm (NS_fa).

**Signals.** Spike counts in 5-ms bins; per unit the condition mean is removed in each bin, leaving trial-to-trial
fluctuations, which are soft-normalised. An area's signal is the mean of 15 random units (area group) or 8 (area
acronym); three random draws (groups) or two (acronyms). *Within-area* coupling uses two disjoint random halves of the
same area (areas with at least 30 or 16 units). 15 trials are drawn per condition (40 for all trials in the quiet
window).

**Measures.** Coherence ({ref('eq-msc')}), multitaper with DPSS tapers on the 5-ms signals, averaged over trials and
tapers and over bands (θ 3–8, α 8–13, β 13–30, low γ 30–60, high γ 60–100 Hz; θ and α only in windows ≥ 0.5 s):

{eq('eq-msc', EQ_MSC)}

The 50-ms correlation ({ref('eq-corr')}) sums the same signals into 50-ms bins and correlates the two areas over all
trials and bins of the window (time-domain coupling):

{eq('eq-corr', EQ_CORR)}

Both are reported as excess over a null in which the trials of one area are shuffled ({ref('eq-excess')}; 20
shuffles per window, 10 in time-resolved windows):

{eq('eq-excess', EQ_EX)}

**Windows.** Stimulus-aligned: quiet −1 to 0 s, early 5–155 ms, execution 0.2–0.5 s, and 200-ms windows every 50 ms.
Jaw-aligned (lick trials only; onset from the DLC jaw-onset table, onsets 0–1 s after the stimulus): pre −0.3 to 0 s,
post 0 to 0.3 s, 200-ms windows every 50 ms, and a follow-up window −0.7 to −0.2 s chosen after seeing the time course.

**Statistics.** Per mouse, the excess is averaged over draws and sessions; R+ vs R− by Mann-Whitney and Welch, both
reported, uncorrected; cohorts need at least 3 mice. *Omnibus*: per mouse the mean over its area pairs (or areas); *pair*
and *area* tests use the mice that have the pair or area. Learners are the main population; all mice the check.

# Results

## Coupling is present in both cohorts

Coherence and the 50-ms correlation exceed the shuffle null in nearly all windows and bands, between and within areas, and rise at the
stimulus and before the jaw movement ({ref('fig-spectra')}, {ref('fig-tr-between')}, {ref('fig-tr-within')}).
Within-area coupling is several times stronger than between-area coupling.

## Stimulus-aligned: no consistent between-area difference

Of the stimulus-aligned between-area pair tests in learners, {pair_frac:.1%} had both p < 0.05 ({ref('tbl-rates')}),
about the rate expected by chance for two correlated tests; where flagged, R− was more often the higher cohort.
In the main cells, {n_omni_flag} of {len(omni_tab)} omnibus tests were flagged ({ref('tbl-omni')}).
Over time, R− tends to have higher β coherence after the stimulus on whisker no-licks and around auditory hits, at
scattered windows ({ref('fig-tr-between')}); per pair, flagged cells are isolated ({ref('fig-pairs')}).

## Within areas: R+ tends to be more coupled in the quiet period

{within_frac:.1%} of within-area tests were flagged, {within_kp} of {within_k} with R+ > R− ({ref('tbl-rates')}). The
clearest cells are low-γ coherence within area acronyms in the quiet window (all trials; R+ {w_gl.mean_Rplus:.4f} vs R−
{w_gl.mean_Rminus:.4f}, p {fmt_p(w_gl.p_mw)} / {fmt_p(w_gl.p_welch)}) and the 50-ms correlation within area groups on
whisker no-lick trials, higher in R+ throughout the trial including before the stimulus (quiet window
{w_nl.mean_Rplus:.3f} vs {w_nl.mean_Rminus:.3f}, p {fmt_p(w_nl.p_mw)} / {fmt_p(w_nl.p_welch)}; {ref('fig-tr-within')}).
A difference present before the stimulus points to brain state rather than to the stimulus response: whisker no-licks
are misses in R+ (often in less engaged moments) but mostly correct withholding in R−. Per area group the differences go both
ways (SSp-orofacial and auditory areas lean R−, SSp-w and PPC R+; few cells flagged; {ref('fig-within-areas')}), so the
R+ lean rests on the acronym level and on the share of flagged tests, and is the weaker of the two leads. After 0.4 s on whisker licks, within-area β
coherence is higher in R+, when R+ mice are consuming the reward and R− mice are not.

## Jaw-aligned: R− has stronger between-area correlation before the movement

Aligned to the jaw onset, whisker licks show higher between-area 50-ms correlation in R− from {pre_lo:+.2f} to
{pre_hi:+.2f} s ({pre_n} windows with both p < 0.05, each p ≤ {pre_pmax:.3f}; {preA_n} such windows in all mice), the
cohorts converging at the onset ({ref('fig-jaw-between')}; all mice {ref('fig-allmice')}). Within areas, the jaw-aligned time courses do not show this difference ({ref('fig-jaw-within')}). With a median stimulus-to-jaw latency of about 0.2 s this
period lies mostly before the stimulus. It is not a shift of the stimulus response in time: jaw-onset latency does not
differ between cohorts ({ref('tbl-latency')}). Auditory hits and false alarms trend in the same direction without
reaching p < 0.05 in both tests.

In the follow-up window −0.7 to −0.2 s, averaged over area-group pairs, the effect was R+ {pw.mean_Rplus:.3f} vs R−
{pw.mean_Rminus:.3f} on whisker licks (p {fmt_p(pw.p_mw)} / {fmt_p(pw.p_welch)}; all mice p {fmt_p(pwA.p_mw)} /
{fmt_p(pwA.p_welch)}), {pa_.mean_Rplus:.3f} vs {pa_.mean_Rminus:.3f} on auditory hits (p {fmt_p(pa_.p_mw)} /
{fmt_p(pa_.p_welch)}) and {pf.mean_Rplus:.3f} vs {pf.mean_Rminus:.3f} on false alarms (p {fmt_p(pf.p_mw)} /
{fmt_p(pf.p_welch)}); over acronym pairs p {fmt_p(pwc.p_mw)} / {fmt_p(pwc.p_welch)} ({ref('fig-prep')}). This window was
chosen from the same data, so it confirms nothing by itself; it serves to locate the effect. The effect is broad
rather than carried by one pathway: {n_g_neg} of {n_g} area groups had R− > R+ when averaging the pairs that include
them, and {n_ar_sig} areas were flagged ({ar_sig_names}; {n_ar_sig_m} with R− > R+; {ref('tbl-prep-areas')}). Over
acronym pairs (8 units per area, fewer mice per pair) the mean difference is in the same direction but not flagged,
consistent with a widespread, low-amplitude shared fluctuation rather than a specific inter-areal channel.

# Discussion

Coupling between areas does not separate R+ from R− after the stimulus. Two exploratory leads remain, and both concern
the period before the stimulus or movement rather than the stimulus response: within areas R+ tends to be more coupled
in the quiet period, and between areas R− is more coupled in the half second before a whisker-triggered lick. Shared
slow fluctuations across areas (arousal, movement) inflate between-area correlation; R− mice may enter whisker trials
they respond to in a more globally synchronised state. Both leads should be tested on held-out data or with the window
fixed in advance, and against pre-stimulus movement and pupil measures.

# Caveats and limitations

- Uncorrected p values over many tests: about 2–3 % of tests are flagged by chance; isolated flags carry no weight.
- The −0.7 to −0.2 s jaw window was chosen after seeing the data.
- Trial-shuffle nulls keep shared slow state that varies within a window; the 50-ms correlation is most sensitive to it.
- Spiking coherence above about 30 Hz is close to the spiking noise floor at these rates.
- Each pair or acronym is recorded in only part of the mice; per-pair tests have small n.

# Next steps

- Pre-stimulus movement (DLC jaw, whisker, nose) and pupil per cohort in the same trials; regress them out of the area
  signals and retest the jaw-aligned effect.
- Fix the windows (quiet within-area; −0.7 to −0.2 s jaw) and test on expert days (held out).
- If the effect survives: directed measures (Granger, phase slope) on the flagged pairs.

# Figures

{figure('fig-sim', RES / 'simulation' / '001_simulation.png', 'Simulation of the measures (001): excess over a 20-shuffle null in the 24–36 Hz band, 30 simulated sessions per scenario, 15 Poisson units per area, 40 trials of 1 s: independent areas (null), A drives B (A→B), unrecorded and recorded common input. Coherence is the measure used in this report.', '001_simulation_validation.py')}

{figure('fig-spectra', FIG['stim'] / 'learners' / 'e1_msc_spectra.png', 'Coherence spectra, learners, stimulus-aligned: excess over the trial shuffle (Eq. (3)) between area groups (top, mean over pairs per mouse) and within area groups (bottom, two disjoint unit halves), mean ± s.e.m. over mice (R+ green, R− magenta) for the main cells. Markers: * Mann-Whitney and Welch p < 0.05, o one test (uncorrected).', '002_run_spikes.py --align stim, 007_exploratory_stats.py')}

{figure('fig-tr-between', FIG['stim'] / 'learners' / 'e2_time_resolved_between.png', 'Time-resolved between-area coupling, learners, stimulus-aligned: coherence in β, low γ and high γ and the 50-ms correlation (rows), per condition (columns), 200-ms windows every 50 ms; per mouse the mean over area-group pairs, mean ± s.e.m. over mice. Markers as Figure 2.', '007_exploratory_stats.py')}

{figure('fig-tr-within', FIG['stim'] / 'learners' / 'e2_time_resolved_within.png', 'As Figure 3 for within-area coupling (two disjoint halves of each area group).', '007_exploratory_stats.py')}

{figure('fig-within-areas', FIG['stim'] / 'learners' / 'e3_within_areas.png', 'Within-area coupling per area group, R+ − R− difference of the mouse means, learners, stimulus-aligned, main cells (columns), β and low-γ coherence and 50-ms correlation (rows). Solid bars and *: both tests p < 0.05.', '007_exploratory_stats.py')}

{figure('fig-jaw-between', FIG['jaw'] / 'learners' / 'e2_time_resolved_between.png', 'As Figure 3, aligned to the jaw onset (DLC), lick trials: whisker licks, auditory hits, false alarms.', '002_run_spikes.py --align jaw, 007_exploratory_stats.py')}

{figure('fig-prep', RES / 'v2_jaw_prep' / 'figures' / 'learners' / 'e5_prep_breakdown.png', 'Pre-movement window −0.7 to −0.2 s before the jaw onset, learners. Top: per mouse the mean over its between-area pairs (dots; bar = mean) for each lick condition, 50-ms correlation and β coherence, area groups and acronyms; numbers = p Mann-Whitney / Welch. Middle: per area (mean over the pairs that include it), Welch t of R+ vs R− on whisker licks, 50-ms correlation; n = mice R+/R−. Bottom: the most extreme pairs. * both p < 0.05.', '002_run_spikes.py --align jaw --windows prep, 008_prep_breakdown.py')}

{table('tbl-rates', rates, 'Fraction of R+ vs R− tests with both Mann-Whitney and Welch p < 0.05 (uncorrected), over all windows, conditions, bands and measures (coherence, 50-ms correlation), per family.')}

{table('tbl-omni', omni_tab, 'Omnibus R+ vs R− tests in the main cells, learners, area groups, stimulus-aligned: per mouse the mean excess (Eq. (3)) over its pairs (between) or areas (within). * both p < 0.05 (uncorrected).')}

{table('tbl-prep-areas', prep_tab, 'Per-area participation in the pre-movement window (−0.7 to −0.2 s before the jaw onset), whisker licks, 50-ms correlation, learners: per mouse the mean excess over the between-area pairs that include the area; the 8 lowest Welch p per level. Uncorrected.')}

# Supplementary

{figure('fig-jaw-within', FIG['jaw'] / 'learners' / 'e2_time_resolved_within.png', 'As Figure 6 for within-area coupling.', '007_exploratory_stats.py', kind='supplementary')}

{figure('fig-pairs', FIG['stim'] / 'learners' / 'e4_pairs.png', 'Between-area pairs, stimulus-aligned, learners: Welch t of R+ vs R− per pair of area groups (green R+ higher) for β and low-γ coherence and the 50-ms correlation (rows) in the main cells (columns). * both p < 0.05.', '007_exploratory_stats.py', kind='supplementary')}

{figure('fig-allmice', FIG['jaw'] / 'all' / 'e2_time_resolved_between.png', 'As Figure 6 for all mice.', '007_exploratory_stats.py', kind='supplementary')}

{table('tbl-latency', lat_tab, 'Jaw-onset latency from the stimulus (median per mouse of onsets 0–1 s on lick trials), mean over mice per cohort.')}

{R.registry_table('tbl-registry')}
"""

summary = (f"# ssl-spectral-coupling — summary (v3, exploratory)\n\nCoupling between and within areas, R+ vs R−, whisker "
           f"day 0, mouse level, uncorrected. Stimulus-aligned between-area tests flagged at chance rate. Leads: "
           f"within-area coupling higher in R+ in the quiet period; between-area 50-ms correlation higher in R− "
           f"before the jaw onset on whisker licks (not a latency difference). Both need confirmation.\n")
R.write(OUT, md, summary=summary)
print("written", OUT)
