"""Article report of ssl-zeta-responses (skills/project-report). Run on haas after 001 / 002:

    ~/code/unit_spikes_analysis/.venv/bin/python report/build_report.py

Reads tables/ and figures/ of combined_results_ks4/ssl-zeta-responses (frozen copies in report/_snapshots/<date>/) and one
session's _zeta_config.json (parameters); writes report/{report.md, numbers.json, figures/, registry.csv, summary.md,
build.sh}. PDF / HTML: bash report/build.sh where Pandoc + TinyTeX are installed.
"""
import glob
import json
import pathlib
import sys

import pandas as pd

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = RES / "ssl-zeta-responses"
REP = OUT / "report"
REPO = pathlib.Path.home() / "code" / "ibl-ai-agent"
sys.path.insert(0, str(REPO / "skills" / "project-report"))
from report_lib import Report  # noqa: E402

R = Report()
R.register([("fig-overview", "Figure 1"), ("tbl-types", "Table 1"), ("fig-areas", "Figure 2"),
            ("fig-latency", "Figure 3"), ("fig-dips", "Figure 4"), ("tbl-s-sizes", "Table S1"),
            ("tbl-s-registry", "Table S2")])
num, pv, ref, figure, table = R.num, R.pv, R.ref, R.figure, R.table

tabs = {p.stem: p for p in (OUT / "tables").glob("*.csv")}
frozen = dict(zip(tabs, R.snapshot(list(tabs.values()) + [OUT / "tables" / "numbers.json"], REP)))
T = pd.read_csv(frozen["analysis_types"]).set_index("analysis_type")
S = pd.read_csv(frozen["sample_sizes"])
LAT = pd.read_csv(frozen["area_latency"])
DIP = pd.read_csv(frozen["area_dips"])
N = json.load(open(REP / "_snapshots" / sorted(p.name for p in (REP / "_snapshots").iterdir())[-1] /
                   "ssl-zeta-responses_tables__numbers.json"))
cfg = json.load(open(sorted(glob.glob(str(RES / "*" / "whisker_*" / "zeta" / "*_zeta_config.json")))[0]))
W = cfg["windows_sec"]
NP = cfg["negative_peak"]
ms = lambda s: f"{1000 * s:.0f}"   # noqa: E731  parameter seconds -> ms (parameters, read from the config)
win = lambda w: f"{ms(w[0])} to {ms(w[1])} ms"  # noqa: E731
FIG = OUT / "figures"
ONE = ["whisker_passive_pre", "whisker_passive_post", "whisker_active", "auditory_passive_pre", "auditory_passive_post",
       "auditory_active", "spontaneous_licks"]
STIM = ["whisker_passive_pre", "whisker_passive_post", "whisker_active", "auditory_passive_pre",
        "auditory_passive_post", "auditory_active"]


def pct(key, v):
    return num(key, 100 * v, "{:.0f}") + " %"


def rng(key, s):
    return f"{pct(key + '_min', s.min())}–{pct(key + '_max', s.max())}"


# ------------------------------------------------------------------------------------------------ numbers
stim = T.loc[STIM]
two = T.drop(index=[t for t in T.index if t in ONE or "miss_active" in t])
two_stim = two.drop(index=[t for t in two.index if "baseline" in t or "pre_vs_post" in t])
pvp = T.loc[["whisker_pre_vs_post_learning", "auditory_pre_vs_post_learning", "wh_vs_aud_pre_vs_post_learning"]]
ws1 = LAT[(LAT.area_group == "Somatosensory-whisker") & (LAT.analysis_type == "whisker_active")].iloc[0]
mid = LAT[(LAT.area_group == "Midbrain") & (LAT.analysis_type == "whisker_active")].iloc[0]
ws1_a = LAT[(LAT.area_group == "Somatosensory-whisker") & (LAT.analysis_type == "auditory_active")].iloc[0]
aud_a = LAT[(LAT.area_group == "Auditory areas") & (LAT.analysis_type == "auditory_active")].iloc[0]
dip_act = DIP[DIP.analysis_type == "whisker_active"]
dip_pas = DIP[DIP.analysis_type == "whisker_passive_pre"]
nS, nM, nU = int(N["n_sessions"]), int(N["n_mice"]), int(N["n_units"])

# ------------------------------------------------------------------------------------------------ text
md = f"""---
title: "ZETA and latenZy responses across the brain (SSL)"
date: "{pd.Timestamp.today():%Y-%m-%d}"
---

Data: Kilosort 4, NWB_ks4; {num('n_sessions', nS)} sessions of {num('n_mice', nM)} mice (all whisker days), \
{num('n_units', nU)} good and multi-unit neurons. Pipeline: unit_spikes_analysis `zeta_analysis` (v2, 2026-11-19), \
zetapy {cfg['packages']['zetapy']}, latenZy {cfg['packages']['latenzy']}.

# Summary

The ZETA test was run on every neuron for the same analysis types, trials and neurons as the ROC pipeline, and latenZy \
gave each neuron's response onset. {R.claim('c1', 'ZETA finds about twice as many stimulus-responsive neurons as the ROC', 'Figure 1A', 'session means, Wilcoxon', nS)}: \
{rng('zeta_stim', stim.zeta_frac)} of neurons per stimulus condition against {rng('roc_stim', stim.roc_frac)} for the \
ROC, because the ROC counts spikes only in {win([0.005, 0.035])} and misses slower responses \
({ref('fig-overview')}, {ref('tbl-types')}). Around spontaneous licks, and for the contrasts of pre-stimulus baselines \
or of passive blocks before vs after the task, ZETA finds similar or fewer neurons than the ROC.

{R.claim('c2', 'latenZy gives an onset for most responsive neurons of the stimulus conditions but rarely for non-responsive ones', 'Table 1', 'coverage', nU)} \
({rng('cov_sig', stim.latency_cov_sig)} vs {rng('cov_ns', stim.latency_cov_nonsig)}). \
{R.claim('c3', 'Whisker onsets are earliest in midbrain and Ss-whisker and auditory onsets earliest in auditory areas', 'Figure 3A', 'session medians', '')} \
({ref('fig-latency')}). Early dips (inhibition within the first {ms(NP['window_s'][1])} ms) are rare and mostly in active whisker trials \
({ref('fig-dips')}).

# Methods

**Neurons and trials.** Good and multi-unit neurons (quality label), with the trials, events and analysis types of the \
ROC pipeline (`roc_utils_new`): the same events per type and no performance or disengagement filtering. Two extra \
one-condition types use active miss trials only (whisker / auditory stimulus without lick). Passive types follow \
`data_utils.keep_passive_units`. A neuron is tested in a condition when it has at least {cfg['min_events']} events. \
Whisker artefact: spikes in {win([-0.010, 0.005])} of each whisker onset are replaced by Poisson spikes at the rate of \
{win([-0.200, -0.010])} (`correct_neuron_spike_train`, as in the ROC). Sample sizes per area group: {ref('tbl-s-sizes')}.

**ZETA test.** One-condition types (stimulus or spontaneous lick vs no event): `zetatest` on {win(W['stim_one_condition'])} \
around the event, which compares the cumulative spike-time distribution with a constant rate and does not use bins. For \
stimuli it is run on both {win(W['stim_one_condition'])} (detects a step that stays up) and {win(W['stim_two_condition'])} \
(detects short transients), and the neuron's p is the smaller p times two (Bonferroni). Two-condition types (the ROC \
contrasts): `zetatest2` on spike counts per trial in {win(W['stim_two_condition'])} after the event, or in the \
pre-stimulus baseline {win(W['baseline'])} for the baseline types. {cfg['n_resampling']} resamplings, \
responsive = p < {cfg['alpha']}, no correction across neurons or types.

**Direction and peaks.** The instantaneous firing rate (IFR, zetapy's multi-scale estimate) is compared with its \
pre-event mean ({win(W['ifr_baseline'])}). Positive or negative = sign of its largest deviation between \
{ms(W['ifr_peak_min'])} ms and the window end, excluding the last {ms(W['ifr_edge_excluded'])} ms, where the IFR is biased. \
For two-condition types, positive = higher mean rate in condition 2 (the ROC convention; the ROC table's sign flip of the \
choice types is undone for the comparison). **Early dip**: in a responsive neuron, the IFR minimum over \
{win(NP['search_window_s'])} falls before {ms(NP['window_s'][1])} ms, is at least {NP['min_z']:.0f} pre-event SDs below \
the pre-event mean and at least {100 * NP['min_drop']:.0f} % below it.

**Onset latency.** latenZy (`latenzy` / `latenzy2`) on {win(W['latenzy_stim'])} around stimuli and \
{win(W['latenzy_lick'])} around licks, for every tested neuron. It returns an onset only when its own test finds a \
significant one, otherwise none; lick onsets can be negative (before the lick).

**ROC comparison.** The ROC result of the same neuron and type (`_roc_stage_analysis/roc_long.parquet`, permutation \
p < 0.05) uses spike counts in 5 to 35 ms after the event against −50 to −15 ms before it for one-condition types, and \
baseline-subtracted counts in that post window for the contrasts. Fractions are averaged within sessions, then across \
sessions; ZETA and ROC fractions are compared with a Wilcoxon signed-rank test over sessions.

# Results

## ZETA finds more stimulus responses than the ROC, mostly slower ones

Per stimulus condition, {rng('zeta_stim2', stim.zeta_frac)} of neurons are ZETA-responsive and \
{rng('roc_stim2', stim.roc_frac)} ROC-responsive (Wilcoxon over sessions p {pv('p_stim_max', stim.zeta_vs_roc_p.max())} \
for every condition; [Figure 1A](#fig-overview), {ref('tbl-types')}). Where both tests are significant, they agree on the \
sign in {rng('sign_stim', stim.sign_agreement)} of neurons. Only {rng('jacc_stim', stim.both_over_either)} of the neurons \
significant by either test are significant by both: ZETA adds neurons with responses after the ROC's 35-ms window \
(mean IFR, [Figure 1B](#fig-overview)), and the ROC adds neurons with small short transients that ZETA's longer window \
dilutes. For spontaneous licks the two fractions are close ({pct('zeta_sl', T.loc['spontaneous_licks', 'zeta_frac'])} \
vs {pct('roc_sl', T.loc['spontaneous_licks', 'roc_frac'])}).

Among the stimulus-locked contrasts (e.g. hit vs correct rejection, whisker vs auditory, choice), ZETA finds more \
neurons in {num('n_two_more', int((two_stim.zeta_frac > two_stim.roc_frac).sum()))} of \
{num('n_two', len(two_stim))}; the exceptions are the contrasts with spontaneous licks, which use a lick-aligned window \
for one condition ({ref('tbl-types')}). \
{R.claim('c4', 'For the pre- vs post-learning contrasts, ZETA and ROC disagree on the sign in most neurons', 'Table 1', 'sign agreement', '', status='holds', note='ZETA compares raw rates, the ROC baseline-subtracted counts')} \
(agreement {rng('sign_pvp', pvp.sign_agreement)}): `zetatest2` compares raw spike counts, so a baseline rate change \
between the passive blocks dominates, whereas the ROC subtracts each trial's baseline. These contrasts are therefore not \
interchangeable between the two methods.

{figure('fig-overview', FIG / 'fig1_overview.png', f'''ZETA and ROC responsive neurons. **A**, fraction of neurons \
responsive by ZETA (p < {cfg['alpha']}, dark) and by the ROC (permutation p < 0.05, light) per analysis type, mean over \
sessions (n per type in Table 1; miss-only types have no ROC). Left: one-condition types (event vs no event); right: \
two-condition contrasts. **B**, mean IFR minus its pre-event mean ({win(W['ifr_baseline'])}) of positive (solid) and \
negative (dashed) ZETA-responsive neurons, averaged within sessions then over sessions (mean ± s.e.m.; n = sessions). \
Whisker yellow, auditory blue, spontaneous licks black. For licks the pre-event period already contains the pre-lick \
change, so the trace is relative to a modulated baseline.''', script='002_figures.py')}

{table('tbl-types', T.reset_index()[['analysis_type', 'sessions', 'units', 'zeta_frac', 'roc_frac', 'zeta_vs_roc_p', 'both_over_either', 'sign_agreement', 'latency_cov_sig', 'latency_cov_nonsig']].round(3).rename(columns={'analysis_type': 'type', 'zeta_frac': 'ZETA', 'roc_frac': 'ROC', 'zeta_vs_roc_p': 'p', 'both_over_either': 'both / either', 'sign_agreement': 'same sign', 'latency_cov_sig': 'onset (resp.)', 'latency_cov_nonsig': 'onset (non-resp.)'}), '''Per analysis type: sessions and neuron-condition pairs tested; \
fraction ZETA- and ROC-responsive (mean over sessions) and Wilcoxon signed-rank p over sessions; neurons significant by \
both / by either test; among those, fraction with the same sign; fraction of responsive and non-responsive neurons with a \
latenZy onset.''')}

## Positive and negative neurons across area groups

Positive neurons are frequent in every area group in active trials and around spontaneous licks; negative neurons are \
rarer, most frequent around spontaneous licks ([Figure 2](#fig-areas)). \
{R.claim('c5', 'Negative neurons are a minority in every stimulus condition', 'Figure 2', 'session means', nS)} \
({rng('neg_stim', stim.neg_frac)} vs {rng('pos_stim', stim.pos_frac)} positive).

{figure('fig-areas', FIG / 'fig2_areas.png', f'''Fraction of positive (**A**) and negative (**B**) ZETA-responsive neurons \
per area group and one-condition type, mean over sessions (sessions with at least 5 neurons of the group; groups with at \
least 3 such sessions). Colour scales clipped at the 98th percentile.''', script='002_figures.py')}

## Onset latencies

latenZy onsets of positive neurons ([Figure 3](#fig-latency)) are earliest in Ss-whisker and midbrain for whisker \
stimuli (active: {num('lat_ws1', 1000 * ws1['mean'], '{:.0f}')} ± {num('lat_ws1_sem', 1000 * ws1['sem'], '{:.0f}')} and \
{num('lat_mid', 1000 * mid['mean'], '{:.0f}')} ± {num('lat_mid_sem', 1000 * mid['sem'], '{:.0f}')} ms, session medians, \
mean ± s.e.m.), and in auditory areas for auditory stimuli ({num('lat_aud', 1000 * aud_a['mean'], '{:.0f}')} ± \
{num('lat_aud_sem', 1000 * aud_a['sem'], '{:.0f}')} ms; Ss-whisker {num('lat_ws1_a', 1000 * ws1_a['mean'], '{:.0f}')} ms). \
Onset distributions have a fast mode within 20 ms and a slow tail; auditory onsets have a second mode near 50 ms \
([Figure 3B](#fig-latency)). Around spontaneous licks, most onsets precede the lick (median \
{num('lat_sl', N['lat_median_ms__spontaneous_licks'], '{:.0f}')} ms; [Figure 3C](#fig-latency)), as expected from the \
licking movement starting before the lick is detected.

{figure('fig-latency', FIG / 'fig3_latency.png', f'''latenZy onset latencies. **A**, onset of positive ZETA-responsive \
neurons per area group: median per session, mean ± s.e.m. over sessions (groups with at least 3 sessions), for passive \
whisker (pre-task, open circles), active whisker (squares) and active auditory (diamonds) stimuli. **B**, distribution \
of all onsets of positive neurons (n = neurons with an onset). **C**, onsets of positive (solid) and negative (dashed) \
spontaneous-lick neurons relative to the lick (latenZy window {win(W['latenzy_lick'])}).''', script='002_figures.py')}

## Early dips

{R.claim('c6', 'Early dips are rare and most frequent in active whisker trials', 'Figure 4A', 'session means', nS)}: \
{rng('dip_act', dip_act['mean'].dropna())} of neurons per area group in active whisker trials, \
{rng('dip_pas', dip_pas['mean'].dropna())} in passive ones ([Figure 4](#fig-dips)). Dips fall at a median of \
{num('dip_med', N['dip_median_ms'], '{:.0f}')} ms and {pct('dip_pos', N['dip_any_dir_pos'])} of them are in neurons whose \
dominant response is positive (a dip before or after an excitation).

{figure('fig-dips', FIG / 'fig4_dips.png', f'''Early dips (IFR minimum within {win(NP['search_window_s'])} falling before \
{ms(NP['window_s'][1])} ms, at least {NP['min_z']:.0f} pre-event SDs and {100 * NP['min_drop']:.0f} % below the pre-event \
mean, ZETA-responsive). **A**, fraction of tested neurons with a dip per area group, mean ± s.e.m. over sessions. **B**, \
dip times (n = neurons).''', script='002_figures.py')}

# Caveats

Every claim, its figure and status are listed in {ref('tbl-s-registry')}.

- Pooled over all whisker days, cohorts and learning stages; no stage or cohort comparison here.
- No correction across neurons or types (as the ROC); with p < {cfg['alpha']}, about {pct('fp', cfg['alpha'])} of \
neurons per type are expected false positives.
- latenZy onsets exist only where its own test is significant; a missing onset is not a long latency.
- The one-condition stimulus test includes the pre-event period; a neuron with a slow drift across the event can be \
responsive.
- Lick IFR traces are relative to the pre-lick period, which is itself modulated.

# Appendix

{table('tbl-s-sizes', S, 'Sessions, mice and neurons per area group (whisker_active type; other types in Table 1).')}

{R.registry_table('tbl-s-registry')}

Code: `projects/ssl-zeta-responses/{{001_assemble.py, 002_figures.py, report/build_report.py}}`; per-session results \
`combined_results_ks4/<mouse>/whisker_<day>/zeta/`; pooled table, figures and tables in \
`combined_results_ks4/ssl-zeta-responses/`.
"""

summary = f"""# ZETA responses: summary

- ZETA finds {rng('s_zeta', stim.zeta_frac)} stimulus-responsive neurons per condition vs {rng('s_roc', stim.roc_frac)} \
for the ROC (5–35 ms window); responses after 35 ms make the difference.
- latenZy onset for {rng('s_cov', stim.latency_cov_sig)} of responsive neurons; earliest in midbrain / Ss-whisker \
(whisker) and auditory areas (auditory).
- Pre- vs post-learning contrasts differ in sign between ZETA (raw rates) and ROC (baseline-subtracted).
- Early dips are rare (mostly active whisker trials).

![Figure 1](figures/fig-overview.png)
"""
R.write(REP, md, summary=summary)
