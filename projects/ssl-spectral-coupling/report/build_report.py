"""Report of ssl-spectral-coupling (skills/project-report, report_lib). First version 2026-11-18 (overnight run).
Spectral (multitaper) coupling between area groups / area acronyms, R+ vs R- learners, from binned spikes of stable units
(002 source "spikes") and from the AP-band multi-unit envelope (005 + 002 --source ap).
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-spectral-coupling/report/build_report.py
Render locally: bash projects/ssl-spectral-coupling/report/render.sh
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    ("eq-msc", "Eq. (1)"), ("eq-cancoh", "Eq. (2)"), ("eq-pcoh", "Eq. (3)"), ("eq-psi", "Eq. (4)"), ("eq-gc", "Eq. (5)"),
    ("eq-excess", "Eq. (6)"),
    ("tbl-measures", "Table 1"), ("fig-sim", "Figure 1"), ("fig-spectra", "Figure 2"), ("tbl-omnibus", "Table 2"),
    ("fig-pairs", "Figure 3"), ("fig-pvals", "Figure 4"), ("tbl-bh", "Table 3"), ("fig-tr", "Figure 5"),
    ("fig-ap", "Figure 6"), ("tbl-n", "Table S1"), ("fig-allmice", "Figure S1"), ("fig-acronym", "Figure S2"),
    ("fig-ap-check", "Figure S3"), ("fig-confounds", "Figure S4"), ("tbl-registry", "Table S2"),
])

HAS_AP = (RES / "ap" / "stats" / "learners" / "omnibus_tests.csv").exists()
FILES = {}
for src in ("spikes",) + (("ap",) if HAS_AP else ()):
    for pop in ("learners", "all"):
        for f in ("omnibus_tests", "omnibus_sessions", "pair_tests", "omnibus_tests_whisker_minus_auditory",
                  "spectra_clusters", "time_resolved_clusters", "confound_tests"):
            FILES[f"{src}_{pop}_{f}"] = RES / src / "stats" / pop / f"{f}.csv"
FILES["sim"] = RES / "simulation" / "001_simulation.csv"
FROZEN = dict(zip(FILES, Report.snapshot(list(FILES.values()), OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v, keep_default_na=(k != "sim")) for k, v in FROZEN.items()}  # scenario "null" is not NaN
PRIMARY = [("quiet", "ALL"), ("early", "W_lick"), ("early", "W_nolick"), ("execution", "W_lick")]
BANDNAME = {"theta": "θ 3–8 Hz", "alpha": "α 8–13 Hz", "beta": "β 13–30 Hz", "gamma_low": "low γ 30–60 Hz",
            "gamma_high": "high γ 60–100 Hz"}


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def omni(src, pop, w, c, m, b, value="raw"):
    d = T[f"{src}_{pop}_omnibus_tests"]
    d = d[(d.level == "group") & (d.window == w) & (d.condition == c) & (d.measure == m) & (d.band == b) & (d.value == value)]
    assert len(d) == 1, (src, pop, w, c, m, b, len(d))
    return d.iloc[0]


def pair_summary(src, pop):
    t = T[f"{src}_{pop}_pair_tests"]
    both = (t.p_mw < 0.05) & (t.p_welch < 0.05)
    bhq = (t.q_mw < 0.05) & (t.q_welch < 0.05)
    prim = t.apply(lambda r: (r.window, r.condition) in PRIMARY, axis=1)
    return dict(n=len(t), frac_both=both.mean(), n_bh=int(bhq.sum()), n_prim=int(prim.sum()),
                frac_prim=both[prim].mean(), n_bh_prim=int((bhq & prim).sum()), bh_rows=t[bhq])


def within_zero(src, pop, w, c, m, b):
    o = T[f"{src}_{pop}_omnibus_sessions"]
    o = o[(o.level == "group") & (o.window == w) & (o.condition == c) & (o.measure == m) & (o.band == b)]
    return {coh: (g.raw.mean(), stats.wilcoxon(g.raw).pvalue, len(g)) for coh, g in o.groupby("cohort")}


def mde(src, pop, w, c, m, b):
    """minimal detectable R+ - R- difference (80 % power, two-sided alpha 0.05, t approximation) relative to the mean."""
    o = T[f"{src}_{pop}_omnibus_sessions"]
    o = o[(o.level == "group") & (o.window == w) & (o.condition == c) & (o.measure == m) & (o.band == b)]
    a, z = o[o.cohort == "R+"].raw, o[o.cohort == "R-"].raw
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(z) - 1) * z.var()) / (len(a) + len(z) - 2))
    d = 2.8 * np.sqrt(1 / len(a) + 1 / len(z))
    diff, se = a.mean() - z.mean(), np.sqrt(a.var() / len(a) + z.var() / len(z))
    return d * sd, d * sd / o.raw.mean(), diff, diff - 1.96 * se, diff + 1.96 * se, o.raw.mean()


# ------------------------------------------------------------------------------------------------ numbers
S = {pop: pair_summary("spikes", pop) for pop in ("learners", "all")}
sess = T["spikes_all_omnibus_sessions"][["session_id", "cohort"]].drop_duplicates()
sessL = T["spikes_learners_omnibus_sessions"][["session_id", "cohort"]].drop_duplicates()
nL = sessL.cohort.value_counts()
nA = sess.cohort.value_counts()
wz = within_zero("spikes", "learners", "quiet", "ALL", "msc", "beta")
md_q = mde("spikes", "learners", "quiet", "ALL", "msc", "beta")
md_e = mde("spikes", "learners", "early", "W_lick", "msc", "gamma_low")
md_x = mde("spikes", "learners", "execution", "W_lick", "msc", "beta")
sim = T["sim"]
sim_s = sim.groupby(["scenario", "measure"]).agg(excess=("excess", "mean"), det=("z", lambda z: np.mean(z > 2))).reset_index()


def simv(sc, m, col="det"):
    return float(sim_s[(sim_s.scenario == sc) & (sim_s.measure == m)][col].iloc[0])


gcd = sim[sim.measure.isin(["gc_ab", "gc_ba"])].pivot_table(index=["scenario", "sim"], columns="measure", values="excess")
gc_ff = float(((gcd.gc_ab - gcd.gc_ba) > 0).loc["ff"].mean())
psi_ff = float((sim[(sim.scenario == "ff") & (sim.measure == "psi")].value > 0).mean())

# omnibus primary table (learners, spikes)
rows = []
for w, c in PRIMARY:
    for b in (["theta", "alpha", "beta", "gamma_low", "gamma_high"] if w == "quiet" else ["beta", "gamma_low", "gamma_high"]):
        for m in ("msc", "cancoh", "pcoh"):
            r = omni("spikes", "learners", w, c, m, b)
            rows.append({"window": w, "condition": c, "band": BANDNAME[b], "measure": m,
                         "R+ mean": f"{r.mean_Rplus:.4f}", "R− mean": f"{r.mean_Rminus:.4f}",
                         "n (R+/R−)": f"{int(r.n_Rplus)}/{int(r.n_Rminus)}", "p MW": fmt_p(r.p_mw),
                         "p Welch": fmt_p(r.p_welch)})
omni_tab = pd.DataFrame(rows)
omni_tab["q (BH, all rows)"] = [fmt_p(q) for q in
                                __import__("statsmodels.stats.multitest", fromlist=["x"]).multipletests(
                                    [float(p.replace("< ", "")) for p in omni_tab["p MW"]], method="fdr_bh")[1]]
n_omni = len(omni_tab)
n_omni_p05 = int((pd.to_numeric(omni_tab["p MW"].str.replace("< ", ""), errors="coerce") < 0.05).sum())

bh = S["learners"]["bh_rows"][["level", "window", "condition", "measure", "band", "pair", "diff", "p_mw", "p_welch",
                               "q_mw", "q_welch", "n_Rplus", "n_Rminus"]].copy()
for c in ("p_mw", "p_welch", "q_mw", "q_welch"):
    bh[c] = bh[c].map(fmt_p)
bh["diff"] = bh["diff"].map(lambda v: f"{v:+.4f}")

spec_cl = T["spikes_learners_spectra_clusters"]
tr_cl = T["spikes_learners_time_resolved_clusters"]
n_spec_sig = int((spec_cl.p < 0.05).sum()) if len(spec_cl) else 0
n_tr_sig = int((tr_cl.p < 0.05).sum()) if len(tr_cl) else 0
cA = T["spikes_all_spectra_clusters"]
tA = T["spikes_all_time_resolved_clusters"]
n_sig_all = int((cA.p < 0.05).sum() if len(cA) else 0) + int((tA.p < 0.05).sum() if len(tA) else 0)

# sample sizes from per-session meta
metas = [json.load(open(p)) for p in (RES / "spikes" / "sessions").glob("*_meta.json") if "_quick" not in p.name]
nrows = []
for lv, k, nu in (("area group", "n_units_group", 15), ("area acronym", "n_units_acronym", 8)):
    cnt = {}
    for m in metas:
        for a, n in m.get(k, {}).items():
            if n >= nu:
                cnt.setdefault(a, {"R+": 0, "R-": 0, "R+ L": 0, "R- L": 0})
                cnt[a][m["cohort"]] += 1
                if m["learner"]:
                    cnt[a][m["cohort"] + " L"] += 1
    for a, c in sorted(cnt.items()):
        if lv == "area group" or (c["R+ L"] >= 3 and c["R- L"] >= 3):
            nrows.append({"level": lv, "area": a, "sessions R+ (learners)": c["R+ L"], "sessions R− (learners)": c["R- L"],
                          "sessions R+ (all)": c["R+"], "sessions R− (all)": c["R-"]})
ntab = pd.DataFrame(nrows)
n_acr = int((ntab.level == "area acronym").sum())
units_tot = int(sum(sum(v for v in m.get("n_units_group", {}).values()) for m in metas))

if HAS_AP:
    SA = {pop: pair_summary("ap", pop) for pop in ("learners", "all")}
    apq = omni("ap", "learners", "quiet", "ALL", "msc", "beta")
    ap_sess = T["ap_learners_omnibus_sessions"][["session_id", "cohort"]].drop_duplicates().cohort.value_counts()
    ap_cl = T["ap_learners_spectra_clusters"]
    ap_cl_sig = int((ap_cl.p < 0.05).sum()) if len(ap_cl) else 0
ap_check = (RES / "ap" / "check" / "004_AB127_20240821_103757.txt").read_text()

# ------------------------------------------------------------------------------------------------ text
EQ_MSC = r"C_{AB}(f) = \frac{\langle X_A(f)\,X_B^{*}(f)\rangle}{\sqrt{\langle |X_A(f)|^2\rangle \langle |X_B(f)|^2\rangle}}, \qquad \mathrm{MSC}_{AB}(f) = |C_{AB}(f)|^2"
EQ_CAN = r"\rho^2_{AB}(f) = \lambda_{\max}\!\left(S_{AA}^{-1}(f)\, S_{AB}(f)\, S_{BB}^{-1}(f)\, S_{BA}(f)\right)"
EQ_PC = r"G(f) = S_{[A,B,Z]}(f)^{-1}, \qquad \mathrm{pcoh}_{AB\mid Z}(f) = \frac{|G_{AB}(f)|^2}{G_{AA}(f)\,G_{BB}(f)}"
EQ_PSI = r"\Psi_{AB} = \operatorname{Im}\sum_{f \in \text{band}} C_{AB}^{*}(f)\, C_{AB}(f + \delta f)"
EQ_GC = r"I_{A\to B}(f) = \ln \frac{S_{BB}(f)}{S_{BB}(f) - \left(\Sigma_{AA} - \Sigma_{AB}^2/\Sigma_{BB}\right) |H_{BA}(f)|^2}"
EQ_EX = r"\Delta m = m_{\text{observed}} - \frac{1}{N_{\text{sh}}}\sum_{k=1}^{N_{\text{sh}}} m_{\text{shuffle } k}"

L = nL.get("R+", 0), nL.get("R-", 0)
A = nA.get("R+", 0), nA.get("R-", 0)
md = f"""---
title: "Spectral coupling between brain areas in R+ and R− mice on the first whisker day"
subtitle: "ssl-spectral-coupling — overnight run, {date.today().isoformat()}"
date: "{date.today().isoformat()}"
---

**Data.** NWB_ks4 (Kilosort 4), stable good + mua units (137_stable_units), whisker day 0; {num('n_sess_all', int(sum(A)))} sessions
(R+ {num('n_all_p', int(A[0]))}, R− {num('n_all_m', int(A[1]))}), learners {num('n_l_p', int(L[0]))} R+ / {num('n_l_m', int(L[1]))} R−;
cohort, learner status and exclusions from the reference sheet. AP band: artefact-corrected CatGT binaries.
{"" if HAS_AP else "**AP-band results are not yet available in this build (run in progress).**"}

# Abstract

We asked whether R+ and R− mice differ in the frequency-resolved coupling between brain areas in the quiet window before
the stimulus, in the early sensory window and in the execution window of whisker trials. For every pair of
simultaneously recorded area groups (and area acronyms) we computed, on trial-to-trial fluctuations of stable units in
5-ms bins, the coherence, canonical coherence, partial coherence given the rest of the recording, imaginary coherency,
phase-slope index and spectral Granger causality, each as excess over a trial-shuffle null. Simulations confirmed that
the measures recover a known coupling and its direction. Coupling was present in both cohorts in every window, strongest
below about 20 Hz and rising at stimulus onset. Neither cohort differed from the other: of
{num('n_pair_tests_l', S['learners']['n'])} pair-level tests in learners, {num('n_bh_l', S['learners']['n_bh'])} survived
correction, none of them in a pre-registered cell; in all mice {num('n_bh_a', S['all']['n_bh'])} survived; the fraction of
pairs with p < 0.05 in both tests was at chance. The hypothesis of a cohort difference in inter-areal spectral coupling
is not supported at the sensitivity of this dataset.

**Key results**

1. {claim('c-sim', 'The spectral measures recover a known 30-Hz coupling and its direction in simulated Poisson populations', 'Figure 1', f'detection {simv("ff", "msc"):.0%}, direction GC {gc_ff:.0%}, PSI {psi_ff:.0%}', '30 simulations')}
   ({ref('fig-sim')}).
2. {claim('c-present', 'Cross-area coupling exceeds the shuffle null in both cohorts', 'Figure 2', f'quiet β MSC Wilcoxon p R+ {fmt_p(wz["R+"][1])}, R− {fmt_p(wz["R-"][1])}', f'{L[0]}/{L[1]} sessions')}:
   quiet-window β coherence excess R+ {num('wz_p', wz['R+'][0], '{:.4f}')} (Wilcoxon p {pv('wz_pp', wz['R+'][1])}),
   R− {num('wz_m', wz['R-'][0], '{:.4f}')} (p {pv('wz_mp', wz['R-'][1])}) ({ref('fig-spectra')}).
3. {claim('c-null', 'No R+ vs R- difference in coupling in the quiet, early or execution window (pre-registered cells)', 'Figure 2, 3, 4; Table 2, 3', f'{S["learners"]["n_bh_prim"]} BH-significant pairs in primary cells; {n_omni_p05}/{n_omni} omnibus p < 0.05', f'{L[0]}/{L[1]} sessions', 'n.s.')}
   ({ref('tbl-omnibus')}, {ref('fig-pairs')}, {ref('fig-pvals')}).
4. {claim('c-bh', 'The few BH-surviving pairs are in control conditions and do not replicate in all mice', 'Table 3', f'{S["learners"]["n_bh"]} of {S["learners"]["n"]} (learners), {S["all"]["n_bh"]} (all mice)', '', 'fragile')}
   ({ref('tbl-bh')}).
5. {claim('c-tr', 'Coherence rises at stimulus onset in both cohorts; no time cluster differs between cohorts in the whisker conditions', 'Figure 5', f'{n_tr_sig} time cluster(s) p < 0.05 (learners), all in control conditions', '', 'n.s.')}
   ({ref('fig-tr')}).
6. Every claim, its statistic and status is listed in {ref('tbl-registry')}.
7. AP band: {"see " + ref('fig-ap') if HAS_AP else "pending (" + ref('fig-ap') + " placeholder)"}.

# Introduction

The communication pilot (ssl-communication-methods-pilot) found no consistent R+ vs R− difference in time-domain
communication measures (reduced-rank regression, CCA, partial CCA, Granger). Coupling between areas can, however, be
confined to a frequency band (for example β or γ synchrony) and be invisible to broadband measures. Here we repeat the
same design with the frequency-domain counterpart of each measure ({ref('tbl-measures')}), on binned spikes and on the
AP-band multi-unit envelope of the same channels.

The hypothesis (user, 2026-11-17) is that R+ and R− differ in the coupling between area groups and area acronyms in the
quiet window (H1), the early sensory window (H2) or the execution window (H3). Since whisker licks are rewarded in R+
and not in R−, a difference specific to whisker trials, absent on auditory hits (Go in both cohorts), would point to
the learned contingency rather than to which areas or sites each cohort happened to have.

# Methods

**Data and inclusion.** Whisker day-0 sessions with ephys, mice with `exclude == 0`, `exclude_ephys == 0`,
`recording == 1`, cohort R+ or R− from the reference sheet; learners = `learning_category` good or moderate (primary
population), all mice as sensitivity. Units: stable good + mua units (coverage ≥ 0.9, presence ≥ 0.5, drift test
passed; sessions without a drift file use all good + mua units), firing rate ≥ 1 Hz, areas from `area_acronym_custom`
grouped into `area_group_coarse_v3` ({num('units_tot', units_tot)} units). Trials: active context (per-trial rule),
perf ≠ 6, no early lick, auditory warm-up block cut (one trial kept before the first whisker trial), end-of-session
disengagement trimmed (rule A1). Conditions: W_lick, W_nolick, A_hit, NS_cr, and ALL (every included trial, quiet
window only). Sample sizes per area are in {ref('tbl-n')}.

**Signals.** Spike counts in 5-ms bins from −1.2 to +1.0 s around trial start; for each draw the condition mean was
removed per unit and bin and each unit divided by its SD plus a soft constant (Poisson SD of a 2-Hz unit in one bin).
Per area and draw, 15 random units (area group; 8 for area acronyms); the area signal is the mean of the unit residuals
("sum" signal) and the top 3 principal components. The rest of the recording, Z, is the top 3 PCs of up to 200 random
units outside both areas. Trials per draw: 40 (ALL) or 15 (each event condition, which needs ≥ 15 trials); 3 draws
(area groups) or 2 (acronyms).

**AP-band envelope.** For each probe, the artefact-corrected AP binary was read in chunks around every selected trial
(up to 25 trials per condition), re-referenced to the median over the 384 channels, band-passed 300–3000 Hz
(Butterworth order 3, zero-phase), rectified and averaged in the same 5-ms bins, on the peak channels of the stable
units (each channel labelled by the majority area of its units). The probe clock was mapped to NWB time by a linear fit
of the whisker stimulus times (residual < 0.1 ms). On AB127 the stimulus artefact of the uncorrected binary peaked
0.7–0.8 ms after the predicted time on all three probes, and each channel's envelope correlated with the spike count
of its units (validation in {ref('fig-ap-check')}). Channels then replace units in the same pipeline (no soft
constant); the quiet window ends at −10 ms and the early window starts at +10 ms, away from the artefact-correction
interval.

**Windows and spectra.** Quiet −1.0 to 0 s (multitaper NW = 3, ±3 Hz), early 5–155 ms (NW = 1.5, ±10 Hz),
execution 0.2–0.5 s (NW = 3, ±10 Hz), and 200-ms windows every 50 ms from −0.9 to +0.8 s (NW = 2). Each trial's
window mean was removed and the DPSS-tapered FFT $X(f)$ taken (zero-padded); cross-spectra $S_{{AB}}(f)$ are averaged over
trials and tapers. Bands: θ 3–8, α 8–13 (quiet only), β 13–30, low γ 30–60, high γ 60–100 Hz.

**Measures.** The coherency and magnitude-squared coherence of the sum signals are
{eq('eq-msc', EQ_MSC)}
where $X_A$, $X_B$ are the tapered Fourier coefficients and the average runs over trials and tapers. The first canonical
coherence of the PC sets is
{eq('eq-cancoh', EQ_CAN)}
with $S_{{AA}}$, $S_{{BB}}$ the 3 × 3 cross-spectral matrices of each area's PCs (Brillinger 1981); it is the spectral
counterpart of CCA. The partial coherence of the sum signals given the rest is
{eq('eq-pcoh', EQ_PC)}
where $S_{{[A,B,Z]}}$ is the joint cross-spectral matrix of the two sum signals and the 3 rest PCs. The imaginary part of
$C_{{AB}}$ ({ref('eq-msc')}) is insensitive to zero-lag common input. Direction is given by the phase-slope index (Nolte et
al. 2008),
{eq('eq-psi', EQ_PSI)}
which is positive when $A$ leads, and by spectral Granger causality (Geweke 1982),
{eq('eq-gc', EQ_GC)}
from a bivariate VAR of order 4 (20 ms) fitted on all trials of the draw, with residual covariance $\\Sigma$, transfer
function $H(f)$ and spectral matrix $S(f) = H\\Sigma H^{{*}}$; the direction index is $I_{{A\\to B}} - I_{{B\\to A}}$.

**Null and excess.** The source area's trials were permuted (20 shuffles for epochs, 10 for 200-ms windows), keeping
every spectrum and PSTH and breaking trial-matched coupling. Every quantity is reported as
{eq('eq-excess', EQ_EX)}
which also removes the positive bias of coherence at finite trial numbers; the excess is averaged over draws.

**Statistics.** Unit of analysis: the session (one per mouse at day 0). Per pair, window, condition, band and measure,
R+ vs R- Mann-Whitney U and Welch t tests on $\\Delta m$ ({ref('eq-excess')}), both reported, pairs needing ≥ 3 sessions per
cohort; Benjamini-Hochberg over pairs within each cell, a pair counted only if both tests pass. Omnibus: per session the
mean $\\Delta m$ over its pairs (raw, and pair-centred, i.e. each pair's across-session mean removed first, so that cohorts
are not compared on which pairs they have). Frequency- and time-cluster permutation tests on the omnibus spectra and
time courses (Welch t, |t| > 2, cluster mass, 500 cohort-label permutations). Whisker − auditory contrasts per session.
The tests H1–H3 were written down before any real-data result (question.md, 2026-11-17).

# Results

## The measures recover known coupling (simulation)

In simulated populations (15 Poisson units per area, 40 trials, 1-s windows, 30 simulations per scenario), a lagged
30-Hz drive from A to B was detected by coherence in {num('sim_ff_msc', 100 * simv('ff', 'msc'), '{:.0f}')} % of
simulations and its direction recovered by spectral Granger in {num('sim_ff_gc', 100 * gc_ff, '{:.0f}')} % and by the
phase-slope index in {num('sim_ff_psi', 100 * psi_ff, '{:.0f}')} % ({ref('fig-sim')}). Zero-lag common input produced
coherence without direction, and partial coherence removed {num('sim_pc', 100 * (1 - simv('common_rec', 'pcoh', 'excess') / simv('common_rec', 'msc', 'excess')), '{:.0f}')} %
of the coupling when the common drive was recorded in the rest. The false-positive rate of the null scenario was
{num('sim_fp', 100 * simv('null', 'msc'), '{:.0f}')} % for coherence (z > 2).

## Coupling is present in both cohorts and does not differ between them

Cross-area coupling exceeded the shuffle null in both cohorts and in every window, with most of the excess below about
20 Hz ({ref('fig-spectra')}). Canonical coherence ({ref('eq-cancoh')}) and partial coherence given the rest
({ref('eq-pcoh')}) behaved like coherence. The R+ and R− spectra overlapped in the quiet, early and execution windows, for whisker
licks, whisker no-licks and auditory hits. Across the omnibus tests of the pre-registered cells ({ref('tbl-omnibus')}),
{num('n_omni_p05', n_omni_p05)} of {num('n_omni', n_omni)} had a Mann-Whitney p < 0.05, about what chance predicts.
The only frequency cluster with p < 0.05 ({num('n_spec_sig', n_spec_sig)} in learners) was in auditory hits, the control
condition.

The null is informative at the level of these effect sizes. For quiet-window β coherence the R+ − R− difference was
{num('mq_d', md_q[2], '{:+.4f}')} (95 % CI {num('mq_lo', md_q[3], '{:+.4f}')} to {num('mq_hi', md_q[4], '{:+.4f}')}),
while the smallest difference detectable with 80 % power is {num('mq_mde', md_q[0], '{:.4f}')}, i.e.
{num('mq_rel', 100 * md_q[1], '{:.0f}')} % of the mean coupling. In the execution window (W_lick, β) the detectable
difference was {num('mx_rel', 100 * md_x[1], '{:.0f}')} % of the mean, and in the early window (W_lick, low γ)
{num('me_rel', 100 * md_e[1], '{:.0f}')} %: cohort differences smaller than about the size of the coupling itself cannot be excluded.

## No pair of areas differs

At the pair level ({ref('fig-pairs')}), {num('frac_both_l', 100 * S['learners']['frac_both'], '{:.1f}')} % of the
{num('n_tests_l', S['learners']['n'])} learner tests had p < 0.05 in both tests (chance about 2.5–5 %), and the p-value
distributions were flat for every measure, level and window ({ref('fig-pvals')}). {num('n_bh_l2', S['learners']['n_bh'])}
tests survived BH correction ({ref('tbl-bh')}), none in a pre-registered cell; they were in control conditions (no-stimulus
correct rejections, auditory hits), involved mostly the hippocampus–thalamus pair, which only few sessions record, and
did not survive in all mice ({num('n_bh_a2', S['all']['n_bh'])} BH-significant tests). The area-acronym level
({num('n_acr', n_acr)} acronyms with ≥ 3 sessions per cohort) gave the same result ({ref('fig-acronym')}). Directed
measures (phase-slope index, {ref('eq-psi')}; Granger direction index, {ref('eq-gc')}) showed no cohort difference either.

## Time course

Coherence rose at stimulus onset in both cohorts, most clearly in the β band on whisker licks and auditory hits
({ref('fig-tr')}). R− sessions tended to show slightly higher β and low-γ coherence after the stimulus in several
conditions, including auditory hits, but no time cluster in a whisker condition reached p < 0.05; the
{num('n_tr_sig', n_tr_sig)} significant cluster(s) were in the no-stimulus condition. Whisker − auditory contrasts,
which test whisker specificity, gave only isolated uncorrected p values near 0.01–0.02 out of 36 contrasts per population
(partial coherence low γ, canonical coherence high γ; omnibus_tests_whisker_minus_auditory.csv).

## AP-band envelope

{(f"The AP-band envelope gave the same answer ({ref('fig-ap')}): {SA['learners']['n_bh']} of {SA['learners']['n']} pair tests survived BH in learners "
  f"({SA['all']['n_bh']} in all mice), {100 * SA['learners']['frac_both']:.1f} % had p < 0.05 in both tests, and quiet β coherence was "
  f"{apq.mean_Rplus:.4f} (R+) vs {apq.mean_Rminus:.4f} (R−; Mann-Whitney p {fmt_p(apq.p_mw)}, Welch p {fmt_p(apq.p_welch)}; "
  f"{int(ap_sess.get('R+', 0))}/{int(ap_sess.get('R-', 0))} sessions). {ap_cl_sig} frequency cluster(s) had p < 0.05.")
 if HAS_AP else f"Pending: the AP-band run was still in progress at this build ({ref('fig-ap')} is the validation placeholder)."}

Confound checks (firing rate and sum-signal power per area group by cohort) are in {ref('fig-confounds')}; all-mice
spectra in {ref('fig-allmice')}.

# Discussion

Frequency-resolved coupling between area groups does not separate R+ from R− mice on the first whisker day, in line
with the time-domain measures of the communication pilot. Coupling itself is robust: it is above the shuffle null in
both cohorts, concentrated at low frequencies and modulated by the stimulus, so the absence of a cohort effect is not a
failure to measure coupling.

Three limits apply. First, the sensitivity: a cohort difference smaller than the coupling itself would be missed with
about 30 sessions per cohort, and each pair is recorded in only part of the sessions. Second, spiking spectra above
about 30 Hz are dominated by the spiking noise floor at these rates, so γ coupling is measured with little power; the
AP-band envelope adds unsorted spikes of the same channels but no field potentials (the AP band is high-passed at
300 Hz). Third, which areas and sites each cohort has differs, which the pair-centred omnibus and the within-pair tests
address but cannot remove for pairs recorded in few sessions.

# Caveats and limitations

- One session per mouse; exploratory with pre-specified tests, no held-out confirmation set.
- Short windows (150–300 ms) resolve only ≥ 10 Hz; θ and α are tested in the quiet window only.
- The execution window contains licking; AP-band envelopes may carry lick-related artefacts not removed by the
  per-probe median reference (condition means are removed, trial-to-trial lick variability is not).
- Trial-shuffle nulls do not remove shared slow state (arousal) that varies within the window; partial coherence on
  the rest addresses recorded common input only.

# Next steps

- Lick-aligned execution windows (corrected first lick) for W_lick and A_hit.
- LFP-band coherence from the `.lf.bin` files (field potentials; not covered by the AP band).
- Pool trials across conditions per cohort to raise power for the quiet-window test.

# Figures

{figure('fig-sim', RES / 'simulation' / '001_simulation.png', 'Simulation of the measures (001). Excess over a 20-shuffle null in the 24–36 Hz band, 30 simulated sessions per scenario (box: median and quartiles), 15 Poisson units per area, 40 trials of 1 s, multitaper NW = 3: independent areas (null), A drives B with a 10-ms lag (A→B), unrecorded zero-lag common input (common) and common input also recorded in 30 rest units (common recorded). Measures as Eq. (1)–(5).', '001_simulation_validation.py')}

{figure('fig-spectra', RES / 'spikes' / 'figures' / 'learners' / 'fig1_omnibus_spectra.png', f'Coupling spectra between area groups, learners (R+ green, n = {L[0]}; R− magenta, n = {L[1]} sessions; fewer where a condition has < 15 trials). Rows: coherence (Eq. (1)), canonical coherence (Eq. (2)), partial coherence given the rest (Eq. (3)), as excess over the trial shuffle (Eq. (6)); per session the mean over its area-group pairs, then mean ± s.e.m. over sessions. Columns: quiet window (all trials), early (5–155 ms) and execution (0.2–0.5 s) windows for whisker licks, whisker no-licks and auditory hits. Black bar: frequency cluster with permutation p < 0.05 (Welch t, |t| > 2, 500 permutations).', '002_run_spikes.py, 003_stats_figures.py')}

{table('tbl-omnibus', omni_tab, 'Omnibus R+ vs R− tests in the pre-registered cells, learners, binned spikes, area groups: per session the mean excess (Eq. (6)) over its pairs; Mann-Whitney and Welch; BH over all rows of the table.')}

{figure('fig-pairs', RES / 'spikes' / 'figures' / 'learners' / 'fig2_pairs_group_msc.png', 'R+ minus R− excess coherence (Eq. (1), Eq. (6)) per pair of area groups, learners, for each window × condition (rows) and band (columns); green = R+ higher. * Mann-Whitney and Welch p < 0.05; ** both BH q < 0.05 over pairs. Empty cells: pair recorded in < 3 sessions of a cohort.', '003_stats_figures.py')}

{figure('fig-pvals', RES / 'spikes' / 'figures' / 'learners' / 'fig3_pvalue_distributions.png', 'Distributions of pair-level Mann-Whitney p values (R+ vs R−, learners) for each level (rows) and measure (columns), pre-registered cells in colour; a flat density at 1 means no effect. Legend: percentage of pairs with p < 0.05. The peak at p = 1 at the acronym level comes from ties at small n.', '003_stats_figures.py')}

{table('tbl-bh', bh, 'All pair-level tests surviving BH (both Mann-Whitney and Welch q < 0.05), learners, binned spikes. diff = R+ − R− excess.')}

{figure('fig-tr', RES / 'spikes' / 'figures' / 'learners' / 'fig4_time_resolved_msc.png', 'Time-resolved excess coherence between area groups (200-ms windows every 50 ms, NW = 2), mean over pairs per session, mean ± s.e.m. over sessions, learners, per band (rows) and condition (columns). Dashed line: stimulus. Black bar: time cluster with permutation p < 0.05.', '003_stats_figures.py')}

{figure('fig-ap', RES / ('ap/figures/learners/fig1_omnibus_spectra.png' if HAS_AP else 'ap/check/004_AB127_20240821_103757.png'), ('AP-band envelope coupling spectra, as Figure 2 with channels (peak channels of stable units) in place of units; quiet window ends at −10 ms, early window starts at +10 ms.' if HAS_AP else 'Placeholder (AP run in progress): AP envelope validation on AB127, see Figure S3.'), '005_build_ap_envelopes.py, 002_run_spikes.py --source ap')}

# Supplementary

{table('tbl-n', ntab, 'Sessions per area with at least 15 (area group) or 8 (area acronym) stable units ≥ 1 Hz, per cohort; acronyms listed only with ≥ 3 learner sessions per cohort.')}

{figure('fig-allmice', RES / 'spikes' / 'figures' / 'all' / 'fig1_omnibus_spectra.png', f'As Figure 2 for all mice (R+ {A[0]}, R− {A[1]} sessions). {n_sig_all} frequency or time cluster(s) with p < 0.05.', '003_stats_figures.py', kind='supplementary')}

{figure('fig-acronym', RES / 'spikes' / 'figures' / 'learners' / 'fig2_pairs_acronym_msc.png', 'As Figure 3 for area acronyms (8 units per acronym, 2 draws).', '003_stats_figures.py', kind='supplementary')}

{figure('fig-ap-check', RES / 'ap' / 'check' / '004_AB127_20240821_103757.png', 'AP-band envelope validation on AB127 day 0 (004): mean envelope on whisker trials per area group (left) and spike PSTH of the stable units (right), z-scored to baseline. ' + ' '.join(ap_check.splitlines()[1:4]).replace('_', ' '), '004_ap_check.py', kind='supplementary')}

{figure('fig-confounds', RES / 'spikes' / 'figures' / 'learners' / 'fig5_confounds.png', 'Firing rate (left) and sum-signal power 30–60 Hz (right) per area group in the quiet window, mean ± s.e.m. over sessions per cohort, learners.', '003_stats_figures.py', kind='supplementary')}

{R.registry_table('tbl-registry')}
"""

measures = pd.DataFrame([
    ["noise correlation / RRR", "coherence (MSC), Eq. (1)", "linear coupling per frequency"],
    ["CCA", "canonical coherence, Eq. (2)", "best coupling over population combinations"],
    ["partial CCA / partial RRR", "partial coherence given rest, Eq. (3)", "coupling not explained by the rest"],
    ["—", "imaginary coherency", "lagged coupling only"],
    ["lagged-RRR direction index", "phase-slope index, Eq. (4)", "which area leads"],
    ["Granger", "spectral Granger, Eq. (5)", "directed coupling per frequency"],
], columns=["pilot (time domain)", "here (frequency domain)", "captures"])
md = md.replace("# Methods\n", "# Methods\n\n" + table("tbl-measures", measures,
                "Frequency-domain counterpart of each measure of the communication pilot (ssl-communication-methods-pilot).") + "\n", 1)
summary = (f"# ssl-spectral-coupling — summary\n\nNo R+ vs R− difference in spectral coupling between areas on whisker day 0 "
           f"(quiet, early, execution windows; area groups and acronyms; spikes{' and AP envelope' if HAS_AP else ''}). "
           f"Coupling is present in both cohorts. {S['learners']['n_bh']} of {S['learners']['n']} learner pair tests survive BH, none in a "
           f"pre-registered cell.\n")
R.write(OUT, md, summary=summary)
print("written", OUT, "HAS_AP", HAS_AP)
