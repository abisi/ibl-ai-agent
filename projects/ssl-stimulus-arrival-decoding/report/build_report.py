"""Article-style report of ssl-stimulus-arrival-decoding (skills/project-report).
Reads the result tables of the project home combined_results_ks4/ssl-stimulus-arrival-decoding/ (active/, passive/,
active_final_n200/, passive_final_n200/, tables/, figures/), writes numbers.json and report.md (Pandoc Markdown with LaTeX
math), copies every figure used into report/figures/ and copies build.sh there. No number in the text is typed by hand.
Onsets at N = 200 come from the final run of an epoch (500 iterations x 20 shuffles) once it covers every area of a level;
until then from the N-sweep run (100 x 10), labelled provisional in the text.
Run (haas):  cd ~/code/unit_spikes_analysis && PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python \
             ~/code/ibl-ai-agent/projects/ssl-stimulus-arrival-decoding/report/build_report.py
Render (locally, Quarto + TinyTeX):  bash projects/ssl-stimulus-arrival-decoding/report/render.sh
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
sys.path.insert(0, str(EA))
AR = importlib.import_module("_areas")
m1 = importlib.import_module("001_arrival_pseudopop")
m2 = importlib.import_module("002_arrival_summary")
m4 = importlib.import_module("004_main_figures")
m6 = importlib.import_module("006_active_vs_passive")

SLUG = PROJ.name
HOME = AR.HOME
OUT = HOME / "report"
NUM: dict = {}
FIGS: dict[str, Path] = {}
LEVELS = ("area_group", "area_acronym_custom")
EPOCHS = ("active", "passive")
EW = {"active": "task (active)", "passive": "passive"}


# ---------------------------------------------------------------------------------------------------------- helpers
def num(key, value, fmt="{:.3f}"):
    """record a quoted number in numbers.json and return it formatted"""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        NUM[key] = int(value)
        return str(int(value))
    v = float(value)
    NUM[key] = None if np.isnan(v) else v
    return "n/a" if np.isnan(v) else fmt.format(v)


def pv(key, value):
    v = float(value)
    NUM[key] = v
    return "< 0.001" if v < 0.001 else f"= {v:.3f}"


def figure(src: Path, dest: str, label: str, caption: str) -> str:
    FIGS[dest] = src
    return f"![**{label}.** {caption}](figures/{dest}){{width=100%}}\n"


def mdtable(df: pd.DataFrame, caption: str) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(":--" if i == 0 else "--:" for i in range(len(cols))) + "|\n"
    body = "".join("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |\n"
                   for r in df.itertuples(index=False))
    return f"\n```{{=latex}}\n\\begingroup\\footnotesize\n```\n\n{head}{body}\n: {caption}\n\n```{{=latex}}\n\\endgroup\n```\n"


def ms(v):
    return "n.s." if not np.isfinite(v) else f"{v:.0f}"


def rng_ms(r):
    return f"{ms(r.onset_ms)} [{ms(r.lo)}, {ms(r.hi)}]" + (" †" if r.flag else "")


# ---------------------------------------------------------------------------------------------------------- data
def epoch_data(e):
    sweep, fin = HOME / e, HOME / f"{e}_final_n200"
    d = dict(sweep=sweep, prov=json.loads((sweep / "provenance_001.json").read_text()),
             w=pd.read_csv(sweep / "window_accuracy.csv"), ov=pd.read_csv(sweep / "onset_vs_window_stats.csv"))
    d["src"], d["ob"], d["fig"], d["w200"] = {}, {}, {}, {}
    for level in LEVELS:
        folder, kind = m6.source(e, level)
        ob = pd.read_csv(folder / "onset_bootstrap_N200.csv")
        ob = ob[ob.level == level].copy()
        ob["flag"] = m4.unreliable(ob)
        w = pd.read_csv(folder / "window_accuracy.csv")
        d["src"][level] = dict(kind=kind, folder=folder, n_iter=int(ob.n_iter.iloc[0]),
                               n_shuf=json.loads((folder / ("provenance_005.json" if kind == "final" else "provenance_001.json"))
                                                 .read_text()).get("n_shuffles"))
        d["ob"][level] = ob.set_index("area")
        d["w200"][level] = w[(w.level == level) & (w.N == m4.N_MAIN)].set_index("area")
        d["fig"][level] = folder / "figures" / f"arrival_main_N200_{level}.png"
    return d


E = {e: epoch_data(e) for e in EPOCHS}
AVP = {lv: pd.read_csv(HOME / "tables" / f"active_vs_passive_{lv}.csv").set_index("area") for lv in LEVELS}
TESTS = pd.read_csv(HOME / "tables" / "active_vs_passive_tests.csv")
ITER = pd.read_csv(HOME / "tables" / "iteration_check_n200.csv")
MATCH = pd.read_csv(HOME / "active" / "matched_accuracy_check.csv")
MATCHN = pd.read_csv(HOME / "active" / "matched_n.csv")
SESS = sorted(E["active"]["prov"]["sessions"])
SESS_P = sorted(E["passive"]["prov"]["sessions"])


def provisional(e, level="area_group"):
    s = E[e]["src"][level]
    return "" if s["kind"] == "final" else f" (provisional: {s['n_iter']} iterations)"


def it_text(e, level="area_group"):
    s = E[e]["src"][level]
    return f"{s['n_iter']} iterations × {s['n_shuf']} shuffles" + ("" if s["kind"] == "final" else ", provisional")


def tst(level, measure):
    return TESTS[(TESTS.level == level) & (TESTS.measure == measure)].iloc[0]


# ---------------------------------------------------------------------------------------------------------- sections
def onset_table(level):
    A, P = E["active"]["ob"][level], E["passive"]["ob"][level]
    wa, wp = E["active"]["w200"][level], E["passive"]["w200"][level]
    areas = sorted(A.index, key=lambda a: (A.loc[a, "onset_ms"] if np.isfinite(A.loc[a, "onset_ms"]) else 1e9,
                                           -wa.loc[a, "mean"]))
    rows = []
    for a in areas:
        rows.append({"Area": m4.label(a), "Onset task, ms [95 %]": rng_ms(A.loc[a]),
                     "Onset passive, ms [95 %]": rng_ms(P.loc[a]) if a in P.index else "",
                     "Acc. 5-50 ms task": f"{wa.loc[a, 'mean']:.3f}", "Acc. 5-50 ms passive":
                     f"{wp.loc[a, 'mean']:.3f}" if a in wp.index else "",
                     "Sessions task / passive": f"{int(wa.loc[a, 'n_eligible_sessions'])} / "
                                                f"{int(wp.loc[a, 'n_eligible_sessions']) if a in wp.index else ''}"})
    return pd.DataFrame(rows)


def reliable_sorted(e, level):
    ob = E[e]["ob"][level]
    return ob[~ob.flag].sort_values("onset_ms")


def sec_front():
    lv = "area_group"
    ra = reliable_sorted("active", lv)
    first = ra[ra.onset_ms == ra.onset_ms.min()]
    last = ra.iloc[-1]
    ov = E["active"]["ov"].query("level == 'area_group' and resolution == 'zoom'").iloc[0]
    t_on, t_acc, t_b = tst(lv, "onset_ms"), tst(lv, "accuracy_5_50ms"), tst(lv, "baseline_-150_0ms")
    mice = sorted({s.split("_")[0] for s in SESS})
    n_unrel = int(E["active"]["ob"][lv].flag.sum())
    num("n_sessions_active", len(SESS)); num("n_sessions_passive", len(SESS_P)); num("n_mice", len(mice))
    s = f"""---
title: "Stimulus arrival across the brain: pseudo-population decoding of whisker vs auditory stimuli"
subtitle: "ssl-stimulus-arrival-decoding"
author: "Axel Bisi (analysis with Claude Code)"
date: "{date.today().isoformat()}"
---

**Data version.** KS4 spike sorting (NWB_ks4), v2 unit table (good + mua units), SSL whisker-training sessions of both
cohorts, learning day and expert days pooled: {num('n_sess_a', len(SESS))} sessions from {num('n_mice_a', len(mice))} mice
with task (active) trials, {num('n_sess_p', len(SESS_P))} sessions with passive trials. Decoding: all {num('n_groups', len(AR.COARSE))}
area groups and the {num('n_fine', len(AR.FINE))} best-sampled areas. Onsets at N = {m4.N_MAIN} neurons: task trials
{it_text('active')}; passive trials {it_text('passive')}.

# Abstract

When does information about stimulus modality reach each brain region? We decoded whisker vs auditory stimuli from
pseudo-populations of N neurons pooled across sessions, in 20-ms bins stepped by 2 ms, and located the first time bin at
which decoding is reliably above a within-session label-shuffle null. With task (active) trials and N = {m4.N_MAIN},
modality became decodable first in {', '.join(m4.label(a) for a in first.index)} ({ms(first.onset_ms.iloc[0])} ms after
stimulus onset){provisional('active')}, and last, among reliable onsets, in {m4.label(last.name)} ({ms(last.onset_ms)} ms).
Onset tracked early decoding strength closely (Spearman ρ = {num('rho_onset_acc_active', ov.rho, '{:.2f}')} across
area × N, n = {num('n_onset_acc_active', int(ov.n))}): part of the arrival order reflects how much early information a
region carries, and matching early accuracy across regions reduces, but does not remove, the onset differences. Passive trials gave the same arrival order;
passive onsets were {num('onset_diff_median', abs(t_on.median_diff), '{:.0f}')} ms earlier (median, n =
{num('onset_diff_n', int(t_on.n))} area groups) and early accuracy was {num('acc_diff_median', t_acc.median_diff, '{:+.3f}')}
higher, but passive decoding was also above zero before stimulus onset ({num('base_diff_median', t_b.median_diff, '{:+.3f}')}
vs task), so these differences are not interpretable as faster arrival without a null that keeps trial order.
{num('n_unrel_active', n_unrel)} area groups have onsets too variable to rank.

# Key results

1. **Arrival order (task trials, N = {m4.N_MAIN}).** First: {', '.join(f"{m4.label(a)} {ms(r.onset_ms)} ms" for a, r in ra.head(6).iterrows())};
   last reliable: {', '.join(f"{m4.label(a)} {ms(r.onset_ms)} ms" for a, r in ra.tail(3).iterrows())}{provisional('active')} (Figure 1, Table 1).
2. **Onset follows early information.** Onset vs mean corrected accuracy 5-50 ms: ρ = {ov.rho:.2f}, p {pv('p_rho_onset_acc_active', ov.p_rho)},
   n = {int(ov.n)} area × N (Figure 2e).
3. **Passive trials, same order.** Onset passive - task: median {t_on.median_diff:+.1f} ms, Wilcoxon p {pv('p_onset_w', t_on.p_wilcoxon)},
   paired t p {pv('p_onset_t', t_on.p_paired_t)}, n = {int(t_on.n)} area groups with reliable onsets in both (Figure 4a).
4. **Passive baseline offset.** Pre-stimulus corrected accuracy (-150..0 ms) passive - task: median {t_b.median_diff:+.3f},
   Wilcoxon p {pv('p_base_w', t_b.p_wilcoxon)}, paired t p {pv('p_base_t', t_b.p_paired_t)}, n = {int(t_b.n)} (Figure 4c);
   the early-accuracy difference ({t_acc.median_diff:+.3f}, Wilcoxon p {pv('p_acc_w', t_acc.p_wilcoxon)}) is of the same size.
5. **Iterations.** 100 vs 1000 iterations give the same onsets within one 2-ms step in
   {num('iter_n_within2', int((abs(ITER.onset_pilot - ITER.onset_final) <= 2).sum()))} of {num('iter_n', len(ITER))} areas
   (Table 4); 500 iterations × 20 shuffles is the default for final onsets.
"""
    return s


def sec_intro():
    return """
# Introduction

In the SSL task, mice learn to lick after a brief whisker deflection (rewarded in the R+ cohort, not in the R- cohort)
and always lick after an auditory tone. Whisker and auditory stimuli travel along separate afferent pathways, so the
first neural responses that distinguish them should appear in primary sensory structures and spread to downstream
regions. The question of this project is where and when modality information first appears across the ~40 regions
recorded with Neuropixels in this dataset, and how this arrival map depends on population size and on the behavioural
context (stimuli delivered within the task or passively, outside it).

We use population decoding rather than single-neuron latencies because it integrates weak signals across neurons and gives
a single, comparable measure per region. Pooling neurons across sessions into pseudo-populations lets every region be
decoded at the same population size N, which removes the most obvious confound of comparing regions: the number of
recorded neurons. Two further confounds remain and are addressed explicitly: decoding strength (a region with weaker
information crosses any threshold later) and the statistical precision of the onset estimate.

Hypotheses: (i) whisker-related structures (barrel cortex SSp-bfd, ventral posteromedial thalamus, superior colliculus) and
auditory structures carry modality information first, within ~10 ms; (ii) motor, frontal and hippocampal regions follow;
(iii) the order is preserved outside the task.
"""


def sec_methods():
    p = E["active"]["prov"]
    return f"""
# Methods

## Data and inclusion

All whisker-training sessions of the v2 unit table with KS4 NWB files (both cohorts, learning day and expert days pooled;
cohort and stage are not analysed separately here, by decision of 2026-10-04). Units: quality label good or mua.
A session contributes to an area when it has ≥ {m1.MIN_UNITS} units of the area and ≥ {m1.MIN_TRIALS} trials per class.

**Task (active) trials**: trials in active context (per-trial context rule: a trial in a fixed ~3 s inter-trial sequence
is passive, otherwise the context column decides), perf ≠ 6, warm-up cut (keep one trial before the first whisker trial),
disengagement tail trimmed with rule A1 (trials after the last lick dropped when that tail holds ≥ 5 whisker and ≥ 1
auditory trials); all whisker vs all auditory trials, regardless of the response.
**Passive trials**: trials in passive context (pre- and post-task blocks pooled; perf 6 kept, as it codes passive trials).

**Spikes**: whisker stimulation produces an electrical artefact; spikes from -10 to +5 ms around every whisker onset are
replaced by a Poisson train at the unit's pre-onset rate (seeded per session). No information can therefore appear on
whisker trials before +5 ms.

## Pseudo-population decoding

One iteration draws {m1.N_SESS} eligible sessions with replacement, then N/{m1.N_SESS} units of the area within each drawn
session (with replacement; the remainder spread over random sessions), and builds {p['t_train']} + {p['t_train']}
pseudo-trials per class by balanced reuse of each session's real trials, separately for training and test folds.
Per time bin, an L2-regularised logistic regression (liblinear; inverse regularisation C chosen from
{len(m1.C_GRID)} values by an inner {p['n_inner']}-fold cross-validation) is trained and tested in a {p['n_outer']}-fold
outer cross-validation built on each session's real trials; features are z-scored in the training fold. The score is the
balanced accuracy $a_i(t)$ of iteration $i$ in bin $t$.

**Null and corrected accuracy.** The same draw is decoded again $S$ times with the trial labels permuted within each
session before pooling, with the same C per bin; the corrected accuracy is
$$d_i(t) = a_i(t) - \\frac{{1}}{{S}}\\sum_{{s=1}}^{{S}} a^{{\\mathrm{{shuf}}}}_{{i,s}}(t),$$
0 at chance and 0.5 for perfect decoding.

**Time resolutions.** Zoom: causal 20-ms bins in 2-ms steps from -20 to +100 ms (a bin labelled $t$ covers
$[t-20, t)$ ms); wide: causal 50-ms bins in 5-ms steps from -200 to +600 ms.

**Above chance.** Bin $t$ is above chance when the 5th percentile of $d_i(t)$ over iterations is > 0.
**Onset.** The first post-stimulus zoom bin that is above chance and is followed by ≥ 80 % above-chance bins over the next
25 ms (11 of 13 bins). **Onset range**: 95 % range of the onset over {m4.N_BOOT} resamples (with replacement) of the
iterations. An onset is **unreliable** (†, not ranked or tested) when that range is wider than {m4.WIDE_RANGE_MS} ms or the
onset is undefined in more than {100 * (1 - m4.MIN_DEFINED):.0f} % of the resamples.
**Early accuracy**: mean of $d_i(t)$ over the zoom bins ending 5-50 ms after onset, averaged over iterations.

**Population size.** N = {', '.join(str(n) for n in m1.N_LIST)} neurons. **Matched accuracy**: for reference areas at
N = 100, each other area's N giving the same early accuracy is read from its accuracy-vs-N curve (log-N interpolation,
extrapolated beyond 500 up to 2000) and decoded again at that N.

**Iterations.** N sweep: 100 iterations × 10 shuffles. Final onsets at N = {m4.N_MAIN}: 500 iterations × 20 shuffles
(default since 2026-10-06; 21 areas of the task-trial run were first run to 1000 iterations, of which iterations 0-499 are
used — each iteration is an independently seeded random draw, so these are the iterations a 500-iteration run produces).

## Task vs passive comparison

Per area at N = {m4.N_MAIN}: onset, early accuracy and pre-stimulus corrected accuracy (mean over the wide bins ending
-150..0 ms, which contain no post-stimulus spikes). Paired over areas (passive - task): Wilcoxon signed-rank test and
paired t-test; onsets only for areas reliable in both epochs.

## Software

Code: `projects/{SLUG}/exploratory-analyses/` — `001_arrival_pseudopop.py` (cache, decoding), `002_arrival_summary.py`
(summaries, N-sweep figures), `003_matched_accuracy.py`, `004_main_figures.py`, `005_final_n200.py` (final onsets),
`006_active_vs_passive.py`, `007_iteration_check.py`; decoder from `ssl-pseudopopulation-area-decoding` 002. Results:
`combined_results_ks4/{SLUG}/` (`active/`, `passive/`, `active_final_n200/`, `passive_final_n200/`, `tables/`, `figures/`).
"""


def sec_results():
    lv = "area_group"
    ra = reliable_sorted("active", lv)
    rp = reliable_sorted("passive", lv)
    wa = E["active"]["w"]
    def acc(area, n):
        r = wa[(wa.level == lv) & (wa.area == area) & (wa.N == n)]
        return r["mean"].iloc[0] if len(r) else np.nan
    sw = "Somatosensory-whisker"
    ovp = E["passive"]["ov"].query("level == 'area_group' and resolution == 'zoom'").iloc[0]
    t_on, t_acc, t_b = tst(lv, "onset_ms"), tst(lv, "accuracy_5_50ms"), tst(lv, "baseline_-150_0ms")
    tf_on, tf_b = tst("area_acronym_custom", "onset_ms"), tst("area_acronym_custom", "baseline_-150_0ms")
    mt = MATCH[MATCH.level == lv].copy()
    mt["ratio"] = mt.achieved / mt.target
    mn = MATCHN[(MATCHN.level == lv)]
    notreached = mn[mn.how == "not reached"]
    ref = mt.reference.iloc[0] if len(mt) else ""
    matched_tbl = mt[["reference", "area", "matched_N", "target", "achieved"]].assign(
        reference=lambda d: d.reference.map(m4.label), area=lambda d: d.area.map(m4.label),
        target=lambda d: d.target.map("{:.3f}".format), achieved=lambda d: d.achieved.map("{:.3f}".format),
        matched_N=lambda d: d.matched_N.astype(int)).rename(columns={"reference": "Reference", "area": "Area",
                                                                      "matched_N": "Matched N", "target": "Target acc.",
                                                                      "achieved": "Achieved acc."})
    it = ITER.copy()
    it["Area"] = it.area.map(m4.label)
    it_tbl = it[["Area", "onset_pilot", "onset_final", "onset_sub100_lo", "onset_sub100_hi", "early_pilot", "early_final",
                 "curve_r", "sig_bins_differ"]].copy()
    it_tbl["100-it. subsets [95 %]"] = it_tbl.apply(lambda r: f"[{r.onset_sub100_lo:.0f}, {r.onset_sub100_hi:.0f}]", axis=1)
    it_tbl = it_tbl.assign(onset_pilot=it_tbl.onset_pilot.map(ms), onset_final=it_tbl.onset_final.map(ms),
                           early_pilot=it_tbl.early_pilot.map("{:.3f}".format), early_final=it_tbl.early_final.map("{:.3f}".format),
                           curve_r=it_tbl.curve_r.map("{:.3f}".format))
    it_tbl = it_tbl[["Area", "onset_pilot", "onset_final", "100-it. subsets [95 %]", "early_pilot", "early_final", "curve_r",
                     "sig_bins_differ"]].rename(columns={"onset_pilot": "Onset 100 it.", "onset_final": "Onset 1000 it.",
                                                         "early_pilot": "Acc. 100 it.", "early_final": "Acc. 1000 it.",
                                                         "curve_r": "r curves", "sig_bins_differ": "Bins differing"})
    tests_tbl = TESTS.assign(level=TESTS.level.map(m4.LEVEL_NAME), measure=TESTS.measure.map(
        {"onset_ms": "onset (ms)", "accuracy_5_50ms": "accuracy 5-50 ms", "baseline_-150_0ms": "baseline -150..0 ms"}),
        mean_diff=TESTS.mean_diff.map("{:+.3f}".format), median_diff=TESTS.median_diff.map("{:+.3f}".format),
        p_wilcoxon=TESTS.p_wilcoxon.map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"),
        p_paired_t=TESTS.p_paired_t.map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")).rename(columns={
            "level": "Level", "measure": "Measure", "n": "n areas", "mean_diff": "Mean passive - task",
            "median_diff": "Median", "p_wilcoxon": "p Wilcoxon", "p_paired_t": "p paired t"})
    unrel = [m4.label(a) for a, r in E["active"]["ob"][lv].iterrows() if r.flag]
    unrel_p = [m4.label(a) for a, r in E["passive"]["ob"][lv].iterrows() if r.flag]
    fa = reliable_sorted("active", "area_acronym_custom")
    s = f"""
# Results

## Modality information arrives in a graded order across regions

With task trials and N = {m4.N_MAIN} neurons per pseudo-population, every area group decoded stimulus modality above chance
shortly after stimulus onset (Figure 1). The earliest onsets{provisional('active')} were in
{', '.join(f"{m4.label(a)} ({ms(r.onset_ms)} ms)" for a, r in ra.head(5).iterrows())}; the latest reliable ones in
{', '.join(f"{m4.label(a)} ({ms(r.onset_ms)} ms)" for a, r in ra.tail(3).iterrows())}. The onsets of
{', '.join(unrel) if unrel else 'no area group'} were too variable to rank (95 % range > {m4.WIDE_RANGE_MS} ms; Table 1).
Among the {len(AR.FINE)} best-sampled areas (Figure S1), the earliest were
{', '.join(f"{a} ({ms(r.onset_ms)} ms)" for a, r in fa.head(8).iterrows())}.
Decoding peaked shortly after onset in most regions and decayed over the rest of the trial (Figure 1a, d).

{figure(E['active']['fig'][lv], 'fig1_active_main_area_group.png', 'Figure 1', f"Stimulus-modality decoding across area groups, task (active) trials, all sessions pooled, N = {m4.N_MAIN} neurons ({it_text('active')}). (a) Corrected balanced accuracy over time (50-ms bins, 5-ms steps); bins not above chance in grey; tick: onset. (b) First 50 ms (20-ms bins, 2-ms steps). (c) Onset ranking, error bars: 95 % range over {m4.N_BOOT} resamples of the iterations; † unreliable onset (range > {m4.WIDE_RANGE_MS} ms), hatched, not ranked. (d, e) Time courses of the 8 best-sampled area groups, mean ± s.d. over iterations; bars: bins above chance; black tick: onset. Colours: area-group palette.")}

{mdtable(onset_table(lv), f"**Table 1.** Onsets (ms, [95 % range]) and early corrected accuracy at N = {m4.N_MAIN}, area groups, sorted by the task-trial onset. Task: {it_text('active')}; passive: {it_text('passive')}. † unreliable onset.")}

## Onset follows the amount of early information, not only the arrival time

Early accuracy grew with the number of neurons in every region (Figure 2d; e.g. {m4.label(sw)}
{num('acc_sw_20', acc(sw, 20))} at N = 20 vs {num('acc_sw_500', acc(sw, 500))} at N = 500), and onsets became earlier as
N grew. Across area × N, onset was strongly anti-correlated with early accuracy (ρ = {ov_rho('active')}, Figure 2e), so the
onset is partly a read-out of decoding strength: a region with weaker but equally fast information crosses the
above-chance criterion later. To separate the two, we matched early accuracy to {m4.label(ref)} at N = 100: the matched
time courses rose at similar times, with differences of a few ms (Figure 2f, Figure S6; Table 2), and
{', '.join(m4.label(a) for a in notreached.area) if len(notreached) else 'no region'} could not reach the reference accuracy
within N ≤ 2000.

{figure(E['active']['sweep'] / 'figures' / 'arrival_summary_area_group.png', 'fig2_active_summary_area_group.png', 'Figure 2', "Summary, task trials, area groups (N sweep: 100 iterations × 10 shuffles). (a) Method. (b) First 50 ms at N = 200. (c) Onset ranking. (d) Corrected accuracy 5-50 ms vs number of neurons (colours: 8 best-sampled groups, grey: others). (e) Onset vs early accuracy over area × N, Spearman correlation and OLS fit with 95 % band. (f) Control: time courses at the N matching the early accuracy of the reference at N = 100. (g) Control: raw balanced accuracy (solid) and the trial-shuffle null (dotted). (h) Time courses of the 8 best-sampled groups.")}

{mdtable(matched_tbl, f"**Table 2.** Matched-accuracy control, area groups: neurons needed to reach the early accuracy of the reference at N = 100, and the accuracy achieved when decoding at that N.")}

## Passive trials: same order, but a pre-stimulus offset

Passive trials gave the same ordering (Figure 3; ρ between onset and early accuracy {num('rho_onset_acc_passive', ovp.rho, '{:.2f}')}).
Earliest passive onsets{provisional('passive')}: {', '.join(f"{m4.label(a)} ({ms(r.onset_ms)} ms)" for a, r in rp.head(5).iterrows())};
unreliable: {', '.join(unrel_p) if unrel_p else 'none'}.

{figure(E['passive']['fig'][lv], 'fig3_passive_main_area_group.png', 'Figure 3', f"As Figure 1, passive trials ({it_text('passive')}).")}

Compared area by area (Figure 4, Table 3), passive onsets were earlier by a median {t_on.median_diff:+.1f} ms
(n = {int(t_on.n)} area groups, Wilcoxon p {pv('p_onset_w2', t_on.p_wilcoxon)}; areas: {tf_on.median_diff:+.1f} ms, n =
{int(tf_on.n)}, p {pv('p_onset_w_fine', tf_on.p_wilcoxon)}) and early accuracy was higher by {t_acc.median_diff:+.3f}.
However, passive decoding was already above zero before the stimulus: the pre-stimulus corrected accuracy exceeded the
task-trial value by {t_b.median_diff:+.3f} (area groups, Wilcoxon p {pv('p_base_w2', t_b.p_wilcoxon)}; areas
{tf_b.median_diff:+.3f}, p {pv('p_base_w_fine', tf_b.p_wilcoxon)}), as large as the early-accuracy difference. Something
in the passive trial sequence predicts the upcoming stimulus beyond what the within-session label shuffle removes (see
Caveats), and an offset of this size moves the above-chance criterion earlier by itself. The earlier passive onset is
therefore not evidence of faster arrival. Later in the trial, task-trial decoding stayed high while passive decoding decayed
(Figure 4e), consistent with licking and reward-related activity on task trials.

{figure(HOME / 'figures' / 'active_vs_passive_area_group.png', 'fig4_active_vs_passive_area_group.png', 'Figure 4', f"Task (active) vs passive trials, area groups, N = {m4.N_MAIN}. (a) Onset per area (filled: task, open: passive), 95 % ranges; grey: unreliable in either epoch, not tested. (b) Early corrected accuracy (5-50 ms), 95 % range over iterations. (c) Pre-stimulus corrected accuracy (wide bins ending -150..0 ms), mean ± 1.96 s.e.m. over iterations; expected 0. Tests: passive - task over areas, Wilcoxon signed-rank and paired t-test. (d, e) First 50 ms and -200..600 ms time courses of the 6 best-sampled groups, task solid, passive dashed, mean ± s.d.")}

{mdtable(tests_tbl, "**Table 3.** Passive - task differences over areas (paired), both levels.")}

## Robustness to the number of iterations

The {len(ITER)} areas run to 1000 iterations × 20 shuffles before the default was set to 500 were compared with the 100 × 10
N-sweep run at N = {m4.N_MAIN} (Table 4). Time courses were indistinguishable (r ≥ {num('iter_min_r', ITER.curve_r.min(), '{:.3f}')};
max |difference| ≤ {num('iter_max_diff', ITER.curve_max_abs_diff.max(), '{:.3f}')}); early accuracy differed by at most
{num('iter_max_acc_diff', (ITER.early_pilot - ITER.early_final).abs().max(), '{:.3f}')}; onsets were identical in
{num('iter_n_same', int((ITER.onset_pilot == ITER.onset_final).sum()))} areas and within one 2-ms step in all but
{', '.join(m4.label(a) for a in ITER[abs(ITER.onset_pilot - ITER.onset_final) > 2].area)}. Onsets recomputed on random
100-iteration subsets of the 1000 varied by ±2 ms for most areas but over tens of ms for Olfactory and Insular areas, the
two onsets flagged as unreliable: their position in the ranking is not determined by the data at this sampling.

{mdtable(it_tbl, "**Table 4.** Task trials, N = 200: 100 iterations × 10 shuffles vs 1000 × 20. Onsets in ms; 100-it. subsets: 95 % range of the onset over 200 random subsets of 100 of the 1000 iterations; r curves: correlation of the mean time courses (all bins); bins differing: bins whose above-chance call differs (of 222).")}
"""
    return s


def ov_rho(e):
    ov = E[e]["ov"].query("level == 'area_group' and resolution == 'zoom'").iloc[0]
    return f"{ov.rho:.2f}"


def sec_discussion():
    return """
# Discussion

Modality information reaches most of the recorded brain within a few tens of ms of the stimulus, with midbrain, thalamic and
whisker-somatosensory populations first and orofacial-somatosensory, insular and olfactory populations last. The order is
broadly the one expected from the afferent pathways: superior colliculus, thalamus and barrel cortex receive whisker
input within a few ms, auditory structures receive auditory input at similar latencies, and the striatum, motor and
frontal cortices follow.

Two aspects of the result call for caution. First, onset and decoding strength are entangled: the anti-correlation
between onset and early accuracy means that a "late" region is often a region with less information. The matched-accuracy
control reduces but does not remove this, because information cannot be matched for regions whose accuracy saturates
below the reference. Second, a population onset is not a first-spike latency: it is the first time a linear read-out of
N neurons separates the two stimuli reliably, which depends on the bin width (20 ms, causal) and on the criterion.
The companion project `ssl-sensory-spatial-maps` relates these onsets to single-neuron latencies and projection zones.

The task vs passive comparison was designed to ask whether engagement speeds up arrival. The data do not answer it yet:
passive decoding sits above zero before the stimulus, which shifts onsets earlier and raises early accuracy by the same
amount. Once that offset is controlled, the arrival order is the same in both contexts.
"""


def sec_caveats():
    unrel = ", ".join(m4.label(a) for a in sorted(set(E["active"]["ob"]["area_group"].query("flag").index)
                                                   | set(E["passive"]["ob"]["area_group"].query("flag").index))) or "none"
    return f"""
# Caveats and limitations

- **Passive pre-stimulus decoding.** The within-session label shuffle breaks the temporal order of the trials. If the
  passive sequence has structure (stimulus types in runs, or slow drift that differs between whisker and auditory
  trials), the real labels are partly predictable from pre-stimulus activity and the corrected accuracy is > 0 before
  onset. A null that keeps trial order (circular or linear shift of the labels within session) would remove this offset;
  until then, passive vs task onset and accuracy differences are not interpretable.
- **Whisker artefact.** Spikes from -10 to +5 ms around whisker onsets are replaced by Poisson spikes, so whisker-evoked
  information can appear only from +5 ms; with causal 20-ms bins, an onset at t means the information is present somewhere
  in [t-20, t) ms.
- **Pooling.** Cohorts (R+, R-) and stages (learning, expert) are pooled; arrival differences between them are not tested.
- **Iterations.** The N sweep (N = 20-500) and the matched-accuracy control use 100 iterations × 10 shuffles; only the
  N = 200 onsets use 500 × 20.
- **Unreliable areas.** Areas flagged † in the tables (among the area groups: {unrel}) have onsets that vary by more
  than {m4.WIDE_RANGE_MS} ms across resamples and are not ranked.
- **Late time points.** On task trials, licking and reward follow both stimuli at different rates; late decoding can
  reflect behaviour rather than the stimulus.

# Open questions and next steps

1. Re-run the passive (and task) decoding with an order-preserving null and re-test the task vs passive difference.
2. Cohort and stage splits of the arrival map (R+ vs R-, learning vs expert).
3. Link population onsets with single-neuron latencies for all area groups (`ssl-sensory-spatial-maps`).
"""


def sec_supp():
    out = ["\n# Supplementary figures\n"]
    out.append(figure(E["active"]["fig"]["area_acronym_custom"], "figS1_active_main_areas.png", "Figure S1",
                      f"As Figure 1, the {len(AR.FINE)} best-sampled areas (colours: shades of their parent group), task trials ({it_text('active', 'area_acronym_custom')})."))
    out.append(figure(E["active"]["sweep"] / "figures" / "arrival_summary_area_acronym_custom.png", "figS2_active_summary_areas.png",
                      "Figure S2", "As Figure 2, areas (panel f: time courses of the best-sampled areas)."))
    out.append(figure(E["passive"]["fig"]["area_acronym_custom"], "figS3_passive_main_areas.png", "Figure S3",
                      f"As Figure 1, areas, passive trials ({it_text('passive', 'area_acronym_custom')})."))
    out.append(figure(E["passive"]["sweep"] / "figures" / "arrival_summary_area_group.png", "figS4_passive_summary_area_group.png",
                      "Figure S4", "As Figure 2, passive trials, area groups (matched-accuracy panel: not run for passive trials)."))
    out.append(figure(HOME / "figures" / "active_vs_passive_area_acronym_custom.png", "figS5_active_vs_passive_areas.png",
                      "Figure S5", "As Figure 4, areas."))
    out.append(figure(E["active"]["sweep"] / "figures" / "matched_accuracy.png", "figS6_matched_accuracy.png", "Figure S6",
                      "Matched-accuracy control, task trials: time courses at the N matching each reference's early accuracy at N = 100."))
    out.append(figure(E["active"]["sweep"] / "figures" / "arrival_curves_area_group.png", "figS7_active_curves_area_group.png",
                      "Figure S7", "Time courses per area group and N (task trials, N sweep, 100 iterations × 10 shuffles); colour: N."))
    out.append(figure(E["active"]["sweep"] / "figures" / "onset_vs_window.png", "figS8_onset_vs_window.png", "Figure S8",
                      "Onset vs early accuracy for every area × N, both levels and resolutions (task trials)."))
    out.append(mdtable(onset_table("area_acronym_custom"),
                       f"**Table S1.** As Table 1, the {len(AR.FINE)} best-sampled areas."))
    return "\n".join(out)


def sec_appendix():
    p = E["active"]["prov"]
    par = pd.DataFrame([
        ("Sessions per draw", m1.N_SESS), ("Min units / trials per class", f"{m1.MIN_UNITS} / {m1.MIN_TRIALS}"),
        ("Pseudo-trials per class (train / test)", f"{p['t_train']} / {p['t_test']}"),
        ("Outer / inner CV folds", f"{p['n_outer']} / {p['n_inner']}"), ("C grid", ", ".join(f"{c:g}" for c in m1.C_GRID)),
        ("N sweep", ", ".join(map(str, m1.N_LIST))), ("Zoom bins", p["zoom"]), ("Wide bins", p["wide"]),
        ("Early window", "5-50 ms (zoom bin ends)"), ("Baseline window", "wide bins ending -150..0 ms"),
        ("Above chance", "5th percentile over iterations > 0"), ("Onset", "≥ 80 % of the next 25 ms above chance (11 / 13)"),
        ("Onset resamples", m4.N_BOOT), ("Unreliable onset", f"95 % range > {m4.WIDE_RANGE_MS} ms or undefined in > 5 %"),
        ("Iterations × shuffles, N sweep", "100 × 10"), ("Iterations × shuffles, N = 200 final", f"{m1.N_ITER} × {m1.N_SHUF}"),
        ("Artefact window (whisker)", "-10..+5 ms, Poisson replacement")], columns=["Parameter", "Value"])
    files = pd.DataFrame([
        ("active/", "task trials, N sweep, matched accuracy"), ("passive/", "passive trials, N sweep"),
        ("active_final_n200/", "task trials, N = 200, final onsets"), ("passive_final_n200/", "passive trials, N = 200, final onsets"),
        ("tables/active_vs_passive_*.csv", "006"), ("tables/iteration_check_n200.csv", "007"),
        ("figures/active_vs_passive_*.png", "006"), ("results_home_manifest_20261006.tsv", "folder migration")],
        columns=["Path (combined_results_ks4/" + SLUG + "/)", "Content"])
    return f"""
# Appendix

{mdtable(par, "**Table A1.** Parameters.")}

{mdtable(files, "**Table A2.** Result files.")}

**Version history.** 2026-10-04: project created (N sweep, matched accuracy, main and summary figures, task trials;
passive trials overnight). 2026-10-06: results moved to `combined_results_ks4/{SLUG}/`; final-onset sampling changed from
1000 to 500 iterations × 20 shuffles after the iteration check (007); task vs passive comparison (006) with the
pre-stimulus baseline; unreliable onsets flagged in the rankings; this report.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = sec_front() + sec_intro() + sec_methods() + sec_results() + sec_discussion() + sec_caveats() + sec_supp() + sec_appendix()
    (OUT / "figures").mkdir(exist_ok=True)
    for dest, src in FIGS.items():
        shutil.copyfile(src, OUT / "figures" / dest)
    (OUT / "report.md").write_text(md, encoding="utf-8")
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1), encoding="utf-8")
    shutil.copyfile(REPO / "skills" / "project-report" / "build.sh", OUT / "build.sh")
    print(f"report.md ({len(md.split())} words), {len(FIGS)} figures, {len(NUM)} numbers -> {OUT}")
    print({e: {lv: E[e]["src"][lv]["kind"] for lv in LEVELS} for e in EPOCHS})


if __name__ == "__main__":
    main()
