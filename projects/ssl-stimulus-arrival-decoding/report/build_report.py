"""Article-style report: whisker vs auditory stimulus decoding across brain regions (skills/project-report).
Reads the result tables of the project results folder (active/, passive/, active_final_n200/, passive_final_n200/,
tables/, figures/), writes numbers.json and report.md (Pandoc Markdown with LaTeX math), copies every figure used into
report/figures/ and copies build.sh there. No number in the text is typed by hand.
Structure (user 2026-10-06): passive trials, then task (active) trials, then their comparison. Every figure and table
has an anchor and is cited with a link in the text; main() refuses to write a report with an uncited figure or table, or
with the word "arrival" (user: "stimulus decoding", not "stimulus arrival").
Onsets at N = 200 come from the final run of an epoch (500 iterations x 20 shuffles) once it covers every area of a level;
until then from the N-sweep run (100 x 10), labelled provisional.
Run (haas):  cd ~/code/unit_spikes_analysis && PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python \
             ~/code/ibl-ai-agent/projects/ssl-stimulus-arrival-decoding/report/build_report.py
Render (locally, Quarto + TinyTeX):  bash projects/ssl-stimulus-arrival-decoding/report/render.sh
"""
from __future__ import annotations

import importlib
import json
import re
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

HOME = AR.HOME
OUT = HOME / "report"
NUM: dict = {}
FIGS: dict[str, Path] = {}
ANCHORS: dict[str, str] = {}          # anchor id -> label ("Figure 3", "Table S2")
LEVELS = ("area_group", "area_acronym_custom")
EPOCHS = ("passive", "active")
LV = {"area_group": "area groups", "area_acronym_custom": "areas"}


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


def ref(anchor):
    """link to a figure / table anchor"""
    return f"[{ANCHORS[anchor]}](#{anchor})"


def figure(anchor, src: Path, caption: str) -> str:
    dest = f"{anchor}.png"
    FIGS[dest] = src
    return f"\n[]{{#{anchor}}}\n\n![**{ANCHORS[anchor]}.** {caption}](figures/{dest}){{width=100%}}\n"


def mdtable(anchor, df: pd.DataFrame, caption: str) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(":--" if i == 0 else "--:" for i in range(len(cols))) + "|\n"
    body = "".join("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |\n"
                   for r in df.itertuples(index=False))
    return (f"\n[]{{#{anchor}}}\n\n```{{=latex}}\n\\begingroup\\footnotesize\n```\n\n{head}{body}\n: **{ANCHORS[anchor]}.** "
            f"{caption}\n\n```{{=latex}}\n\\endgroup\n```\n")


def ms(v):
    return "n.s." if not np.isfinite(v) else f"{v:.0f}"


def rng_ms(r):
    return f"{ms(r.onset_ms)} [{ms(r.lo)}, {ms(r.hi)}]" + (" †" if r.flag else "")


def short(a):
    return m4.label(a)


# ---------------------------------------------------------------------------------------------------------- data
def epoch_data(e):
    sweep = HOME / e
    d = dict(sweep=sweep, prov=json.loads((sweep / "provenance_001.json").read_text()),
             w=pd.read_csv(sweep / "window_accuracy.csv"), ov=pd.read_csv(sweep / "onset_vs_window_stats.csv"),
             src={}, ob={}, fig={}, w200={})
    for level in LEVELS:
        folder, kind = m6.source(e, level)
        ob = pd.read_csv(folder / "onset_bootstrap_N200.csv")
        ob = ob[ob.level == level].copy()
        ob["flag"] = m4.unreliable(ob)
        w = pd.read_csv(folder / "window_accuracy.csv")
        prov = folder / ("provenance_005.json" if kind == "final" else "provenance_001.json")
        d["src"][level] = dict(kind=kind, n_iter=int(ob.n_iter.iloc[0]), n_shuf=json.loads(prov.read_text()).get("n_shuffles"))
        d["ob"][level] = ob.set_index("area")
        d["w200"][level] = w[(w.level == level) & (w.N == m4.N_MAIN)].set_index("area")
        d["fig"][level] = folder / "figures" / f"arrival_main_N200_{level}.png"
    return d


E = {e: epoch_data(e) for e in EPOCHS}
TESTS = pd.read_csv(HOME / "tables" / "active_vs_passive_tests.csv")
ITER = pd.read_csv(HOME / "tables" / "iteration_check_n200.csv")
MATCH = pd.read_csv(HOME / "active" / "matched_accuracy_check.csv")
MATCHN = pd.read_csv(HOME / "active" / "matched_n.csv")
SS = pd.read_csv(HOME / "tables" / "sample_sizes.csv")
SEQ = pd.read_csv(HOME / "tables" / "trial_sequence_check.csv")
RANK = pd.read_csv(HOME / "tables" / "area_ranking.csv")
SESS = sorted(E["active"]["prov"]["sessions"])
SESS_P = sorted(E["passive"]["prov"]["sessions"])
FIGDIR = {e: HOME / e / "figures" for e in EPOCHS}


def it_text(e, level="area_group"):
    s = E[e]["src"][level]
    return f"{s['n_iter']} iterations × {s['n_shuf']} shuffles" + ("" if s["kind"] == "final" else ", provisional")


def prov_note(e, level="area_group"):
    s = E[e]["src"][level]
    return "" if s["kind"] == "final" else f" (provisional, {s['n_iter']} iterations)"


def tst(level, measure):
    return TESTS[(TESTS.level == level) & (TESTS.measure == measure)].iloc[0]


def ovw(e, level="area_group", set_="N200_all_areas"):
    ov = E[e]["ov"]
    return ov[(ov.level == level) & (ov["set"] == set_)].iloc[0]


def reliable_sorted(e, level):
    ob = E[e]["ob"][level]
    return ob[~ob.flag].sort_values("onset_ms")


def ss_range(e, level, col):
    q = SS[(SS.epoch == e) & (SS.level == level) & (SS.n_sessions > 0)][col]
    return f"{int(q.min())}-{int(q.max())}"


def unrel(e, level="area_group"):
    return [short(a) for a, r in E[e]["ob"][level].iterrows() if r.flag]


# anchors in reading order (labels fixed before any text cites them)
for _a, _l in [("fig-passive-main", "Figure 1"), ("fig-active-main", "Figure 2"), ("fig-onset-acc", "Figure 3"),
               ("fig-matched", "Figure 4"), ("fig-peak", "Figure 5"), ("fig-onset-accuracy", "Figure 6"),
               ("fig-avp", "Figure 7"),
               ("fig-s-passive-main-areas", "Figure S1"), ("fig-s-passive-summary-groups", "Figure S2"),
               ("fig-s-passive-summary-areas", "Figure S3"), ("fig-s-passive-curves-groups", "Figure S4"),
               ("fig-s-passive-curves-areas", "Figure S5"), ("fig-s-passive-n", "Figure S6"),
               ("fig-s-passive-onset-acc", "Figure S7"), ("fig-s-active-main-areas", "Figure S8"),
               ("fig-s-active-summary-groups", "Figure S9"), ("fig-s-active-summary-areas", "Figure S10"),
               ("fig-s-active-curves-groups", "Figure S11"), ("fig-s-active-curves-areas", "Figure S12"),
               ("fig-s-active-n", "Figure S13"), ("fig-s-avp-areas", "Figure S14"),
               ("tbl-onsets", "Table 1"), ("tbl-matched", "Table 2"), ("tbl-tests", "Table 3"),
               ("tbl-s-sizes-groups", "Table S1"), ("tbl-s-sizes-areas", "Table S2"), ("tbl-s-onsets-areas", "Table S3"),
               ("tbl-s-iter", "Table S4"), ("tbl-s-seq", "Table S5"), ("tbl-a-params", "Table A1"),
               ("tbl-a-files", "Table A2")]:
    ANCHORS[_a] = _l


# ---------------------------------------------------------------------------------------------------------- tables
def onset_table(level):
    P, A = E["passive"]["ob"][level], E["active"]["ob"][level]
    wp, wa = E["passive"]["w200"][level], E["active"]["w200"][level]
    areas = sorted(P.index, key=lambda a: (P.loc[a, "onset_ms"] if np.isfinite(P.loc[a, "onset_ms"]) else 1e9, -wp.loc[a, "mean"]))
    return pd.DataFrame([{"Area": short(a), "Onset passive, ms [95 %]": rng_ms(P.loc[a]),
                          "Onset task, ms [95 %]": rng_ms(A.loc[a]) if a in A.index else "",
                          "Acc. 5-50 ms passive": f"{wp.loc[a, 'mean']:.3f}",
                          "Acc. 5-50 ms task": f"{wa.loc[a, 'mean']:.3f}" if a in wa.index else ""} for a in areas])


def sizes_table(level):
    q = SS[SS.level == level]
    rows = []
    for a in AR.LEVELS[level]:
        r = {"Area": short(a)}
        for e, lab in (("passive", "passive"), ("active", "task")):
            x = q[(q.area == a) & (q.epoch == e)]
            if len(x) and x.n_sessions.iloc[0] > 0:
                x = x.iloc[0]
                r[f"Sessions / mice ({lab})"] = f"{int(x.n_sessions)} / {int(x.n_mice)}"
                r[f"Units ({lab})"] = f"{int(x.n_units)}"
                r[f"Trials W / A ({lab})"] = f"{int(x.n_whisker_trials)} / {int(x.n_auditory_trials)}"
            else:
                r[f"Sessions / mice ({lab})"], r[f"Units ({lab})"], r[f"Trials W / A ({lab})"] = "0", "", ""
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- sections
def sec_front():
    lv = "area_group"
    rp, ra = reliable_sorted("passive", lv), reliable_sorted("active", lv)
    fp = rp[rp.onset_ms == rp.onset_ms.min()]
    fa = ra[ra.onset_ms == ra.onset_ms.min()]
    ova = ovw("active", lv, "all_areas_all_N")
    t_on, t_acc, t_b = tst(lv, "onset_ms"), tst(lv, "accuracy_5_50ms"), tst(lv, "baseline_-150_0ms")
    mice = sorted({s.split("_")[0] for s in SESS})
    return f"""---
title: "Whisker vs auditory stimulus decoding across brain regions"
author: "Axel Bisi (analysis with Claude Code)"
date: "{date.today().isoformat()}"
---

**Data version.** KS4 spike sorting (NWB_ks4), v2 unit table (good + mua units), SSL whisker-training sessions of both
cohorts, learning day and expert days pooled: {num('n_sess_a', len(SESS))} sessions from {num('n_mice', len(mice))} mice
with task (active) trials and {num('n_sess_p', len(SESS_P))} sessions with passive trials. Decoding of whisker vs auditory
stimuli in all {num('n_groups', len(AR.COARSE))} area groups and the {num('n_fine', len(AR.FINE))} best-sampled areas;
sample sizes per area in {ref('tbl-s-sizes-groups')} and {ref('tbl-s-sizes-areas')}. Onsets at N = {m4.N_MAIN} neurons:
passive trials {it_text('passive')}; task trials {it_text('active')}.

# Abstract

When and where can whisker and auditory stimuli be told apart from neural population activity? We decoded the stimulus
(whisker vs auditory) from pseudo-populations of N neurons pooled across sessions, in 20-ms bins stepped by 2 ms, and
located the first time bin at which decoding was reliably above a within-session label-shuffle null.

On passive trials (N = {m4.N_MAIN}), the stimulus became decodable first in {', '.join(short(a) for a in fp.index)}
({ms(fp.onset_ms.iloc[0])} ms after stimulus onset){prov_note('passive')}; on task trials first in
{', '.join(short(a) for a in fa.index)} ({ms(fa.onset_ms.iloc[0])} ms){prov_note('active')}, with the same overall order.

Onset decreased with early decoding strength along a decreasing exponential (task trials, every area × N: Spearman
ρ = {num('rho_all_active', ova.rho, '{:.2f}')}, exponential R² = {num('r2exp_all_active', ova.r2_exp, '{:.2f}')} vs linear
R² = {num('r2lin_all_active', ova.r2_ols, '{:.2f}')}), so part of the onset order reflects how much early information a
region carries; matching early accuracy across regions reduces but does not remove the onset differences.

Compared area by area, passive onsets were {num('onset_diff_median', abs(t_on.median_diff), '{:.0f}')} ms earlier (median)
and early accuracy {num('acc_diff_median', t_acc.median_diff, '{:+.3f}')} higher than on task trials, but passive decoding
was also above zero before the stimulus ({num('base_diff_median', t_b.median_diff, '{:+.3f}')} vs task trials), an offset
of the same size.

# Key results

1. **Passive trials, onset order (N = {m4.N_MAIN}).** First: {', '.join(f"{short(a)} {ms(r.onset_ms)} ms" for a, r in rp.head(6).iterrows())};
   last reliable: {', '.join(f"{short(a)} {ms(r.onset_ms)} ms" for a, r in rp.tail(3).iterrows())}{prov_note('passive')}
   ({ref('fig-passive-main')}, {ref('tbl-onsets')}).
2. **Task trials, same order.** First: {', '.join(f"{short(a)} {ms(r.onset_ms)} ms" for a, r in ra.head(6).iterrows())};
   last reliable: {', '.join(f"{short(a)} {ms(r.onset_ms)} ms" for a, r in ra.tail(3).iterrows())}{prov_note('active')}
   ({ref('fig-active-main')}).
3. **Onset follows early information.** Every area × N, task trials: ρ = {ova.rho:.2f}, p {pv('p_rho_all_active', ova.p_rho)},
   n = {int(ova.n)}; exponential fit R² = {ova.r2_exp:.2f} ({ref('fig-onset-acc')}).
4. **Task vs passive.** Onset passive - task: median {t_on.median_diff:+.1f} ms, Wilcoxon p {pv('p_onset_w', t_on.p_wilcoxon)},
   paired t p {pv('p_onset_t', t_on.p_paired_t)}, n = {int(t_on.n)} area groups; pre-stimulus corrected accuracy passive -
   task: median {t_b.median_diff:+.3f}, Wilcoxon p {pv('p_base_w', t_b.p_wilcoxon)}, n = {int(t_b.n)} ({ref('fig-avp')}, {ref('tbl-tests')}).
"""


def sec_intro():
    return f"""
# Introduction

In the SSL task, mice learn to lick after a brief whisker deflection (rewarded in the R+ cohort, not in the R- cohort)
and always lick after an auditory tone; outside the task, the same stimuli are delivered passively before and after the
session. Whisker and auditory stimuli travel along separate afferent pathways, so the first neural responses that
distinguish them should appear in sensory structures and spread to downstream regions.

This report asks where and when the two stimuli first become distinguishable across the {len(AR.FINE)} best-sampled
regions recorded with Neuropixels, how this depends on population size, and whether it differs between passive delivery
and the task. Passive trials are presented first, as the condition without licking or reward; task trials second; their
comparison last.

We use population decoding rather than single-neuron latencies because it pools weak signals across neurons and yields one
comparable measure per region. Pooling neurons across sessions into pseudo-populations lets every region be decoded at the
same population size N, which removes the most obvious confound between regions: the number of recorded neurons.

Two further confounds are addressed explicitly: decoding strength (a region with weaker information crosses any threshold
later) and the precision of the onset estimate.
"""


def sec_methods():
    p = E["active"]["prov"]
    return f"""
# Methods

## Data and inclusion

All whisker-training sessions of the v2 unit table with KS4 NWB files (both cohorts, learning day and expert days pooled;
cohort and stage are not analysed separately). Units: quality label good or mua. A session contributes to an area when it
has ≥ {m1.MIN_UNITS} units of the area and ≥ {m1.MIN_TRIALS} trials per class.

The resulting numbers of sessions, mice, units and trials per area and condition are in {ref('tbl-s-sizes-groups')}
(area groups: {ss_range('passive', 'area_group', 'n_sessions')} sessions per group on passive trials,
{ss_range('active', 'area_group', 'n_sessions')} on task trials) and {ref('tbl-s-sizes-areas')} (areas).

**Passive trials**: trials in passive context (a trial in a fixed ~3 s inter-trial sequence, or labelled passive),
pre- and post-task blocks pooled (perf 6 kept, as it codes passive trials); all whisker vs all auditory trials.

**Task (active) trials**: trials in active context, perf ≠ 6, warm-up cut (keep one trial before the first whisker trial),
disengagement tail trimmed with rule A1 (trials after the last lick dropped when that tail holds ≥ 5 whisker and ≥ 1
auditory trials); all whisker vs all auditory trials, regardless of the response.

**Spikes**: whisker stimulation produces an electrical artefact; spikes from -10 to +5 ms around every whisker onset are
replaced by a Poisson train at the unit's pre-onset rate (seeded per session). No information can therefore appear on
whisker trials before +5 ms.

## Pseudo-population decoding

One iteration draws {m1.N_SESS} eligible sessions with replacement, then N/{m1.N_SESS} units of the area within each drawn
session (with replacement; the remainder spread over random sessions), and builds {p['t_train']} + {p['t_train']}
pseudo-trials per class by balanced reuse of each session's real trials, separately for training and test folds.

Per time bin, an L2-regularised logistic regression (liblinear; inverse regularisation C chosen from {len(m1.C_GRID)}
values by an inner {p['n_inner']}-fold cross-validation) is trained and tested in a {p['n_outer']}-fold outer
cross-validation built on each session's real trials; features are z-scored in the training fold. The score is the
balanced accuracy $a_i(t)$ of iteration $i$ in bin $t$.

**Null and corrected accuracy.** The same draw is decoded again $S$ times with the trial labels permuted within each
session before pooling, with the same C per bin; the corrected accuracy is
$$d_i(t) = a_i(t) - \\frac{{1}}{{S}}\\sum_{{s=1}}^{{S}} a^{{\\mathrm{{shuf}}}}_{{i,s}}(t),$$
0 at chance and 0.5 for perfect decoding. The passive trial sequence is randomised, so the within-session shuffle is the
null for both conditions.

**Time resolutions.** Zoom: causal 20-ms bins in 2-ms steps from -20 to +100 ms (a bin labelled $t$ covers
$[t-20, t)$ ms); wide: causal 50-ms bins in 5-ms steps from -200 to +600 ms.

## Onset, unreliable onsets and accuracy measures

**Above chance.** Bin $t$ is above chance when the 5th percentile of $d_i(t)$ over iterations is > 0.

**Onset.** The first post-stimulus zoom bin that is above chance and is followed by ≥ 80 % above-chance bins over the next
25 ms (11 of 13 bins). **Onset range**: 95 % range of the onset over {m4.N_BOOT} resamples (with replacement) of the
iterations.

**Unreliable onset.** An onset is unreliable when its 95 % range is wider than {m4.WIDE_RANGE_MS} ms or the onset is
undefined in more than {100 * (1 - m4.MIN_DEFINED):.0f} % of the resamples: the data do not fix its position in the
ranking. Unreliable onsets are shown hatched with a dagger (†) and are excluded from rankings and tests.

**Early accuracy**: mean of $d_i(t)$ over the zoom bins ending 5-50 ms after onset, averaged over iterations.

**Peak accuracy**: the maximum over the post-stimulus wide bins of the mean corrected accuracy, with its time.

**Accuracy at onset**: the mean corrected accuracy in the onset bin.

## Population size, matched accuracy and onset vs accuracy

N = {', '.join(str(n) for n in m1.N_LIST)} neurons. **Matched accuracy** (task trials): for reference areas at N = 100,
each other area's N giving the same early accuracy is read from its accuracy-vs-N curve (log-N interpolation, extrapolated
beyond 500 up to 2000) and decoded again at that N.

**Onset vs early accuracy.** Spearman correlation, and a decreasing exponential
$\\mathrm{{onset}} = c + a\\,e^{{-x/\\tau}}$ fitted by least squares ($x$: early accuracy) and compared with a straight line
(R², AIC); 95 % band: 500 bootstrap resamples of the points. Shown for every area at N = {m4.N_MAIN} and for the 8
best-sampled areas at every N.

## Task vs passive comparison

Per area at N = {m4.N_MAIN}: onset, early accuracy and pre-stimulus corrected accuracy (mean over the wide bins ending
-150..0 ms, which contain no post-stimulus spikes). Paired over areas (passive - task): Wilcoxon signed-rank test and
paired t-test; onsets only for areas reliable in both conditions.

## Iterations, colours and software

**Iterations.** N sweep: 100 iterations × 10 shuffles. Onsets at N = {m4.N_MAIN}: 500 iterations × 20 shuffles. The
number of iterations was set after comparing 100 with 1000 iterations ({ref('tbl-s-iter')}).

**Colours.** Area groups: `ephys_utilities.allen_utils.get_custom_area_groups_colors()` (one colour per group); areas:
shades of their group's colour (group membership from `get_custom_area_groups_from_name()` of the same module).

**Code** (`exploratory-analyses/`): `001` decoding, `002` summaries and N-sweep figures, `003` matched accuracy, `004` main
figures, `005` final onsets, `006` task vs passive, `007` iteration check, `008` sample sizes and trial-sequence check,
`009` area rankings; decoder from the pseudo-population area-decoding project (002).
"""


def sec_results():
    lv = "area_group"
    rp, ra = reliable_sorted("passive", lv), reliable_sorted("active", lv)
    fpa, faa = reliable_sorted("passive", "area_acronym_custom"), reliable_sorted("active", "area_acronym_custom")
    ovp, ova = ovw("passive", lv), ovw("active", lv)
    ovpa, ovaa = ovw("passive", lv, "all_areas_all_N"), ovw("active", lv, "all_areas_all_N")
    ovb = ovw("active", lv, "best8_all_N")
    wa = E["active"]["w"]

    def acc(area, n):
        r = wa[(wa.level == lv) & (wa.area == area) & (wa.N == n)]
        return r["mean"].iloc[0] if len(r) else np.nan
    sw = "Somatosensory-whisker"
    t_on, t_acc, t_b = tst(lv, "onset_ms"), tst(lv, "accuracy_5_50ms"), tst(lv, "baseline_-150_0ms")
    tf_on, tf_b = tst("area_acronym_custom", "onset_ms"), tst("area_acronym_custom", "baseline_-150_0ms")
    mt = MATCH[MATCH.level == lv]
    mn = MATCHN[MATCHN.level == lv]
    notreached = sorted(set(mn[mn.how == "not reached"].area))
    refs = list(dict.fromkeys(mt.reference))
    matched_tbl = mt.assign(reference=mt.reference.map(short), area=mt.area.map(short), target=mt.target.map("{:.3f}".format),
                            achieved=mt.achieved.map("{:.3f}".format), matched_N=mt.matched_N.astype(int))[
        ["reference", "area", "matched_N", "target", "achieved"]].rename(columns={
            "reference": "Reference (N = 100)", "area": "Area", "matched_N": "Matched N", "target": "Target acc.",
            "achieved": "Achieved acc."})
    tests_tbl = TESTS.assign(level=TESTS.level.map(LV), measure=TESTS.measure.map(
        {"onset_ms": "onset (ms)", "accuracy_5_50ms": "accuracy 5-50 ms", "baseline_-150_0ms": "baseline -150..0 ms"}),
        mean_diff=TESTS.mean_diff.map("{:+.3f}".format), median_diff=TESTS.median_diff.map("{:+.3f}".format),
        p_wilcoxon=TESTS.p_wilcoxon.map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"),
        p_paired_t=TESTS.p_paired_t.map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")).rename(columns={
            "level": "Level", "measure": "Measure", "n": "n areas", "mean_diff": "Mean passive - task",
            "median_diff": "Median", "p_wilcoxon": "p Wilcoxon", "p_paired_t": "p paired t"})
    R = RANK.copy()

    def top(e, level, key, k=4):
        q = R[(R.epoch == e) & (R.level == level)].dropna(subset=[key])
        if key == "onset_acc":                         # unreliable onsets are not ranked
            q = q[~q.onset_flag.astype(bool)]
        q = q.sort_values(key, ascending=False).head(k)
        return ", ".join(f"{short(r.area)} ({r[key]:.2f})" for _, r in q.iterrows())
    pk = R[(R.level == lv)].pivot(index="area", columns="epoch", values="peak")
    rho_pk = pk.corr(method="spearman").loc["passive", "active"]
    seqp = SEQ[SEQ.epoch == "passive"]
    imb = (seqp.p_whisker_first_half - seqp.p_whisker_second_half).abs()

    def n_ss(e, lv_):
        return ss_range(e, lv_, "n_sessions")

    return f"""
# Results

## Passive trials: the order in which regions separate whisker from auditory stimuli

With passive trials and N = {m4.N_MAIN} neurons per pseudo-population, every area group decoded the stimulus above chance
shortly after onset ({ref('fig-passive-main')}).

The earliest onsets{prov_note('passive')} were in {', '.join(f"{short(a)} ({ms(r.onset_ms)} ms)" for a, r in rp.head(5).iterrows())};
the latest reliable ones in {', '.join(f"{short(a)} ({ms(r.onset_ms)} ms)" for a, r in rp.tail(3).iterrows())}
({ref('tbl-onsets')}). The onsets of {', '.join(unrel('passive')) or 'no area group'} were unreliable.

Among the {len(AR.FINE)} areas ({ref('fig-s-passive-main-areas')}, {ref('tbl-s-onsets-areas')}), the earliest were
{', '.join(f"{a} ({ms(r.onset_ms)} ms)" for a, r in fpa.head(8).iterrows())}. Time courses for every area and N are in
{ref('fig-s-passive-curves-groups')} and {ref('fig-s-passive-curves-areas')}, and summaries in
{ref('fig-s-passive-summary-groups')} and {ref('fig-s-passive-summary-areas')}.

Early accuracy grew with N and onsets became earlier as N grew ({ref('fig-s-passive-n')}). Onset decreased with early
accuracy along a decreasing exponential (every area at N = {m4.N_MAIN}: ρ = {num('rho_n200_passive', ovp.rho, '{:.2f}')},
exponential R² = {num('r2exp_n200_passive', ovp.r2_exp, '{:.2f}')}; every area × N: ρ = {num('rho_all_passive', ovpa.rho, '{:.2f}')};
{ref('fig-s-passive-onset-acc')}).

{figure('fig-passive-main', E['passive']['fig'][lv], f"Whisker vs auditory decoding across area groups, passive trials, all sessions pooled, N = {m4.N_MAIN} neurons per pseudo-population ({it_text('passive')}; {n_ss('passive', lv)} sessions per group, {ref('tbl-s-sizes-groups')}). (a) Corrected balanced accuracy (decoding minus trial-shuffle null; 0 = chance, 0.5 = perfect) over time, causal 50-ms bins in 5-ms steps; colour: magma, dark = low accuracy; grey: bins not above chance (5th percentile over iterations ≤ 0); tick: onset. Rows sorted by onset. (b) The same for the first 50 ms, causal 20-ms bins in 2-ms steps (bin labelled at its end). (c) Onset per group; error bars: 95 % range over {m4.N_BOOT} resamples of the iterations; † hatched: unreliable onset (95 % range > {m4.WIDE_RANGE_MS} ms), not ranked. (d, e) Time courses of the 8 best-sampled groups, mean ± s.d. over iterations, -200..600 ms and the first 50 ms; bars above: bins above chance; black tick: onset. Colours: area-group palette (ephys_utilities.allen_utils).")}

{mdtable('tbl-onsets', onset_table(lv), f"Onsets (ms, [95 % range over resamples of the iterations]) and early corrected accuracy (5-50 ms) at N = {m4.N_MAIN}, area groups, sorted by the passive onset. Passive: {it_text('passive')}; task: {it_text('active')}. † unreliable onset.")}

## Task trials: the same order

On task trials ({ref('fig-active-main')}), the earliest onsets{prov_note('active')} were in
{', '.join(f"{short(a)} ({ms(r.onset_ms)} ms)" for a, r in ra.head(5).iterrows())} and the latest reliable ones in
{', '.join(f"{short(a)} ({ms(r.onset_ms)} ms)" for a, r in ra.tail(3).iterrows())}; unreliable:
{', '.join(unrel('active')) or 'none'} ({ref('tbl-onsets')}).

Among the areas ({ref('fig-s-active-main-areas')}), the earliest were
{', '.join(f"{a} ({ms(r.onset_ms)} ms)" for a, r in faa.head(8).iterrows())}. Per-N time courses and summaries are in
{ref('fig-s-active-curves-groups')}, {ref('fig-s-active-curves-areas')}, {ref('fig-s-active-summary-groups')} and
{ref('fig-s-active-summary-areas')}.

{figure('fig-active-main', E['active']['fig'][lv], f"As {ANCHORS['fig-passive-main']}, task (active) trials ({it_text('active')}; {n_ss('active', lv)} sessions per group).")}

## Onset follows the amount of early information

Early accuracy grew with the number of neurons in every region (e.g. {short(sw)} {num('acc_sw_20', acc(sw, 20))} at
N = 20 and {num('acc_sw_500', acc(sw, 500))} at N = 500; {ref('fig-s-active-n')}), and onsets became earlier as N grew.

Onset decreased with early accuracy along a decreasing exponential rather than a line ({ref('fig-onset-acc')}). Across
every area at N = {m4.N_MAIN}: ρ = {num('rho_n200_active', ova.rho, '{:.2f}')}, exponential R² = {num('r2exp_n200_active', ova.r2_exp, '{:.2f}')},
linear R² = {num('r2lin_n200_active', ova.r2_ols, '{:.2f}')}. Within the 8 best-sampled groups across N:
ρ = {num('rho_best8_active', ovb.rho, '{:.2f}')}, exponential R² = {num('r2exp_best8_active', ovb.r2_exp, '{:.2f}')}.
Pooling every area × N: exponential R² = {ovaa.r2_exp:.2f} vs linear {ovaa.r2_ols:.2f}.

The exponential shape means that onsets saturate near the earliest values for strongly decodable regions, while weakly
decodable regions have long and variable onsets. The onset of a region is therefore partly a read-out of how much early
information it carries.

To separate the two, early accuracy was matched to a reference region at N = 100 ({ref('fig-matched')}, {ref('tbl-matched')};
references: {', '.join(short(r) for r in refs)}). At matched accuracy the time courses rose at similar times, with
differences of a few ms. {', '.join(short(a) for a in notreached) if notreached else 'No region'} could not reach at
least one reference accuracy within N ≤ 2000.

{figure('fig-onset-acc', FIGDIR['active'] / 'onset_vs_window.png', f"Onset vs early accuracy, task trials (N sweep, 100 iterations × 10 shuffles). (a) Every area group at N = {m4.N_MAIN}; (b) the 8 best-sampled groups at every N (dot size: N = 20-500; lines join a group's values); (c, d) the same for areas. Black line: least-squares decreasing exponential, onset = c + a·exp(-accuracy/τ), solid when Spearman p < 0.05; grey band: 95 % range of the fit over 500 bootstrap resamples of the points. Text: Spearman ρ, p, n points; R² of the exponential and of a straight line; ΔAIC (exponential - linear). Onset: 20-ms bins, 2-ms steps. Colours: area palette.")}

{figure('fig-matched', FIGDIR['active'] / 'matched_accuracy.png', "Matched-accuracy control, task trials, area groups. Each row: one reference group decoded with 100 neurons; every other group decoded at the N that gives the same corrected accuracy in 5-50 ms (read from its accuracy-vs-N curve, log-N interpolation; * extrapolated beyond 500; groups not reaching the target within 2000 neurons are listed as > 2000). (a, d, g) Time courses at the matched N, 50-ms bins, mean ± s.d. over iterations; (b, e, h) first 100 ms, 20-ms bins, shaded: 5-50 ms window; legend: group (matched N); thick line: reference. (c, f, i) Matched N per group, log scale; dotted line: 100 neurons. 100 iterations × 10 shuffles.")}

{mdtable('tbl-matched', matched_tbl, "Matched-accuracy control, task trials, area groups: target = the reference's corrected accuracy (5-50 ms) at N = 100; achieved = the group's accuracy when decoded at its matched N.")}

## Which regions decode best

Ranked by peak accuracy ({ref('fig-peak')}), the best area groups were {top('passive', lv, 'peak')} on passive trials
and {top('active', lv, 'peak')} on task trials. The ranking of group peaks was similar in the two conditions (Spearman
ρ = {num('rho_peak_epochs', rho_pk, '{:.2f}')}, n = {len(pk)} groups). Among areas, the best were
{top('passive', 'area_acronym_custom', 'peak')} (passive) and {top('active', 'area_acronym_custom', 'peak')} (task).

The accuracy reached in the onset bin ({ref('fig-onset-accuracy')}) was much lower than the peak in every region, as
expected for the first bin passing the criterion. Among reliable onsets it was highest in {top('passive', lv, 'onset_acc', 3)} (passive) and
{top('active', lv, 'onset_acc', 3)} (task). Its wide error bars show that the onset bin falls on the rising edge of the
time course, where iterations differ most.

{figure('fig-peak', HOME / 'figures' / 'area_peak_accuracy.png', f"Peak whisker vs auditory decoding per region, N = {m4.N_MAIN}, ordered by decreasing peak. Peak: maximum over post-stimulus causal 50-ms bins (5-ms steps, up to 600 ms) of the corrected balanced accuracy (mean over iterations); error bars: 2.5-97.5 percentiles over iterations in that bin; text: time of the peak (bin end). (a) Area groups, passive trials; (b) area groups, task trials; (c) areas, passive; (d) areas, task. Dotted line: 0.5 (perfect decoding). Iterations per panel in the titles. Colours: area palette.")}

{figure('fig-onset-accuracy', HOME / 'figures' / 'area_onset_accuracy.png', f"Corrected balanced accuracy in the onset bin per region, N = {m4.N_MAIN}, ordered by decreasing value; layout as {ANCHORS['fig-peak']}. Onset bin: causal 20-ms bin, 2-ms steps (onset rule in Methods); error bars: 2.5-97.5 percentiles over iterations in that bin; text: onset time; † hatched: unreliable onset. Regions without an onset are omitted.")}

## Task vs passive trials

Compared area by area ({ref('fig-avp')}, {ref('tbl-tests')}; areas: {ref('fig-s-avp-areas')}), passive onsets were earlier
than task onsets (passive - task, median {t_on.median_diff:+.1f} ms, n = {int(t_on.n)} area groups, Wilcoxon p {pv('p_onset_w2', t_on.p_wilcoxon)};
areas {tf_on.median_diff:+.1f} ms, n = {int(tf_on.n)}, p {pv('p_onset_w_fine', tf_on.p_wilcoxon)}), and early accuracy was
higher by {t_acc.median_diff:+.3f}.

Passive decoding was, however, already above zero before the stimulus: the pre-stimulus corrected accuracy exceeded the
task-trial value by {t_b.median_diff:+.3f} (area groups, Wilcoxon p {pv('p_base_w2', t_b.p_wilcoxon)}; areas
{tf_b.median_diff:+.3f}, p {pv('p_base_w_fine', tf_b.p_wilcoxon)}), as large as the early-accuracy difference. An offset
of this size moves the above-chance criterion earlier by itself, so the earlier passive onset is not evidence of faster
processing.

The passive trial order does not explain the offset: consecutive passive trials repeated the stimulus type as often as an
independent sequence would (median P(repeat) {num('seq_rep_passive', seqp.p_repeat.median(), '{:.3f}')} vs
{num('seq_rep_indep_passive', seqp.p_repeat_independent.median(), '{:.3f}')}, n = {num('seq_n_passive', len(seqp))}
sessions; {ref('tbl-s-seq')}). The whisker fraction differed by more than 0.2 between the first and second half of the
passive trials (roughly the pre- and post-task blocks) in {num('seq_imb_passive', int((imb > 0.2).sum()))} sessions; in
those sessions, a state difference between the blocks would make pre-stimulus activity informative about the stimulus.

Later in the trial, task-trial decoding stayed high while passive decoding decayed ({ref('fig-avp')}e), consistent with
licking and reward-related activity on task trials.

{figure('fig-avp', HOME / 'figures' / 'active_vs_passive_area_group.png', f"Task (active) vs passive trials, area groups, N = {m4.N_MAIN}. (a) Onset per group (filled: task, open: passive), 95 % ranges over resamples of the iterations; grey: unreliable in either condition, not tested. (b) Early corrected accuracy (5-50 ms), 2.5-97.5 percentiles over iterations. (c) Pre-stimulus corrected accuracy (wide bins ending -150..0 ms), mean ± 1.96 s.e.m. over iterations; expected 0. Rows sorted by the task-trial onset. Under each axis: passive - task median, Wilcoxon signed-rank and paired t-test over groups. (d, e) First 50 ms and -200..600 ms time courses of the 6 best-sampled groups, task solid, passive dashed, mean ± s.d. over iterations. Passive: {it_text('passive')}; task: {it_text('active')}.")}

{mdtable('tbl-tests', tests_tbl, "Passive - task differences over regions (paired), area groups and areas: onset (reliable in both conditions only), early accuracy (5-50 ms) and pre-stimulus corrected accuracy (-150..0 ms).")}
"""


def sec_discussion():
    return """
# Discussion

Whisker and auditory stimuli become distinguishable in most recorded regions within a few tens of ms, first in midbrain,
thalamic and whisker-somatosensory populations and last in orofacial-somatosensory, insular and olfactory populations. The
order is broadly the one expected from the afferent pathways: superior colliculus, thalamus and barrel cortex receive
whisker input within a few ms, auditory structures receive auditory input at similar latencies, and striatum, motor and
frontal cortices follow.

Onset and decoding strength are entangled. The exponential relation between onset and early accuracy means that a "late"
region is often a region with little early information; the matched-accuracy control reduces the onset differences but
cannot be applied to regions whose accuracy saturates below the reference.

A population onset is also not a first-spike latency: it is the first time a linear read-out of N neurons separates the two
stimuli reliably, and it depends on the bin width (20 ms, causal) and on the criterion. The companion sensory-maps project
relates these onsets to single-neuron latencies and projection zones.

The comparison of passive and task trials asks whether engagement changes how fast the two stimuli are told apart. The
onset order is the same in both conditions; the earlier and stronger passive decoding coincides with a pre-stimulus
offset of the same size and cannot be read as faster processing.
"""


def sec_caveats():
    up = sorted(set(unrel("passive")) | set(unrel("active")))
    return f"""
# Caveats and limitations

- **Passive pre-stimulus decoding.** Corrected accuracy is above zero before stimulus onset on passive trials. The trial
  order is random, but the pre- and post-task blocks are pooled and their whisker fractions differ in some sessions; a
  state difference between blocks would make baseline activity informative. Splitting pre and post blocks would test this.
- **Whisker artefact.** Spikes from -10 to +5 ms around whisker onsets are replaced by Poisson spikes, so whisker-evoked
  information can appear only from +5 ms; with causal 20-ms bins, an onset at t means the information is present somewhere
  in [t-20, t) ms.
- **Pooling.** Cohorts (R+, R-) and stages (learning, expert) are pooled; differences between them are not tested.
- **Iterations.** The N sweep (N = 20-500), the matched-accuracy control and the onset-vs-accuracy fits use 100 iterations ×
  10 shuffles; only the N = {m4.N_MAIN} onsets use 500 × 20.
- **Unreliable onsets.** {', '.join(up) or 'None'} (area groups) have onsets that vary by more than {m4.WIDE_RANGE_MS} ms
  across resamples in at least one condition; they are not ranked.
- **Late time points.** On task trials, licking and reward follow both stimuli at different rates; late decoding can
  reflect behaviour rather than the stimulus.

# Open questions and next steps

1. Split passive trials into pre- and post-task blocks to locate the pre-stimulus offset.
2. Decode whisker vs catch and auditory vs catch trials to attribute early information to one stimulus per region.
3. Cohort and stage splits (R+ vs R-, learning vs expert) of onsets and early accuracy.
4. Relate population onsets to single-neuron latencies for all area groups (sensory-maps project).
"""


def sec_supp():
    o = ["\n# Supplementary figures\n"]
    o.append(figure("fig-s-passive-main-areas", E["passive"]["fig"]["area_acronym_custom"],
                    f"As {ANCHORS['fig-passive-main']}, the {len(AR.FINE)} best-sampled areas, passive trials ({it_text('passive', 'area_acronym_custom')}; {ss_range('passive', 'area_acronym_custom', 'n_sessions')} sessions per area, {ref('tbl-s-sizes-areas')}). Colours: shades of the area's group colour."))
    for e, lab, tag in (("passive", "passive trials", "passive"), ("active", "task (active) trials", "active")):
        for lv, an, word in (("area_group", f"fig-s-{tag}-summary-groups", "area group"),
                             ("area_acronym_custom", f"fig-s-{tag}-summary-areas", "area")):
            f_txt = ("time courses at the N matching the early accuracy of SS-whisker at 100 neurons."
                     if (e == "active" and lv == "area_group") else "time courses of the 8 best-sampled, first 100 ms.")
            o.append(figure(an, FIGDIR[e] / f"arrival_summary_{lv}.png",
                            f"Summary, {lab}, {word}s (N sweep: 100 iterations × 10 shuffles). (a) Method. (b) First 50 ms at N = 200 (magma, dark = low corrected accuracy; grey: not above chance). (c) Onset at N = 200, 95 % range; † unreliable. (d) Corrected accuracy 5-50 ms vs number of neurons (colour: 8 best-sampled, grey: others). (e) Onset vs early accuracy over every {word} × N, decreasing-exponential fit with 95 % bootstrap band (dot size: N). (f) {f_txt} (g) Control: raw balanced accuracy (solid) and the trial-shuffle null (dotted) for three regions. (h) Time courses of the 8 best-sampled, -200..600 ms; bars: bins above chance; tick: onset."))
        for lv, an, word in (("area_group", f"fig-s-{tag}-curves-groups", "area group"),
                             ("area_acronym_custom", f"fig-s-{tag}-curves-areas", "area")):
            o.append(figure(an, FIGDIR[e] / f"arrival_curves_{lv}.png",
                            f"Corrected balanced accuracy over time for every {word} and every N (colour: N = 20-500), {lab}. Left of each pair: 50-ms bins, -200..600 ms (grey: zoomed window); right: 20-ms bins, -20..100 ms; mean ± s.d. over iterations (100 × 10); triangles: onset per N."))
        o.append(figure(f"fig-s-{tag}-n", FIGDIR[e] / "accuracy_onset_vs_N.png",
                        f"Effect of the number of neurons, {lab}. (a, c) Corrected accuracy 5-50 ms vs N, area groups and areas, 95 % range over iterations (100 × 10); (b, d) onset (20-ms bins, 2-ms steps) vs N; missing points: no onset at that N. Colours: area palette."))
        if e == "passive":
            o.append(figure("fig-s-passive-onset-acc", FIGDIR["passive"] / "onset_vs_window.png",
                            f"As {ANCHORS['fig-onset-acc']}, passive trials."))
            o.append(figure("fig-s-active-main-areas", E["active"]["fig"]["area_acronym_custom"],
                            f"As {ANCHORS['fig-passive-main']}, areas, task trials ({it_text('active', 'area_acronym_custom')}; {ss_range('active', 'area_acronym_custom', 'n_sessions')} sessions per area)."))
    o.append(figure("fig-s-avp-areas", HOME / "figures" / "active_vs_passive_area_acronym_custom.png",
                    f"As {ANCHORS['fig-avp']}, the {len(AR.FINE)} areas (d, e: the 6 best-sampled areas)."))
    return "\n".join(o)


def sec_supp_tables():
    it = ITER
    it_tbl = pd.DataFrame({"Area": it.area.map(short), "Onset 100 it.": it.onset_pilot.map(ms), "Onset 1000 it.": it.onset_final.map(ms),
                           "100-it. subsets [95 %]": [f"[{a:.0f}, {b:.0f}]" for a, b in zip(it.onset_sub100_lo, it.onset_sub100_hi)],
                           "Acc. 100 it.": it.early_pilot.map("{:.3f}".format), "Acc. 1000 it.": it.early_final.map("{:.3f}".format),
                           "r curves": it.curve_r.map("{:.3f}".format), "Bins differing": it.sig_bins_differ})
    rows = []
    for e, lab in (("passive", "passive"), ("active", "task")):
        q = SEQ[SEQ.epoch == e]
        d = (q.p_whisker_first_half - q.p_whisker_second_half).abs()
        rows.append({"Condition": lab, "Sessions": len(q), "Trials (median)": f"{q.n_trials.median():.0f}",
                     "P(repeat)": f"{q.p_repeat.median():.3f}", "P(repeat), independent": f"{q.p_repeat_independent.median():.3f}",
                     "|Δ whisker fraction|, halves": f"{d.median():.3f}", "Sessions with |Δ| > 0.2": int((d > 0.2).sum())})
    return "\n# Supplementary tables\n" + "\n".join([
        mdtable("tbl-s-sizes-groups", sizes_table("area_group"), f"Sample sizes per area group and condition: eligible sessions (≥ {m1.MIN_UNITS} good + mua units of the group and ≥ {m1.MIN_TRIALS} trials per class) and their mice, units (sum over eligible sessions) and whisker / auditory trials (sum)."),
        mdtable("tbl-s-sizes-areas", sizes_table("area_acronym_custom"), f"As {ANCHORS['tbl-s-sizes-groups']}, the {len(AR.FINE)} areas."),
        mdtable("tbl-s-onsets-areas", onset_table("area_acronym_custom"), f"As {ANCHORS['tbl-onsets']}, the {len(AR.FINE)} areas."),
        mdtable("tbl-s-iter", it_tbl, "Task trials, N = 200: 100 iterations × 10 shuffles vs 1000 × 20, for the areas run to 1000 iterations. Onsets in ms; 100-it. subsets: 95 % range of the onset over 200 random subsets of 100 of the 1000 iterations; r curves: correlation of the mean time courses (all bins); bins differing: bins whose above-chance call differs (of 222)."),
        mdtable("tbl-s-seq", pd.DataFrame(rows), "Trial sequences per condition (sessions with ≥ 10 trials): P(repeat) = probability that a trial has the same stimulus type as the previous one, and its value for an independent sequence with the session's whisker fraction (medians over sessions); |Δ whisker fraction|: absolute difference of the whisker fraction between the first and second half of the session's trials (passive: roughly pre- and post-task blocks), median over sessions.")])


def sec_appendix():
    p = E["active"]["prov"]
    par = pd.DataFrame([
        ("Sessions per draw", m1.N_SESS), ("Min units / trials per class", f"{m1.MIN_UNITS} / {m1.MIN_TRIALS}"),
        ("Pseudo-trials per class (train / test)", f"{p['t_train']} / {p['t_test']}"),
        ("Outer / inner CV folds", f"{p['n_outer']} / {p['n_inner']}"), ("C grid", ", ".join(f"{c:g}" for c in m1.C_GRID)),
        ("N sweep", ", ".join(map(str, m1.N_LIST))), ("Zoom bins", p["zoom"]), ("Wide bins", p["wide"]),
        ("Early window", "5-50 ms (zoom bin ends)"), ("Baseline window", "wide bins ending -150..0 ms"),
        ("Peak", "max over post-stimulus wide bins"), ("Above chance", "5th percentile over iterations > 0"),
        ("Onset", "≥ 80 % of the next 25 ms above chance (11 / 13)"), ("Onset resamples", m4.N_BOOT),
        ("Unreliable onset", f"95 % range > {m4.WIDE_RANGE_MS} ms or undefined in > 5 %"),
        ("Iterations × shuffles, N sweep", "100 × 10"), ("Iterations × shuffles, N = 200", f"{m1.N_ITER} × {m1.N_SHUF}"),
        ("Artefact window (whisker)", "-10..+5 ms, Poisson replacement")], columns=["Parameter", "Value"])
    files = pd.DataFrame([
        ("active/, passive/", "N sweep per condition (raw, summaries, figures); active/: matched accuracy"),
        ("active_final_n200/, passive_final_n200/", "N = 200 onsets, 500 × 20"),
        ("tables/", "sample_sizes, trial_sequence_check, area_ranking, active_vs_passive_*, iteration_check_n200"),
        ("figures/", "active_vs_passive_*, area_peak_accuracy, area_onset_accuracy")], columns=["Results folder", "Content"])
    return f"""
# Appendix

{mdtable('tbl-a-params', par, "Parameters.")}

{mdtable('tbl-a-files', files, "Result files (project results folder in combined_results_ks4/ on the NAS).")}

**Version history.** 2026-10-04: N sweep, matched accuracy, main and summary figures (task trials); passive trials.
2026-10-06: N = 200 onsets at 500 iterations × 20 shuffles after the iteration check; task vs passive comparison with the
pre-stimulus baseline; unreliable onsets flagged; exponential fit of onset vs early accuracy; region rankings by peak and
onset accuracy; sample-size and trial-sequence tables; allen_utils colours; report restructured (passive, task,
comparison). Parameters and result files: {ref('tbl-a-params')}, {ref('tbl-a-files')}.
"""


def check(md):
    """every registered figure / table defined and cited at least once; no 'arrival' wording"""
    bad = [f"{a}: not defined" for a in ANCHORS if f"{{#{a}}}" not in md]
    bad += [f"{a}: never cited" for a in ANCHORS if not re.search(rf"\(#{re.escape(a)}\)", md)]
    bad += ["contains 'arrival'"] if "arrival" in md.lower() else []
    if bad:
        raise SystemExit("report not written:\n  " + "\n  ".join(bad))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = (sec_front() + sec_intro() + sec_methods() + sec_results() + sec_discussion() + sec_caveats() + sec_supp()
          + sec_supp_tables() + sec_appendix())
    check(md)
    (OUT / "figures").mkdir(exist_ok=True)
    for old in (OUT / "figures").glob("*.png"):
        if old.name not in FIGS:
            old.unlink()
    for dest, src in FIGS.items():
        shutil.copyfile(src, OUT / "figures" / dest)
    (OUT / "report.md").write_text(md, encoding="utf-8")
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1), encoding="utf-8")
    shutil.copyfile(REPO / "skills" / "project-report" / "build.sh", OUT / "build.sh")
    print(f"report.md ({len(md.split())} words), {len(FIGS)} figures, {len(ANCHORS)} anchors, {len(NUM)} numbers -> {OUT}")
    print({e: {lv: E[e]["src"][lv]["kind"] for lv in LEVELS} for e in EPOCHS})


if __name__ == "__main__":
    main()
