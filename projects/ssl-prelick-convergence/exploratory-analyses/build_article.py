"""Build the pre-lick convergence article for one reference x population from the 062 outputs.
usage: python build_article.py <publication_dir> <ref: fa|sl> <pop: all|learners>
reads  <publication_dir>/stats_<pop>.csv and captions_<pop>.md; writes <publication_dir>/prelick_convergence_<ref>_<pop>.qmd
(figures referenced by file name; render with quarto next to the PNGs)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PUB, REF, POP = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
S = pd.read_csv(PUB / f"stats_{POP}.csv")
CAP = (PUB / f"captions_{POP}.md").read_text(encoding="utf-8")
G = {"RpL": "R+ learning", "RpE": "R+ expert", "RmL": "R− learning", "RmE": "R− expert"}
RA = "FA" if REF == "fa" else "SL"
RN = "false alarm" if REF == "fa" else "spontaneous lick"
RNP = RN + "s"
FA = REF == "fa"


def row(panel):
    r = S[S.panel == panel]
    return r.iloc[0] if len(r) else None


def P(x):
    if x is None or pd.isna(x):
        return "n/a"
    return "p < 0.001" if x < 0.001 else (f"p = {x:.3f}" if x < 0.01 else f"p = {x:.2f}")


def m(panel, g, f=2):
    r = row(panel)
    return "n/a" if r is None or pd.isna(r.get(f"mean_{G[g]}")) else f"{r[f'mean_{G[g]}']:.{f}f}".replace("-", "−")


def n(panel, g):
    r = row(panel)
    return "n/a" if r is None or pd.isna(r.get(f"n_{G[g]}")) else str(int(r[f"n_{G[g]}"]))


def p(panel, test):
    r = row(panel)
    key = {"R+": "R+ L vs E p_MWU", "R-": "R- L vs E p_MWU", "exp": "expert R+ vs R- p_MWU",
           "int": "interaction_p_perm", "R+w": "R+ L vs E p_Welch", "expw": "expert R+ vs R- p_Welch"}[test]
    return "n/a" if r is None else P(r.get(key))


def v(panel, col, f=2):
    r = row(panel)
    return "n/a" if r is None or pd.isna(r.get(col)) else f"{r[col]:.{f}f}".replace("-", "−")


def an(term):
    r = S[(S.panel == "4a ANOVA") & (S.measure == term)]
    if not len(r):
        return "n/a"
    r = r.iloc[0]
    return f"F = {r.F:.1f}, {P(r.p)} (F test), {P(r.p_perm)} (mouse-level permutation)"


def an2(term):
    r = S[(S.panel == "2g ANOVA") & (S.measure == term)]
    if not len(r):
        return "n/a"
    r = r.iloc[0]
    return f"F = {r.F:.2f}, {P(r.p_perm)} (mouse-level permutation)"


r2g = S[S.panel == "2g ANOVA"]
_pc = r2g[r2g.measure == "cohort"].p_perm.iloc[0] if len(r2g) else np.nan
_pi = r2g[r2g.measure == "cohort x area"].p_perm.iloc[0] if len(r2g) else np.nan
G2_COH = ("converging neurons were more frequent in R+ than in R− experts" if _pc < 0.05 else
          "the fraction of converging neurons did not differ significantly between R+ and R− experts at the area level")
G2_INT = ("the cohort difference varied between areas" if _pi < 0.05 else
          "the cohort difference did not detectably depend on the area, i.e. a broad shift rather than a few hotspots")
n_g = int(r2g.n_areas.iloc[0]) if len(r2g) else "n/a"


def pp(ro, key):
    r = row(f"5 pp {ro}")
    if r is None:
        return "n/a"
    if key in ("dRp_p_boot", "dRm_p_boot", "interaction_p_perm"):
        return P(r[key])
    return f"{r[key]:+.2f}".replace("-", "−")


def caption(header_start):
    """caption text of one figure from captions_<pop>.md, joined into one paragraph per figure"""
    blocks = re.split(r"\n## ", CAP)
    for b in blocks:
        if b.startswith(header_start):
            body = b.split("\n", 1)[1].strip()
            return " ".join(body.split("\n")).replace("[", "\\[").replace("]", "\\]")
    return ""


GEN = re.search(r"\*\*General\.\*\*(.*?)\n", CAP)
GEN = GEN.group(1).strip() if GEN else ""
POPTXT = "all mice" if POP == "all" else "learners only (day-0 sessions of non-learning mice removed; their expert sessions kept)"
REFTXT = ("false alarms (FA), licks on no-stimulus catch trials" if FA else
          "spontaneous licks (SL), licks emitted during no-stimulus periods between trials")
RT_SENT = ("" if not FA else
           f" Because whisker reaction times shorten in R+ mice, we repeated the analysis with reaction-time-matched trials: "
           f"Δd still increased in R+ mice ({p('3g', 'R+')}), but the expert R+ vs R− difference ({p('3g', 'exp')}) and the "
           f"interaction ({p('3g', 'int')}) were no longer significant, so part of the cohort difference may relate to "
           f"reaction time.")


FIGW = {"Fig1_task_data.png": 100, "Fig2_single_neurons.png": 78, "Fig2S_converging_neurons.png": 72,
        "Fig3_population_distance.png": 100, "Fig3S_lambda_lambdaLDA.png": 92, "Fig4_areas.png": 100,
        "Fig5_decoders.png": 100}
NUM = {"Figure 1": "Figure 1", "Figure 2 |": "Figure 2", "Figure 2—supplement": "Figure 2—supplement",
       "Figure 3 |": "Figure 3", "Figure 3—supplement": "Figure 3—supplement", "Figure 4": "Figure 4",
       "Figure 5": "Figure 5"}


def typst_escape(t):
    t = t.replace("\\[", "[").replace("\\]", "]")
    for ch in ["\\", "#", "$", "@", "<", ">", "_", "[", "]", "`", "~"]:
        t = t.replace(ch, "\\" + ch)
    return t.replace("**", "*")


def fig(file, header, width=None):
    """image (no float, sized to fit one page) followed by the full caption as a small-font paragraph (can break)"""
    w = width or FIGW.get(file, 100)
    cap = typst_escape(caption(header))
    return (f"![]({file}){{width={w}%}}\n\n```{{=typst}}\n#block(inset: (x: 0.3em))[#text(size: 8pt)"
            f"[*{NUM[header]}.* {cap}]]\n```\n")


r2f = {c: row(f"2f {c}") for c in ["R+", "R-"]}


def f2f(c, k, f=2):
    r = r2f[c]
    return "n/a" if r is None else f"{r[k]:.{f}f}".replace("-", "−")


txt = f"""---
title: "Reward contingency moves whisker-triggered licks toward auditory-triggered licks across the mouse brain"
subtitle: "Pre-lick single-neuron, population and decoding analyses — reference: {RNP}; population: {POPTXT.split(' (')[0]} (working draft)"
author: "Axel Bisi"
date: 2026-10-03
format:
  typst:
    papersize: a4
    margin:
      x: 1.8cm
      y: 1.8cm
    fontsize: 10pt
    section-numbering: "1."
---

# Summary

Mice learned a whisker/auditory detection task in which an auditory-triggered lick is always rewarded, whereas a
whisker-triggered lick is rewarded in one cohort (R+) and not in the other (R−). With brain-wide Neuropixels recordings
on the first whisker-training day (learning) and on later days (expert), we asked whether the activity that precedes a
whisker-triggered lick comes to resemble the activity that precedes an auditory-triggered lick once whisker licks are
rewarded. We compared whisker hits (WH) and auditory hits (AH) with an unrewarded-lick reference, {RNP} ({RA}), in the
100 ms before the first lick. In R+ mice, whisker-hit activity departed from the {RN} state and moved in the
direction of auditory hits from learning to expert stage. In R− mice it did not. The change appeared in single
neurons (converging neurons: learning × cohort interaction {p('2c', 'int')}), in cross-validated population distances
(distance difference: {p('3c', 'int')}) and in chance-corrected decoders (probability readout: {p('5d', 'int')}). It was
distributed across the brain rather than confined to one area group. Whisker hits moved toward auditory hits but did
not converge onto them: their distance to auditory hits did not shrink, and part of their displacement was orthogonal
to the reward-lick axis.

# Introduction

Associative learning attaches value and action to sensory cues. When two cues of different modalities lead to the same
rewarded action, does the brain come to represent them alike once the action is initiated? Our task dissociates the cue
modality from its outcome. Auditory-triggered licks are rewarded in all mice; whisker-triggered licks are rewarded only in
R+ mice, although the stimulus, the action and the definition of a whisker hit are identical in both cohorts. If reward
contingency shapes the state preceding the action, whisker hits should become auditory-hit-like in R+ but not in R− mice.
Licks without a reward-predicting stimulus provide the reference for an unrewarded lick. We focused on the 100 ms
before the first lick, so that all event types are aligned to the same motor event. We asked the question at four
levels: single neurons, population geometry, linear decoders and brain areas.

# General methods

**Task and cohorts.** On each trial a whisker stimulus, an auditory stimulus or no stimulus was presented. A lick in the
response window after an auditory stimulus was always rewarded; after a whisker stimulus it was rewarded only in R+ mice;
no-stimulus trials were never rewarded. Learning is the first whisker-training day (day 0), expert any later day.

**Events, trials and units.** {GEN} The reference used throughout this report is {REFTXT}.

**Unit of analysis and statistics.** The session is the unit of analysis. Each mouse contributes one learning session;
experts may contribute several sessions. For each measure we report two-sided Mann-Whitney U and Welch t tests for
learning vs expert within each cohort and for R+ vs R− at each stage. The learning × cohort interaction
[Δ(R+) − Δ(R−)] is tested by permuting cohort labels across mice, which respects the nesting of sessions within mice.
Sessions per group (Fig. 2b): R+ learning {n('2b', 'RpL')}, R+ expert {n('2b', 'RpE')}, R− learning {n('2b', 'RmL')},
R− expert {n('2b', 'RmE')}.

# Pre-lick activity and its quantification

**Motivation.** Before testing for convergence, we described the raw signal: how firing before the first lick differs
between event types, cohorts and stages, and whether the effect is visible without any model.

**Method.** For every tested unit we computed first-lick-aligned PSTHs, subtracted the pre-trial baseline, averaged
over units within each session and then across sessions. We quantified the mean change in the 100 ms pre-lick window
per event type. Because overall excitability may change with training, we also expressed the hit responses relative to
the {RN} response of the same sessions (ratio WH/{RA} and AH/{RA}).

**Results.** Pre-lick activity rose before every lick type (Fig. 1g).

- **Whisker hits.** The pre-lick change was {m('1h WH', 'RpL')} → {m('1h WH', 'RpE')} Hz in R+ mice and
  {m('1h WH', 'RmL')} → {m('1h WH', 'RmE')} Hz in R− mice (learning → expert); at expert stage R+ exceeded R−
  ({p('1h WH', 'exp')}).
- **Auditory hits and {RNP}.** Responses did not differ systematically between cohorts (AH: expert R+ vs R−
  {p('1h AH', 'exp')}; {RA}: {p('1h ref', 'exp')}).
- **Ratios to the reference.** The WH/{RA} ratio rose in R+ mice ({m('1h WH/ref', 'RpL')} → {m('1h WH/ref', 'RpE')},
  {p('1h WH/ref', 'R+')}) and not in R− mice ({p('1h WH/ref', 'R-')}). The AH/{RA} ratio did not change
  ({p('1h AH/ref', 'R+')} in R+).

Hence, relative to an unrewarded lick, whisker hits in R+ experts carry an excess of pre-lick activity of the kind
carried by auditory hits. This is visible in the PSTHs: in R+ experts the whisker-hit trace reaches the auditory-hit
level, whereas in R− experts it stays near the {RN} trace. In R+ mice, whisker reaction times also shortened to
auditory-like values (Fig. 1d).

{fig('Fig1_task_data.png', 'Figure 1')}

# Single neurons

**Motivation.** A population-level resemblance could arise from a few strongly modulated neurons or from many weakly
modulated ones. We therefore asked whether individual neurons that distinguish rewarded from unrewarded licks also come
to distinguish whisker hits, with the same sign.

**Method.** For each unit we computed rate-based ROC selectivities (2·AUC − 1) for AH vs {RA} and WH vs {RA} in the
pre-lick window, with significance from label permutations. Reward-lick neurons are those with a significant AH vs {RA}
ROC. Converging neurons are reward-lick neurons also significant for WH vs {RA} with the same sign. Per session we
computed four measures:

- the fraction of units separating WH from {RA};
- the fraction of converging neurons among reward-lick neurons;
- the AH-likeness of reward-lick neurons, c = (|WH − {RA}| − |WH − AH|) / |AH − {RA}|;
- the across-unit correlation of the two selectivities.

At the unit level we compared the relation between the two selectivities across stages, with separate fits per stage
and a session bootstrap. Finally, we mapped the converging neurons on 500 µm slabs of the Allen CCF.

**Results.**

- **Selective units.** The fraction of units separating whisker hits from {RNP} rose in R+ mice from
  {m('2b', 'RpL')} to {m('2b', 'RpE')} ({p('2b', 'R+')}). It did not change in R− mice ({m('2b', 'RmL')} →
  {m('2b', 'RmE')}, {p('2b', 'R-')}); interaction {p('2b', 'int')}.
- **Converging neurons.** Their fraction among reward-lick neurons rose from {m('2c', 'RpL')} to {m('2c', 'RpE')} in
  R+ mice ({p('2c', 'R+')}) and stayed at {m('2c', 'RmL')} → {m('2c', 'RmE')} in R− mice ({p('2c', 'R-')});
  interaction {p('2c', 'int')}, expert R+ vs R− {p('2c', 'exp')}.
- **AH-likeness.** Reward-lick neurons became more auditory-hit-like on whisker hits in R+ mice ({m('2d', 'RpL')} →
  {m('2d', 'RpE')}, {p('2d', 'R+')}; interaction {p('2d', 'int')}), but remained below 0: whisker hits do not reach
  auditory-hit rates in these neurons.
- **Shared code.** The across-unit correlation of the selectivities increased in R+ mice ({m('2e', 'RpL')} →
  {m('2e', 'RpE')}, {p('2e', 'R+')}; interaction {p('2e', 'int')}).
- **Unit level (Fig. 2f).** In R+ mice the correlation went from r = {f2f('R+', 'r_learning')} to
  r = {f2f('R+', 'r_expert')} (slope {f2f('R+', 'slope_learning')} → {f2f('R+', 'slope_expert')}; Δr =
  {f2f('R+', 'dr')}, session bootstrap {P(r2f['R+']['p_boot']) if r2f['R+'] is not None else 'n/a'}). In R− mice it went
  from r = {f2f('R-', 'r_learning')} to r = {f2f('R-', 'r_expert')} (Δr = {f2f('R-', 'dr')},
  {P(r2f['R-']['p_boot']) if r2f['R-'] is not None else 'n/a'}). The unit-level change is therefore modest and should be
  read together with the session-level measures, which weight each session equally.
- **Areas (Fig. 2g).** Across the {n_g} areas recorded in expert sessions of both cohorts, {G2_COH}
  (ANOVA, cohort: {an2('cohort')}); {G2_INT} (cohort × area: {an2('cohort x area')}). The anatomical density
  maps (Fig. 2—supplement c, d) show the same picture on coronal and sagittal slabs.
- **Functional types (Fig. 2—supplement b).** Converging neurons were mostly lick-related: a majority also responded
  to spontaneous licks and to licking versus correct rejections, with modest enrichment for sensory-responsive types.

{fig('Fig2_single_neurons.png', 'Figure 2 |')}

{fig('Fig2S_converging_neurons.png', 'Figure 2—supplement')}

# Population geometry

**Motivation.** Single-neuron counts depend on significance thresholds and ignore weakly selective neurons. A
population measure that uses all units, without fitting a decoder, tests directly whether whisker hits move closer to
auditory hits than to {RNP}.

**Method.** For each session we computed cross-validated squared Euclidean distances between the mean pre-lick
population vectors of WH, AH and {RA} (trial halves, 50 splits; unbiased by noise). The main readout is the distance
difference Δd = d(WH, {RA}) − d(WH, AH), which is positive when whisker hits are closer to auditory hits. To ask whether
whisker hits move *toward* auditory hits or merely *away* from {RNP}, we split the displacement of WH from {RA} into a
component along the {RA} → AH direction and an orthogonal component. We also computed λ, the along-axis component
normalised by the axis length, and λ_LDA, the same projection on a noise-whitened (shrinkage LDA) axis.

**Results.**

- **Distance difference.** At learning, whisker hits were closer to {RNP} than to auditory hits in both cohorts
  (Δd {m('3c', 'RpL', 3)} in R+, {m('3c', 'RmL', 3)} in R−). In R+ mice Δd became positive at expert stage
  ({m('3c', 'RpE', 3)}, {p('3c', 'R+')}). It did not increase in R− mice ({m('3c', 'RmE', 3)}, {p('3c', 'R-')});
  interaction {p('3c', 'int')}, expert R+ vs R− {p('3c', 'exp')}.{RT_SENT}
- **Components.** The change came from an increase of d(WH, {RA}) ({m('3d', 'RpL', 3)} → {m('3d', 'RpE', 3)},
  {p('3d', 'R+')}; interaction {p('3d', 'int')}). d(WH, AH) did not decrease ({p('3e', 'R+')}), and the separation of
  rewarded and unrewarded licks, d(AH, {RA}), was stable ({p('3f', 'R+')}).
- **Toward or away?** The along-axis displacement grew in R+ mice ({m('3i', 'RpL', 3)} → {m('3i', 'RpE', 3)},
  {p('3i', 'R+')}; expert R+ vs R− {p('3i', 'exp')}; interaction {p('3i', 'int')}). So did the orthogonal part
  ({m('3j', 'RpL', 3)} → {m('3j', 'RpE', 3)}, {p('3j', 'R+')}; interaction {p('3j', 'int')}).
- **λ and λ_LDA (Fig. 3—supplement).** Normalised, the along-axis component (λ) rose from {m('3S d', 'RpL')} to
  {m('3S d', 'RpE')} in R+ mice ({p('3S d', 'R+')}). The noise-whitened λ_LDA rose from {m('3S j', 'RpL')} to
  {m('3S j', 'RpE')} ({p('3S j', 'R+')}). Both stayed flat in R− mice, but their interactions were weaker
  ({p('3S d', 'int')} and {p('3S j', 'int')}). λ and λ_LDA agreed closely across sessions (Spearman
  r = {v('3S i', 'r')}).

In R+ mice, whisker hits therefore leave the unrewarded-lick state, and a substantial part of the departure lies in the
direction of auditory hits (λ ≈ {m('3S d', 'RpE', 1)} at expert stage). They end up closer to auditory hits than to
{RNP}, but not closer to auditory hits in absolute distance. The distance difference carries the cohort-specific
effect most clearly; the projection measures confirm its direction.

{fig('Fig3_population_distance.png', 'Figure 3 |')}

{fig('Fig3S_lambda_lambdaLDA.png', 'Figure 3—supplement')}

# Brain areas

**Motivation.** We asked whether the convergence is carried by specific regions, for instance motor and frontal areas
that plan licking, or is distributed across the brain.

**Method.** We recomputed Δd per session × area group (≥ 5 units). An area group was included for a cohort if that
cohort had ≥ 3 sessions at both stages. Interactions and family-wise p values (max-|z| permutation) were computed for
area groups included in both cohorts. An ANOVA (Δd ~ cohort × stage × area) tested whether the cohort-specific learning
effect differed between area groups, with p values from mouse-level permutations of cohort labels.

**Results.** In R+ mice Δd increased in most area groups (Fig. 4a, b). For example, in motor areas it went from
{m('4c Motor areas', 'RpL', 3)} to {m('4c Motor areas', 'RpE', 3)} ({p('4c Motor areas', 'R+')}; interaction
{p('4c Motor areas', 'int')}). R− mice showed no consistent change. Across area groups included in both cohorts:

- the cohort × stage term was {an('cohort x stage')};
- the cohort × stage × area term was {an('cohort x stage x area')}, so the size of the effect did not differ detectably
  between area groups;
- no single area group survived family-wise correction.

The convergence is a brain-wide change of state rather than the signature of one region.

{fig('Fig4_areas.png', 'Figure 4')}

# Decoders

**Motivation.** Distances describe geometry; a linear readout tests whether downstream neurons could tell rewarded
from unrewarded licks, and whether such a readout would classify whisker hits as rewarded licks. Licking and engagement
are autocorrelated in time, so chance levels must respect the temporal structure of the session. Single sessions have
limited numbers of neurons, so we complemented them with pseudo-populations pooled across mice.

**Method.**

- **Single sessions.** Logistic-regression decoders were trained on AH vs {RA} and applied unchanged to whisker hits.
  Chance came from linear shifts of the neural activity relative to the time-ordered labels, and every readout is
  reported above chance.
- **Readouts.** Accuracy measures the reward-lick axis itself. The transfer measures (yes/no or probability, normalised
  between {RA} = 0 and AH = 1) and the probability numerator P(AH | WH) − P(AH | {RA}) measure where whisker hits fall
  on that axis. The numerator has no denominator that can approach zero.
- **Pseudo-populations.** A hierarchical bootstrap (mice → sessions → neurons → trials) builds populations of up to
  thousands of neurons per group, with the same shift-based chance correction.

**Results.**

- **Accuracy.** AH and {RNP} were decoded equally well in all groups ({m('5a', 'RpL')}–{m('5a', 'RmE')} above chance;
  interaction {p('5a', 'int')}). The reward-lick representation itself did not change.
- **Where whisker hits fall.** The probability numerator rose from {m('5d', 'RpL')} to {m('5d', 'RpE')} in R+ mice
  ({p('5d', 'R+')}) and stayed at {m('5d', 'RmL')} → {m('5d', 'RmE')} in R− mice ({p('5d', 'R-')}); interaction
  {p('5d', 'int')}. The yes/no transfer agreed (interaction {p('5b', 'int')}). The probability ratio, limited by its
  denominator, was weaker ({p('5c', 'int')}).
- **Pseudo-populations.** At the largest population size, accuracy did not change (interaction
  {pp('bacc', 'interaction_p_perm')}). Whisker-hit transfer increased in R+ mice (yes/no: Δ = {pp('transfer_bin', 'dRp')},
  {pp('transfer_bin', 'dRp_p_boot')}; probability: Δ = {pp('transfer', 'dRp')}, {pp('transfer', 'dRp_p_boot')};
  numerator: Δ = {pp('num_prob', 'dRp')}, {pp('num_prob', 'dRp_p_boot')}) and not in R− mice (numerator: Δ =
  {pp('num_prob', 'dRm')}, {pp('num_prob', 'dRm_p_boot')}). Interactions: yes/no {pp('transfer_bin', 'interaction_p_perm')},
  probability {pp('transfer', 'interaction_p_perm')}, numerator {pp('num_prob', 'interaction_p_perm')}.

{fig('Fig5_decoders.png', 'Figure 5')}

# Conclusion

When whisker-triggered licks are rewarded, the brain-wide pre-lick state on whisker hits departs from the unrewarded-lick
state and acquires part of the auditory-hit (rewarded-lick) pattern. This happens in single neurons, in population
geometry and in linear readouts, and it is absent when whisker licks are not rewarded. The reward-lick axis is as
decodable before as after learning, so the effect is a re-mapping of whisker-hit activity onto an existing reward-lick
representation rather than a sharpening of that representation.
"""
out = PUB / f"prelick_convergence_{REF}_{POP}.qmd"
out.write_text(txt, encoding="utf-8")
print("wrote", out)
