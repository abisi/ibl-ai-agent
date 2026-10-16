"""Report of ssl-within-day-two-axes (first version 2026-10-16; skills/project-report, report_lib): within-session changes on the
reward-lick coding direction and on the hit - miss axis, day 0 vs expert sessions, whole brain and area groups; reaction-time and jaw
controls. Replaces the within-day part of the former ssl-within-day-remapping report.
Run (haas, after 001-003): cd ~/code/ibl-ai-agent/projects/ssl-within-day-two-axes/exploratory-analyses && PRELICK_REF=sl PRELICK_UNITS=stable \
  PYTHONPATH=$HOME/code/NWB_reader:$HOME/code/unit_spikes_analysis:$HOME/code/ephys_utilities ~/code/unit_spikes_analysis/.venv/bin/python ../report/build_report.py
Render locally: bash projects/ssl-within-day-two-axes/report/render.sh
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

OUT = C.ROOT / "report"
HMPUB = C.HM / "figures" / "publication"
R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = num
num = lambda *a, **k: _num(*a, **k).replace("-", "−")
R.register([
    ("fig-two", "Figure 1"), ("fig-wb", "Figure 2"), ("fig-lam", "Figure 3"), ("fig-mm", "Figure 4"), ("fig-areas", "Figure 5"),
    ("fig-rt", "Figure 6"), ("fig-pass", "Figure 7"),
    ("eq-drift", "Eq. (1)"), ("eq-hm", "Eq. (2)"), ("eq-lam", "Eq. (3)"), ("eq-mm", "Eq. (4)"),
    ("tbl-classes", "Table 1"), ("fig-s-tc", "Figure S1"), ("fig-s-units", "Figure S2"), ("tbl-registry", "Table S1"),
])
SRC = {"003": C.TAB / "003_stats.csv", "001": C.TAB / "001_rl_controls_stats.csv", "002": C.TAB / "002_passive_kinematics_stats.csv",
       "005": C.TAB / "005_stats.csv", "005lv": C.TAB / "005_levels.csv", "006": C.TAB / "006_mixed_model.csv", "007": C.TAB / "007_stats.csv"}
FROZEN = dict(zip(SRC, Report.snapshot([SRC[k] for k in SRC], OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}
W3 = T["003"][T["003"].level == "whole_brain"]
AR = T["003"][T["003"].level != "whole_brain"]
LV = T["005lv"].set_index(["cohort", "test"])
MM = T["006"]


def mm(key, ref, c, term):
    r = MM[(MM.ref == ref) & (MM.cohort == c) & (MM.term == term)].iloc[0]
    return f"{num(key, r.coef, '{:+.2f}')} ± {num(key + '_se', r.se, '{:.2f}')} (p {pv(key + '_p', r.p)})"


def lv(key, c, test):
    r = LV.loc[(c, test)]
    return f"{num(key + '_d0', r.mean_d0, '{:.2f}')} vs {num(key + '_e', r.mean_e, '{:.2f}')} (p {pv(key + '_p', r.p_mwu)})"


def g(df, m, c):
    d = df[(df.measure == m) & (df.cohort == c)]
    assert len(d) == 1, (m, c)
    return d.iloc[0]


def per(key, df, m, c, fmt="{:+.3f}"):
    r = g(df, m, c)
    return (f"day 0 {num(key + '_d0', r.mean_d0, fmt)} (p {pv(key + '_p0', r.p_d0)}), expert {num(key + '_e', r.mean_e, fmt)} "
            f"(p {pv(key + '_pe', r.p_e)}), day 0 vs expert p {pv(key + '_ps', r.p_stage_mwu)}")


def coh(key, df, m):
    r = g(df, m, "R+ vs R-")
    return (f"R+ vs R− day 0 p {pv(key + '_c0', r.p_d0)}, expert p {pv(key + '_ce', r.p_e)}, cohort × stage p "
            f"{pv(key + '_ci', r.p_interaction)}")


LAB = {"rl_WH_raw": "reward-lick CD: WH drift, raw", "rl_WH": "reward-lick CD: WH drift beyond null", "rl_WHSL": "reward-lick CD: WH − SL drift beyond null",
       "hm_W": "hit − miss axis: passive whisker post − pre", "hm_A": "hit − miss axis: passive auditory post − pre", "hm_WA": "hit − miss axis: whisker − auditory"}


def classes_table():
    f = lambda v: "" if not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")
    rows = []
    for m in LAB:
        for c in ("R+", "R-", "R+ vs R-"):
            r = g(W3, m, c)
            rows.append({"measure": LAB[m], "cohort": c.replace("-", "−"),
                         "day 0": "" if c == "R+ vs R-" else f"{r.mean_d0:+.3f} ({int(r.n_d0)})",
                         "expert": "" if c == "R+ vs R-" else f"{r.mean_e:+.3f} ({int(r.n_e)})",
                         "p day 0": f(r.p_d0), "p expert": f(r.p_e),
                         "p day 0 vs expert": f(r.p_interaction if c == "R+ vs R-" else r.p_stage_mwu), "class": r.cls})
    return pd.DataFrame(rows)


def area_list(m, c, cls):
    d = AR[(AR.measure == m) & (AR.cohort == c) & (AR.cls == cls)]
    return ", ".join(f"{a.replace('Somatosensory-', 'Ss-').replace(' areas', '')}{' (uncorrected)' if fl == 'uncorrected' else ''}"
                     for a, fl in zip(d.level, d.flag)) or "none"


def build():
    SHOWN = AR[AR.cohort.isin(["R+", "R-"]) & (AR.measure != "rl_WH_raw")]
    nls = int((SHOWN.cls == "learning-specific").sum()); ncells = int(SHOWN.cls.ne("n/a").sum())
    CC = AR[(AR.cohort == "R+ vs R-") & (AR.measure != "rl_WH_raw") & (AR.cls == "learning-specific")]
    cc_txt = "; ".join(f"{LAB[m]} in {a.replace('Somatosensory-', 'Ss-').replace(' areas', '')}" for m, a in zip(CC.measure, CC.level)) or "none"
    md = rf"""---
title: "Within-session changes on two coding directions: what is specific to the learning day"
subtitle: "ssl-within-day-two-axes · day 0 vs expert sessions · learners · stable units · Kilosort 4"
date: "{date.today().isoformat()}"
---

## Abstract

On the learning day, activity changes along two population directions: the reward-lick coding direction, which separates the 100 ms
before rewarded auditory-cued licks from spontaneous licks, and the hit − miss axis, which separates licked from unlicked whisker trials
5–35 ms after the stimulus. Expert sessions, run with the same task once the contingency is learned, show which of these within-session
changes come with running the task and which are specific to the learning day. In R+ mice, whisker-hit pre-lick activity drifts toward
the reward-lick pattern over the session, and expert R+ sessions show the same drift: measured as the relative position of whisker hits
between spontaneous licks and auditory hits (λ), the change within the session is the same at both stages, whereas the level differs:
day-0 sessions start far below the expert level and end close to it, and expert sessions start higher. The learning-related change is
therefore a shift of the level from which each session starts, not a different within-session drift. In R− mice, the passive whisker response moves away from the hit side of the hit − miss
axis on day 0; relative to the auditory response, this cohort difference is present on day 0 and absent in expert sessions, the only
change classified as learning-specific at the whole-brain level. Reaction-time matching and jaw kinematics do not explain the reward-lick
drift, and movement state around passive trials changes equally in both cohorts. Per area, most changes cannot be separated from the expert
sessions, which are few.

**Key results** (whole brain; sessions are the unit; Wilcoxon vs 0 within group, Mann–Whitney between stages and cohorts, mouse-level
permutation for cohort × stage).

1. Reward-lick CD, R+ whisker-hit drift relative to spontaneous licks beyond the null: {per('k_whsl', W3, 'rl_WHSL', 'R+')}: present at both stages ({ref('fig-wb')}c).
2. Reward-lick CD, R+ whisker-hit drift beyond the null: {per('k_wh', W3, 'rl_WH', 'R+')}: day 0 only, not distinguishable from expert ({ref('fig-wb')}b).
3. Hit − miss axis, R− passive whisker change: {per('k_hmw', W3, 'hm_W', 'R-')} ({ref('fig-wb')}d).
4. Hit − miss axis, whisker − auditory, cohort contrast: {coh('k_hmwa', W3, 'hm_WA')}: learning-specific ({ref('fig-wb')}f).
5. Relative position λ (R+): day-0 start vs expert level {lv('k_lvs', 'R+', 'day-0 start vs expert mean')}; day-0 end vs expert level {lv('k_lve', 'R+', 'day-0 end vs expert mean')}; per-event mixed model, expert − day 0 start offset {mm('k_mmo', 'SL', 'R+', 'W:E')}, relative slope {mm('k_mms', 'SL', 'R+', 'tau:W:E')} ({ref('fig-lam')}, {ref('fig-mm')}).
6. Controls: the R+ drift holds with reaction times matched ({per('k_rtm', T['001'], 'd_cd_rtm', 'R+')}) and with jaw opening as a covariate; whisker-hit jaw opening does not drift ({ref('fig-rt')}).

## Introduction

Two coding directions were studied separately: the reward-lick CD in ssl-prelick-convergence and the hit − miss axis in
ssl-hitmiss-axis-passive. On day 0 each showed a cohort-specific within-session change. A change within one session can arise from
learning, but also from anything that accompanies a session: satiety, fatigue, adaptation to repeated stimuli, slow drifts of the
recording, changes in reaction time or licking movements. Expert sessions share all of these and differ in that the contingency is
already learned. Comparing the within-session change on day 0 with the same measure in expert sessions separates changes specific to
learning from changes that come with running the task ({ref('fig-two')}). Every claim, with its figure, statistic and status, is listed in
{ref('tbl-registry')}.

## Methods

{R.methods('task', 'learners', 'trials', 'units', 'lick_alignment', 'shift_null')}
*Reward-lick CD (ssl-prelick-convergence, within_day 026).* Per session, units are z-scored over auditory-hit (AH) and spontaneous-lick
(SL) events and the CD is the unit-norm AH − SL mean difference, estimated by 5-fold cross-validation; every whisker-hit (WH) event is
projected on the folds' axes (per neuron, / √N). The drift is the excess of the OLS slope over normalised session time $\tau$:
{eq('eq-drift', r"\Delta\beta_{WH} = \beta_{WH} - \frac{1}{K}\sum_{k=1}^{K}\beta_{WH}^{(k)}, \qquad p_{WH,i} = \alpha + \beta_{WH}\,\tau_i + \varepsilon_i")}
where $p_{{WH,i}}$ is the projection of whisker hit $i$ and $\beta^{{(k)}}$ the slope on the CD rebuilt from AH / SL labels shifted against
the time-ordered events ($K = 50$). The same is computed for held-out spontaneous licks, and the WH − SL contrast is the drift of whisker
hits relative to the unrewarded reference measured at the same times. Sessions need ≥ 4 WH, 8 AH and 8 SL events.

*Hit − miss axis (ssl-hitmiss-axis-passive, 135).* Per session, the unit-norm mean difference of active whisker hits and misses
5–35 ms after the stimulus (harmonised scaling; ≥ 4 hits and misses, so that expert sessions can enter); passive whisker (W) and auditory
(A) responses before (pre) and after (post) the active block are projected per neuron. The change is the excess over shifted-label axes:
{eq('eq-hm', r"\Delta\pi_s = \left(\pi^{\mathrm{post}}_s - \pi^{\mathrm{pre}}_s\right) - \frac{1}{K}\sum_{k}\left(\pi^{\mathrm{post},(k)}_s - \pi^{\mathrm{pre},(k)}_s\right), \quad s \in \{W, A, W - A\}")}
with $\pi$ the projection and $k$ the shifted axes.

*Relative position λ and auditory hits (005).* The within_day 002 scores place every event on its session's fixed CD, scaled so that held-out
spontaneous licks average 0 and held-out auditory hits average 1 (sessions with held-out AH-vs-SL $d' \ge 0.3$). For each class, a line is
fit to the scores against session time, $f_c(\tau) = a_c + b_c \tau$, and the relative position of whisker hits is
{eq('eq-lam', r"\lambda(\tau) = \frac{f_{WH}(\tau) - f_{SL}(\tau)}{f_{AH}(\tau) - f_{SL}(\tau)}")}
evaluated at the start ($\tau = 0$) and end ($\tau = 1$) of the session ($\Delta\lambda$ = end − start; fitted AH − SL separation ≥ 0.3 at both
ends). λ = 0 means that whisker hits sit with spontaneous licks, λ = 1 with auditory hits. Because numerator and denominator are differences
between classes measured at the same time, a drift shared by all pre-lick activity cancels, and so does a change of the AH − SL separation.
Day-0 start and end levels are compared with the expert session-mean λ (Mann–Whitney). The slopes $b_c$ of each class and the contrasts
$b_{{WH}} - b_{{SL}}$ (whisker hits relative to spontaneous licks), $b_{{WH}} - b_{{AH}}$ (relative to auditory hits) and $b_{{AH}} - b_{{SL}}$ (the
auditory-hit drift, the non-learned cue) are compared between stages. These per-event scores have no shift null; the class contrasts take
its place by cancelling shared drift.

*Mixed model (006).* All events of two classes (WH and SL, or WH and AH) per cohort enter
{eq('eq-mm', r"y_{i} = \beta_0 + \beta_1 \tau_i + \beta_2 W_i + \beta_3 E_i + \beta_4 \tau_i W_i + \beta_5 \tau_i E_i + \beta_6 W_i E_i + \beta_7 \tau_i W_i E_i + u_{0s} + u_{1s}\tau_i + \varepsilon_i")}
where $W_i = 1$ for whisker hits, $E_i = 1$ for expert sessions and $(u_{{0s}}, u_{{1s}})$ are random intercept and slope of session $s$
(REML). $\beta_4$ is the day-0 slope of whisker hits relative to the reference class, $\beta_7$ the expert − day 0 difference of that slope,
$\beta_2$ the day-0 start offset and $\beta_6$ its expert − day 0 difference; the cohort × stage terms come from the same model with a cohort
factor. Time is also expressed per trial, per whisker hit (also over the first 30 whisker hits only), per minute and per reward collected
(007).

*Classification.* For each measure and cohort: day 0 vs 0 and expert vs 0 (Wilcoxon), day 0 vs expert (Mann–Whitney; Welch in the tables).
**Learning-specific**: a day-0 change, none at expert, day 0 ≠ expert. **Stronger with learning**: changes at both stages, larger on day 0,
day 0 ≠ expert. **Both stages**: changes at both stages, day 0 = expert. **Day 0 only (stage n.s.)**: a day-0 change that cannot be
separated from the expert value (inconclusive). For the cohort contrast, the same rule uses R+ vs R− at each stage and the cohort × stage
interaction. Absence of an expert effect is not tested for equivalence; with 9–14 expert sessions, "learning-specific" requires the
stage contrast to be significant, not only the expert test to fail.

{R.methods('statistics', 'areas')}
Per area, the day-0 p values are corrected with the Benjamini–Hochberg procedure across areas within each measure and cohort; cells whose
day-0 effect does not survive are marked.

*Controls.* Reward-lick axis (001): per-event reaction time (corrected first lick − stimulus) and pre-lick jaw opening (DLC, mm, mean over
[−100, 0) ms) joined to the CD scores of within_day 002; the whisker-hit drift with reaction time or jaw opening as a covariate, late −
early difference with reaction times matched (20-ms bins, 50 draws), and the drift of reaction time, of jaw opening and of the whisker-hit vs
auditory-hit jaw similarity (signed ROC). Hit − miss axis (002): pre-stimulus whisking and jaw opening ([−200, 0) ms) and evoked whisking
([0, 100) ms) and jaw opening ([100, 300) ms) of passive trials, post − pre.

## Results

### Two within-session changes on day 0

On day 0 ({ref('fig-two')}), R+ whisker hits drift toward the auditory-hit side of the reward-lick CD relative to spontaneous licks, beyond
the shift null, while R− whisker hits do not; and the passive whisker response of R− mice moves away from the hit side of the hit − miss axis,
relative to the auditory response, more than in R+.

### Reward-lick CD: the R+ drift is present in expert sessions too

The raw whisker-hit drift was present in R+ at both stages ({per('wh_raw', W3, 'rl_WH_raw', 'R+')}) and absent in R− ({per('wh_raw_m', W3, 'rl_WH_raw', 'R-')})
({ref('fig-wb')}a). Beyond the shift null, the R+ drift was significant on day 0 but not at expert, and the two stages did not differ
({per('wh', W3, 'rl_WH', 'R+')}; {ref('fig-wb')}b). Relative to spontaneous licks, the R+ drift beyond the null was present at both stages with the same
size ({per('whsl', W3, 'rl_WHSL', 'R+')}; R− {per('whsl_m', W3, 'rl_WHSL', 'R-')}; {coh('whsl_c', W3, 'rl_WHSL')}; {ref('fig-wb')}c). The time course per fifth of the session ({ref('fig-s-tc')}) shows why the whisker-hit drift alone is "day 0 only": the real
projection rises as much in expert as in day-0 R+ sessions, but the CDs rebuilt from shifted labels also rise more in expert sessions,
so the excess is smaller and more variable there. The R+ within-session drift on the
reward-lick CD therefore comes with running the task in R+ mice, whether or not the contingency is still being learned.

### Reward-lick CD: learning shifts the starting level, not the within-session drift

Placing whisker hits between spontaneous licks (λ = 0) and auditory hits (λ = 1) at each time of the session ({ref('eq-lam')}, {ref('fig-lam')}a–c)
separates the drift from the level. In R+, λ rose within the session at both stages by the same amount ({per('dlam', T['005'], 'dlam', 'R+', '{:+.2f}')}).
The levels differed: day-0 sessions started far below the expert level (day-0 start vs expert mean λ {lv('lvs', 'R+', 'day-0 start vs expert mean')})
and ended close to it ({lv('lve', 'R+', 'day-0 end vs expert mean')}), and expert sessions started higher than day-0 sessions
({per('lams', T['005'], 'lam_start', 'R+', '{:.2f}')}). In R−, λ stayed near 0 and did not change significantly within the session at either stage
({per('dlam_m', T['005'], 'dlam', 'R-', '{:+.2f}')}).

Auditory hits, the cue rewarded in both cohorts throughout, did not drift along the CD within R+ sessions ({per('bah', T['005'], 'b_AH', 'R+', '{:+.2f}')}),
so the R+ whisker-hit drift is specific to whisker hits: relative to auditory hits measured at the same times it was present at both stages
({per('bwa', T['005'], 'b_WH-AH', 'R+', '{:+.2f}')}). In R−, auditory hits rose along the CD over the session at both stages, as did spontaneous licks (auditory − spontaneous slope n.s. on day 0)
({per('bah_m', T['005'], 'b_AH', 'R-', '{:+.2f}')}; {ref('fig-lam')}d–g).

The per-event mixed model ({ref('eq-mm')}, {ref('fig-mm')}) gave the same picture with every event. Relative to spontaneous licks, the R+ day-0 whisker-hit
slope was {mm('mm_s', 'SL', 'R+', 'tau:W')}, its expert − day 0 difference {mm('mm_se', 'SL', 'R+', 'tau:W:E')}, and the start offset was higher in
expert sessions by {mm('mm_oe', 'SL', 'R+', 'W:E')}; in R− the start offset did not change ({mm('mm_oem', 'SL', 'R-', 'W:E')}). The cohort × stage
interaction was significant for the start offset ({mm('mm_ci_o', 'SL', 'R+ vs R-', 'W:E:R')}) and not for the slope ({mm('mm_ci_s', 'SL', 'R+ vs R-', 'tau:W:E:R')}).
Measured per whisker hit, per reward or over the first 30 whisker hits, the R+ day-0 drift was larger than the expert drift but not
significantly so ({per('wh10', T['007'], 's_wh', 'R+')}; {ref('fig-s-units')}).

### Hit − miss axis: the R− change relative to the auditory response is specific to day 0

The passive whisker response moved away from the hit side in R− on day 0 ({per('hmw_m', W3, 'hm_W', 'R-')}) and slightly in R+
({per('hmw_p', W3, 'hm_W', 'R+')}) ({ref('fig-wb')}d). The auditory response did not change in either cohort ({per('hma_m', W3, 'hm_A', 'R-')} in R−;
{ref('fig-wb')}e). The whisker − auditory change was negative in R− on day 0 ({per('hmwa_m', W3, 'hm_WA', 'R-')}) and absent in R+; the cohort
contrast was present on day 0 and not at expert, with a significant cohort × stage interaction ({coh('hmwa_c', W3, 'hm_WA')}; {ref('fig-wb')}f). This is the
only whole-brain change classified as learning-specific ({ref('tbl-classes')}).

### Area groups

Per area ({ref('fig-areas')}), {nls} of the {ncells} area × measure × cohort cells shown were classified as learning-specific: the R− passive whisker
change on the hit − miss axis in {area_list('hm_W', 'R-', 'learning-specific')}, the R− whisker − auditory change in
{area_list('hm_WA', 'R-', 'learning-specific')}, and the R− reward-lick whisker-hit drift in {area_list('rl_WH', 'R-', 'learning-specific')}. The R+ reward-lick
drift relative to spontaneous licks was present at both stages in {area_list('rl_WHSL', 'R+', 'both stages')}, and on day 0 only, without a
stage difference, in several others ({area_list('rl_WHSL', 'R+', 'day 0 only (stage n.s.)')}). Most cells are "day 0 only (stage n.s.)": the expert sessions per area are too few to separate the
stages. For the cohort contrast (R+ vs R−, not shown in the figure), the cohort × stage classification was learning-specific for {cc_txt}.

### Controls

*Reaction time.* Whisker reaction times shortened over the R+ session at both stages ({per('rt', T['001'], 'b_rt', 'R+')}). The R+ drift on the
reward-lick CD remained with reaction time as a covariate ({per('cdrt', T['001'], 'b_cd_rt', 'R+')}) and when early and late whisker hits were
matched on reaction time ({per('rtm', T['001'], 'd_cd_rtm', 'R+')}) ({ref('fig-rt')}a–d, f).

*Jaw kinematics.* Pre-lick jaw opening on whisker hits did not drift over the session (R+ {per('jaw', T['001'], 'b_jaw', 'R+')}), the drift
remained with jaw opening as a covariate ({per('cdjaw', T['001'], 'b_cd_jaw', 'R+')}), and the similarity of whisker-hit and auditory-hit jaw
opening did not change from early to late in R+ ({per('sjaw', T['001'], 'd_s_jaw', 'R+')}) ({ref('fig-rt')}e, g, h). The R+ drift is therefore not
a change of reaction time or of the licking movement.

*Movement state around passive trials.* The hit − miss window precedes any movement, so movement can only act through the state at stimulus
time. Pre-stimulus whisking and jaw opening did not change from passive pre to post in either cohort; whisker-evoked whisking decreased in both
cohorts on day 0 by the same amount (R+ {per('wev_p', T['002'], 'W d_whisk_ev', 'R+', '{:+.1f}')}; R− {per('wev_m', T['002'], 'W d_whisk_ev', 'R-', '{:+.1f}')};
{coh('wev_c', T['002'], 'W d_whisk_ev')}) ({ref('fig-pass')}). Movement state does not explain the R− change on the hit − miss axis.

## Discussion

Expert sessions change the interpretation of the day-0 drift on the reward-lick CD: R+ whisker hits drift toward the rewarded-lick pattern
within every R+ session, and the day-0 drift is not larger than the expert drift. What learning changes is the level: expert sessions start
where day-0 sessions end. The day-0 session therefore contains the drift common to every R+ session plus a level that is still low at the
start; the within-session ramp may reflect a session-level process (warm-up, rising reward expectation) that is re-expressed every day on
top of a baseline that learning raises. This within-session drift is thus a property of performing
the rewarded whisker task (for example the build-up of reward expectation or of task engagement within a session), and the learning-related
change of the reward-lick CD is the across-days shift reported in ssl-prelick-convergence. On the hit − miss axis, by contrast, the R− change
relative to the auditory response is present on day 0 and not in expert sessions: when R− mice learn that the whisker stimulus is not rewarded,
their passive whisker response turns away from the direction that predicts licking, and this does not recur once the contingency is known.

## Caveats and limitations

- Expert sessions are few (9–14 per cohort; per area often 3–5), so "day 0 only (stage n.s.)" is common and absence of an expert effect is not
  equivalence.
- The learner definition is under review; day-0 sessions are learners, expert sessions all mice.
- The two axes come from different session sets (event minima differ); they are compared through classes, not session by session.
- Area-level learning-specific cells are uncorrected for the number of measures; one HPF cell does not survive correction across areas.

## Open questions and next steps

- More expert sessions (or several per mouse) to separate day 0 from expert per area.
- Within-session time courses of both axes in the same sessions (joint model).

## Figures

{figure('fig-two', HMPUB / '165_two_axes_compact_angle.png', '''The two axes within day 0 (ssl-hitmiss-axis-passive 165, learners). Top row, hit − miss
axis: projections over time, angle of the passive whisker response to the axis pre / post, change of angle and of projection beyond the shift null
(whisker yellow, auditory blue), per-area change. Bottom row, reward-lick CD: projections around the lick, example sessions, projections per fifth of
the session on the real (solid) and shifted (grey) CDs, slope beyond the null for whisker hits and spontaneous licks, per-area WH − SL slope.''',
                 "ssl-hitmiss-axis-passive 165_two_axes_figure.py")}
{figure('fig-wb', C.FIG / '003_whole_brain.png', rf'''Within-session changes, day 0 (open) vs expert (filled), whole brain (003). **a**, Raw whisker-hit
drift on the reward-lick CD; **b**, the same beyond the shift null ({ref('eq-drift')}); **c**, whisker-hit minus spontaneous-lick drift beyond the null;
**d–f**, passive whisker, auditory and whisker − auditory change on the hit − miss axis beyond the null ({ref('eq-hm')}). Dots: sessions; mean ±
s.e.m.; numbers: sessions. Above each panel per cohort: class, Wilcoxon vs 0 at each stage and Mann–Whitney day 0 vs expert; below: R+ vs R− per
stage and cohort × stage (mouse-level permutation, 10,000).''', "003_classification.py")}
{figure('fig-lam', C.FIG / '005_lambda_auditory.png', rf'''Relative position of whisker hits and auditory-hit drift on the reward-lick CD (005). **a**,
Mean CD score per third of the session for spontaneous licks (black), whisker hits (yellow) and auditory hits (blue), per cohort and stage (score:
held-out SL = 0, AH = 1; mean ± s.e.m. over sessions). **b**, λ ({ref('eq-lam')}) per third, day 0 (dashed, open) vs expert (solid, filled).
**c**, λ at the start and end of the session from the linear fits; above: day-0 start and end vs the expert session-mean λ (Mann–Whitney).
**d**, Auditory-hit slope; **e**, whisker-hit − spontaneous-lick slope; **f**, whisker-hit − auditory-hit slope; **g**, Δλ (layout as Figure 2).''',
                 "005_lambda_auditory.py")}
{figure('fig-mm', C.FIG / '006_mixed_model.png', rf'''Per-event mixed model ({ref('eq-mm')}; 006): coefficients ± 95 % CI per cohort for whisker hits relative to
spontaneous licks (**a**) and to auditory hits (**b**): day-0 relative slope, its expert − day 0 difference, day-0 start offset and its expert −
day 0 difference (filled: p < 0.05); titles: cohort × stage of the relative slope.''', "006_mixed_model.py")}
{figure('fig-areas', C.FIG / '003_areas.png', '''Class of the within-session change per area group (rows, allen_utils order) and cohort (columns) for the
reward-lick CD and the hit − miss axis (003). Numbers: day-0 mean change; colour: class; †: day-0 effect not significant after Benjamini–Hochberg
correction across areas; grey: fewer than 3 sessions at a stage.''', "003_classification.py")}
{figure('fig-rt', C.FIG / '001_rl_controls.png', '''Reaction-time and jaw controls of the whisker-hit drift on the reward-lick CD (001; CD score: held-out SL =
0, AH = 1). **a**, Drift (slope over the session); **b**, with reaction time as a covariate; **c**, late − early difference; **d**, late − early with
reaction times matched; **e**, drift with jaw opening as a covariate; **f**, reaction-time drift; **g**, jaw-opening drift of whisker hits; **h**, change of
the whisker-hit vs auditory-hit jaw similarity (signed ROC, late − early). Layout as Figure 2.''', "001_reward_lick_controls.py")}
{figure('fig-pass', C.FIG / '002_passive_kinematics.png', '''Movement state around passive trials, post − pre (002): pre-stimulus whisking and jaw opening,
evoked whisking and jaw opening, for passive whisker (top) and auditory (bottom) trials. Layout as Figure 2.''', "002_passive_kinematics.py")}

{figure('fig-s-tc', C.FIG / '004_rl_timecourse_stages.png', '''Reward-lick CD across the session, day 0 (dashed, open) vs expert (solid, filled) (004): projections of whisker hits, spontaneous licks and their difference per fifth of the session (per neuron, relative to the first fifth; mean ± s.e.m. over sessions; n in legend), on the real CD (colour) and on CDs rebuilt from shifted labels (grey, mean over 50 shifts). Right: the quantity tested (slope over normalised session time, real − shifted, per neuron), with Wilcoxon vs 0 per stage and Mann–Whitney day 0 vs expert.''', "004_rl_timecourse_stages.py", kind="supplementary")}
{figure('fig-s-units', C.FIG / '007_time_units.png', '''Whisker-hit drift relative to spontaneous licks per session (τ), per 100 events, per 10 whisker hits, per 10
whisker hits over the first 30, per 10 minutes and per 10 rewards (007; score units; layout as Figure 2).''', "007_time_units.py", kind="supplementary")}
{table('tbl-classes', classes_table(), 'Whole-brain classification of the within-session changes (mean change and sessions per stage; p: Wilcoxon vs 0 at each stage, Mann–Whitney day 0 vs expert; for R+ vs R−: Mann–Whitney per stage and cohort × stage permutation).')}
"""
    claims()
    md += "\n## Appendix\n\n" + R.registry_table("tbl-registry") + APPENDIX
    return md


def claims():
    r = lambda m, c: g(W3, m, c)
    claim("C1", "R+ whisker-hit drift on the reward-lick CD relative to SL is present at day 0 and expert (not learning-specific)", "Fig. 2c",
          f"D0 p = {r('rl_WHSL', 'R+').p_d0:.4f}; E p = {r('rl_WHSL', 'R+').p_e:.3f}; D0 vs E p = {r('rl_WHSL', 'R+').p_stage_mwu:.2f}")
    claim("C2", "R+ whisker-hit drift beyond the null: day 0 only, not separable from expert", "Fig. 2b",
          f"D0 p = {r('rl_WH', 'R+').p_d0:.3f}; E p = {r('rl_WH', 'R+').p_e:.2f}; D0 vs E p = {r('rl_WH', 'R+').p_stage_mwu:.2f}", status="fragile")
    claim("C3", "R− passive whisker change on the hit − miss axis on day 0", "Fig. 2d", f"D0 p = {r('hm_W', 'R-').p_d0:.1e}; D0 vs E p = {r('hm_W', 'R-').p_stage_mwu:.2f}")
    claim("C4", "Whisker − auditory cohort contrast on the hit − miss axis is learning-specific", "Fig. 2f, Table 1",
          f"cohort D0 p = {r('hm_WA', 'R+ vs R-').p_d0:.3f}; E p = {r('hm_WA', 'R+ vs R-').p_e:.2f}; cohort × stage p = {r('hm_WA', 'R+ vs R-').p_interaction:.3f}",
          status="fragile", note="few expert sessions; single test")
    claim("C2b", "R+ λ: same within-session change at both stages; day-0 start below the expert level, end close to it", "Fig. 3",
          f"Δλ D0 vs E p = {g(T['005'], 'dlam', 'R+').p_stage_mwu:.2f}; start vs expert p = {LV.loc[('R+', 'day-0 start vs expert mean')].p_mwu:.1e}")
    claim("C2c", "Learning shifts the start offset of WH relative to SL (mixed model), not the slope", "Fig. 4",
          f"W:E p = {MM[(MM.ref == 'SL') & (MM.cohort == 'R+') & (MM.term == 'W:E')].p.iloc[0]:.1e}; tau:W:E p = {MM[(MM.ref == 'SL') & (MM.cohort == 'R+') & (MM.term == 'tau:W:E')].p.iloc[0]:.2f}")
    claim("C2d", "Auditory hits do not drift along the CD within R+ sessions", "Fig. 3d", f"p = {g(T['005'], 'b_AH', 'R+').p_d0:.2f}")
    claim("C5", "R+ drift not explained by reaction time (matched, covariate)", "Fig. 4", f"RT-matched D0 p = {g(T['001'], 'd_cd_rtm', 'R+').p_d0:.1e}")
    claim("C6", "R+ drift not explained by jaw kinematics", "Fig. 4", f"jaw drift D0 p = {g(T['001'], 'b_jaw', 'R+').p_d0:.2f}")
    claim("C7", "Movement state around passive trials changes equally in both cohorts", "Fig. 5",
          f"evoked whisking cohort p = {g(T['002'], 'W d_whisk_ev', 'R+ vs R-').p_d0:.2f}")
    claim("C8", "Area-level learning-specific cells (R− hit − miss in Ss-whisker)", "Fig. 3", "uncorrected over measures", status="fragile")


APPENDIX = """
### Figure versions and provenance

New figures: 001_rl_controls, 002_passive_kinematics, 003_whole_brain, 003_areas (this project). Figure 1 is 165_two_axes_compact_angle of
ssl-hitmiss-axis-passive. Inputs: ssl-prelick-convergence within_day/sl_stable (002 trial scores, 026 cd_shift_null), ssl-hitmiss-axis-passive
(135 min2 harmonised tables). The within-day figures of the former ssl-within-day-remapping report (within_day 001–025) are superseded.
Source files, hashes and dates: figures.json.

### Result folders

combined_results_ks4/ssl-within-day-two-axes/: tables (001–003), figures, report. Statistics read by this report are frozen in report/_snapshots/.
"""


def summary():
    r = lambda m, c: g(W3, m, c)
    return f"""# ssl-within-day-two-axes: summary, {date.today().isoformat()}

Question: which within-session changes on the reward-lick CD and the hit − miss axis are specific to the learning day, compared with the same
measures in expert sessions?

1. Reward-lick CD: the R+ whisker-hit drift toward the rewarded-lick pattern (relative to spontaneous licks, beyond the null) is present in expert sessions with the same size (day 0 vs expert p = {r('rl_WHSL', 'R+').p_stage_mwu:.2f}): not specific to learning.
1b. What learning changes on this axis is the level: λ (whisker hits between spontaneous licks = 0 and auditory hits = 1) rises within the session by the same amount at both stages, but day-0 sessions start far below the expert level and end close to it; mixed model: expert − day 0 start offset p < 0.001, slope n.s.
2. Hit − miss axis: the R− passive whisker change relative to auditory separates the cohorts on day 0 but not at expert (cohort × stage p = {r('hm_WA', 'R+ vs R-').p_interaction:.3f}): the only learning-specific whole-brain change.
3. Controls: reaction-time matching and jaw kinematics do not explain the R+ drift; movement state around passive trials changes equally in both cohorts.
4. Areas: mostly not separable from expert (few expert sessions per area).

![Whole-brain classification](figures/fig-wb.png)

Full report: report.pdf / report.html; claims registry: registry.csv.
"""


def main():
    md = build()
    R.write(OUT, md, summary=summary())


if __name__ == "__main__":
    main()
