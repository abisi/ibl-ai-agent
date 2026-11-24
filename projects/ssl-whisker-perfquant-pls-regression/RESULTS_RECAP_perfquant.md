# Results recap: decoding behavioral learning curves from neural activity ("perfquant")

Chronological recap of the full arc (`exploratory-analyses/049`-`065`). The
method itself is documented separately in `METHOD_perfquant_decoding.md`;
this file is the results record. **Working method: PLS+1SE, single window,
single-session.** dPCA and CEBRA were evaluated and NOT adopted (see §6).

## 1. Question

Can session-level behavioral learning trajectories (`whisker_curve`,
`falsealarm_curve`, `performance_curve` — smoothed, per-trial trajectories)
be decoded/regressed from single-session neural population activity?

## 2. Headline answer: yes, robustly, at the group level

Full 88-session pool, PLS+1SE, sensory window (5-50ms post-stimulus),
1000-shuffle null (each shuffle scored against its own shifted target):

- **All 9 target x metric (R2/Pearson/Spearman) combinations reject the
  null at the group level**, Wilcoxon p between 1.4e-9 and 8.1e-14.
- Above-null margins: whisker_curve ~0.14-0.17, falsealarm_curve
  ~0.11-0.13, performance_curve ~0.10-0.12 (R2/Pearson/Spearman units).
- Train scores ~0.75-0.88 vs. test ~0.24-0.57 (expected gap given p >> n,
  ~600-2400 neurons vs ~100-300 trials per session).
- **Individual-session significance is much sparser than the group
  effect**: on a dedicated 1000-shuffle-per-session test on 12 sessions
  spanning best-to-worst, only 5/36 (R2), 5/36 (Pearson), 8/36 (Spearman)
  session x target pairs reached p<0.05 alone. This is expected, not a
  contradiction -- it is exactly the situation a group-level test is for.

## 3. Feature window comparison: sensory vs. baseline vs. baseline-corrected

Three conditions run full-pool, same method, same rigor:

| Window | test Pearson (mean) | null Pearson (mean) | **above-null** |
|---|---|---|---|
| Sensory (5-50ms post-stim) | 0.46-0.57 | 0.35-0.42 | **0.108-0.151 (best)** |
| Baseline (wide, -2000 to -10ms) | 0.64-0.74 (highest raw) | 0.57-0.63 (also highest) | 0.073-0.117 |
| Corrected (sensory minus short baseline) | 0.21-0.27 (lowest raw) | 0.13-0.16 (also lowest) | 0.087-0.107 |

**Raw score ranking (baseline > sensory > corrected) is highly significant
but misleading** -- paired Wilcoxon across all 88 sessions confirms all
27/27 condition-pair comparisons significant (p < 4e-10), but baseline's
huge raw advantage is mostly a null-inflation artifact: it is the most
slowly-varying, most autocorrelated window (2 full seconds), so its
shift-null is inflated by the same amount as its real score. **Once
corrected for chance (above-null), sensory is the best or tied-best
window on all three targets.** This is why sensory (not baseline) is the
canonical feature window.

## 4. Cohort (R+/R-) comparison: no effect, anywhere

Tested at lag=0 across all three windows, all three targets, all three
metrics (27 comparisons, Mann-Whitney + Welch, project convention):
**only 1/27 significant**, and that one (performance_curve, corrected
window, R2 only, Welch p=0.042) is borderline, not replicated in
Pearson/Spearman, and R2 is the metric already known to be outlier-prone.
Reward contingency does not detectably shape decodability of these curves
from population activity, under any feature window tested.

## 5. Diagnostics that shaped the method (read before reusing)

- **+/-20 trial lag sweep** (`057`): broad, not sharply peaked at lag=0,
  for both cohorts and both windows tested. The target curves are
  autocorrelated well beyond a 20-trial window, so a narrow lag sweep
  cannot by itself separate genuine same-trial coupling from shared slow
  drift (arousal, satiety, electrode drift).
- **Null-band-on-real-axis pitfall** (`060`): remapping shift-null
  predictions onto the real trial axis and scoring them against the real
  Y inflates the apparent null almost to real-model levels in several
  sessions -- NOT evidence the effect is fake, but an artifact of the
  same long-autocorrelation confound. This construction is kept ONLY as a
  qualitative "does the model track fine structure" visual (`060`, `065`);
  every actual significance number in this project uses the valid
  construction (§ below).
- **The valid null**: score each shuffle's prediction against ITS OWN
  shifted target, never the real one. This is the only null used for any
  p-value or Wilcoxon test reported here.
- **`SCALE_FLOOR` numerical bug** (found in `052`): a neuron column can be
  near-constant within one CV fold's training subset purely by chance,
  causing `StandardScaler` to blow up (`1/std` -> R2 ~ -1e28). Fixed with
  a scale-floored scaler (`SCALE_FLOOR=1e-3`); caught only because R2 is
  scale-sensitive (Pearson/Spearman were unaffected) -- the reason all
  three metrics are always reported together.

## 6. Method comparison (all on sensory window, single-session)

**Linear methods** (`055`, `056`): Ridge / Lasso / ElasticNet / PLS
compared on 5 example sessions with a train-vs-test gap analysis. No
single method dominated raw test score; PLS with 1-SE-rule component
selection gave the cleanest, most reproducible train/test gap reduction
(gap: Ridge ~0.39-0.53 -> PLS+1SE ~0.34-0.45), confirmed at full-pool
scale later. Stability-selection Lasso had an even smaller gap but
consistently lower test scores. **PLS+1SE adopted as the working method.**

**PLS-Forest** (`056`, bagged PLS across bootstrap trial resamples + random
neuron subspaces): no improvement over single PLS+1SE on any axis (test
score, train/test gap, or null separation) despite ~30x the compute.
Dropped.

**dPCA** (`064`, adapted to continuous single-trial regression via
tertile-discretized conditions + 4 coarse time bins -- NOT its native
use case): mean test Pearson 0.31-0.51 across targets, clearly below
PLS+1SE (0.49-0.58), but with a MUCH smaller train/test gap (train
0.48-0.62) and a competitive above-null margin on whisker_curve and
performance_curve (0.155 / 0.103 vs PLS's 0.137 / 0.078) -- it essentially
fails on falsealarm_curve (above-null ~0.014, indistinguishable from
chance). A genuinely lower-capacity, harder-to-overfit alternative, but
not a raw-score improvement.

**CEBRA** (`064`, behavior-contrastive, `offset1-model`, 8-D embedding,
kNN decoder, single-session, 5 example sessions only): mean test Pearson
0.41-0.54, competitive with PLS+1SE on `whisker_curve` (even beat it on
2/5 sessions) but trailing on `performance_curve`, and with the LARGEST
train score of all methods tested (~0.98) -- a red flag for the largest
train/test gap. **Its null was never run** (explicit user instruction to
skip the expensive 100-shuffle null for CEBRA), so its real above-chance
effect size is unknown -- on `performance_curve` its raw test score
(0.409) sits suspiciously close to PLS+1SE's OWN null level (0.409) from
the earlier full-pool run, which is a real reason for caution, not
reassurance. CEBRA was run single-session with fixed, untuned
hyperparameters (output_dim=8, 500 iterations); its natural strength
(joint multi-session embedding alignment) was never exercised.

**Decision: keep PLS+1SE as the working method.** It is simpler, faster,
fully validated (proper null, full pool, three metrics, group
significance), and neither dPCA nor CEBRA beat it on the evidence
gathered so far. CEBRA's multi-session mode remains the one genuinely
promising unexplored option if pseudopopulation pooling is revisited.

## 7. Scope / limitations (unchanged from `METHOD_perfquant_decoding.md`)

- Single-animal / within-session only; no pseudopopulation/multi-session
  pooling implemented anywhere in this pipeline -- repeatedly flagged as
  the single biggest untried lever for increasing effective n.
- The +/-20 lag sweep cannot, by itself, rule out slow shared drift as a
  contributor -- only the properly-constructed shift-null is a valid
  significance test.
- Cross-animal generalization is possible in principle (`050`, LOSO,
  area-averaged features) but was not pursued further this session (user
  decision).

## 8. Canonical scripts

| Script | Role |
|---|---|
| `052` | Original single-session multi-head Ridge validation; SCALE_FLOOR fix. |
| `055`/`056` | Ridge/Lasso/ElasticNet/PLS + PLS-Forest comparison; motivated PLS+1SE. |
| `057` | +/-20 trial lag-profile diagnostic + R+/R- cohort comparison, full pool. |
| **`058`** | **Canonical full-pool validation** (PLS+1SE, 1000-shuffle null, R2/Pearson/Spearman, Wilcoxon). |
| `059`/`060`/`061` | Example-session inspection: best-to-worst ranking, null-band overlay (qualitative only), per-session 1000-shuffle significance. |
| `062` | Sensory vs. baseline vs. baseline-corrected full-pool comparison. |
| `063` | Cross-condition (paired Wilcoxon) + cross-cohort (Mann-Whitney/Welch) statistical comparison. |
| `064` | dPCA and CEBRA comparison against PLS+1SE, 5 example sessions. |
| `065` | All good/moderate-learner session predictions with null overlay (sensory window, PLS+1SE). |

## 9. Area-level breakdown (`066`-`068`, 2026-09-21/22 -- added to this recap 2026-09-23)

Same canonical recipe as `058` (sensory window, floored scaler, PLS+1SE
per session x area, shift-null scored against each shuffle's own shifted
target) but run per `area_group` instead of `whole_brain`, `N_SHUF_NULL=100`
(not 1000 -- ~20 areas x up to 88 sessions makes the canonical budget
~20x `058`'s cost; 100 still gives ~1% p resolution, adequate for a
ranking screen). `MIN_UNITS_PER_AREA=5`; an area is only reported for a
cohort with >= 15 sessions, which leaves 8 areas: Frontal, Hippocampus,
Midbrain, Motor, Retrosplenial, Somatosensory-whisker, Striatum, Thalamus.

**Ranking (`066`, mean above-null Pearson).** Best areas for
`whisker_curve`: R+ Midbrain 0.164 > Striatum 0.145 > Motor 0.124 >
Thalamus 0.120 > Frontal 0.114; R- (by per-session median) Midbrain 0.189
> Thalamus 0.176 > Frontal 0.174 > Hippocampus 0.168 > Motor 0.166.
Somatosensory-whisker is mid-pack, not top, on every target.

**Significance (`067`, Wilcoxon on per-session real-minus-mean-null,
BH-FDR across the 45 area x cohort x target tests).** Significance is
near-ubiquitous: 39/45 tests at `pearson_q<0.05` (42/45 R2, 40/45
Spearman), i.e. 22/24 R+ and 17/21 R- cells. Every one of the 8 areas is
significant for at least one target in at least one cohort.

**Interpretation caution (the reason this is a screen, not a result).**
That near-every-area-significant pattern is exactly what a broadly
distributed OR a shared-slow-drift signal both predict; combined with the
§5 lag-sweep and null-band diagnostics, the area ranking should be read as
"which areas carry the most of it", not as evidence of area specificity.
Effect sizes are also compressed relative to whole_brain (~0.07-0.19 vs.
~0.10-0.17 above-null with far more neurons), consistent with per-area
unit counts being the limiting factor.

**`068`**: per-area (+ whole_brain reference) true-vs-predicted curves for
the 5 example sessions (AB092, AB154, AB158, MH011, MH022), null band by
the `060`/`065` real-axis-remapping construction -- QUALITATIVE only,
`N_SHUF=20`, never a p-value.

| Script | Role |
|---|---|
| `066` | Per-`area_group` full-pool PLS+1SE + area ranking (`..._area_ranking.csv`). |
| `067` | Per-area/cohort Wilcoxon + BH-FDR on `066`'s CSV; significance heatmap. No new fits. |
| `068` | Per-area example-session prediction curves, 5 sessions, qualitative null band. |

## 10. Expert-stage replication (`070`, 2026-09-23)

Same recipe as `058`, run on `day_stage=='expert'` sessions instead of
learning-stage (23/29 usable -- 6 excluded, no day>0 curve file or a
whisker trial-count mismatch), `N_SHUF_NULL=500` (not `058`'s 1000, user's
explicit number for this run).

**Headline replicates**: all 9 target x metric combinations reject the
null (Wilcoxon p 2.4e-7 to 0.023), same as learning-stage. Above-null
margins are comparable to slightly larger than learning-stage:
whisker_curve 0.156-0.220 (vs. learning's 0.14-0.17), falsealarm_curve
0.143-0.162 (vs. 0.11-0.13), performance_curve 0.070-0.112 (vs.
0.10-0.12, the one target/metric combo -- spearman -- that's weaker here
than at learning stage, p=0.023 the loosest of the 9). Train/test gap is
similar in shape to learning-stage (train ~0.83-0.93, test ~0.27-0.66).

**Reading this result**: within-session behavioral-state decodability
from neural population activity is NOT specific to the learning stage --
it replicates once the mouse is already expert, i.e. this is not "decoding
the learning process" so much as "decoding whatever slow within-session
performance drift exists," which is present at both stages. This argues
against interpreting the learning-stage result as evidence for a
learning-specific neural signal on its own (a caveat already implicit in
§5's slow-drift discussion, now empirically confirmed rather than assumed).

| Script | Role |
|---|---|
| `070` | Expert-stage replication of `058` (PLS+1SE, N_SHUF_NULL=500, 23/29 sessions). |

## 11. Within-mouse test, by condition (learning/expert) x cohort (R+/R-) (`071`, 2026-09-23)

Unit of analysis: **MOUSE** (sessions averaged within mouse first --
matters only for expert stage, where 23 sessions come from 16 mice; a
no-op at learning stage, 1 session/mouse there). Per (day_stage, cohort,
target): Wilcoxon signed-rank AND one-sample t-test on per-mouse
above-null Pearson vs 0, both reported (this project's standing
"always report both" convention, [[ssl_rplus_rminus_test_pair]], applied
here to a one-sample test). BH-FDR across the 12 (2 stage x 2 cohort x 3
target) tests, per metric x test-type.

**11/12 cells significant at BH-FDR q<0.05 (t-test); 10/12 (Wilcoxon,
Pearson)**. Every learning-stage cell (6/6) is highly significant
(q<1e-5) regardless of test type. Expert-stage: R+ is significant on all
3 targets (q<0.006 Wilcoxon, q<0.0004 t-test, n=10 mice); **R- is the one
weak cell** -- only `whisker_curve` clears Wilcoxon (q=0.0375, n=6 mice),
`falsealarm_curve` clears the t-test but not Wilcoxon (q=0.025 vs 0.068),
`performance_curve` clears neither (q=0.094/0.124). Given n=6 mice, this
reads as an underpowered cell, not evidence of a real R+/R- difference at
expert stage -- effect sizes (median above-null Pearson 0.12-0.21) are
comparable to or larger than every other cell; this is a significance-vs-n
gap, not an effect-size gap. No R+/R- comparison was run directly (this
tests each cohort against its own null, not against each other) -- matches
§4's earlier finding of no cohort effect and doesn't contradict it.

| Script | Role |
|---|---|
| `071` | Within-mouse Wilcoxon + t-test, above-null vs 0, by day_stage x cohort. |

## 12. Caveats, CV/stratification details, and fuller figures for §11 (`072`, 2026-09-23)

**Cross-validation**: plain `KFold(shuffle=True, random_state=seed)`
throughout `058`/`070` -- NOT stratified, at either the component-
selection step (`select_pls_components`, its own separate nested 5-fold
CV + 1-SE rule) or the pooled-CV scoring step. No stratification by
target value (continuous regression, no classes), by trial type (whisker-
only design matrix, `is_whisker` all-True), or by mouse (each model is
fit within ONE session -- no cross-mouse pooling in a single model, so
mouse-level group-CV, used elsewhere in this project for cross-animal
designs, doesn't apply here). PLS component count is selected once per
session from real Y only, then reused unchanged for the real fit AND
every null shuffle (matched, not re-tuned per shuffle). Pooled scoring:
2 repeats x 5-fold, out-of-fold predictions concatenated across repeats
before scoring once.

**Caveats on §11's within-mouse test**:
- Expert R- (n=6 mice) is underpowered, not a null result -- effect sizes
  are as large as every significant cell (visible directly in `072`'s
  heatmap: expert-R- cells are as dark/darker than significant
  learning-stage cells, just starless).
- Cohort n is unbalanced and shrinks sharply at expert stage (R+/R- =
  50/38 learning, 10/6 expert) -- asymmetric power across cohorts, not
  itself evidence of a cohort difference (§4/§11 already found none
  directly).
- BH-FDR corrected per metric x test-type family (12 tests each), not
  across all 3 metrics jointly -- consistent with `067`'s established
  convention, but worth knowing when comparing "11/12" style counts.
- R2/Pearson/Spearman are correlated metrics on the same fit, not
  independent replications -- their agreement is reassuring, not
  additional independent evidence.
- Per-mouse averaging weights a 1-session and a 2-session mouse equally
  (correct, avoids double-counting) but doesn't account for the 1-session
  mouse's value being noisier.
- Target is a point estimate (`p_mean`) from the smoothed learning curve;
  its own CI (`p_low`) isn't used, so target measurement uncertainty
  isn't propagated into the decoding scores.
- Deeper caveat carried from §5/§10: a significant cell here means
  "real != null," not "signal specific to this behavioral variable" --
  the effect is still more consistent with slow within-session drift than
  a target-specific neural code.

**Figures added** (`072_perfquant_withinmouse_permouse_strip.png`,
`072_perfquant_withinmouse_heatmap.png`): per-mouse strip plot (individual
mice, all 3 metrics, 4 cells x 3 targets) and a 3-metric heatmap (mean
above-null value, BH-FDR stars) -- both make the expert-R- power issue
visually obvious rather than only stated in text.

| Script | Role |
|---|---|
| `072` | Per-mouse strip plot + 3-metric heatmap for §11's within-mouse test. |

## 13. Neuron-count control, hitmiss time-resolved decoding (`073`, 2026-09-23)

(Not a perfquant-branch result -- logged here alongside §11/§12 since it's
part of the same "is this pipeline confound-robust" investigation.)

5 example sessions (`068`'s `EXAMPLE_MICE`), `hitmiss_stim_whole_brain`,
exact production recipe, subsampled to N=100,200,...,1000 (+true full
count), 3 draws/N averaged, real accuracy only (no null -- see `073`'s
own docstring for the cost rationale).

**Result: reassuring.** Peak accuracy rises steeply from N=100 to
~N=300-500, then **flattens for every session** -- including the
best-decoding ones. Critically, **peak decodability does NOT track total
unit count**: MH011 (n_total=687, the SMALLEST pool of the 5) reaches the
2nd-highest peak accuracy (~0.94), clearly above AB154 (n_total=2199,
peak ~0.80) and AB092 (n_total=1217, peak ~0.57, never really rises above
chance-adjacent regardless of N). AB158 (n_total=2704, the largest pool)
plateaus at ~0.93 by N~=600-700, indistinguishable from its N=2704 value.
This argues directly against "accuracy differences are just proxying how
many neurons happened to be recorded" -- the informative sessions are
informative already at N~300-500, and more units past that point buy
almost nothing. AB092's flat-near-chance curve across the ENTIRE N range
(not just at low N) shows it's a genuinely poor-decoding session, not an
N-starved one.

| Script | Role |
|---|---|
| `073` | hitmiss_stim_whole_brain accuracy vs N, time-resolved, 5 example sessions. |
| `074` | PLS+1SE perfquant accuracy vs N, single sensory window, same 5 sessions (running). |

## 14. Neuron-count control, PLS+1SE perfquant (`074`, 2026-09-23)

Same 5 example sessions, same N grid as `073`, PLS+1SE recipe (sensory
window, single fixed window -- not time-resolved, per user's scope
decision), real test Pearson only (no null).

**Result: does NOT clearly saturate, unlike `073`'s hitmiss result --
opposite conclusion, stated explicitly so the two aren't conflated.**
Test Pearson keeps climbing across most of the tested range for most
sessions/targets: AB158 whisker 0.68->0.85 still rising at N=2704;
AB154 falsealarm 0.32->0.52, performance 0.06->0.23, neither flat;
AB092 performance jumps sharply at its full-pool endpoint (N=1217, the
single biggest step in the figure); MH022 performance still visibly
rising at N=1487. MH011 (n_total=687, the smallest pool of the 5) is the
one session that looks close to plateaued by N~400-500 across all three
targets -- but being the smallest pool, whether it would keep climbing
with more units is untestable here.

**This reinforces, rather than resolves, §9's area-level caveat**
("effect sizes are compressed relative to whole_brain... consistent with
per-area unit counts being the limiting factor") -- that was previously a
plausible-but-unverified guess; this result makes it the more likely
explanation, since PLS regression scores in this pilot clearly have NOT
hit a ceiling in the N range areas typically operate in (`066`'s areas
mostly have well under 1000 units). Area-level PLS rankings (`066`-`068`)
should be read with this in mind until a proper unit-count-matched or
covariate-adjusted re-analysis is done (the two concrete options
discussed with the user before this control was built).

## 15. `071` redone for R2 specifically -- test R2 vs null R2 side by side (`076`, 2026-09-23)

Correction of a misread request (`075` -- the N-subsample control with
null -- was NOT this; that's a separate, still-useful control logged in
§13/§14 territory). This is `071`'s own within-mouse test, replotted to
show **test R2 and null R2 as paired bars** (not just the above-null
delta) per (day_stage, cohort, target) cell. Pure post-hoc on `058`/`070`
CSVs, no new compute.

Numbers match `071`'s original r2/Wilcoxon count exactly (9/12 cells
significant at BH-FDR q<0.05) -- the 3 non-significant cells are all
expert/R- (n=6 mice): whisker_curve q=0.068, falsealarm_curve q=0.068,
performance_curve q=0.156. Every learning-stage cell and every expert-R+
cell is significant (q<0.006).

**What the paired-bar view adds over `071`'s delta-only bars**: null R2
is never near zero -- it sits at roughly half the test R2 in most cells
(e.g. learning R+ whisker_curve: test=0.329, null=0.167; expert R+
falsealarm_curve: test=0.575, null=0.359). This is the R2 counterpart of
the point made in §5/§10: a large share of the RAW score is explained by
whatever the shift-null already captures (shared slow drift); the test-
vs-null GAP, not the raw test R2, is what should be read as the genuine
decoding claim -- now visible directly as two bars rather than requiring
mental subtraction.

| Script | Role |
|---|---|
| `076` | `071` redone for R2 specifically: test R2 vs null R2 paired bars, by day_stage x cohort x target. |

**2026-09-23 follow-up ("Also shown individual mouse lines")**: `076`'s
figure now overlays each mouse's own (test R2, null R2) pair as
connected points (filled=test, open=null, `046`'s `_paired_panel`
convention) on top of the (now lighter) mean bars. Confirms the
expert-R- "underpowered, not null" reading directly rather than just by
argument: most individual mice still show a clear downward test->null
slope in those panels, it's the n=6 that keeps it from clearing
significance, not an absence of real per-mouse separation.

## 16. `074` corrected: above-null R2, not raw score, is what matters (`075`, 2026-09-23)

**Revises §14's practical conclusion** -- §14 read `074`'s rising raw
test-Pearson-vs-N curves as evidence PLS is unit-count-limited, which
would reinforce the area-level confound worry from §9. That conclusion
was incomplete: `074` had no null. `075` reruns the same 5 sessions x N
grid WITH null (full retrain-under-shift, `N_SHUF_NULL=30`), computing
test R2, null R2, and above-null R2 (=test-mean(null)) at every N.

**Result: both test R2 AND null R2 climb together with N, staying a
roughly constant gap apart** -- visible directly in `075`'s test-vs-null
line-overlay figure (solid vs dashed track each other across the whole
range, for every session x target panel). Consequently **above-null R2
is roughly FLAT vs N** for most session/target combinations (AB158
whisker ~0.22-0.30, AB158 performance ~0.15-0.21, MH011 whisker/
falsealarm flat, MH022 roughly flat with noise) -- much closer to `073`'s
hitmiss finding (early-saturating, not unit-count-limited) than §14's
raw-score-only reading suggested. AB092 stays noisy/near-zero throughout,
consistent with `073`'s "genuinely poor session, not N-starved" reading.

**Corrected takeaway**: the apparent unit-count-sensitivity in `074` was
substantially a NULL-INFLATION artifact (more features = more room for a
signal-free shift-null to overfit too), not the real signal needing more
neurons. This softens, but doesn't fully retire, §9's area-level caveat:
areas with very few units (well below this pilot's N=100 floor) could
still be genuinely underpowered -- the pilot's smallest N tested was 100,
which is already above several areas' actual unit counts in `066` -- but
the "PLS results are mostly a unit-count artifact" reading is now less
supported than §14 stated.

| Script | Role |
|---|---|
| `075` | `074` rerun with real null shuffles (N_SHUF_NULL=30); test R2 vs null R2 and above-null R2, vs N. |

## 17. Single-trial face validity: does the model make sense? (`077`, 2026-09-23)

Direct answer to "gain confidence the model makes sense": does the
single-trial PREDICTED performance state (`Y_pred`, from `065`'s already-
cached predictions -- no PLS recompute) track independent single-trial
behavioral signals, in the expected direction? Two metrics, deliberately
NOT trial-index/session-time (that would just re-confirm the already-
known slow-drift confound, not test against it): RT (hit trials only)
and previous-trial outcome (lag-1 hit/miss within the session's own
whisker-trial sequence). Per-session Pearson r, then group-level
Wilcoxon + t-test on those per-session r values against 0 (avoids any
trial-autocorrelation p-value trap -- same logic as every above-null
test in this project).

**Result: strong, consistent, and in exactly the expected direction.**
17/24 (day_stage x cohort x target x metric) cells significant at BH-FDR
q<0.05 (12 RT cells + 12 prev_outcome cells):
- **RT**: negative in every single cell tested (median r -0.03 to -0.38)
  -- higher predicted performance state -> faster responses. Significant
  in 7/12 cells: 3/3 learning-R+, 2/3 learning-R- (falsealarm_curve n.s.),
  2/3 expert-R+ (performance_curve n.s.), 0/3 expert-R- (all n.s., n=6 --
  same power story as §11, sign is still right in all 3).
- **Previous-trial outcome**: positive in every single cell tested
  (median r +0.04 to +0.38) -- a recent hit predicts a higher current
  predicted state. Significant in 10/12 cells: 3/3 learning-R+, 3/3
  learning-R-, 2/3 expert-R+ (performance_curve n.s.), 2/3 expert-R-
  (falsealarm_curve n.s.).
- **Every one of the 24 cells has the theoretically-expected sign** --
  not just "mostly significant," genuinely unanimous in direction, which
  is much stronger face-validity evidence than the significance count
  alone. This is independent evidence (RT and trial history are not
  derived from spike counts or from the target curve itself) that the
  model's predictions track real, meaningful behavioral variation, not
  just noise that happens to beat a shift-null.

| Script | Role |
|---|---|
| `077` | Single-trial `Y_pred` vs RT / previous-trial outcome, per-session r + group test. |

## 18. Cross-mouse session-permutation nulls, pilot (`079`, 2026-09-23)

User-proposed Harris-style nulls (see `079`'s docstring for full mechanics
and the alignment-simplification rationale confirmed with the user before
building): **R- null** (recipient R+ mouse's own target, donor R- mouse's
neural activity -- tests reward-dependence) and **within-R+ null**
(recipient's own neural activity, donor OTHER R+ mouse's target -- tests
individual specificity). Pilot: 5 top-decoding R+ learning-stage
recipients (MH070, MH069, MH031, AB162, AB087) x all 37 R- donors x all
33 other-R+ donors x 3 targets, PLS+1SE, whisker-trial-index alignment
(collapses to whisker-stim-count alignment in this pipeline, confirmed
before building).

**Two implementation bugs found and fixed during this run** (both now
documented in `079`/`079b`'s own code comments): (1) a misplaced inner
`import numpy as np` after two earlier `np.where(...)` calls in the same
function caused an `UnboundLocalError` on every cell, and an unguarded
`fut.result()` let that one bug silently kill the entire 355-cell job
after only 2 cells -- fixed (removed the bad import, added a try/except
so one bad cell can never again kill the run). (2) The CSV-saved list
columns for pearson/spearman stringify as `"[np.float64(0.93), ...]"`,
and a naive float-regex parser matched the literal "64" inside
"float64" as a spurious extra number, corrupting 2 of every 3 parsed
values (produced impossible |r|>1 "correlations" in the first summary
pass) -- fixed by stripping the `np.float64(...)` wrapper before
regex-parsing.

**Result (pilot scale, 5 recipients, no group-level test yet -- read as
suggestive, not confirmatory)**:
- **R- null (reward-dependence)**: real Pearson beats the donor median in
  every one of 15 (5 recipients x 3 targets) cells, corrected margin
  +0.08 to +0.38, rank-based p mostly 0.026-0.16 (minimum possible p at
  n=37 donors is ~0.026 -- several cells are AT that floor, meaning real
  beat literally every donor). Best separation: AB162 whisker_curve
  (+0.378), MH031 whisker_curve (+0.365).
- **Within-R+ null (specificity)**: real still usually beats the donor
  median (12/15 cells positive, +0.09 to +0.32) but separation is
  visibly weaker in the figure -- donor violins are WIDE, often reaching
  0.8-0.9, so "beats the median" is a much softer claim than it sounds.
  **AB162 is the clear exception**: all 3 targets non-significant
  (p=0.21-0.27), the real diamond sits inside the bulk of the donor
  violin rather than above it -- for this one session, its neural
  activity predicts OTHER R+ mice's trajectories nearly as well as its
  own, i.e. weak evidence of individual specificity there specifically.
- **Honest reading**: the R- null is cleared fairly convincingly at this
  pilot scale -- the signal is not purely generic drift. The within-R+
  (specificity) claim is real but visibly weaker and not uniform across
  recipients -- AB162 argues against over-claiming individual specificity
  as a general property, at least not yet, at n=5 recipients with wide
  donor distributions limiting resolution.
- **Not yet done**: no BH-FDR or group-level test across the 5 recipients
  (pilot too small to be a real group claim); donor pool capped to
  learning-stage only; only Pearson inspected visually (r2/spearman are
  in the CSV, not yet plotted).

| Script | Role |
|---|---|
| `079` | Cross-mouse R- and within-R+ nulls, compute stage (355 cells, PLS+1SE fits). |
| `079b` | Summary + figure from `079`'s raw CSV (fixes the two bugs above; no recompute). |

## 19. Full cross-mouse sweep, all 75 sessions as recipients (`081`, 2026-09-23)

Extends §18 from a 5-recipient (cherry-picked top-decoding) pilot to
**all 75 learning-stage sessions as recipients** (38 R+, 37 R-), full 2x2
design: R+ recipient x {R- donor (`rminus_null`), other-R+ donor
(`withinrplus_null`)}, and R- recipient x {R+ donor (`rplus_null`,
NEW), other-R- donor (`withinrminus_null`, NEW -- user's own follow-up
question, "Should there be within R- null too?"). ~5600 cells, all
succeeded (0 exceptions) -- see `080`/`081`'s own docstrings for the
precompute+global-cache speed redesign that made this tractable (finished
in under a minute of actual compute once launched).

**Group-level test this time** (per cohort x null_type x target, Wilcoxon
+ t-test on each recipient's corrected score across all recipients in
that cohort, BH-FDR across the 12 cells) -- not just per-recipient
inspection.

**Result: meaningfully weaker and more mixed than the 5-recipient pilot
suggested -- the pilot's strength was partly a selection-bias artifact of
testing only the best-decoding sessions, exactly the caveat flagged when
it was built.** Only **5/12 cells** survive Wilcoxon+BH-FDR; only **1/12**
by t-test (the Wilcoxon/t-test disagreement itself signals a skewed,
outlier-heavy distribution of corrected scores, visible directly in the
figure -- range roughly -0.6 to +0.4 per recipient, not a tight cluster
around a positive median).

- **R+ reward-dependence** (`rminus_null`): all 3 targets significant by
  Wilcoxon (q=0.008 each) but NONE by t-test (q=0.09-0.34) -- median
  corrected +0.07 to +0.11, but only 2-5/38 mice (5-13%) individually
  significant. A real but modest, right-skewed group effect, not a
  strong uniform one.
- **R+ specificity** (`withinrplus_null`): only `whisker_curve`
  survives (Wilcoxon q=0.008, t-test q=0.013 -- the ONE cell where both
  tests agree). `falsealarm_curve`/`performance_curve` are clearly n.s.
  (q>0.32). Specificity, where it exists, is whisker-curve-specific, not
  general across all three behavioral targets.
- **R- reward-dependence, mirrored** (`rplus_null`): only `whisker_curve`
  marginally survives (Wilcoxon q=0.043, right at the boundary; t-test
  n.s.). Weaker than the R+ direction.
- **R- specificity** (`withinrminus_null`, the new test): **completely
  null, all 3 targets, both test types** (q=0.32-0.64). R- mice do NOT
  show individual specificity to each other. Read together with the R+
  `whisker_curve` specificity result: this is the reassuring half of the
  answer to the question that motivated adding this null -- individuality
  in this decode is NOT simply "any two animals differ," since R- mice
  fail the exact same test R+ mice (weakly) pass. Whatever individual
  specificity exists for `whisker_curve` in R+ mice is not explained by
  generic animal-to-animal idiosyncrasy alone.

**Revised overall reading** (supersedes §18's pilot-based optimism):
reward-dependence is a real but modest, skewed, right-tailed effect
(median mice show a small positive margin, individual significance is
rare); specificity survives only for `whisker_curve` and only in the R+
direction; R- mice provide a clean negative control showing that
specificity is NOT a trivial any-animal property. The cherry-picked pilot
overstated how strong and how general these effects are -- an object
lesson in why the full, unbiased sweep mattered here.

| Script | Role |
|---|---|
| `080` | Precompute each session's (X, Y_raw) once, cached -- speed optimization for the full sweep. |
| `081` | Full 2x2 cross-mouse sweep, 75 recipients, ~5600 cells, compute stage only. |
| `081b` | Group-level summary (Wilcoxon+t-test+BH-FDR, per-recipient CSV) + figure from `081`'s raw CSV. |

## 20. Within-R+ specificity extended to R2 and Spearman (`081b`, 2026-09-23)

User: "Extend the within-R+ specificity test to R2 and Spearman too."
Pure post-hoc -- `081`'s raw CSV already carried all 3 metrics, `081b`
just wasn't reporting them; extended to all 3 for all 4 null types (36
cells total, up from 12), no new compute.

**Confirms and strengthens §19's `whisker_curve`-only reading -- not a
Pearson artifact.** `whisker_curve` within-R+ specificity is significant
across ALL 3 metrics, BOTH test types (q=0.003-0.013 every time --
r2/pearson/spearman all agree, Wilcoxon AND t-test both agree for this
one target/null combination, unlike most other cells in this project
where the two tests disagree). `falsealarm_curve`/`performance_curve`
remain clearly non-significant across all 3 metrics (q>0.10 throughout)
-- the narrow, whisker-curve-specific reading from §19 holds up under
the metric-triangulation check, it isn't a Pearson-specific fluke.

Full 36-cell table: 14/36 significant (Wilcoxon, BH-FDR), 4/36 (t-test)
-- same broad pattern as §19's 12-cell Pearson-only version (roughly
40%/11%), scaled to the larger family.

| Script | Role |
|---|---|
| `081b` (updated) | Group-level summary extended to all 3 metrics; added a dedicated within-R+-specificity, 3-metric figure. |

## 21. RRR + LG-AR1 pilot: null-controlled smoothing does NOT help here (`082`, 2026-09-23)

User-specified alternative pipeline (full spec, not this project's own
design -- see `082`'s docstring for the two implementation deviations
required to make it tractable: dual-ridge/Woodbury reformulation of the
alternating-ridge RRR fit, and direct marginal-likelihood maximization
instead of EM for the LG-AR1 parameters). 3 sessions (MH070/MH069/MH031,
same top R+ recipients as `079`/`081`), all 3 targets, whole_brain,
`N_SHUF_NULL=20`, null applied through the ENTIRE pipeline (RRR
out-of-fold fit + LG-AR1 fit + smoothing, all on the shifted target) per
the user's own explicit caveat -- not compared against an unsmoothed
baseline.

**RRR alone (single-trial decoder) shows real above-null decoding in
9/9 cells** (+0.025 to +0.226), same order of magnitude as this
project's own PLS+1SE pipeline despite a completely different
architecture (multi-bin spike time-course + low-rank regression vs.
single aggregated window + PLS) -- a genuinely independent
cross-validation of the headline "yes, decodable" finding using an
unrelated method.

**LG-AR1 smoothing does NOT increase the above-null margin -- it
decreases it, in EVERY ONE of the 9 cells tested** (e.g. MH070
whisker_curve: RRR above-null +0.127 -> smoothed +0.113; MH069
falsealarm_curve: +0.226 -> +0.202; MH031 whisker_curve: +0.074 ->
+0.053). This is exactly the failure mode the user's own caveat
predicted, now empirically demonstrated: smoothing raised BOTH the real
correlation AND the null correlation by a similar amount in every cell
(e.g. MH070 whisker_curve real 0.96->0.98, null 0.83->0.87), so the
NET above-null gain is a wash or a net loss, never a win, once the null
is properly run through the same smoothing step rather than compared to
an unsmoothed baseline.

**Why, mechanistically**: fitted AR(1) persistence `rho` is extremely
high in every single cell (0.970-0.993), implying an effective smoothing
timescale of ~1/(1-rho) ~ 33 to 140+ trials -- comparable to or LONGER
than these sessions' own trial counts (73-100). At that persistence, the
smoother is essentially tracking each session's whole-session slow
trend, which the shift-null (by construction, preserving label
autocorrelation) shares almost as much as the real data does -- so a
smoother this aggressive can't add discriminating power beyond what the
per-trial RRR estimate already had.

**Practical reading for this project's task specifically** (short, fast-
timescale sessions -- 70-250 trials, sensory-evoked, NOT the original
method's long slow-prior ITI design it was built for): LG-AR1 smoothing
on top of RRR does not currently offer a validated improvement here.
This is a real, useful negative result from a small, deliberately-scoped
pilot (3 sessions) -- not yet tested at full-pool scale, and the
RRR-alone result (independently confirming decodability via a different
architecture) is arguably the more valuable finding of the two.

**Known caveat on the figure** (`082_rrr_lgar1_pilot.png`): the plotted
smoothed trace is the raw latent-state estimate m_s, not affine-
corrected back to the observation scale via the fitted theta/mu -- causes
a visible vertical scale/offset mismatch in some panels (e.g. MH031
whisker_curve) despite a high correlation, since Pearson r is scale/
offset-invariant but the plot isn't. Doesn't affect any reported number,
only the visual comparison.

| Script | Role |
|---|---|
| `082` | RRR (dual-ridge alternating fit) + LG-AR1 (marginal-likelihood Kalman smoother) pilot, 3 sessions, full-pipeline null. |

## 22. Matched PLS+1SE vs RRR comparison, baseline/ITI window (`083`, 2026-09-23)

Direct follow-up to §21, matching what §21/the discussion flagged as
unmatched: SAME window for both decoders (`WIDE_BASELINE_WINDOW=
(-2.0,-0.01)`, this project's own canonical baseline window, `062`/§3 --
chosen by the user as the harder test, deliberately not the sensory
window `082` used), SAME feature scaling (RRR now uses the floored
`StandardScaler`, matching PLS's `052`-bug-fix convention -- `082`'s RRR
only mean-centered), and LG-AR1 applied to BOTH decoders' outputs, not
just RRR's. RRR still gets T=20 time-resolved 100ms bins across the same
window (its structural point); PLS still aggregates to one flat mean
(its structural point) -- deliberately not forced to match, or there
would be nothing left to compare.

**Result: once fairly matched, PLS+1SE is the more reliable method on
this window -- RRR does not show a consistent advantage.**
- **PLS above-null: positive in 8/9 cells**, mean +0.055.
- **RRR above-null: positive in only 5/9 cells, mean -0.014** (i.e. on
  average RRR does not beat its own shift-null on this window) --
  strongly negative in 3/9 (MH070 falsealarm -0.178, MH070 performance
  -0.118, MH031 performance -0.059).
- **RRR+LG-AR1 is highly volatile**, unlike the uniformly-mild loss
  pattern §21 found on the sensory window: MH070 whisker_curve jumps
  from RRR-only -0.003 to RRR+LGAR1 +0.227 (the single largest bar in
  the whole figure), but MH070 performance_curve gets WORSE (-0.118 ->
  -0.187) and MH031 whisker_curve collapses (+0.074 -> +0.008). When the
  underlying per-trial RRR estimate is already weak/noisy (as it often
  is on this window), the Kalman smoother can either accidentally
  align with real structure (large win) or amplify the wrong pattern
  (large loss) -- much higher variance than smoothing a stronger
  signal.
- **PLS+LG-AR1 stays close to PLS alone** in most cells (mean +0.053 vs
  +0.055) -- smoothing a more reliable base estimate doesn't help much,
  but doesn't hurt much either, consistent with §21's general "smoothing
  doesn't add above-null separation once null-controlled" finding, just
  with much less volatility than pairing it with RRR.

**One confound this comparison still does NOT remove, stated plainly**:
RRR's hyperparameter selection (plain CV-best over an R x lambda grid)
is more liberal than PLS's 1-SE rule (deliberately biased toward the
simplest adequate model). On this particular window -- long (2s), high
raw autocorrelation, T=20 bins x up to ~2000+ units giving RRR a very
large effective parameter space via its R*(N+T) factorization -- a more
liberal selection rule is plausibly more prone to overfitting than PLS's
conservative one. This pilot does NOT distinguish "RRR the architecture
underperforms here" from "RRR's hyperparameter selection wasn't as
conservative as PLS's" -- both remain live explanations for the gap.

**Bottom line**: for this project's baseline/ITI window specifically, at
this pilot scale (3 sessions), PLS+1SE is the safer choice -- RRR's
extra structural flexibility (learned temporal kernel) did not translate
into a reliable advantage, and LG-AR1 smoothing remains unhelpful on
average and additionally unstable when paired with RRR's noisier
per-trial estimates.

| Script | Role |
|---|---|
| `083` | Matched PLS+1SE vs RRR (+/- LG-AR1) comparison, same window+scaling, baseline/ITI. |

## 23. CORRECTION to §22: RRR outperforms PLS once properly regularized

**§22 is superseded, not just caveated -- its headline ("PLS beats RRR")
was substantially an artifact of an under-regularized RRR fit.** User
asked to plot the actual per-trial curves (`083`'s new
`083_pls_rrr_lgar1_curves.png`), which revealed RRR's raw held-out
predictions reaching ~250-300x the standardized target's own scale for
at least one session -- not visible in the summary correlation numbers
alone. Traced to the hyperparameter grid search selecting `lambda=100`,
the OLD `LAM_GRID`'s maximum -- the search had been hitting its own
ceiling rather than finding a genuine interior optimum. With R up to 3
and N up to ~2000+ units, the U-step fits up to ~R*N~6000 parameters
from ~80 training trials/fold; under-regularization there doesn't just
overfit, it destabilizes the alternating-ridge fit on held-out folds.

**Fixed**: `LAM_GRID` widened to 9 values up to 30000; an automatic
warning now fires if a selected lambda hits the grid's max (so this
can't silently recur). Rerun: zero warnings this time -- every cell
found a genuine interior optimum.

**Corrected result: RRR clearly outperforms PLS on this window.**
- **RRR above-null: positive in 8/9 cells, mean +0.291** (vs the old,
  buggy run's mean -0.014).
- **PLS above-null: positive in 9/9 cells, mean +0.080** (essentially
  unchanged from §22 -- PLS was never affected by the RRR bug).
- RRR beats PLS in 8/9 direct cell-by-cell comparisons, several by a
  large margin (MH070 falsealarm_curve: PLS +0.110 vs RRR +0.797; MH069
  falsealarm_curve: PLS +0.124 vs RRR +0.523).
- LG-AR1: mixed and mostly small now that RRR itself is stable (3/9
  cells better, 6/9 worse vs RRR alone; PLS+LGAR1 mean +0.063 vs PLS's
  +0.080, same "doesn't help" pattern as §21 for PLS specifically).

**New finding from the curves themselves, distinct from the bug**: RRR's
raw per-trial predictions remain much LARGER in magnitude than the true
curve in most panels (reaching tens to 100+, vs. true's ~0-10 range),
even after the fix. This is not a leftover version of the same bug --
Pearson correlation is scale-invariant, so a real, properly null-
controlled correlation does not require the prediction to be well-
calibrated in absolute scale. **RRR here is a good relative predictor
(tracks the right direction/shape) but a poorly calibrated one** (doesn't
match the target's own magnitude) -- an inherent property of this high-
dimensional ridge setup (p >> n even after regularization), not
something the lambda fix was meant to or does resolve. A simple final
rescaling step (fit `d_calibrated = a*d_raw + b` via OLS on training
folds) would fix this for visualization/absolute-value purposes without
changing the correlation-based significance claims at all -- not yet
done.

**Revised bottom line**: once genuinely comparable (same window, same
scaling, AND properly regularized), RRR's extra structural flexibility
DOES pay off on the baseline/ITI window -- the opposite of §22's
conclusion. §22's own "unmatched hyperparameter-selection philosophy"
caveat (PLS's 1-SE rule vs RRR's plain CV-best) still stands as an
open question, but is now secondary to the fact that the original
comparison wasn't using a properly-fit RRR at all.

| Script | Role |
|---|---|
| `083` (corrected) | Widened LAM_GRID (was hitting its own ceiling), curve capture, affine-corrected smoothed curves. |

## 24. Scaling problem fixed (`083`, 2026-09-23, continued)

User: "How to fix the scaling problem?" -- the raw (unsmoothed) curves in
`083_pls_rrr_lgar1_curves.png` were still on the RRR predictor's own
native scale (correctly correlated, but not visually calibrated to
`y`). Fixed with a correlation-preserving OLS calibration
(`calibrate_to_target`: fit `a*d+b` regressing d on y; the OLS slope's
sign always matches the correlation's sign, so `corr(a*d+b, y) ==
corr(d, y)` exactly -- changes nothing about any reported above-null
number, purely a plotting fix). Applied to BOTH the raw curves (`d_pls`,
`d_rrr`) and, as a second layer, the LG-AR1 smoothed curves (whose own
theta/mu correction maps back to the raw predictor's native scale, not
necessarily y's scale -- a second same-signed affine map on top is still
correlation-preserving). Also added a results cache
(`083_pls_rrr_lgar1_cache.pkl`, same pattern as `065`) so any future
cosmetic change to this figure is free.

Verified numerically: above-null numbers unchanged in direction/
magnitude across the calibration-only reruns (small run-to-run
differences are from fresh null-shuffle/CV-fold draws, not the
calibration fix -- Python's per-process string-hash randomization means
`hash(session_id)`-seeded RNGs aren't bit-identical across separate
launches, a pre-existing property of this pipeline, not introduced by
this fix).

**Figure now genuinely interpretable**: every curve sits on a comparable
scale to `true` in every panel, and visual fit quality now matches the
reported numbers (e.g. MH070 falsealarm_curve and MH031 performance_curve
visibly show RRR/RRR+LGAR1 tracking the true curve's shape better than
PLS, consistent with their large above-null scores in those cells).

## 25. PLS-only figures, RRR shelved (`085`, 2026-09-23)

User: "Forget about RRR for now, update figures with the PLS+1SE
results." RRR's above-null advantage over PLS (§23) remains an open
question -- whether it reflects genuine specificity or a null-side
artifact from reusing real-data-optimized hyperparameters on shift-null
refits was raised but not yet resolved (RRR is far more sensitive to its
regularization strength than PLS, and both methods reuse one real-fit
hyperparameter set across all null shuffles -- the same reuse convention
this project has used throughout, but RRR may be more exposed to it).
Rather than resolve that now, RRR is dropped from the figures entirely
and PLS+1SE (+/- LG-AR1) is presented on its own, pure post-hoc from
`083`'s already-computed cache/CSV -- no new compute.

**`085_pls_only_curves.png`**: same 9-panel true-vs-predicted grid as
`083`/`084`, RRR removed. PLS+1SE tracks the true curve closely in
essentially every panel (baseline/ITI window) -- visually confirms the
already-reported numbers (real r 0.84-0.96, above-null +0.013 to
+0.164 across the 9 cells) without RRR's parallel story competing for
attention.

**`085_pls_vs_pls_lgar1_paired.png`**: paired PLS-vs-PLS+LGAR1 above-null,
one line per (mouse, target) cell. **LG-AR1 helps only 2/9 cells**,
both tiny: MH070 whisker_curve (0.113->0.116) and MH069 performance_curve
(0.017->0.019). The other 7/9 get worse, several substantially (MH069
falsealarm_curve 0.164->0.123, MH031 falsealarm_curve 0.051->0.035, MH031
performance_curve 0.040->0.028). Net mean above-null 0.065 -> 0.056 --
consistent with §21's original PLS+LGAR1 finding (smoothing doesn't
reliably help once null-controlled) and now shown as the primary,
uncluttered PLS-focused figure for this window.

**Open item, explicitly carried forward, not resolved**: whether RRR's
apparent advantage (§23) is real or an artifact of null-side
hyperparameter reuse. The proposed test (re-select R/lambda per null
shuffle rather than reusing the real-data value) has not been run.

| Script | Role |
|---|---|
| `085` | PLS+1SE-only curves + PLS-vs-PLS+LGAR1 paired figure, RRR dropped. |

## 26. Pairwise-line real-vs-null figure (`086`, 2026-09-24)

Pure post-hoc on `058`'s CSV, no new compute -- same rows=target x
cols=metric facet layout as `058`'s Figure 1 (`058_perfquant_real_vs_
null_distributions.png`), but replaces the two marginal violin
distributions (real, null) with one line per session connecting that
session's own null (mean across its 1000 shift-null shuffles) to its own
real (out-of-fold test) score. Makes the within-session pairing visible
directly. Session counts with real > null (out of 88): whisker_curve
80/78/75 (r2/pearson/spearman), falsealarm_curve 73/73/69,
performance_curve 69/70/71 -- matches §2's group-level finding exactly,
now visible as majority-solid, minority-faint-dashed lines per panel.

| Script | Role |
|---|---|
| `086` | Paired real-vs-null lines, one per session, `058`'s facet layout. |

## 27. Cohort-pooled trial-index-wise learning curves (`087`, 2026-09-24)

Pure post-hoc on `065`'s cache (`065_perfquant_allmice_cache.pkl`), no
new compute. `065` plots one panel PER SESSION (true vs predicted curve
on that session's own real trial axis); this instead pools sessions
WITHIN each cohort (R+/R-) onto a common axis -- each session's trial
index rescaled to a 0-1 fraction of its own length, binned into 20 bins
-- and plots the cohort-mean true vs predicted curve (+/- SEM across
sessions), one panel per (day_stage, target), both cohorts overlaid.

Qualitative read: predicted (dashed) curves track the real (solid)
curves' slow within-session decline closely for both cohorts at both
stages -- consistent with §5's point that a large share of the
decodable signal is shared slow drift rather than something highly
target-specific. R+/R- separate visibly early in the session for
whisker_curve/falsealarm_curve but converge by the session's end, both
stages. performance_curve is the noisiest target and where real/
predicted diverge most, especially expert-stage R+.

**Caveat inherited directly from the binning method**: averaging on a
normalized (fraction-of-session) axis across sessions of very different
lengths trades away exactly the real-trial-order fine structure that
`065`'s per-session panels show -- this figure answers "what does the
average trajectory look like, cohort-wise, independent of exact session
length" not "does the model track any one session's actual trial
sequence" (065's question). The two are complementary, not
interchangeable.

| Script | Role |
|---|---|
| `087` | Cohort-pooled (not per-session), trial-index-normalized true-vs-predicted curves. |

## 28. Cohort-combined cross-mouse null summary (`088`, 2026-09-24)

Pure post-hoc on `081b`'s per-recipient CSV (`081_perfquant_crossmouse_
fullsweep_persession.csv`), no new compute. §19/§20 tested each of the 4
null types separately, split by recipient cohort (4 cells: R+ x
{rminus_null, withinrplus_null}, R- x {rplus_null, withinrminus_null}).
This pools the two MIRROR-IMAGE null types per underlying question --
**reward-dependence** (`rminus_null` + `rplus_null`, both cohorts) and
**specificity** (`withinrplus_null` + `withinrminus_null`, both cohorts)
-- giving n=75 recipients per test instead of 37-38, and re-runs
Wilcoxon + one-sample t-test vs 0 on the pooled sample (BH-FDR across 2
roles x 3 targets x 3 metrics = 18 tests). Dots in the figure stay
colored by the recipient's own cohort so the pooling is visible, not
hidden.

**Reward-dependence becomes robust once pooled**: significant (q<0.05,
mostly q<0.01, both Wilcoxon AND t-test) across all 3 targets x all 3
metrics except falsealarm_curve/spearman (n.s.) -- a markedly cleaner
picture than §19's per-cohort split, where the R+ direction cleared
Wilcoxon-only (t-test q=0.09-0.34) and the R- direction only marginally
cleared for whisker_curve.

**Specificity stays narrow even after pooling**: only whisker_curve is
significant across all 3 metrics; falsealarm_curve is n.s. everywhere;
performance_curve is mixed (only r2 clears q<0.05) -- consistent with
and reinforcing §20's whisker_curve-only reading, now at higher power
and still not generalizing to the other two targets.

**Caveat -- this is a power move, read alongside §19/§20, not instead of
them**: pooling cohorts increases n and therefore power, which is
expected to produce more, not fewer, significant cells even with no
change in the underlying effect. The more positive reward-dependence
picture here should not be read as contradicting §19's more cautious,
per-cohort "real but modest, right-skewed" characterization -- both are
correct descriptions of the same underlying data at different levels of
aggregation, and the per-cohort split remains the reference for whether
the effect is symmetric across reward groups (§19 found the R- direction
distinctly weaker than R+ before pooling).

| Script | Role |
|---|---|
| `088` | Cohort-combined (reward-dependence + specificity, both roles pooled across R+/R-) cross-mouse null summary. |

## 29. Cross-target specificity + PLS weight-vector comparison (`089`/`090`, 2026-09-24)

Direct follow-up to the "drift vs. code" open question (REPORT.md Caveat 1
/ open item 3): does each target's decoder carry target-specific
information, or does one shared "generic drift" axis explain all three
targets about equally well? Unlike `058`/`070` (one joint 3-output PLS
model per session), this fits **three separate single-output PLS+1SE
models per session**, one per target, each independently choosing its own
component count -- a fairer test of whether the separately-optimal
neuron-weightings converge (shared) or diverge (specific). Pilot: the same
12 "best to worst" example sessions `059`-`061` use, `N_SHUF_NULL=200`
(reduced from `058`'s 1000, since this is 3x the single-output null cost).
Run on haas (`089`, compute); figures pure post-hoc (`090`).

**Cross-target specificity matrix** (`090_perfquant_crosstarget_matrix.png`):
every diagonal cell (own target) is positive; every off-diagonal cell is
negative, several strongly so (falsealarm-trained decoder predicting
performance_curve: -0.569; performance-trained decoder predicting
falsealarm_curve: -0.532). **Paired test, own-target vs. mean cross-target,
36 (session x train_target) rows: 33/36 wins, Wilcoxon p=3.2e-9, t-test
p=6.0e-9** -- about as clean a result as this project has produced. This
argues against the "one shared drift axis explains everything" reading:
if it were pure generic drift, cross-target transfer should track the
diagonal closely, not sit reliably below the target's own null reference.

**Caveat on the null reference** (stated in `089`'s own docstring, worth
repeating): the "above-null" value for an off-diagonal cell uses the
TRAIN target's own shift-null as the chance-level reference, not a null
built by literally shuffling the (train, test) pairing -- the best
available same-units reference without a second, more expensive null
construction, but an approximation, not a formal permutation test of the
cross-target claim specifically.

**Per-cell significance is uneven** (`090_perfquant_crosstarget_cellstats.csv`,
BH-FDR across 9 cells): only 4/9 cells individually survive at q<0.05
(falsealarm-falsealarm, falsealarm-performance, performance-falsealarm,
and marginally whisker-performance at raw p but not after FDR). The
AGGREGATE paired test is the trustworthy number here (n=36 pairs, not
n=12 per cell) -- individual-cell power is limited at this pilot's n=12,
same lesson as §5's per-session-vs-group-level point.

**PLS weight-vector cosine similarity** (`090_perfquant_crosstarget_weightcosine.png`):
a more nuanced picture than the specificity matrix alone suggests.
whisker_curve vs performance_curve share substantial weight-vector overlap
(mean cosine +0.629, q=0.0015) despite their cross-target above-null
scores both being clearly negative (-0.181, -0.141) -- shared linear
direction in neuron-space does not automatically translate into shared
predictive power for the other target's specific curve shape.
whisker_curve vs falsealarm_curve: moderate positive overlap (+0.313,
q=0.0068). **falsealarm_curve vs performance_curve: significantly
NEGATIVE cosine (-0.437, q=0.0068)** -- their weight vectors point in
largely opposite directions, consistent with (and likely explaining) that
pair's strongly negative cross-target transfer in both directions: a
decoder built to predict one would push its predictions the wrong way for
the other.

**Reading, honestly**: this pilot's evidence leans toward target-specific
coding over a single shared generic-drift axis -- the aggregate
specificity test is strong and the weight-vector geometry is clearly not
"one shared direction, use it for anything." But it does NOT rule out
slow drift entirely (Caveat 1 in REPORT.md still applies to the diagonal
itself -- own-target decoding could still substantially reflect shared
slow structure that just happens to have a different shape per target),
and it is a 12-session pilot, not a full-pool sweep. The proper next step,
not yet done, would be the full 88-session version of this same design.

| Script | Role |
|---|---|
| `089` | Compute: 3 separate single-output PLS+1SE fits/session, cross-target prediction matrix + weight-vector `coef_`, 12-session pilot, haas. |
| `090` | Figures: specificity matrix, paired diagonal-vs-cross-target, weight-vector cosine similarity -- pure post-hoc on `089`'s CSVs. |

## 30. Weight-vector comparison scaled to full pool, additional visualizations (`091`/`092`, 2026-09-24)

User follow-up to §29's 12-session pilot: "For the weight vector, show
other visualization and do on all data." Scoped deliberately to the
weight-vector comparison only (not the null-controlled specificity
matrix, which stays at pilot scale -- rescaling that would require
re-running the expensive 200-shuffle null on all 88 sessions, not asked
for here). `091` computes only the 3 single-output PLS+1SE fits per
session needed for each target's `coef_` (component selection + one
final full-data fit, no null shuffling at all) -- cheap enough to run
on the full 88-session learning-stage pool directly on haas. 88/89
sessions usable.

**Full-pool result confirms and sharpens the pilot's pattern, now with
enormous statistical power** (BH-FDR q<1e-9 for all 3 pairs, n=88):
- whisker_curve vs performance_curve: mean cosine **+0.707** (up from the
  pilot's +0.629) -- the two targets whose PLS decoders lean most heavily
  on a shared population direction.
- whisker_curve vs falsealarm_curve: mean cosine **+0.348** (pilot: +0.313)
  -- consistent, moderate positive overlap.
- falsealarm_curve vs performance_curve: mean cosine **-0.301** (pilot:
  -0.437) -- confirmed anti-correlated, though the full-pool mean is
  somewhat less extreme than the 12-session pilot's.

**Two additional visualizations, beyond the pilot's per-pair strip plot**:
1. **Summary heatmap** (`092_perfquant_weightvector_matrix.png`) -- same
   visual language as `090`'s specificity matrix, for direct side-by-side
   comparison; makes the "whisker bridges the other two, falsealarm and
   performance pull apart" structure legible at a glance.
2. **Cohort-split violin+strip distribution** (`092_perfquant_weightvector_
   distribution.png`) -- R+/R- points fully interleaved in all three
   panels, no visible cohort separation, consistent with §4/§11/§28's
   repeated "no cohort effect" finding, now checked for this analysis too.
3. **Neuron-level loadings scatter** (`092_perfquant_weightvector_
   loadings_scatter.png`) -- lowest/median/highest-cosine EXAMPLE SESSION
   per pair, one point per neuron, weight_A vs weight_B. Makes concrete
   what a given cosine number actually looks like: falsealarm_curve vs
   performance_curve's lowest example (MH036, n=1150 units) is a clean
   anti-diagonal cloud; its highest example (AB082, n=302 units) is a
   near-perfect diagonal line (cosine +0.998). **Caveat**: both
   whisker-paired "highest cosine" panels happen to land on the same
   session, MH015 -- which has only 19 units, a regime where cosine
   similarity is inherently noisier/more extreme-prone from sampling
   variance alone; that particular extreme should be read as illustrative
   of what near-perfect alignment looks like, not as this analysis's most
   trustworthy high-cosine example (AB082's n=302 case is more convincing).

| Script | Role |
|---|---|
| `091` | Full-pool (88-session) weight-vector-only compute, no null, haas. |
| `092` | Three figures (summary heatmap, cohort-split distribution, neuron-level loadings scatter) -- pure post-hoc on `091`'s CSV/cache. |
