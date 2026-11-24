# Report: decoding behavioral learning curves from neural activity ("perfquant")

This project is a **split-off** of `ssl-whisker-hitmiss-timeresolved-decoding`
(forked 2026-09-24), containing only the PLS regression / "perfquant" branch
of that project's work: predicting continuous behavioral learning curves
from neural population activity, plus the RRR/LG-AR1 method comparison and
the cross-mouse/within-mouse permutation-null tests built on top of it. The
classification pipeline (hit/miss, modality, performance-state decoding via
`024`/`034`) stays in the source project and is NOT part of this one.

Full detail lives in two companion documents, copied unchanged from the
source project (this file is a higher-level entry point, not a replacement):
- **`METHOD_perfquant_decoding.md`** -- canonical method reference.
- **`RESULTS_RECAP_perfquant.md`** -- chronological results record, 28
  sections, `049`-`088`.

All 147 scripts/figures/CSVs/caches numbered `049`-`088` live in
`exploratory-analyses/`.

## 1. Method, briefly

**Targets**: three per-trial, smoothed behavioral trajectories per session
-- `whisker_curve`, `falsealarm_curve`, `performance_curve`.

**Features**: one feature per neuron (binned firing rate, 5-50ms
post-whisker-stimulus), all units, whole-brain, single-session (no
cross-animal pooling of features).

**Model**: PLS regression, component count chosen once per session via a
5-fold CV sweep + the **1-SE rule** (smallest component count within 1 SE
of the best CV score) -- "PLS+1SE," the project's one working method after
comparing against Ridge/Lasso/ElasticNet/PLS-Forest/dPCA/CEBRA (none beat
it; see `RESULTS_RECAP` §6).

**Cross-validation**: pooled out-of-fold prediction, 2 repeats x 5-fold
`KFold(shuffle=True)`, plain (not stratified -- continuous regression, no
classes to stratify by).

**Null model**: full retrain-under-linear-shift null, 1000 shuffles per
session. Shift X relative to Y by a random 10-50% of session length, refit
EVERYTHING (including the null's own PLS fit) from scratch, and **score
each shuffle against its own shifted target, never the real one** -- this
last point is load-bearing (see Caveats).

**Metrics**: R², Pearson, Spearman, always reported together (R² is
scale-sensitive and catches numerical bugs Pearson/Spearman miss; see
Caveats' `SCALE_FLOOR` item).

**Significance**: per-session, empirical one-sided p-value against the
session's own 1000-shuffle null. Group-level, Wilcoxon signed-rank
(+ one-sample t-test, both always reported) on paired (real - null)
differences across the session/mouse pool.

## 2. Caveats

These are the caveats that matter for interpreting anything in this
project -- most carried forward verbatim from `RESULTS_RECAP_perfquant.md`
§5/§7/§12/§19/§21-23/§25/§27/§28, consolidated here in one place.

1. **The decodable signal is more consistent with shared slow within-session
   drift than a target-specific neural code.** The ±20-trial lag-profile
   diagnostic (`057`) is broad, not sharply peaked at lag=0 -- these curves'
   own autocorrelation extends well beyond 20 trials, so no diagnostic used
   here can cleanly separate genuine same-trial coupling from shared slow
   drift (arousal, satiety, electrode drift). §27's cohort-pooled curves
   visually reinforce this: predicted curves mostly track the real curves'
   slow decline, not fine trial-to-trial structure.
2. **Never trust a null that's remapped onto the real trial axis.** Doing
   so (the `060`/`065` qualitative visualization) inflates the apparent
   null almost to real-model levels in some sessions -- purely a
   long-autocorrelation artifact, not evidence the effect is fake. Every
   actual significance number in this project uses the valid construction
   (score each shuffle against its own shifted target).
3. **Single-animal / within-session only.** Per-neuron features aren't
   comparable across subjects, so this pipeline cannot be pooled across
   animals in one model as-is. Pseudopopulation/multi-session pooling to
   genuinely increase n (rather than work around p >> n per session) is
   flagged repeatedly as the single biggest untried lever, and is not
   implemented anywhere here.
4. **The `SCALE_FLOOR` numerical bug (fixed, but a lesson for reuse):** a
   neuron column can be near-constant within one CV fold's training subset
   purely by chance, causing a vanilla `StandardScaler` to blow up
   (R² ~ -1e28). Invisible to Pearson/Spearman. Fixed with a floored
   scaler (`SCALE_FLOOR=1e-3`) -- always use it if this pipeline is
   extended or copied elsewhere.
5. **Individual-session power is much lower than the group effect.** On a
   dedicated 1000-shuffle-per-session test (12 sessions, best-to-worst),
   only 5-8/36 session x target pairs reached p<0.05 alone, even though
   every group-level test rejects the null overwhelmingly (p<1e-9). Don't
   expect single sessions to individually "prove" the effect.
6. **Cross-mouse "reward-dependence" and "specificity" nulls disagree
   between Wilcoxon and t-test at the per-cohort scale** (§19) -- a signal
   of skewed, outlier-heavy corrected-score distributions, not a clean
   normal effect. §28's cohort-combined re-test is a power move (more n,
   not a different question) and should be read alongside, not instead of,
   the per-cohort split -- the per-cohort split is what shows the R+ and
   R- directions are NOT symmetric (§19).
7. **Cross-mouse specificity narrows to `whisker_curve` only**, confirmed
   across all 3 metrics and both test types (§20), and confirmed again
   after cohort-pooling (§28). Don't generalize a specificity claim to
   `falsealarm_curve`/`performance_curve` -- the evidence there is
   consistently null.
8. **RRR's apparent advantage over PLS+1SE (§23) is an open question, not
   a settled finding.** RRR is far more sensitive to its regularization
   strength than PLS (a hard-learned lesson: the first RRR-vs-PLS run had
   an under-regularized RRR fit that inverted the entire conclusion, §23).
   Whether RRR's corrected advantage reflects genuine architectural benefit
   or an artifact of reusing real-data-optimized hyperparameters on every
   null shuffle (rather than re-selecting per shuffle) was raised but never
   resolved -- explicitly left open in §25.
9. **RRR's raw predictions are well-correlated but poorly scale-calibrated**
   (§23) -- an inherent property of a p>>n ridge setup, not a bug. Pearson
   correlation doesn't require calibration, but don't read RRR's raw
   predicted values as being on the same scale as the true curve without
   the `calibrate_to_target` affine correction applied (§24).
10. **LG-AR1 smoothing does not reliably help, on either decoder, once
    properly null-controlled** (§21/§23/§25) -- it decreased the RRR
    above-null margin in every pilot cell (§21) and helped PLS in only
    2/9 cells on the baseline window (§25). The fitted AR(1) persistence
    is extremely high (ρ=0.97-0.99), meaning the smoother tracks each
    session's whole-session trend, which the null shares almost as much
    as the real data.
11. **No R+/R- cohort difference in decodability was found anywhere this
    project looked** (§4, 27/27 comparisons, only 1 borderline
    non-replicated hit) -- but expert-stage R- (n=6 mice) is underpowered,
    not a genuine null result (§11/§12); effect sizes there are as large as
    significant cells, just short of significance.
12. **§27's cohort-pooled trial-index curves trade away real-trial-order
    fine structure** for a normalized, averaged trajectory -- a
    complementary view to `065`'s per-session panels, not a replacement.

## 3. Report: what this body of work found

**Headline (§1-2)**: yes, session-level behavioral learning curves are
robustly decodable from single-session whole-brain population activity at
the group level. Full 88-session pool, PLS+1SE, sensory window: all 9
target x metric combinations reject the null (Wilcoxon p between 1.4e-9
and 8.1e-14), above-null margins ~0.10-0.17 depending on target/metric.

**Feature window (§3)**: sensory window (5-50ms post-stimulus) is the best
or tied-best window once corrected for chance -- a wider baseline/ITI
window has much higher raw score but an equally inflated null (it's the
most autocorrelated window), so raw-score ranking is misleading without
the correction.

**Cohort (§4, §11, §28)**: no R+/R- difference in decodability at lag 0,
robustly (see Caveat 11 for the one power caveat).

**Method choice (§6)**: PLS+1SE beats or matches Ridge/Lasso/ElasticNet/
PLS-Forest/dPCA/CEBRA on every axis tested (raw score, train/test gap,
robustness) -- adopted as the working method throughout.

**Area-level (§9, §13-14)**: area-level PLS rankings exist but should be
read cautiously -- a neuron-count control (§14) found PLS scores had NOT
saturated in the unit-count range areas typically operate in, unlike the
classification pipeline's own neuron-count control (§13), which did
saturate. Area-level comparisons may still be partly unit-count-limited.

**Within-mouse (§11-12, §15-16)**: 11/12 (t-test) / 10/12 (Wilcoxon)
day_stage x cohort x target cells significant; the one weak cell
(expert R-, n=6 mice) is underpowered, not a real null result.

**Cross-mouse (§18-20, §28)**: a Harris-style permutation design (swap
neural activity or behavioral target between a recipient and a donor
mouse) finds a real but modest, right-skewed **reward-dependence** effect
(stronger once cohort-pooled, §28) and a narrower **specificity** effect
confined to `whisker_curve` (confirmed across all 3 metrics, both test
types, both before and after cohort-pooling). R- mice fail the specificity
test cleanly (§19), which rules out "any two animals differ" as the
explanation for the `whisker_curve` specificity result in R+ mice.

**Method comparison, RRR + LG-AR1 (§21-25)**: RRR (reduced-rank
regression) independently confirms decodability via a completely
different architecture (§21). A properly-regularized, matched comparison
(§23, correcting an earlier under-regularization bug that had inverted
the conclusion in §22) finds RRR outperforms PLS+1SE on the baseline/ITI
window -- but this remains an open question (Caveat 8), and the project's
figures currently ship PLS-only (RRR shelved, §25) pending that
resolution. LG-AR1 smoothing does not reliably help either decoder once
null-controlled (§21/§23/§25).

**Newest additions (§26-30, 2026-09-24)**: a pairwise-line recasting of
the canonical real-vs-null figure (§26, one line per session, makes the
within-session pairing visible); cohort-pooled trial-index-normalized
learning curves replacing per-session panels (§27); a cohort-combined
re-test of the cross-mouse nulls (§28, more power, same qualitative
pattern as the per-cohort split, reward-dependence stronger, specificity
still `whisker_curve`-only); a cross-target specificity test + PLS
weight-vector comparison arguing against pure shared drift (§29, 12-session
pilot); and that weight-vector comparison scaled to the full 88-session
pool (§30, q<1e-9 for all 3 target pairs) -- whisker_curve and
performance_curve share the strongest population axis (cosine +0.71),
falsealarm_curve and performance_curve are significantly anti-correlated
(-0.30), with no visible R+/R- cohort separation in any pair.

## 4. What's genuinely still open

- RRR-vs-PLS: is RRR's advantage real, or a null-side hyperparameter-reuse
  artifact? (Caveat 8) -- the proposed test (re-select R/λ per null
  shuffle) has never been run.
- Pseudopopulation/multi-session pooling to increase effective n -- the
  single most-repeated "biggest untried lever" note in this project,
  never implemented.
- Whether the decodable signal reflects genuine target-specific neural
  coding vs. shared slow within-session drift was **partially addressed**
  2026-09-24 (§29-30): a 12-session pilot fitting each target's own
  single-output PLS decoder found strong evidence AGAINST pure shared
  drift -- own-target prediction beats cross-target transfer in 33/36
  (session x target) rows (Wilcoxon p=3.2e-9). The weight-vector half of
  that finding was then scaled to the full 88-session pool (§30, q<1e-9
  for all 3 pairs): whisker_curve/performance_curve share a strong
  population axis (cosine +0.71), whisker_curve/falsealarm_curve overlap
  moderately (+0.35), and falsealarm_curve/performance_curve are
  significantly ANTI-correlated (-0.30) -- not a single shared drift
  direction. Still not fully resolved: the cross-target SPECIFICITY test
  itself (the null-controlled half) remains at 12-session pilot scale,
  the off-diagonal null reference is an approximation, and Caveat 1
  otherwise still applies to each target's OWN decoding, which could
  still substantially reflect shared slow structure with a
  target-specific shape. See `RESULTS_RECAP_perfquant.md` §29-30 for full
  detail.
