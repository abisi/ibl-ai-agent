## Purpose
Concrete, reusable test recipes for comparing `reward_group` (R+/R-) or any
other **mouse-level** factor on a **unit-level** metric. `SKILL.md`'s Default
analysis policy (items 8-9) and `ssl_analysis_patterns.md`'s R+ vs R- cohort
comparison / PERMANOVA sections already state the governing principles
(mouse-block permutation, always report both parametric and non-parametric);
this file adds the actual formulas/algorithms, since three genuinely
different questions ("does the typical value differ", "does the balance of
positive vs negative units differ", "does the full distribution shape
differ") each need a different test, and getting this wrong silently
mixes them up. Worked, reusable implementation:
`projects/ssl-history-modulation-indices/exploratory-analyses/history_lib.py`
(functions named identically to the sections below) -- copy from there
rather than re-deriving.

## The three questions are not interchangeable
For a unit-level metric (e.g. a per-unit modulation index) and a mouse-level
factor (cohort, `learning_category`, ...):
1. **Location**: does the typical (median/mean) value differ between groups?
2. **Proportion**: does the balance of units above/below some threshold
   (e.g. positive vs negative) differ between groups?
3. **Distribution**: does the *full* distribution shape (spread, skew, tails)
   differ, even absent a location or proportion difference?

These can disagree with each other on the same data -- treat a disagreement
as a finding to report and investigate (see the LMM/mouse-level divergence
note below), not as license to pick whichever test supports the story.

## 1. Location tests
- **Mouse-level** (`mouse_level_test`): one value per mouse (median of that
  mouse's valid units), then **both** Mann-Whitney U and Welch's t between
  groups -- per the R+ vs R- required-test-pair rule. Naturally avoids
  unit-level pseudoreplication (one row per mouse by construction).
- **Neuron-level** (`neuron_level_lmm`): `index ~ C(group)` with a
  mouse-level random intercept (`statsmodels.formula.api.mixedlm(...,
  groups=mouse_id)`), REML fit. Uses the full unit-level N while accounting
  for non-independence of units within a mouse.

**Known failure mode, found 2026-08-21/22 (`ssl-history-modulation-indices`):**
the mouse-level test and the neuron-level LMM can disagree **in direction**,
not just significance. Cause: unit count per mouse can span two orders of
magnitude (tens to >1000 in this dataset); the LMM's fixed effect is an
implicitly N-weighted average across units, so a handful of high-unit-count
mice dominate it, while the mouse-level test weights every mouse equally
regardless of its unit count. When they disagree, **do not report either as
"the" result** -- state both, and treat the mouse-level test as primary
(it's the one that matches the actual replicate structure) unless a random
slope or explicit per-mouse weighting is added to the LMM to fix the
imbalance.

## 2. Proportion test
`mouse_level_proportion_test(df, metric, threshold=0.0)`: per mouse, the
fraction of units with `index > threshold` (default: sign test, fraction
"enhanced" vs "suppressed"), then Mann-Whitney + Welch between groups on
that per-mouse fraction -- same mouse-as-replicate logic as the location
test, applied to a proportion instead. Answers a different question than
location: a metric can have identical medians between groups while one
group has a much more lopsided positive/negative split.

## 3. Distribution tests (Kolmogorov-Smirnov)
- **Neuron-level** (`neuron_level_ks_test`): standard `scipy.stats.ks_2samp`
  on pooled units. **Pseudoreplication-caveated** exactly like a naive
  neuron-level test of a mouse-level factor -- pools units across mice with
  unequal counts, so a "significant" result here alone proves nothing.
- **Mouse-block permutation KS** (`mouse_block_permutation_ks_test`, the
  test to actually cite): observed KS statistic on the real group labels,
  then a null built by permuting the R+/R- (or other factor) label across
  **whole mice** (not units) and recomputing the pooled-unit KS statistic
  each time (2000 permutations is a reasonable default), `p = (#null >=
  observed + 1) / (n_perm + 1)`. Same mouse-block-permutation logic that
  fixed the PERMANOVA pseudoreplication bug in `ssl-reward-history-modulation`
  (see `ssl_analysis_patterns.md`) -- this is that fix generalized from a
  PERMANOVA statistic to a KS statistic; the same construction (permute
  mouse labels, recompute any pooled-unit statistic, rank the observed
  value against the null) works for other distributional statistics too
  (e.g. a variance-ratio or Cramer-von Mises statistic), not just KS.

**Observed pattern worth expecting**: in
`ssl-history-modulation-indices`, the naive neuron-level KS was
"significant" (p<0.001) on data where only 2-9 of 68 mice had any valid
unit at all -- a textbook pseudoreplication artifact. The mouse-block
permutation version correctly returned null (p>0.3) on the same data. Do
not trust a neuron-level KS result without also running the permutation
version, especially when `n_mice` is small.

## Reporting checklist
- State which of the three questions (location / proportion / distribution)
  is being answered -- do not present a distribution-test result as if it
  answered the location question or vice versa.
- For location and proportion: report both the non-parametric and
  parametric mouse-level test.
- For distribution: report both the neuron-level KS (labeled as
  pseudoreplication-caveated) and the mouse-block permutation KS (labeled
  as the primary result), and state `n_mice` per group for the permutation
  test -- a permutation test on 2-3 mice per group is underpowered even
  when it returns a clean p-value, say so explicitly rather than letting a
  clean-looking p-value imply adequate power.
- If mouse-level and neuron-level tests disagree (location or otherwise),
  report the disagreement itself as a result, with the per-mouse
  unit-count-imbalance hypothesis as the default first explanation to check.

## Quality gates
- Reject any unit-level test of a mouse-level factor (cohort, reward_group,
  `learning_category`) that does not either aggregate to one row per mouse
  first, use a `(1|mouse)` mixed-effects term, or use mouse-block
  permutation -- naive unit-level testing/permutation of a mouse-level
  factor is pseudoreplication, full stop.
- Reject a location-only test (mouse-level median/LMM) presented as
  answering whether the "distribution" or "proportion" differs -- these are
  different questions requiring the tests in sections 2-3 above.
- Reject a distributional claim ("the shapes differ") backed only by the
  neuron-level KS test, without the mouse-block permutation version.
- Reject silently picking whichever of the mouse-level vs neuron-level
  location tests happens to agree with a hypothesis when they disagree in
  direction -- report the disagreement per the checklist above.
