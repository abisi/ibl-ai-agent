# Method: single-animal PLS+1SE decoding of behavioral learning curves ("perfquant")

Canonical reference for the decoding pipeline developed in
`exploratory-analyses/049`-`061`. Predicts session-level behavioral
learning trajectories from single-session neural population activity,
within-animal (per-neuron features, not pooled across subjects).

## 1. Targets

Three per-trial, smoothed/interpolated behavioral trajectories per session
(`prep_perfquant_curve_targets` in `ssl_timeresolved_decoding.py`):
- `whisker_curve`
- `falsealarm_curve`
- `performance_curve`

These are session-level, single-animal quantities. They are themselves
smoothly varying across trials (autocorrelated well beyond 20 trials —
see §7), which matters for how the null must be constructed.

## 2. Features

- One feature per neuron: binned firing rate in `SENSORY_WINDOW = (0.005, 0.050)` s
  post whisker-stimulus onset, excluding `DEAD_ZONE = (-0.001, 0.004)` s.
- All units, `whole_brain` (no area restriction) — "one neuron = one feature."
- Trials with any NaN feature are dropped.
- Columns with near-zero whole-session variance (`std <= 1e-6`) are dropped
  (necessary but NOT sufficient — see §8).
- Minimum session inclusion: `MIN_UNITS_PER_AREA = 5` units,
  `MIN_TRIALS_FOR_REGRESSION = 25` trials.

## 3. Preprocessing / standardization

- Input features: z-scored via a **floored** `StandardScaler` (see §8 for why).
- Targets: z-scored using the **session's own mean/std** — valid because
  this is single-animal decoding; a cross-animal version would need to
  skip this step (targets already comparable across subjects) and use
  area-averaged rather than per-neuron features.

## 4. Model: PLS regression with 1-SE-rule component selection

- `sklearn.cross_decomposition.PLSRegression`, jointly predicting all 3
  targets ("three heads") from all neurons in one fit.
- Component count selected ONCE per session via a 5-fold CV sweep over
  candidates `{2, 5, 10, 15, 20, 30}` (capped at `min(30, p-1, 0.6n)`):
  compute mean + standard error of CV R² per candidate, find the best
  (highest-mean) candidate, then pick the **smallest** component count
  whose mean CV score is still within one standard error of the best.
- Rationale: plain "argmax CV score" component selection is itself noisy
  and biased toward whichever candidate got a lucky fold split (seen
  directly: 2 components on one session, 10-30 on another, no real
  difference in achievable signal). The 1-SE rule trades a small amount
  of achievable CV score for a much less overfit, more reproducible model.
- The selected component count is fixed and reused unchanged for the real
  test/train scoring AND every null shuffle (never re-selected per null
  draw — that would let model complexity chase noise in the shuffled data).
- Plain (argmax) PLS was compared directly against PLS+1SE on the full
  88-session pool: statistically indistinguishable on raw test score, but
  PLS+1SE avoids the worst-case overfitting sessions plain PLS produces.
  **PLS+1SE is the method kept going forward**; plain PLS is retired.

## 5. Cross-validation (train vs. test)

- **Test**: pooled cross-validation — `N_REPEATS = 2` independent 5-fold
  `KFold(shuffle=True)` partitions, one out-of-fold prediction per trial
  per repeat, predictions averaged across repeats, then scored once on
  the pooled out-of-fold predictions.
- **Train**: refit on all trials, predict the same trials (in-sample).
  Reported ONLY to quantify the train/test overfitting gap — never
  treated as a real performance number on its own.

## 6. Null model (significance testing)

Full retrain-under-linear-shift null, `N_SHUF_NULL = 1000`:

1. Draw a random shift of 10-50% of session length (`MIN_SHIFT_FRAC,
   MAX_SHIFT_FRAC = 0.1, 0.5`).
2. Shift X relative to Y by that amount (direction chosen at random),
   trimming to the overlapping range — this gives a decorrelated
   `(X_shift, Y_shift)` pairing.
3. Refit and pooled-CV-score **everything from scratch** on this shifted
   pairing, using the SAME fixed n_components as the real model.
4. **Score each shuffle's prediction against its OWN shifted target
   `Y_shift`** — never against the real, correctly-aligned `Y`.

Point 4 is critical and was found the hard way (§7): remapping a
shuffle's predictions back onto the real trial axis and scoring against
the real Y inflates the apparent null score, because both the neural
signal and the behavioral curves are smoothly autocorrelated over many
dozens of trials — a model trained on a decorrelated pairing still tracks
the real curve's slow trend decently at nearby trials. The only valid
null score compares a shuffle's prediction to the target it was actually
trained against.

## 7. Two diagnostics that shaped the design (worth knowing before reusing this)

- **±20-trial systematic lag sweep** (`057`): pairing X[t] with Y[t+lag]
  for lag in [-20, 20] gives a BROAD profile, not a sharp peak at lag=0,
  for both reward cohorts. This means the target curves' own
  autocorrelation length exceeds 20 trials, so a narrow lag sweep alone
  cannot cleanly separate genuine same-trial coupling from shared slow
  drift (arousal, satiety, electrode drift, etc.).
- **Null-band-on-real-axis pitfall** (`060`): visualizing the null as a
  band on the real trial axis (by remapping shifted-shuffle predictions
  back to original indices and scoring against real Y) looked almost as
  good as the real prediction in several sessions — not because the
  effect is fake, but because of the same long-autocorrelation confound.
  This construction is useful ONLY as a qualitative "does the model track
  fine structure" visual; never cite its null-r numbers as a significance
  test. Use §6's construction for any actual significance claim.

## 8. Numerical pitfall: the `SCALE_FLOOR` bug

A neuron column can have fine whole-session variance but be exactly or
near-exactly constant within one particular CV fold's *training subset*,
purely by chance. Vanilla `StandardScaler`'s fitted std for that fold is
then a tiny-but-nonzero float (sklearn's zero-variance guard never
triggers, since it isn't exactly zero), so `1/std` explodes — scaled test
features reaching ~1e17, R² around -1e28. This is invisible to
Pearson/Spearman (scale-invariant) and was only caught because R² is
scale-sensitive. A whole-session variance pre-filter (`std > 1e-6`) does
**not** catch it, since it's a per-fold coincidence, not a whole-column
property.

**Fix**: always use a floored scaler, never a plain `StandardScaler`:

```python
class _FlooredScaler(StandardScaler):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)  # SCALE_FLOOR = 1e-3 (Hz)
        return self
```

**Lesson for metric choice**: always compute R², Pearson, AND Spearman
together. R² is the most useful sanity check (it will scream if something
like this breaks); Pearson/Spearman are the robust, scale/rank-invariant
numbers to actually trust for the size of an effect.

## 9. Statistical reporting conventions

- **Per-session significance**: empirical one-sided p-value,
  `p = (1 + #{null >= real}) / (1 + N_SHUF_NULL)`, computed separately per
  target and per metric (R², Pearson, Spearman).
- **Group-level significance** (across the full session pool): Wilcoxon
  signed-rank test on the paired per-session `(real - mean(null))`
  differences. This is the right tool here — individual-session power is
  often insufficient (only ~15-22% of sessions reach p<0.05 alone) even
  when the group effect is overwhelming (p < 1e-9 on every
  target x metric combination, full 88-session pool).
- **Cohort (R+ / R-) comparisons**: always report both a non-parametric
  (Mann-Whitney U) and a parametric (Welch's t) test.

## 10. Headline full-pool result (88 sessions, PLS+1SE, `058`)

All 9 target x metric combinations reject the null at the group level
(Wilcoxon p between 1.4e-9 and 8.1e-14). Typical above-null margins:
whisker_curve ~0.14-0.17, falsealarm_curve ~0.11-0.13, performance_curve
~0.10-0.12 (R²/Pearson/Spearman units). Train scores sit around 0.75-0.88
vs. test 0.24-0.57 across the three targets — a substantial but expected
gap given p >> n.

No R+/R- cohort difference was found at lag 0 for any target or metric
(Mann-Whitney p = 0.21-0.99, Welch p = 0.40-0.93) — reward contingency
does not detectably change how decodable these curves are from
sensory-window population activity.

## 11. Scope / limitations

- Single-animal / within-session only. Per-neuron features are not
  comparable across subjects, so this pipeline cannot be pooled across
  animals as-is. A cross-animal version exists (`050`, LOSO, area-averaged
  features, no per-session target standardization) but is a separate,
  less powerful design (fewer, coarser features).
- Pseudopopulation/multi-session pooling to actually increase n (rather
  than work around p >> n per session) was explicitly identified as the
  single biggest open lever, and is NOT implemented anywhere in this
  pipeline.
- The ±20-trial lag diagnostic cannot, by itself, rule out slow shared
  drift as a contributor to the effect (see §7) — only the properly
  constructed shift-null (§6) is a valid significance test.

## 12. Canonical scripts

| Script | Role |
|---|---|
| `052_perfquant_singleanimal_multihead_test.py` | Original single-session multi-head Ridge validation; introduced the floored-scaler fix. |
| `056_perfquant_best_recipe.py` | Ridge / PLS+1SE / Stability-Lasso / PLS-Forest comparison on 5 example sessions; motivated the 1-SE rule and ruled out PLS-Forest bagging (no benefit over PLS+1SE). |
| `057_perfquant_pls_lagprofile_fullpool.py` | ±20-trial lag-profile diagnostic + R+/R- cohort comparison, full 88-session pool. |
| **`058_perfquant_pls_nulldist_fullpool.py`** | **Canonical full-pool validation.** PLS+1SE only, pooled-CV test/train, 1000-shuffle null (correct construction), R²/Pearson/Spearman throughout, above-null distributions, Wilcoxon group significance. |
| `059`/`060`/`061` | Example-session inspection: best-to-worst ranking, real-vs-null curve overlays (qualitative only, see §7), per-session 1000-shuffle empirical significance (the valid per-session test). |
