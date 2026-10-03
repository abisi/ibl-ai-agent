# Pre-lick convergence analyses (051–071): locked state, 2026-10-03

## Locked analysis set (do not change without a new version tag)
- Pre-lick window: 100 ms before the corrected first lick (start + lick_time − response_window_start).
- Trials: active, perf ≠ 6, warm-up cut (1 trial before first whisker trial kept), A1 disengagement trim.
- Units: KS4 good + mua, mean raw pre-lick rate ≥ 0.1 Hz. Unit of analysis: session.
- References: spontaneous licks (SL; `PRELICK_REF=sl`, headline) and false alarms (FA; `PRELICK_REF=fa`, control).
- Populations: all mice (headline) and learners (day-0 sessions of non-learners dropped; expert sessions kept).
- Nulls: single-neuron ROC = label permutation (both refs); single-session decoders (068/069) and pseudo-populations
  (064) = linear shift of neural activity vs time-ordered labels (both refs).
- Stats: MWU + Welch per contrast; learning × cohort interaction by mouse-level cohort permutation; area ANOVA with
  mouse-level permutation; no correction across panels.
- Main population measure: distance difference Δd = d(WH, ref) − d(WH, AH) (cv squared distances per unit, 057);
  λ / λ_LDA in Fig 3S; decoder readout of choice: chance-corrected probability numerator.
- Figures: 062 → Fig1–5, Fig2S, Fig3S, COSYNE v2/v3 + captions + stats; article PDFs per variant
  (`publication/<pop>/prelick_convergence_<ref>_<pop>.pdf`, generator `build_article.py`).

## Headline results (SL, all mice; learning × cohort interaction, permutation)
- Converging neurons p = 0.02 (learners 0.06); WH-vs-SL selective units p = 0.01.
- Δd p = 0.014 (learners 0.02); d(WH, SL) p = 0.03; orthogonal displacement p = 0.03; along-axis p = 0.39.
- Decoder probability numerator p = 0.015 (learners 0.02); accuracy unchanged (p = 0.84).
- Areas: ANOVA cohort × stage p = 0.048 (perm), × area p = 0.55; no area survives family-wise correction.
- Pseudo-population (M = 2000): R+ increases (p ≤ 0.005), interactions n.s. (prob. transfer 0.07, numerator 0.22).
- FA reference: R+ increase and expert R+ > R− hold; Δd interaction weaker (0.09); RT-matched interaction n.s. (0.61).

## Caveats
1. Reaction time: WH RTs shorten to auditory-like values in R+ only; RT matching possible only with FA, where it removes
   the cohort difference (Δd interaction 0.61). Main open confound.
2. Reference choice changes the strength of the population interaction (SL 0.014 vs FA 0.09); SL timing control (071)
   did not change Δd.
3. Few expert sessions (14 R+ / 13 R−; 11 / 9 mice); interactions tested at mouse level.
4. Stage = day 0 vs later days: confounded with time on task, recording day, probe placement.
5. Toward, not onto: WH moves away from the reference with a component toward AH; d(WH, AH) does not shrink;
   λ / λ_LDA interactions n.s.
6. Unit-level selectivity correlation (Fig 2f) does not change significantly (R+ Δr = +0.10, p = 0.14).
7. Areas under-powered; absence of area specificity is not evidence of uniformity.
8. Converging neurons are mostly lick-responsive: reward expectation / vigour vs modality-general code not separable.
9. Pseudo-population repetitions (bootstraps, permutation B) are pilot values; increase before final runs.
10. No multiple-comparison correction across panels; conclusions rest on agreement across measures.

## Remaining TODOs
- [ ] Within-session remapping test (072): early vs late halves (or pre/post learning trial) on day 0 and expert,
      Δd + λ + decoder numerator, trial-count matched, odd/even null, per-half shift null.
- [ ] Increase pseudo-population repetitions for final runs (064).
- [ ] RT-matched single-session decoders (068/069) and pseudo-population for FA.
- [ ] Decide git home for the analysis scripts (projects/ is gitignored in ibl-ai-agent).
- [ ] COSYNE: choose v3 (Δd heatmap) as final; check learners version.
