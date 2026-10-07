# TODO

Steps performed in `ssl-rastermap-psth-variants` (2026-09-29 → 2026-10-03) before the split; outputs in
`combined_results_ks4/_roc_prelick{,_sl}/` (see the README there).
- [X] 051–056 pre-lick ROC (WH/AH/ref), variants (all, short/long RT, RT-matched), min FR 0.1 Hz, stage × area
      summaries, PSTHs.
- [X] 057 λ and cross-validated distances; 058/059 convergence summary and single-neuron transfer; 060 λ_LDA.
- [X] 061 learners-only rule (bad mice lose only their day-0 session).
- [X] 063 decoder sweep; 068/069 single-session decoders with linear-shift null (both references).
- [X] 064 pseudo-population hierarchical bootstrap (mice → sessions → neurons → trials), shift null; 2026-10-03 rerun
      with transfer_bin and num_prob readouts (SL first; FA running).
- [X] 065–067 converging-unit locations, area attrition, CCF density maps; 070 recap; 071 SL timing control.
- [X] 062 publication figures (distance-based Fig 3, Fig 3S λ/λ_LDA, Fig 1h, Fig 2f/g, Fig 2S, Fig 5 pseudo-pop rows,
      COSYNE v2/v3), complete captions, article PDFs per variant (build_article.py + render.sh).
- [X] Split into this project; LOCKED.md (locked set, results, caveats, TODOs).

Planned
- [X] Within-session remapping (072, 2026-10-03; SL reference; moved to project ssl-within-day-remapping as 001): halves split at the median auditory hit (equal AH per
      half; session end trimmed by A1), count-matched subsamples (20), Δd / λ / components per half, whole-session CV
      decoder evaluated per half and cross-half decoder (train one half, read out the other), linear-shift chance,
      odd/even null split, controls (d(AH,SL), event rates, RT, raw rates). Output `_roc_prelick_sl/within_session/`
      (within_session_sessions.csv, <pop>/within_session_tests.csv, within_session_{time,oddeven,controls}.png).
- [ ] Increase pseudo-population repetitions for final runs.
- [ ] RT-matched single-session decoders and pseudo-populations (FA reference).
- [ ] Final COSYNE figure choice (v3) and abstract text.

## Next iteration (decided 2026-10-07)
Spontaneous-lick reference (072/073 example sessions, figures in combined_results_ks4/ssl-prelick-convergence/sl_definitions/):
- [ ] New SL definition "C8" in 051 (replaces the current one): lick train = piezo licks + corrected first lick of every
      trial (piezo detection misses many false-alarm licks), licks < 50 ms apart collapsed; SL = first lick after >= 1 s
      without licking (bout onset, as now); false-alarm licks (no-stim response windows) COUNT as SL (user: same
      behaviour, measured in a window); exclude SL in [stim - 0.5 s, stim + W] of every whisker / auditory trial, hit or
      miss (stimulus processing, reward collection).
- [ ] Set W from the data first: duration of post-hit licking (reward collection) across sessions; default W = 8 s.
- [ ] Rerun 051 (SL reference, new tag) and everything downstream (057/058/059/060/061/062, 063/064/068/069, 065-067,
      070, 071) and ssl-within-day-remapping; side-by-side comparison of the headline statistics old vs C8; then update
      LOCKED.md with the user's choice.
- [ ] 071 timing control becomes redundant for exclusions <= W; keep only longer windows.
- [ ] Upstream (NWB conversion, behaviour_analysis): false-alarm / weak licks missing from piezo_lick_times (TODO in
      spontaneous_licks_utils.py); spontaneous_licks_utils exclusion only around hits lets false-alarm-window and
      post-miss licks in, and its defaults lack trial_exclusion_window_s.
Harmonisation with ssl-within-day-remapping (see the method map in README):
- [ ] One minimum of events per class across analyses (now 3 in 051 ROC, 4 in 057 / 068 / within-day 001-002, 6 in 064).
- [ ] One implementation each of the decoder + linear-shift null and of the mouse-level cohort-permutation interaction
      (now 068 cv_bacc / within-day 001 ws_predict, and 062 group_stats / within-day 001 perm_interaction).
- [ ] Migrate results to combined_results_ks4/<slug>/ (now _roc_prelick{,_sl}/ and _within_day_sl/) and the report to
      the project-report standard (report_lib).


# Part II: within days (formerly ssl-within-day-remapping/TODO.md)


- [X] 001 halves analysis (moved from ssl-prelick-convergence 072, 2026-10-03): `within_day/sl/halves/`
      (within_session_sessions.csv; <pop>/within_session_tests.csv; within_session_{time,oddeven,controls}.png).
- [X] 002 trial-level slopes on a fixed session axis with expert comparison (2026-10-03): `within_day/sl/slopes/`
      (trial_slopes_sessions.csv, trial_slopes_trajectories.csv, <pop>/trial_slopes_tests.csv, <pop>/trial_slopes.png).
      First run had per-fold normalisation blowing up; fixed: unit axis per fold, one session normalisation, d′ ≥ 0.3.
- [X] 003 within-day vs across-day on a common footing (2026-10-03): phase-matched halves (split at median AH) in every
      session, fixed 6 events per class and 150 units, session-anchored measures (WH − SL position on the session axis,
      whole-session decoder with shift null read out per epoch, Δd / session d(AH, SL)); contrasts within-day,
      across-day (early, late), carry-over; hierarchical bootstrap (mice → sessions), mouse-level cohort permutation,
      MixedLM. Output `within_day/sl/epochs/` (epoch_sessions.csv, <pop>/epoch_contrasts.csv, <pop>/epoch_comparison.png).
      Only 5 R− expert sessions (3 mice) pass 6 events per class per half.
- [X] 003 with 4 events per class, both cohorts (`--n-fix 4`, output `within_day/sl/epochs_n4/`): R− experts 7 sessions /
      5 mice (vs 5 / 3 with 6 events).
- [X] 004 per-half first-lick PSTHs (spikes re-binned per session half, 051 events): `within_day/sl/psth_halves/`.
- [X] 005 COSYNE 5-panel summary "R− whisker-hit activity changes more within the learning day than within expert
      sessions" (PSTHs per half, whisker-hit rate, trial-level WH − SL trajectory, per-half normalised distance,
      |within-session change| day 0 vs expert): `within_day/sl/cosyne/COSYNE_within_day.{png,pdf,svg}` + stats csv.
- [X] 006 compact COSYNE timeline, 007 caption compiler, 008 expanded COSYNE figure (methods with examples, results,
      controls), coding-direction (CD) wording and equations in captions (2026-10-04).
- [X] 009 single-trial mixed model (CD projection ~ whisker hit × time × cohort; per-session random intercept, time slope
      and whisker-hit offset; cohort shuffles across mice): `within_day/sl/mixed_model/`.
- [X] 003 decoder schemes (`--n-units`, `--unit-draws`): 150 × 10, all units, 50 / 100 × 10, 400 × 5; 010 comparison figure.
- [ ] Learning-aligned version (trials since each mouse's learning trial).
- [ ] Robustness splits for 001: session midpoint (trial index) and median whisker hit.
- [ ] Drift control: all classes drift up the SL → AH axis within sessions (all groups); model the drift explicitly
      (e.g. regress WH on time with SL-trial scores as a time-varying reference).
- [ ] Per-mouse saturating fit on day 0 (when does R− WH reach SL?) against the behavioural drop in whisker licking.
- [ ] Report.

## Next iteration (decided 2026-10-07)
- [ ] Spontaneous-lick reference "C8" (false-alarm licks counted, exclusion [stim - 0.5 s, stim + W] after every
      whisker / auditory trial, trial licks merged into the lick train): inherited from ssl-prelick-convergence 051 --
      rerun 001-010 after the 051 rerun; see ssl-prelick-convergence/TODO.md "Next iteration".
- [ ] Harmonise with ssl-prelick-convergence: event minima, one decoder / shift-null / interaction implementation.

## Review decisions (user, 2026-10-07, item by item)
Order: figures first on the current results, then the rerun, then the merged report.

Data and settings (rerun of 051 + everything downstream, both parts):
- [ ] 1. SL reference C8 replaces the current one (FA licks counted; exclusion [stim - 0.5 s, stim + 8 s] after every
      whisker / auditory trial; trial first licks merged into the lick train, 50 ms collapse; bout onset >= 1 s).
- [ ] 2. W = 8 s fixed.
- [ ] 3. Units: stable good + mua as main (coverage + presence + drift joint test), all good + mua as a supplementary
      control.
- [ ] 4. Minimum events per class: 4 in every analysis (now 3 in 051 ROC, 6 in 064).
- [ ] 5. One shared implementation of the decoder + linear-shift null and of the mouse-level cohort-permutation
      interaction (068 / 062 / within_day 001 duplicates removed).
- [X] 6. LOCKED.md untracked from the public fork (kept locally / NAS; .gitignore).
- [ ] Geometry normalisation: diagonal crossnobis (each unit divided by its pooled within-class noise SD, equal weight
      per class) as main; raw spikes/s geometry as a supplement. Decoders keep training-fold z-scoring.
- [ ] Axis inclusion rule (lambda / CD): to decide after seeing results (options: axis length >= 0.01 / unit, held-out
      d' >= 0.3, cross-validated axis length above its shift / permutation null).

Single neurons (Figure 2):
- [ ] 7. 2a: square panels (same example-selection rule).
- [ ] 8. 2b: add the fraction of WH-vs-AH selective units per cohort x stage (051 ROC wh_vs_aud_hit_prelick).
- [ ] 9. Converging neurons / AH-likeness: no change (cross-validated versions skipped).
- [ ] 10. One row: shared-code schematic | Spearman r per session | selectivity scatter R+ | R-.
- [ ] 11. 2f: no change (no mixed model).
- [ ] 12. New supplement: change in |selectivity| (equivalents of 2b-e), all units and split by sign, plus fraction
      significant per sign.
- [ ] 13. Time-resolved pre-lick ROC: skipped.

Population geometry (Figure 3 and supplements):
- [ ] 14. Linear-shift correction for geometry (delta d, lambda, angles) as a control supplement (raw stays main).
- [ ] 15. Supplement: angles (cross-validated cos theta between WH - SL and AH - SL), length ratio |WH - SL| / |AH - SL|
      and lambda side by side (lambda = ratio x cos theta); sessions with a too-short axis excluded.
- [ ] 16. Supplement: quiet windows [-200, -100] ms before every trial start (raw rates, no baseline: the baseline /
      spontaneous activity itself) placed on the readouts (distance, lambda, lambda_LDA).
- [ ] 17. Clarity: 3b axes (real data: along SL -> AH, orthogonal; units sqrt(squared distance per unit)); 3i relabelled
      cross-validated dot product (WH - SL).(AH - SL) per unit; 3j orthogonal part; 3S h "orthogonal PC1" replaced by the
      WH off-axis direction; definitions in captions and report.
- [ ] 18. Coding direction: apply BOTH estimators in BOTH parts for comparison -- (a) unified split-half estimator (trial
      scores with the session normalisation D, whose WH session mean equals lambda), (b) the current 5-fold CD; agreement
      supplement.

Areas and decoders:
- [ ] 19. 4d / 4e: same y-label (rate - baseline, spikes/s); averaging (over neurons / over sessions) in the titles;
      note SL pre-lick vs hit pre-trial baselines.
- [ ] 20. 5a: raw balanced accuracy with the per-session shift-null chance (median, 95 % band) first; current 5a -> 5b.

Cross-cutting:
- [ ] 21. Supplement: per-session quantities vs session performance (whisker hit rate, FA rate, d'), by cohort, analysis
      and stage (scatter + fit + 95 % CI, Spearman).

Report (after figure review): one merged report on report_lib -- Part I across days, Part II within the learning day and
within expert sessions; math definitions with short derivations (squared distance per unit, crossnobis normalisation,
lambda and its split-half estimator, along / orthogonal decomposition, cos theta and length ratio, AH-likeness, LDA d',
decoder readouts, linear-shift chance); every figure and table linked; sample-size tables.
