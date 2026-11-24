# change-log

- 2026-11-21: project created; question.md (request, review of tca_pipeline_bis.py, draft refined question) and TODO.md
  (plan, awaiting user consultation on plan / split / compute).
- 2026-11-21: user decisions recorded (pilot then all, expert as reference); TODO steps 001 / 002 pilot done;
  artefact window set to -10..+5 ms (empirical, ssl-s1m2-communication 006).
- 2026-11-21: split corrected in question.md / TODO.md: pilot = exploration set, remaining learning sessions =
  confirmation set (as the user's chosen option states).
- 2026-11-21: added epoch `trial` (-0.1..0.5 s, user) and norm `bsub` (baseline-subtracted, signed CP); user chose
  bsub as the main decomposition for learning (question.md 'Main decomposition').
- 2026-11-21: added 003b_tca_jaw.py (jaw-onset-aligned TCA, variants jaw / jaw_all, from the 002 stimulus tensors,
  jaw onsets from the per-mouse dlc_jaw_onset_times.pkl); pilot launched; 004 handles jaw fits (--epochs).
- 2026-11-21: full run re-queued without bsub x whole (user agreed); run_full_bsub_v2.sh replaces run_full_bsub.sh.
- 2026-11-22: pilot rank selection: bsub fits explain 1-4 % at rank 12 and held-out error exceeds 1 from rank 3-4
  (10-ms bins). User: use 50-ms bins -> epochs *_b50 in 003 and jaw_b50 in 003b (rate over the valid part of bins
  overlapping the artefact). Full run held. AB123 jaw table rewritten 2026-11-21 19:01 by another process (104 -> 103
  lick trials); 10-ms jaw fits started before that.
- 2026-11-22: event-aligned redesign (user): 005_build_aligned.py (single builder from spike times, 50-ms rates,
  per-trial baseline incl. jaw / lick), 006_tca_aligned.py (10 TCA types, rank selection, condition-mean model),
  007_aligned_figures.py (trial markers: type colour, filled lick, open no lick, light passive). Pilot AB123.
- 2026-11-22: user: sensory window 10 ms only; normalisations rate / soft / z-score (min-max dropped); rank selection
  by the Williams et al. 2018 error-vs-similarity trade-off only (held-out folds removed). AB123 rerun (12 types).
- 2026-11-22: no-stim trials dropped from stimulus-aligned tensors (user). haas slowness = own superseded runs +
  BLAS oversubscription (load ~390): old 003/003b runs stopped, BLAS threads capped; 006 split into one job per
  session x type x rank (AB123, ranks 1-10: 6 min). Rank rule (sim >= 0.85, gain >= 5 %) picked rank 10 nearly
  everywhere (no elbow) -> ranks 1-20 rerun.
- 2026-11-22: Kuma benchmark (kuma_bench_gpu.py, job 5031381, L40S, torch cu126): tensorly GPU float32 0.04-0.23 s
  per fit vs tensortools CPU 0.6-20 s (40-90x at rank 10-20), same errors; best solutions agree (similarity
  0.83-1.0; lower at high rank = different restarts / local optima at equal error).
- 2026-11-22: 008_match_test.py (tensorly congruence_coefficient + Hungarian permutation) on AB123, MH032, AB144
  (R+), AB092 (R-), soft, rank 6: time-factor match 0.88-0.94 (stim_long), 0.59-0.97 (jaw); stacked (time + area +
  class profiles) 0.54-0.78; restart reference 0.99. No null yet: time factors are similar broad ramps.
- 2026-11-22: permutation check (AB123 stim_long soft rank 6, tensorly): congruence_coefficient and cp_permute_factors
  recover a known permutation + sign flips + noise in 200/200 draws. On real restart pairs the tensortools rule
  (product of per-mode cosines) and tensorly's stacked congruence agree in 2/6 pairs; disagreements are swaps of
  2-3 similar components (matched product score 0.71-0.83). Time-only matching differs again.
- 2026-11-22 CORRECTION (from the sources): tensortools kruskal_align = MEAN over modes of |cosine|, Hungarian,
  mean matched; tensorly congruence_coefficient with a list of factor matrices (and cp_permute_factors) = PRODUCT
  over modes; with one stacked matrix = |mean of signed cosines|. The previous entry labelled the product rule as
  tensortools' (wrong). Re-run (6 restart pairs): mean (tensortools) vs product (tensorly) permutations agree in 4/6;
  a 5-line scipy re-implementation of the mean rule reproduces kruskal_align exactly (0.8772-0.9342).
- 2026-11-22: GPU port 006g_tca_gpu.py (tensorly float32, Kuma L40S; kuma_tca_array.sh, rclone in/out) validated on
  AB123 vs haas 006 (006v_compare_gpu.py): input identical; 10/12 types pass preset criteria; misses: stim_long soft
  max error diff 0.0021 (> 0.002), stim_sensory soft component similarity 0.888 (< 0.9) at rank 6 (GPU rank 7);
  GPU error +0.0001-0.0004. ~30 s per type on one GPU vs 0.8-3.7 CPU-h. Port accepted.
- 2026-11-22: pilot 6 day-0 mice on Kuma (R+ AB123, MH032, AB144; R- AB092, AB120, MH062), 009_pilot_summary.py:
  chosen ranks 3-9 (mean ~7 for every type); TCA at chosen rank fits better than the condition-mean model in all
  types except the sensory window (soft / zscore ~ equal); R+ curves fall further than R- (n = 3 each, confounds
  unchecked).
- 2026-11-22: user: ranks 1-12 per mouse, comparisons at one fixed rank (tbd); z-score main normalisation. Outliers
  (AB123 trials 162-163: large stimulus-locked auditory responses on 2 probes after long ITI / low baseline, not
  artefacts): z entries clipped at +-10 SD, trials with RMS > 4 robust SD masked out of the fit and given trial
  weights by least squares (0-7 trials per fit in the 6 mice). 007 rewritten (fixed grids, area names in style
  spelling, legends off the data, reconstruction page with removal loss + component congruence, class strip).
- 2026-11-22: CP degeneracy: signed CP on baseline-subtracted z-scores gives cancelling component pairs (congruence
  < -0.5) in most mice from rank 4-6 on (median strongest -0.6 to -0.85 at rank 7, all alignments); AB123 stim_long
  r7: C3/C6 and C4/C7 at -0.8, removal losses ~60 % vs 13.5 % full-model R2. Options pending (ridge, split
  non-negative, lower rank).
- 2026-11-22: 10-SD cap check: touches <= 0.06 % of entries (mostly single 50-ms bins of ~2 Hz units) but 0.5-4.4 %
  of the signal sum of squares; kept, to be checked in a sensitivity run.
- 2026-11-22: degeneracy variants (006g --methods cp splitnn --l2 ..., 010_variant_compare.py; 6 mice, stim_long + jaw,
  ranks 1-12, Kuma job 5031523). Rank 7 medians: signed CP cancelling pairs in 6/6 (stim_long) and 5/6 (jaw) mice;
  ridge 100 removes them in stim_long at no error cost (0.950 vs 0.949) but drops jaw restart similarity to 0.57
  (ridge 10: jaw clean, stim_long 2/6 still cancel; ridge 1000 collapses) -> strength depends on tensor size; split
  non-negative: no cancelling pairs, restart similarity 0.91-0.93, error +0.008-0.009 (AB123 R2 11.7 vs 13.5 %),
  no tuning. 007: shared y per column, aligned labels, data / model matrices side by side, --variant.
- 2026-11-22: capping removed (user); outlier-trial masking kept.
- 2026-11-22: outlier-trial masking removed (user); CLIP_SD / OUTLIER_Z kept as optional settings, both off.
- 2026-11-22: semi-non-negative CP (006g --methods seminn: trial + time >= 0, unit signed; HALS nn_modes [0, 2];
  singular-solve restarts counted, 3 in 12 fits) vs the others, no capping / no outlier masking (Kuma 5031562 +
  5031613). Rank 7 medians: seminn cancelling pairs in 6/6 (stim_long) and 5/6 (jaw) mice, overlap -0.84 / -0.80
  (worse than signed CP -0.66 / -0.64): components with the same time / trial profiles cancel through opposite unit
  weights. Ridge 100: stim_long clean, jaw restart similarity 0.57; ridge 10: jaw clean, stim_long 2/6. Split
  non-negative: 0/6 both, similarity 0.92-0.93, error +0.008-0.009.
- 2026-11-22: split non-negative (006g --methods splitnn) is the main method (user); default in 006g.
- 2026-11-22: methods recap + analysis plan drafted in question.md (awaiting approval before the full run).
- 2026-11-22: 005 pre-event bins for the sensory window (-50..-10 ms, 10 ms bins, artefact gap kept out); full rebuild of all sessions launched (logs/005_all.log). User: rank per cohort after the full fit; all sessions exploratory.
- 2026-11-22: sensory grid shifted by -2.5 ms (user choice). Note: a tool call the user rejected still applied that
  edit, killed the first full rebuild at 12:22 and rebuilt AB123; detected and reported. All 126 sessions rebuilt with
  the shifted grid (tables/aligned_build_log.csv, 126 ok); AB080, AB082, MH026 have no jaw-onset table -> stimulus
  alignments only (005 now skips alignments with no trials).
- 2026-11-22: full run submitted on Kuma (job 5031731, 126 tasks, %24, 1 h walltime, estimate CHF 26.47 upper bound):
  006g --methods splitnn nn, 4 alignments, ranks 1-12, 10 restarts -> artifacts/tca_aligned_gpu/<sid>/.
- 2026-11-22: full run done (Kuma 5031731 l40s + 5032384 mig24gb after l40s priority stall): 977 fits ok; no jaw / lick fits for 7 sessions (AB080, AB082, MH026 day 0: no jaw table; MH064 0114 / 0117, MH065 0114 / 0115 expert: absent from their jaw tables); their stale 0-trial jaw / lick files deleted.
- 2026-11-22: analyses on the full run: 011 (rank per cohort; saturates for split jaw / lick), 012 (9,260 components;
  descriptors, area-label null, shift nulls; learn_ctrl = learning beyond trial outcome added), 013 (Pearson
  similarity, within-session 95th percentile cut, cohort-bias permutation, area coverage). Outcome control removes
  most learning effects (split stim_long R+ 11 -> 4, R- 8 -> 1). Interim summary in report/.
- 2026-11-22: 014_method_agreement.py: stim_long cl. 67 and 56 replicate in the reference fit; sensory cl. 68 / 67 direction only; R--specific cl. 61 and jaw / lick cohort biases do not.
- 2026-11-22: 015_candidates_figure.py (candidate learning components, main vs reference fit). First article report
  (report/build_report.py -> combined_results_ks4/ssl-tca-session-wide/report/: report.pdf / .md / .html, numbers.json,
  registry.csv, summary.md; statistics frozen in report/_snapshots/2026-11-22). 4 main figures (example session, rank,
  stim_long clusters, candidates), 4 supplementary (example reference fit, variant pilot, reference clusters, sensory
  clusters). Not yet published to the private reports repo (awaiting user).
- 2026-11-22: 016_candidate_details.py: detail page per candidate cluster and its reference match
  (figures/aligned_gpu/candidates/). Finding: within active whisker hits the candidate weights do not rise over the
  session (panel D); cl. 67 / 56 are largest on rewarded-lick trials (auditory hits, R+ whisker hits) and scale with
  lick latency in R+ -> lick / reward responses whose outcome-controlled "learning" correlation follows non-monotonic
  session shape (engagement / end-of-session decline), not a learning rise. Sensory cl. 68 is mostly auditory-evoked;
  its weights rise over the R+ session (the negative partial correlation is a detrending effect). Report wording to
  revise.
- 2026-11-22: 017_trend_contrasts.py (user: split by outcome instead of regressing it out; whisker hits vs auditory
  hits; R- as contrast). Per member: slope of normalised trial weight over the active block for whisker hits / misses
  / auditory hits (trend kept, no shift null); d = slope(W hit) - slope(A hit); candidate = R+ d != 0 and R+ vs R- d
  (MW) p < 0.05. Result: no stim_long candidate (old cl. 67 / 56 / sensory 68 all n.s.); sensory cl. 1 (Ss-whisker,
  peak 12.5 ms, area-specific) rises over the R+ session on whisker hits (+0.61) and misses, flat / falling in R-
  (MW p = 0.005, Welch 0.025), replicated by reference cl. 34 (MW 0.005, Welch 0.117). Sensory cl. 59 (variability,
  6 R- mice) not pursued. tables/trend_members.parquet, trend_clusters.csv; figures trend_overview, trend_candidates_*.
- 2026-11-22: artefact correction moved before the tensor (user). Check of the valid-time tensors (126 sessions): no
  residual artefact in the mean (0.93 of pre-stimulus in the two 2.5-ms-valid sensory bins), but trial-to-trial
  variance x3.5 and input RMS x1.9 there (long window -50..0 ms bin: x1.18); 28 % / 22 % of split sensory components
  peaked in those bins (-7.5 / +2.5 ms), 13 % of split stim_long at -25 ms. 005 now fills the removed window
  (-10..+5 ms) with Poisson surrogate spikes at the unit's baseline rate of that trial, before binning (FILL; seeded per
  session); all bins full width. AB123 check: corrected bins mean 0.98 / 1.03, variance 0.96 / 1.03 of pre-stimulus.
  Old tensors / fits / tables archived: artifacts/aligned_validtime_20261122, artifacts/tca_aligned_gpu_validtime_20261122,
  tables/validtime_20261122 (Kuma: data_ / out_validtime_20261122). Full rebuild (126 ok) and Kuma refit job 5034152
  (l40s,mig24gb; splitnn + nn); 011-017 to rerun; report and interim results based on valid-time tensors superseded.
- 2026-11-22: refit on filled tensors done (Kuma 5034152, 980 fits); 011-014, 017 rerun. Corrected-bin peaks: split
  sensory -7.5 / +2.5 ms 6 % / 7 % of components (was 28 / 22 %), stim_long -25 ms 1.5 % (was 13 %). Ranks (R+ / R-,
  split): stim_long 8 / 9, sensory 8 / 8, jaw 12 / 12, lick 12 / 11; reference 7 / 8, 8 / 10, 9 / 8, 10 / 12.
  017 fixed: clusters can hold several components of one session (up to 4; 013 average-linkage does not prevent it),
  slopes now averaged per mouse before the tests. Only trend candidate per fit: stim_sensory split cl. 10 (Ss-whisker,
  peak 22.5 ms, area-specific) <-> reference cl. 40: R+ whisker hits +0.33 (p = 0.015) and misses +0.26 (p < 0.001),
  auditory hits flat; R- whisker hits -0.46 (p = 0.074), misses flat; d R+ +0.33 (p = 0.026), R- -0.46 (p = 0.043);
  R+ vs R- MW p = 0.005, Welch p = 0.076 (reference MW 0.001, Welch 0.064). No stim_long candidate. 016 rerun for it.

## 2026-11-23: candidate cl. 10 deep dive (018) + candidate report
- 018_candidate_deep.py (haas): per session (90 day-0 + 36 expert) member score in SD units of the active whisker
  trials (mean-normalisation was unstable for weak members / PCA), area-restricted projections, single-unit evoked
  slopes, per-area fifths, OLS trial factors (model 1: position, hit, previous reward, pre-stim population rate, log ITI;
  model 2: + P(lick|whisker), P(lick|auditory)), other methods (passive-pre template, best whisker PC, cross-validated
  whisker - auditory axis), expert matching (mean 013 similarity >= 10th pct of members' LOO; 8 % false positives).
  Run for main cl. 10 and reference cl. 40. Outputs tables/deep_{splitnn_cl10,nn_cl40}_*, figures candidates/deep_*.
- Findings: SD-unit d R+ +0.44 (p 0.003) vs R- -0.46, MW 0.002 / Welch 0.004 (ref MW < 0.001 / Welch 0.004);
  position coefficient survives model 2 in R+; pre-stim rate lowers the component in both cohorts; trend difference in
  Ss-whisker (projection and single units without TCA); template finds only the R- decrease, PCA / stim axis nothing;
  expert R+ sessions still show the within-session rise (hits p 0.035, misses 0.005) -> reward-contingency
  within-session effect, not yet day-0 learning.
- report/build_candidate_report.py -> report/candidate_sensory_cl10/ (report.pdf/.md/.html, 6 figures + 5 supp.,
  4 tables + 2 supp., 11 equations). Built 2026-11-23, 0 undefined refs. Not published.

## 2026-11-23: active-specific components (019)
- 019_active_specific.py (haas): per member component (recurring clusters, stim alignments, both fits), whisker-miss
  context steps in SD units: pre step = active 1st half - passive pre, post step = active 2nd half - passive post (misses:
  no lick / reward); same for hits and auditory. Active-specific = both pooled steps > 0, Wilcoxon p < 0.05.
- 9 / 35 main-fit and 9 / 29 reference clusters active-specific. The strongest (sensory 29, 30, 32; long 61, 16) are
  variability components (flat time factor, directionality ~0, 10-16 mice): a state / noise-magnitude signal, higher in
  the active block with an inverted-U over it, alike on auditory trials. Evoked ones: long 82 (thalamus, 875 ms, mostly
  hits / auditory = lick), 79 (thalamus 225 ms), sensory 66 (striatum 92 ms). None has a whisker-specific trend or an
  R+ / R- difference (017 d n.s.); all have an auditory post step too -> general task context, not whisker learning.
- Candidate cl. 10 is not active-specific: passive pre > early active (pre step -0.20, p 0.005), post step ~0 for
  misses (p 0.47), no cohort difference in the steps.

## 2026-11-23: passive post vs pre (020)
- 020_passive_pre_post.py (haas): (A) TCA clusters, (post - pre whisker) - (post - pre auditory) weight; (B) units
  without TCA, evoked z in 10-45 / 50-250 / 250-1000 ms, same contrast, per area group, all / whisker-responsive units;
  (C) population per area group: cross-validated post-pre distance (W - A) and gain index along the pre response. All
  passive trials, jaw movements kept (user 2026-11-23: do not remove jaw movements for now); jaw-movement rate reported.
- Result: whisker-specific post-pre change is a DECREASE in R- (brain-wide responsive units, middle window: contrast
  -0.29 z, p < 0.001; midbrain, motor, frontal, hippocampus, striatum), R+ ~0; R+ vs R- MW 0.01-0.03 across windows,
  Welch similar. Expert sessions: no R- decrease (n = 9 mice). TCA: stim_long thalamus 875 ms (split 82 / ref 50) and
  striatal clusters (split stim_long 67, 68; sensory 68) with the R- decrease. No R+ increase anywhere. Population
  distance / gain: few, inconsistent differences. Jaw movements post-pre: whisker equal across cohorts; auditory more in
  R- (MW p < 0.001) -> part of the auditory term in R-.

## 2026-11-23: go-signal scan (021)
- 021_go_signal.py (haas): five criteria per recurring stim_long cluster (both fits, day 0): (1) hit > still miss for
  whisker (both cohorts) and auditory (AUC > 0.6, few auditory misses); (2) hit > miss with jaw movement and passive
  jaw-movement trials not raised (AUC within modality < 0.6); (3) present in R-; (4) onset before median jaw onset and
  pre-movement AUC (bins before each trial's jaw onset) above a unit-shuffle null; (5) weight predicts shorter lick
  latency. Jaw-aligned clusters summarised (pre-movement share, false-alarm / hit weight).
- No cluster meets all five. Best go-like: striatal cl. 67 (main; ref cl. 40), peak 275 ms, onset ~125 ms vs jaw
  ~150 ms: hit vs still miss AUC W 0.91 / 0.81 (R+ / R-), A 0.98; hit vs moving miss 0.82; latency rho -0.25; but
  passive jaw movements raise it (AUC 0.64 within modality) and the pre-movement excess over shuffled units is small
  (+0.03; 50-ms bins). Thalamic cl. 82 (875 ms) = licking after movement onset (onset 375 ms, no latency relation, no
  passive-movement response). Early striatal clusters 68 / 73 are auditory-evoked. Jaw-aligned components entirely
  before jaw onset are modality-specific and near 0 on false alarms (stimulus-driven, not a general go).
- Fix during the run: passive classes split by modality (passive auditory trials have more jaw movements).

## 2026-11-23: go-signal tensors + exploratory gallery (005, 006g, 011-013 _go, 022)
- 005: --align option (partial rebuild); new alignments stim_long_ns (stim_long + no-stim trials) and jaw_all (jaw
  onset +-0.3 s, 10-ms bins, all trials; lick / move at real onset, still at type-matched median lick onset;
  jaw_class column). Built for 126 sessions (AB080 / AB082 / MH026 lack jaw onsets -> excluded from jaw_all analyses).
- 006g: ALIGNMENTS extended (appended, seeds of old alignments unchanged); Kuma job 5038229, 504 fits, 0 errors.
- 011-013: TCA_ALIGNMENTS / TCA_SUFFIX env options -> rank_by_cohort_go.csv, components_go.parquet, clusters_go.csv,
  cluster_members_go.parquet (original tables untouched). 012: no-stim classes added.
- 022_go_signal_v2.py: exploratory per user (no pass / fail): per cluster AUC lick vs still / vs jaw move without
  lick, passive move vs still, FA vs CR, FA / hit, onset, pre-movement AUC (lick vs move) over unit-shuffle null,
  latency rho (support only); galleries figures/aligned_gpu/go2/go2_gallery_<al>_<fit>_p*.png.
- First look: jaw_all main cl. 69 (ref cl. 49; 83 / 82 mice): brain-wide, rises from ~-100 ms before jaw onset, peaks
  ~+100 ms; lick vs move AUC 0.90, FA vs CR 0.90, FA / hit 0.58, passive move 0.66, R+ = R-; pre-movement excess +0.03.
  cl. 73 (42 mice): peaks at jaw onset, similar profile, passive move 0.75. cl. 79 / 115 = auditory / whisker evoked
  responses before the (pseudo-)onset.

## 2026-11-23: go component and learning (023)
- 023_go_learning.py (haas): jaw_all main cl. 69 / 73, reference cl. 49 / 48; per session member weights (SD of active
  licks, 0 = active still); fifths per class (W lick, A lick, FA, move), slopes, d = slope W - slope A, OLS on licks
  (position x whisker, latency controlled), W / A lick ratio first vs last third, performance bins, model-free
  projection time courses, expert matching (013 similarity, 10th pct threshold).
- cl. 69 (ref 49 alike): R+ day 0 d +0.41 (p 0.003), R- -0.23 (n.s.), MW 0.011 / Welch 0.014; position x whisker
  +0.10 (p 0.003), R+ vs R- 0.014 / 0.026. Auditory-lick weight declines in R+ (slope -0.45), whisker licks stay ->
  W / A ratio 0.77 -> 0.88 (p 0.028), expert R+ ~1.0; R- 0.57 -> 0.50, expert 0.47. W-lick weight rises with
  P(lick | whisker) (R+ 1.2 -> 1.85 SD). FA also stays relative to A (d_FA +0.47). Expert R+: W and A licks both decline
  within session (d ~ 0). cl. 73 / 48: no clear effect.

## 2026-11-23: go-component report; project ON HOLD
- report/build_go_report.py -> report/candidate_go_cl69/ (report.pdf / .md / .html, 2 figures + 3 supp., 3 tables +
  2 supp., 11 equations; 0 undefined references). Not published.
- Project put on hold by the user; resume points in TODO.md ("ON HOLD").
