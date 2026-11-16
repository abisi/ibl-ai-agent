# Change log

- 2026-10-23 Project created (user): reactivation decoding as its own project; design decisions of 2026-10-16 / 10-18 recorded in question.md.
- 2026-10-23 Pilot on MH030 (whisker): all units per group, per-trace robust normalisation, s.d. floor 0.2, sensitivity-based threshold (90 % of held-out quiet stimuli); clean vs spont templates compared (003). The spont template separates whisker stimuli from spontaneous activity far better (Ss-whisker pAUC 0.97); the 90 % threshold produced many detections (8/min Ss-whisker spont, 79/min clean).
- 2026-10-24 Threshold (user): no longer sensitivity-based and no longer per session; one threshold per area group, the same for every session and both templates. Other-modality (auditory) stimuli crossing are reported, not penalised. 001 is now threshold-free (fit_summary.csv, peaks, null curves); apply_thresholds() / --apply-only derive validation.csv and events.parquet. 004 picks the group threshold on the pooled pilot: smallest value with time-shift null <= 0.01 events/min in every session and template. Fix: fold-model peri-event readouts normalised by each fold model's own continuous median / MAD (were the final model's). Launched 8 pilot sessions x {clean, spont}, whisker (haas ~/react_pilot_all.sh).
- 2026-10-24 Pilot fits done (8 sessions x 2 templates, whisker). 004 rule refined: only decoders with pAUC >= 0.8 and no artefact set a group's threshold (a decoder without a template has a heavy-tailed readout, e.g. MH035 Ss-whisker clean: 21 null events/min at threshold 9.65). Thresholds: Ss-whisker 9.65 (12 eligible decoders), Midbrain 7.95 (14), Thalamus 8.5 (5), Striatum 8.4 (2), Motor and Auditory: no eligible decoder (fallback over all decoders: 22.0, 8.0; not meaningful).
  Artefact: MH070 ALM (Motor) whisker template pAUC 1.00 is stimulus artefact (periodic spikes -5..+8 ms after whisker onsets only, past the -10..+5 ms replacement); excluded from the threshold choice.
  AB150_20241219_144646 (day 0 R+): no whisker response in Ss-whisker, Midbrain or Motor (1-ms PSTH flat) while auditory responses are present; no whisker template in any group (pAUC ~0.5). Pending user decision: drop / replace.
- 2026-10-24 Rewind (user): decoders trained on stimulus data only (clean template), threshold per decoder from its own held-out stimulus data, applied to the continuous readout; AB150 kept. 005_stim_threshold_examples.py (no refit; uses the 001 clean outputs): threshold = 95th percentile of held-out no-stimulus decision values (held-out FPR 5 %), plus a stricter alternative (max of held-out no-stimulus values, FPR 0). Summary stim_threshold_summary.csv; example figures fig7_examples_<session> (2 strongest, 2 median, 2 near-threshold events per group, different epochs). At 5 % held-out FPR: sensitivity ~1 for the good decoders but 40-300 events / min, mostly at the time-shift null level; at the strict threshold Ss-whisker 7-45 / min (null 0-5). Strongest events: synchronous volleys in the positive-weight units, sometimes coincident across groups (AB136 task, MH036 t ~2198 s); median and near-threshold events look like noise.
- 2026-10-29 Threshold per decoder from the continuous data (user, after the literature check; Sugden et al. 2020): smallest decision value at which the mean time-shift null rate is <= 5 % of the data rate (there and above). 006_fdr_threshold_heldout_controls.py refits the clean template per session x group for whisker and auditory (auditory = specificity control; previously only the whisker decoder's crossings on auditory stimuli were reported, 001 --target auditory was never run) and for four training sets: all, passive pre held out, passive post held out, both held out. Held-out passive stimuli tested for pre, post and pre + post: direct detection (window ending 10-100 ms after onset above threshold) and injection of the trial's evoked increment into random quiet continuous windows. Content controls: shuffled template (weights permuted over units) at the decoder's threshold, cell-identity permutation test per event, whisker-auditory and cross-area coincidence. Continuous windows and valid / quiet time unchanged. Figures fig8_examples_<session>_<target>, fig9_fdr_summary.
- 2026-10-29 006 results (8 sessions): held-out passive whisker stimuli detected well by the good whisker decoders (median direct / injected: pre 0.95 / 0.89, post 0.82 / 0.77, pre+post 0.84 / 0.78; shams <= 0.1); Ss-whisker decoders whisker-specific (injected auditory 0.02-0.06, except MH036 0.48), Midbrain / Thalamus decoders also cross on auditory stimuli. But the time-shift rule is met at ~3.5 MAD (median 34 events / min), shuffled templates give as many events (median ratio 1.07), only 5 % of events have identity p < 0.05, and 23 % of whisker events coincide with an auditory event (chance 1.7 %): the events are mostly generic population bursts.
- 2026-10-29 User: threshold per decoder against the shuffled template (null <= 5 % of the data rate) plus a cell-identity gate per event (p < 0.05), and leave-one-unit-out participation for all units (categories afterwards: evoked whisker response, weight sign, waveform type, area). 007_content_threshold_participation.py; 006 kept as the record of the time-shift rule (refactor: fit_all(), fig_examples(sub_dir, rule, fig_name)).
- 2026-10-29 007 shuffled template: free weight permutation is confounded (sparse units get large z on a single spike, heavy-tailed shuffled readouts); weights are permuted within 10 firing-rate strata (deciles of the training mean), for both the shuffled template and the identity test. Shuffled readouts normalised by their own median / MAD.
- 2026-10-29 User: first 30 s of each recording excluded from valid time (006 prepare(), START_EXCL); every rate = events in valid time / valid minutes. 007 saves curves.parquet (real vs shuffled rate per threshold); new fig12_real_vs_shuffled.
- 2026-10-29 007 results (8 sessions): real / shuffled ratio ~1 at low thresholds for nearly all decoders. Rule reached by 4/17 good whisker decoders (median 11.5 MAD, kept 0.03 / min; AB136 Ss-whisker 0.43 / min) and 12/24 good auditory decoders (median 12.8 MAD, kept 0.08 / min; auditory decoders, mostly Midbrain / Thalamus, exceed the shuffled template more often). At these thresholds held-out passive stimuli almost never cross (good whisker decoders: median direct crossing 0 for pre, post and pre+post; auditory 0.02-0.08): the rule leaves no sensitivity, so absence of events is not evidence of absence. Leave-one-out participation higher for whisker-responsive units (Ss-whisker median 0.85 vs 0.30), based on few events.
- 2026-10-29 Sync check (8 sessions): piezo licks vs corrected trial licks agree within tens of ms (median -49..+11 ms); lick-rate x jaw-speed cross-correlation peaks within +-50 ms in 6 sessions; MH030 (r 0.02) and MH036 (lag 0.4 s) fail, their jaw traces are not shown.
- 2026-10-29 User: threshold too strict; implement the sensitivity-matched threshold, continuous-detection figures with lick rate (piezo), whisking and jaw below the readout, cohort comparison figures. 008_sensitivity_threshold_cohorts.py: task-trained decoders (passive pre + post held out), whisker and auditory, per area group plus a whole-brain decoder (all units); threshold = value crossed by 25 / 50 / 80 % of held-out passive target stimuli (primary 50 %), in MAD units; rate vs rate-matched shuffled template (excess, ratio, p), time-shift null by circular shifts of the binned counts; identity p reported, not gated; per-epoch and quiet rates. Figures fig13_sens_real_vs_shuffled, fig14_continuous_<session>_<target>, fig15_examples_<session>_<target>, fig16_cohort_rates_<target>.
- 2026-10-29 008 results (8 sessions, 50 % sensitivity): thresholds 4-11 MAD for good whisker decoders; whisker decoders specific (held-out auditory crossing 0 in Ss-whisker, Midbrain, Striatum; 0.14 Thalamus). Real exceeds shuffled for Ss-whisker in 3/5 good sessions (ratio 3-18, AB136, AB157, MH036), whole brain 3/6; Midbrain, Striatum, Motor ~1. Most events fall outside quiet time (e.g. MH070 Ss-whisker 2.2 vs 0.09 / min quiet), except MH036. Cohort comparison: n = 0-2 sessions per stage x cohort, no inference possible. Runtime 7-18 min per session (6 threads), the whole-brain decoder ~60 % of it.
- 2026-10-29 User: run the full dataset (whole brain and Ss-whisker first, to debug); sessions without passive pre and post are left out of every threshold-dependent figure and number (no substitute threshold); save everything needed to re-threshold; stable units only (ssl-valid-data: good / mua AND stable in 137_stable_units, joined on electrode_group + NWB cluster id; 82 % of good / mua units). 009_full_dataset_events.py: per session x group x target, threshold-free event counts (real, 20 shuffled templates, 10 time shifts) x masks (all valid, quiet, passive pre, task, passive post) x thresholds 1-30 MAD (0.1 steps), held-out peak per stimulus, readout z; thresholds recomputed from these files at any sensitivity (decoders_table()). Cohort from the mouse reference sheet (R+ / R-, exclude flags), stage day 0 / expert (day >= 1): 122 sessions. Sessions without drift results (MH064_20260114, MH065_20260115, MH067_20260120, MH067_20260121) have no stable unit. Smoke test MH035: 70 s for whole brain + Ss-whisker (008: ~7 min for the session; column-major spike counts, vectorised event counting). Fix: 008 continuous-figure title (literal newline in an f-string broke the earlier figure regeneration).
- 2026-10-29 009 phase 1 (whole brain + Ss-whisker, 122 sessions, stable units): no failure; 103 sessions with passive pre and post. Real vs shuffled at 50 % sensitivity: most decoders > 10 events / min and real ~ shuffled; real exceeds shuffled mainly where the shuffled rate is < 1 / min. No consistent R+ / R- difference (isolated p < 0.05 in one of the two tests only).
- 2026-10-29 User: be autonomous; run every area group. 009 --groups auto (default): whole brain + every area group of the session with >= 10 stable units (18 groups in the dataset). Summary: excess also as z = (real - shuffled mean) / shuffled s.d. (comparable across decoders with different rates); R+ vs R- tested per stage only where both cohorts have >= 3 sessions (area inclusion rule), Mann-Whitney and Welch, BH correction across groups per metric x stage; fig17 rows = groups, columns = metrics; fig18 one figure per target. --figures <sessions>: 008 continuous / example figures from the saved files (raster unit indices refer to the stable set).
- 2026-10-29 009 full dataset, all groups (122 sessions, stable units; no failure; 103 sessions with passive pre and post; 547 good decoders with passive). At 50 % held-out sensitivity (z = (real - shuffled) / shuffled s.d.; frac = sessions with p_shuf < 0.05, i.e. real above all 20 shuffles):
  auditory decoders exceed the shuffled template widely: whole brain z 7.1, 82 % of sessions (quiet: z 3.9, 65 %); Midbrain 4.3 / 74 %, Thalamus 3.6 / 71 %, Striatum 3.6 / 66 %, Auditory areas 9.4 / 78 %; real rates low (whole brain 0.8 / min).
  whisker decoders weakly: whole brain z 1.35, 34 % (quiet 0.37, 24 %); Ss-whisker 0.19, 19 % (quiet -0.17, 14 %); Midbrain 1.2 / 33 %; Thalamus 1.6 / 40 %.
  Task events: no accumulation at a fixed time after trial onset (reward, valve), but 25-29 % follow a lick within 250 ms (movement); quiet-time metrics are the cleaner measure.
  R+ vs R-: 120 tests (per stage, groups with >= 3 sessions per cohort); 4 with p < 0.05 in both tests (whisker Midbrain day 0 z and quiet z: R- > R+; whisker Thalamus day 0 ratio; auditory Midbrain day 0 quiet rate), none after BH correction.
  Example figures (fig14 / fig15) for the sessions with the largest quiet excess per target x cohort: figures/full_stable/continuous, examples.

## 2026-10-29 — no multiple-comparison correction for now; day-0 whole-brain cohort comparison
- User decision: do not correct for multiple comparisons yet. fig17 stars now mean both uncorrected p < 0.05 (BH q columns kept in the csv for reference only).
- New `009 --day0-wholebrain`: day 0, whole-brain decoders, R+ vs R-, all decoders vs pAUC >= 0.8 (sessions with passive pre and post), Mann-Whitney + Welch uncorrected -> fig19_day0_wholebrain_cohorts, full_day0_wholebrain_stats_stable.csv.
- Result: whisker event rate higher in R+ (all decoders: median 37.8 vs 10.9 /min, p 0.027/0.040; good: 10.2 vs 4.2, p 0.069/0.046), but no difference in excess over the shuffled template (z, ratio) or in quiet time -> higher raw rate is not template-specific. Auditory: no difference (lowest p 0.12, z excess good decoders, R- > R+).
- z_shuf is undefined when all shuffles give 0 events, so n for z/ratio is smaller than for rate (auditory: 29/25 vs 42/33).
- Focus measures (user): rate, excess = real - mean shuffled rate (events/min), and both within the pre-start quiet window. New mask `prestart` in 009: windows in the last 2 s before each task trial start (task quiet window is Uniform(2, 5) s and not stored per trial, so only the guaranteed 2 s; windows within 0.1 s of onset are already invalid), trial kept only if no piezo lick in that 2 s; no whisking criterion for now (to be discussed). The old `quiet` mask (> 1 s from licks, whisker speed < 200 deg/s) stays in the files. fig17/fig19 now show rate, excess, rate_prestart, excess_prestart. Full rerun (all groups, --force) queued on haas: ~/react_full_prestart.sh, logs ~/log/react_full_pre/.
- Correction (user): the trial table stores the quiet window: abort_window_start_time..abort_window_stop_time (stop = start_time in all 122 sessions, duration 2.0-5.0 s, median 3.5 s; a few zero-length trials in AB128_20240829, AB117_20240723, MH018_20250228). `prestart` now = windows ending inside each task trial's abort window, trial dropped if the piezo shows a lick in it; no whisking criterion. 006 prepare() now also returns trials.
- Shuffle (user): no rate matching; the control only needs to disrupt the spatial pattern -> decoder weights permuted across all units of the decoder (20 permutations), each shuffled trace normalised by its own median/MAD. Identity p (reported only, not used) still uses rate-matched permutations.
- Full rerun (all groups, --force) relaunched with both changes: ~/react_full_prestart.sh, logs ~/log/react_full_pre/, then --summary --day0-wholebrain.
- Day-0 cohort comparison extended to the per-area decoders (user): `009 --day0` (alias --day0-wholebrain) -> fig19_day0_wholebrain_cohorts (rows = targets) and fig20_day0_areas_<target> (rows = area groups with >= 3 day-0 sessions per cohort among all decoders); columns rate, excess, rate_prestart, excess_prestart; all vs good decoders; MW + Welch uncorrected; full_day0_cohort_stats_<unit_set>.csv (replaces full_day0_wholebrain_stats).
- Rerun done (122 sessions, 0 failures; pre-start = abort window; plain permutation shuffle). Day 0 R+ vs R-, whole brain: whisker rate higher in R+ (37.8 vs 10.9/min, p 0.027/0.040), pre-start rate trend (19.9 vs 5.3, p 0.076/0.099); excess no consistent difference. Auditory: no rate difference; good-decoder excess R- > R+ in Welch only (p 0.048; pre-start 0.039). Per area: 184 tests, 8 with both p < 0.05 (whisker Midbrain rate + pre-start rate R+ > R-, whisker Hippocampus rate + pre-start rate R+ > R-, whisker Striatum pre-start excess R- > R+, auditory Striatum excess + pre-start excess R- > R+ (good decoders)).
- Problem with the plain permutation: whisker excess is now negative (whole brain median -1.1/min; real < shuffled), auditory excess shrank (64% of sessions p < 0.05 vs 82% before). In all 118 sessions |w| correlates with the unit's training mean count (Spearman median 0.45 whisker / 0.54 auditory, > 0 in every session); top-10% |w| units fire ~3x the median unit. Permuting without rate matching moves large weights onto sparse units -> spikier readout -> more threshold crossings, so the excess is biased negative by firing rate, not by pattern. Raised with the user.

## 2026-10-30 — shuffle weight space test; quiet-window example figure (010)
- Test (8 day-0 sessions, whole brain, 20 shuffles each; scratch/shuffle_space_test.csv): real - shuffled event rate, median, whisker / auditory: permute standardised weights (current plain) -1.79 / +0.14; permute raw-count weights (w/sd, no standardisation) +1.13 / +0.14; rate-matched standardised +0.11 / +0.14. Shuffled readout kurtosis (whisker): 1.86 / 0.92 / 1.01 (real 1.83). Unmatched permutation changes the tail shape in either space (heavier when permuting standardised weights, lighter when permuting raw weights) and biases the whisker excess in opposite directions; rate matching keeps per-unit statistics. Auditory unaffected. Recommendation to user: rate-matched shuffle (coarse strata), pending decision.
- 010_quiet_window_examples.py -> fig21_quiet_window_examples_<target>: day 0, whole brain, good decoders; quiet-window rate and excess per session R+ vs R-; sessions at the 25/50/75th percentile of each cohort's quiet-window rate, 30 quiet windows concatenated, readout + threshold + events, whisker speed below. Whisker good decoders: quiet-window rate R+ 6.6 vs R- 2.7/min median, p 0.18 (MW) / 0.093 (Welch) -> trend only (all decoders 0.076 / 0.099). Mice whisk a lot inside the no-lick window and many events coincide with whisking bursts.
- 011_standardization_test.py (user: with vs without z-scoring the units): 8 day-0 sessions, whole brain, zscore (current) vs raw (centred native counts, no scaling), each with plain and rate-matched shuffles; scratch/standardization_test/, scratch/standardization_test.csv, fig22_standardization_test. Medians zscore vs raw: pAUC whisker 0.940 vs 0.932, auditory 0.984 vs 0.985 (no loss); Spearman |native weight| vs mean count whisker 0.04 vs 0.66, auditory 0.19 vs 0.73 (raw template = high-rate units); whisker rate 7.7 vs 3.8/min, quiet window 3.9 vs 2.2; only ~40-65% of whisker events shared between variants; whisker excess plain -0.68 vs +0.24, rate-matched +0.16 vs +0.75; raw auditory readout kurtosis ~26 in 2 sessions (MH030_20250501, AB126_20240822): threshold > 30 MAD, no threshold. 009 gains SHUF_RATE_MATCH switch (default False).

## 2026-10-30 — shuffle strata test; 009 shuffle = rate quintiles
- User: keep z-scoring; coarser rate strata, or a shuffle without rate matching and without instability. Test (8 day-0 sessions, whole brain, 20 shuffles; /tmp/shufstrata_*.csv on haas, scratchpad shuffle_strata_test.py): whisker median excess (/min) plain -1.79, deciles +0.17, quintiles +0.33, tertiles +0.32, halves +0.39 (shuffled rate back up to 7.3, matching lost), sign-flip +3.25 (shuffled rate 2.3, lenient); shuffled kurtosis real 1.83 / plain 1.86 / q10-q3 ~1.0; time-shift null 0.095/min. Auditory ~+0.14 in all nulls.
- 009: SHUF_RATE_MATCH replaced by SHUF_STRATA = 5 (rate_strata(); 1 = plain); 011 sets 10 / 1. Full rerun not yet launched (user question on the 0.2 s.d. floor pending).

## 2026-10-30 — 009 v2: continuous z-scoring, rate-quintile shuffle, linear shift, passive blocks out, events / s; 012 browser
- User decisions: keep all units (no firing-rate exclusion); z-score each unit on continuous valid time before training (001 fit/nested gain `scale`); shuffle within 5 firing-rate quantiles (continuous mean count); linear time shift instead of circular (np.roll + core mask: BORDER_FRAC 0.05 or >= max lag 300 s cut at each recording end, so no wrapped count is ever used); passive blocks removed from valid time explicitly from the trial table context column; rates in events / s; main measure = events in the quiet (abort) windows of the kept active trials (task_events: perf != 6, warm-up cut, rule A1; prestart previously used every non-passive trial). Metrics: rate_prestart, excess_prestart (vs shuffled template), excess_shift_prestart (vs time shift; also p_shift).
- Confirmed: training = active trials only (perf != 6, warm-up, A1, quiet), target vs correct rejection, 5-35 ms window; passive pre / post held out, threshold only.
- Previous (v1) outputs archived: <HOME>/archive_v1_20261029/ (full/, full_*.csv, figures/full_stable).
- 8-session test (whole brain, v1 -> v2 medians, /s): whisker pAUC 0.940 -> 0.918, threshold 4.5 -> 4.9 MAD, quiet rate 0.065 -> 0.038, excess vs shuffle -0.007 -> +0.001 (shuffle > real in 4/8 sessions), time shift ~0; |w| vs mean count Spearman 0.46 -> 0.14. Auditory pAUC 0.984 -> 0.925 (MH038 0.98 -> 0.87, AB139 0.99 -> 0.91), quiet rate 0.002 -> 0.027 /s.
- 012_event_browser.py: --browse multi-page PDF per session x target (reference passive stimulus + events: raster of top-60 |w| units, readout, whisker / jaw speed, licks, +-2 s context strip) + page-1 PNG; --eta fig23_event_movements_<target> (movements aligned to event peaks vs random quiet-window times, R+ / R-, day 0 / expert; fraction of events during whisking > 200 deg/s). First look (AB130 whisker): quiet-window events ride on slow (0.5-1 s) elevations of the readout with broad firing of positive-weight units, unlike the sharp 30-ms stimulus response.
- Full v2 rerun launched on haas (~/react_full_v2.sh: 009 all sessions -> summary + day0 -> 010 -> 012 --browse --eta).
- Held-out passive decoding (user 2026-10-30): 009 quantify gains auc_{sham,other}_{pre,post,prepost} = AUC of the readout in the training window (30-ms window ending 35 ms after onset; win_values) for target stimuli vs shams / vs the other modality, in decoders.csv (from saved z, no rerun); fig24_passive_heldout_auc (whole brain, one point per session, R+ vs R- MW / Welch) in --summary; AUCs added to the 010 example titles and the 012 browser header. 8 test sessions, whisker median: vs sham pre 1.00 / post 0.95, vs auditory 0.77 / 0.72 (AB139 pre 0.01, AB130 pre 0.42, MH030 / AB116 post 0.17-0.25); fraction of target stimuli above the event threshold pre 0.69 / post 0.20.
- Full v2 rerun done (122 sessions; 19 died silently with empty logs in the 12-parallel run, rerun at 6-parallel, all DONE). Whole brain, day 0, quiet window (/s), R+ vs R- (MW / Welch, uncorrected): rate all decoders 0.113 vs 0.059 (0.19 / 0.24), good decoders 0.031 vs 0.031 (0.62 / 0.67) -> v1 trend gone; excess vs shuffled template good -0.028 vs +0.008 (0.0085 / 0.033; R- > R+), all -0.019 vs +0.010 (0.012 / 0.002); excess vs time shift ~+0.02-0.03 both cohorts (0.65 / 0.68). Day-0 whisker sessions with p_shuf < 0.05: R+ 14 %, R- 33 %. Auditory: no rate difference; excess good 0.028 / 0.066.
- fig23 (whisker): whisker speed peaks at event peaks (~600 deg/s vs ~170 at random quiet-window times) in both cohorts at day 0; 50 % of events during whisking > 200 deg/s vs 13-20 % of random quiet times; no licks around events (by construction).
- fig24: held-out passive AUC vs sham whisker ~0.94-0.99 median all stages; vs auditory 0.76-0.86 with many sessions < 0.5 (both cohorts); no R+ / R- difference (no test with both p < 0.05).

## 2026-10-30 -- v3: narrower pre-onset exclusion, no lick drop in quiet windows, passive target readout (fig25)
- Valid time now excludes -10 ms .. +600 ms around onsets (was -100 ms); 10 ms = the spike-artefact window (user).
- Task quiet windows (abort window of kept active trials) are no longer dropped when the piezo shows a lick; the number of
  such windows is counted in params.json (n_prestart_lick; expected 0).
- decoders.csv: win_{target,sham,other}_{pre,post,prepost} = median training-window readout (MAD units) of held-out passive
  events; fig25_passive_target_readout: whisker decoder on passive whisker stimuli (auditory on auditory), pre, post,
  post - pre, fraction above threshold, R+ vs R-.
- v2 summary CSVs and figures archived in archive_v2_20261030/; full rerun of 122 sessions launched on haas.

## 2026-10-30 -- v4: events in all valid time, movement flags, null-percentile thresholds; 013 one-page overview
- v3 batch stopped after ~17 sessions (superseded); AB131 / AB162 / AB164 v3 overview pages were made (013).
- Valid time: outside every trial from onset - 10 ms to onset + 5 s (was +600 ms), passive blocks and the 5 % borders; events
  are detected in all of it, not only in the task quiet window (user).
- Movement (per 30-ms window): whisking (speed > 200 deg/s) or jaw movement (speed > median + 5 MAD of valid time; only where
  lick-jaw sync passes) within 250 ms, lick within 1 s. Masks: all, still (every window still; main), whisk / jaw / lick (any
  window), prestart (first window, task quiet window). count_curve counts per mask with these modes, so shuffled and shifted
  nulls are filtered the same way as the real events. Event table: whisk / jaw / lick / still flags, movement label,
  whisk_max, jaw_max, lick_dt, dt_prev_onset / dt_next_onset, prestart.
- AB164 test: still = 11 % of valid time (whisk 70 %, jaw 74 %, lick 52 % incl. margins); 0 of 265 whisker events still.
  Whisker speed noise floor (median 14-47 deg/s) is well below 200 deg/s, so the fractions reflect behaviour.
- Null-percentile thresholds (Renard, Foustoukos et al. 2026: median over shuffles of the 99th percentile): null_q saved per
  trace; decoders.csv rows thr_source = shift99 / shift99.5 / shift99.9 next to the passive-sensitivity rows (main unchanged).
  AB164: whisker passive 5.2 MAD vs shift99 2.0 / shift99.9 2.6; auditory passive 1.3 vs shift99 2.2.
- 010 -> fig21_still_examples, 012 browser shows every event with its movement label (ETA vs random valid times), 013 rows use
  still. Full v4 rerun launched (~/react_full_v4.sh: 009 -> summary -> 010 -> 012 -> 013).

## 2026-10-30 -- 014 shift-null threshold (Renard rule, linear shifts); balanced accuracy as decoder criterion
- 014_shift_null_threshold.py: 100 linear time shifts per decoder (30-300 s per unit), threshold = median over shifts of the
  shifted readout's 99.9th percentile in all valid time (99.9 at 200 windows / s ~ one null window above threshold every 5 s,
  close to Renard's 99 % of 30-Hz frames); null counts per mask from the same shifts; passive crossing fractions as a check.
  AB164 whole brain (2 min): whisker threshold 2.6 MAD (s.d. over shifts 0.06); all valid 0.90 /s vs shift null 0.076 /s
  (p 0.01); still 0.045 vs 0.059 /s (p 0.81); passive pre crossing: whisker 98 %, auditory 36 %, sham 7 %.
  Queued on haas after the v4 run (~/react_v4_shiftnull.sh -> full_shiftnull_stable.csv, fig26_shiftnull_day0_wholebrain).
- Decoder check (118 sessions, whole brain): task training trials per session median 112 whisker vs 26 auditory (vs 79 N);
  the quiet filter removes ~1 trial; out-of-fold pAUC median 0.89 both, balanced accuracy 0.85 / 0.87, AUC 0.98 / 0.99;
  pAUC barely tracks trial number (Spearman 0.05 whisker, 0.21 auditory). Weak auditory decoders show up on held-out passive,
  not in the task test folds.
- Balanced accuracy (outer-fold decision values at 0) added to decoders.csv / shiftnull tables (bacc, n_train_stim,
  n_train_N); good decoder = bacc >= 0.75 (GOOD_BACC) in every figure, replacing pAUC >= 0.8 (pAUC kept in the tables).

## 2026-10-30 -- v5 (not launched): task + passive training, shift-null threshold in 009, bacc >= 0.7, effects vs bacc
- v4 batch and the queued 014 run stopped (12 sessions done). 014 removed: the 100-shift threshold now lives in 009
  (N_NULL = 100; THR_MAIN = shift99.9; decoders.csv has only shift99 / 99.5 / 99.9 rows, no passive-sensitivity rows).
- Training (user): every target stimulus of the task and of passive pre + post vs task correct rejections, no quiet filter,
  no shams (user: forget shams for now); classes balanced overall. Out-of-fold quality: bacc (all), bacc_task,
  tpr_passive_pre / post; GOOD_BACC = 0.7. fig24 = decoder quality (out-of-fold) + other-modality AUC; fig27_effects_vs_bacc
  (per cohort OLS on symlog scale + 95 % CI, Spearman, ANCOVA metric ~ cohort + bacc). 010 / 012 / 013 use THR_MAIN.
- AB164 test (whole brain, 2.5 min, 4 GB): whisker bacc 0.83 (task 0.83), passive stimuli detected 100 %; threshold 3.0 MAD.
  Auditory: bacc 0.65 but task bacc 0.45 (below chance) with passive detection 100 / 90 %: 67 passive vs 19 task auditory
  positives and no passive negatives, so the decoder can learn "passive block" rather than "auditory stimulus". Needs passive
  negatives (or context balancing) before the full run.

## 2026-10-30 -- v5 full run + 015 large-scale analyses
- v5 sham training: passive negatives = shams (user), classes balanced within context; out-of-fold bacc per context
  (bacc_task, bacc_passive_pre / post). 12 example sessions: whisker bacc 0.74-0.94, auditory 0.63-0.95 (AB164 0.63, AB162
  0.75); thresholds 2.1-3.8 MAD; all-valid rates 0.23-1.08 /s vs time-shift null 0.04-0.10; still events 0.01-0.57 /s, p_shift
  <= 0.01 for 11 / 24 decoders; some whisker decoders respond more to auditory than whisker stimuli (AB139, AB147: AUC vs
  auditory 0.06 / 0.21 pre) -- user: keep, decoding from the other modality is fine; all decoders used, weak ones flagged.
- Full v5 run launched (122 sessions, day 0 first; ~/react_full_v5.sh), then 015 (~/react_v5_015.sh).
- 015_cohort_analyses.py (user plan 2026-10-30): A primary = still-event excess over the time-shift null (+ quasi-Poisson GLM
  still events ~ cohort + bacc, offset null expectation), B movement (rates per mask, event rate around whisking / lick bout
  onsets), C content (identity p, joint with the other decoder, modality-specific still events), D trials (previous-trial
  type / outcome, W hit - miss, time since trial, quiet window before W hit vs miss), E time course (10-min bins, thirds,
  cohort x stage), F performance (hit / FA / d', trial-level logistic), G HMM learning curves (ssl-learning-trial-
  identification 024_avg_curves_per_mouse.pkl; source to confirm with the user), H area-group heatmaps, I robustness.
  Tested on the 12 examples (figures in analysis_015_test/): events peak at whisking-bout onsets and just before lick-bout
  onsets in both cohorts.

## 2026-11-01 -- v5 results; v6: shuffle-based threshold
- v5 full run: 118 / 118 sessions, no failure. No R+ vs R- difference in still-event excess over the time-shift null, still
  rate or all-valid rate (day 0 / expert, whisker / auditory, all thresholds). Still-time rate ~= time-shift null (half the
  sessions below it): the time shift moves movement-period activity into still time (null not matched to behavioural state).
  At the time-shift threshold, the 20 rate-matched shuffled templates cross as often as the real decoder in all valid time
  (0.76 vs 0.71 /s) and more often in still time (0.20 vs 0.08 /s): most v5 events are population fluctuations that any
  rate-matched projection picks up, not the template's own pattern. Pre-trial W hit < miss events in R+ (p 0.003-0.04) are
  events during movement only (still: p 0.26-0.86). Expert R+: still excess vs time shift ~ d' (rho 0.89 / 0.83, n 12).
- v6 (user): threshold = median over N_SHUF = 100 shuffled templates of their 99.9th percentile (THR_MAIN = shuf99.9;
  rows shuf99 / 99.5 / 99.9 / 99.99 and shift99 / 99.5 / 99.9 / 99.99); null traces processed one at a time (memory);
  shuffled readouts = raw decision values of the shuffled decoders (permuted w, same b) on the same data, in the real
  readout's median / MAD units for all traces (display units only; user: no per-trace scaling). Check on 12 examples
  (100 shuffles): per-trace own-MAD scaling inflates the shuffles (raw MAD 0.77 x real) and raises the threshold 4.7 -> 6.2. 015: primary = still-event rate; excess and
  GLM offset vs the shuffled templates; I = shuffle percentiles + v5 time-shift threshold.

## 2026-11-01 -- v7: detection everywhere but artefact / training windows; decoder + correlation readouts
- User: detect everywhere that is not training data, excluding readout windows touching the artefact window. Valid time = every
  30-ms readout window not touching onset - 10 ms .. + 35 ms of a trial onset or sham (EXCL = (-0.01, 0.065) on window ends),
  passive blocks included, START_EXCL (30 s) after the first spike to END_EXCL (30 s) before the last (end-of-recording
  artefact seen in the AB147 test). Time-shift null and the 5 % border cut dropped (N_NULL = 0).
- New masks: evoked (window end within EVOKED_S = 1 s after any stimulus onset) and passive; still now also excludes the evoked
  tail. Events carry readout, passive, evoked, dt_prev_stim.
- Correlation readout (user: keep both readouts): Pearson correlation across units between the z-scored population vector and
  the evoked template (mean z of target stimuli - mean z of no-stimulus training events, training window); 100 template
  shuffles within rate quintiles; same threshold rule; files / columns _corr, rows corr_shuf<q>. Formula checked against
  np.corrcoef (max error 4e-8).
- 015: --readout decoder / corr; A = total rate (coarse primary), still rate, movement-adjusted quasi-Poisson GLM (events ~
  cohort + fractions of valid time whisking / jaw / lick / evoked / passive + bacc, offset log valid s), excess vs shuffles.
- 013: two A4 pages per session x target; page 2 = 3-min stretch + 5 x 30-s stretches with raster, both readouts with 95 %
  bands of 20 template shuffles, population activity, whisker / jaw speed, licks, onsets.
- AB147 test: decoder 0.31 events/s vs shuffles 0.05 (p 0.01), still 0.037 vs 0.0045; whisker correlation threshold 17.8 MAD
  (raw r 0.25), real inside the shuffle band (sparse units with large z dominate shuffled correlations); auditory correlation
  0.21 vs 0.05 /s (p 0.01).
- Raw units (user): decoder readout = decision value (log-odds), correlation readout = Pearson r; thresholds exact (median over
  100 shuffles of their 99.9th pct, no grid snapping); counts at the exact thresholds (counts_q); MAD kept as thr_norm.
- Conditions (user, overlapping): qw = task quiet window and no lick within 1 s; qw_still = qw and no whisking / jaw movement;
  iti = task epoch outside trials (start..stop) and the evoked tail, includes qw, movement allowed; iti_still = iti and no
  whisking / jaw / lick. Movement per body part: speed > median + 5 robust s.d. (MOVE_K) of the session's speed over valid
  time (AB147 whisker: 199 deg/s). Movement tuning masks (outside the evoked tail): whisker / jaw speed robust-z bins
  (<1, 1-2, 2-3, 3-5, 5-10, 10-20, >20), time to nearest lick (0-0.25, 0.25-0.5, 0.5-1, 1-2, 2-5, >5 s); each with real and
  shuffle counts. 015: fig39 conditions (R+ vs R-, rate and excess), fig40 movement tuning; 013 page 1 row E shows both.
- Pilot: 20 day-0 sessions, one per mouse, 10 R+ / 10 R- (the 12 examples + 8 by v5 bacc >= 0.7 both decoders;
  ~/react_v7_pilot20.txt, ~/react_v7_pilot20.sh).
- 2026-11-02 Pilot (20 day-0 sessions) done; 015 fix: event-content frac_still renamed frac_ev_still (clashed with the decoder
  table's still fraction of valid time). No R+ vs R- difference in any condition; still / quiet rates ~0.01/s at shuffle level;
  decoder readout coupled to movement (rho ~0.5 with whisker / jaw speed, both targets); R+ readout rise after whisker hits =
  reward licking (gone in still windows, same for the auditory decoder).
- 2026-11-02 016_soft_norm_example.py (AB147 whisker): correlation readout with z_u = (x_u - mu_u) / (sd_u + c), c = 0, 0.5,
  1, 2 x median unit s.d., plus a movement-partial variant (whisking-minus-still pattern partialled out across units). c = 0:
  top-5 units hold 96 % of the positive contribution at event peaks, real below shuffles (0.009 vs 0.048 /s); c >= 0.5:
  top-5 share ~0.29, real 1.1 vs 0.10 /s, still 0.95 vs 0.09 /s, rho with whisker speed 0.16 (decoder 0.50); results
  nearly identical for c 0.5-2. Partial | movement: rho -0.02, still rate unchanged.
- 2026-11-02 User: soft normalisation in the 009 correlation readout, SOFT_C_FRAC = 1 (saved sd = sd + c, sd_raw, soft_c).
  Tested on AB147 (whole brain): whisker 1.13 vs 0.10 /s (still 0.93 vs 0.09), auditory 0.76 vs 0.11 (still 0.08 vs 0.08).
  The other 19 pilot sessions still hold the c = 0 correlation readout until the pilot is rerun.
- 2026-11-02 017_whisking_cohort.py (user hypothesis: R+ whisk more in spontaneous periods, whisking goes with a higher
  readout). Spontaneous = iti and > 1 s from licks (passive blocks < 1 min, dropped). Pilot: R+ whisk less (fraction 0.44
  vs 0.53, p 0.008 / 0.005; bouts 0.35 vs 0.50 s, p 0.02 / 0.07), raw speed not different; whisking raises every readout
  equally in both cohorts (bout-onset transient, whisker and auditory decoders alike); no post-bout rise; no across-session
  link between whisking amount and event rate. Correlation readouts were still c = 0.
- 2026-11-02 Movement-partial correlation as a third 009 readout (_pcorr, rows pcorr_shuf99.9): r(population, template | movement
  pattern) per window, movement pattern = mean soft-normalised count while whisking or jaw moving minus still (valid, outside
  the evoked tail; without whisker tracking: jaw or lick vs neither), same template shuffles. 013 draws every readout present
  (timeline, onsets, stats rows; page 2 partial on the right axis); 015 --readout pcorr. AB131 rerun (whole brain): whisker
  decoder still 0.000 /s (shuffles 0.0003), soft correlation still 0.60 (0.07), partial still 0.93 (0.13).
- 2026-11-02 017 absolute whisking (user: angular velocity): NWB whisker angle, likelihood < 0.5 removed, gaps <= 50 ms
  interpolated, 30 Hz low-pass, |d angle / dt|; amplitude = Hilbert envelope of the 5-25 Hz angle; fractions above fixed
  thresholds; effect sizes (Hedges g, rank-biserial, ratio of medians + bootstrap CI). Pilot: R- > R+ only in relative
  measures (session threshold: fraction 0.10 vs 0.14, p 0.046 / 0.058, g -0.98); absolute angular velocity / amplitude
  differ by g -0.4 to -0.7, p 0.14-0.37. 009 whisker_speed (|gradient| of the unfiltered angle) kept for masks.
- 2026-11-02 User: the partial correlation partials out whisker velocity (DLC), replacing the movement-pattern version. 009
  _pcorr: whisker |angular velocity| (whisker_angvel: likelihood-masked angle, gaps <= 50 ms interpolated, 30 Hz low-pass,
  |d angle / dt|; 001 load_session now keeps dlc["whisker_angle"]) averaged in each 30-ms readout window; per unit, OLS of
  the counts on [1, velocity, log(1 + velocity / 25 deg/s)] over valid time outside the evoked tail; residual counts,
  soft-normalised, Pearson r with the same template and shuffles (corr_resid). No whisker tracking: no _pcorr files.
  Saved: vel_beta, vel, vel_fill. AB131 (whole brain): Spearman of the readout with whisker velocity 0.65 decoder, 0.37
  correlation, -0.13 partial; whisker partial 1.33 vs 0.11 /s, still 0.88 vs 0.10, ITI still 0.90 vs 0.11; auditory
  partial still 0.50 vs 0.10.
