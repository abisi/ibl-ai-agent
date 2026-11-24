# change-log

- 2026-10-15: project created by moving the passive hit / miss axis analyses (132-135b, 140, 145-147, 149-151, 153, 154, 156,
  158, 160-175) out of ssl-whisker-hitmiss-timeresolved-decoding; results moved on the NAS with symlinks at the old paths
  (one locked file, figures/publication/175_across_days_passive_projection.pdf, left in place); shared modules re-exported;
  figures now written to this project's figures/publication. Script history before this date is in the change-log of
  ssl-whisker-hitmiss-timeresolved-decoding.
- 2026-10-15: first report (report/build_report.py on report_lib; report/ on the NAS): Figure 1 = 158 (within day 0), Figure 2 = 165 hit - miss row (new SSL_165_ROW1=1 with SSL_165_ANGLE=1: angle and projection beyond session time, areas), Figure 3 = new 177_report_across_days (119 row-1 panels + 175 v2 panels with passive-post auditory and the per-area panel). Supplementary: 174 schematic, 160 decoder, 153 exposure, 156 expert control. Not included: 162 and 164 (same quantities as Figures 2e / 1d), 175 v1 / v2 (replaced by 177), Part III figures of the old decoding report (135b, 145, 147-151).
- 2026-11-23: step 1 of the clean Figure 4 rerun (data selection; reconciliation with panels A-G of the colleague,
  tables/session_reconciliation_AG_vs_CD_20261123.csv). Fixes: (1) 132.session_trials bug -- post was "passive after the LAST
  active trial", so 1-5 active-labelled trials after the passive-post block emptied post and dropped the session silently
  (AB120, AB124, AB136, AB150, AB155, MH007, MH015, MH017, MH068, MH070); now main active block = longest active run, post = the
  passive run right after it, later active trials dropped (MH009 post 41 -> 80 trials); every other session unchanged.
  (2) AB144 re-included in ephys_utilities keep_passive_mice (NAS checkout, uncommitted; excluded since 2026-08-24 without a
  documented reason, passive data normal). (3) 137b tracked-unit list rebuilt (backup 137b_tracked_units_backup_20261123).
  (4) user: the hit - miss axis needs >= 3 hits and >= 3 misses (was 10 / 10); 169 SSL_169_MIN_CLASS, suffix _hm3. The missing
  shift null of AB123 / AB139 was the same cause (exactly 10 trials in the minority class: no shift kept 10).
- 2026-11-23: 137b rebuilt (77 sessions; +11 recovered sessions; units changed only in MH009 (post fix) and AB127 / AB138 /
  MH028 (NWBs reconverted 2026-11-20)). 169 rerun with SSL_169_MIN_CLASS=3 -> 169_along_orthogonal_hm3.parquet: learners 32 R+ /
  30 R- (was 27 / 24); = A-G set + MH004, minus MH015 (15 tracked units < MIN_UNITS 20) and MH036 (2 whisker licks). K-type
  quick check (d_along - null, all units): W R+ -0.015 vs R- -0.051 (MW .08, Welch .10); W - A interaction MW .05, Welch .014
  (10/10 set on the same table: W MW .02 / Welch .04, W - A .017 / .006).
- 2026-11-23: 132.session_trials made robust for expert sessions (context NaN = non-passive; pre = longest passive run before
  the main block): day 0 unchanged, 12 expert sessions recovered (MH001, MH022, MH027, MH034, MH037, MH064 x3, MH067, MH070 x2;
  MH062 / MH068 post = the run right after the main block). Expert 137b rebuilt (36 sessions; backup _backup_20261123).
- 2026-11-23: NEW 179_cd_clean.py -- single clean CD pipeline for Figure 4 (replaces 135 / 168 / 169 for the figure): axis from
  all active whisker trials (>= 3 hits / misses), shift null, projection / angle / gain-rotation / stimulus axis / areas (group +
  fine) / evoked change / LOO time course / unit table; learning (63 learners: 33 R+ / 30 R-) and expert (18 / 16 sessions).
  Whole-brain values identical to 169 _hm3. NEW 180_cd_clean_overview.py: one review page of all candidate panels + stats csv.
- 2026-11-24: 179 adds a_ / th_{pre,post}_{h1,h2} (+ _null): projection and angle on the temporal halves of each passive epoch
  (own rng; all existing values identical, checked against backups 179_cd_clean{,_expert}_backup_20261124). NEW
  181_split_half_floor.py: within-epoch noise floor (Kingsbury et al. 2025): within = h2 - h1, across = post h1 - pre h2.
- 2026-11-24: NEW 182_roc_neuron_classes.py: neuron classes from the per-mouse ROC (whisker_choice, wh_vs_aud_active, 5-35 ms)
  joined to 179_cd_units (cluster_id % 1e6); exact split of the passive whisker projection change by class; overlap of
  hit / miss and stimulus coding.
- 2026-11-24: NEW 183_subspace_alignment.py: task (hit and miss, 5-45 ms bins, K=3), decision (hit - miss, K=2, + shift
  null) and stimulus subspaces; cross-validated fraction of the passive W / A response in each, pre / post and halves.
- 2026-11-24: NEW 184_subspace_summary.py (183 results; 179 session set; unstable fractions outside [-0.5, 1.5] -> NaN) and
  185_orthogonalisation_across_learning.py (pre-task angle to the axis, cos(axis, stimulus axis), d', day 0 vs expert).
