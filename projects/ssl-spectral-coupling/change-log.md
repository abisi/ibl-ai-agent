# Change log

- **2026-11-17 23:10: project created** (user request, overnight autonomous run). question.md written with the
  pre-registered tests H1-H3 before any real-data result; TODO.md with the step list.
- **2026-11-17 23:15-23:59: spike run and stats.** 001 simulation passed; 002 run on 89 sessions; 003 written.
  Warm-up cut added (the pilot loader keeps the auditory warm-up block). Cohort / learner / exclusions re-read from the
  current reference sheet (the pilot coverage table dates from 2026-11-06).
- **2026-11-17: AP-band source.** Clock from whisker_stim_times_to_imecP.txt (linear fit, mode-of-differences offset +
  nearest matching: handles unequal counts and unsorted files, MH021 / AB080 / MH038); AB155 (joined session) has no
  single AP binary and is skipped. AP epochs shifted 10 ms away from t = 0 (artefact-correction dip in the 0-5 ms bin).
  ap_lib.select_trials returned index labels instead of positions (fixed before any cache was used).
- **2026-11-18 00:15: report v1** (spikes only; AP section marked pending): build_report.py, render.sh.
- **2026-11-18 01:40: AP clock drift fix.** AB138 imec1 drifts > 10 ms against the NWB clock over the session, so a
  fixed-offset match failed (153 of 182). clock_fit now refines the matching twice with the linear fit (residual
  0.05 ms on all 4 probes). AB138 rebuilt and added to the AP run (~/ssl_spectral_ap_chain2.sh; 003 --source ap rerun).
  AP source: 88 of 89 sessions (AB155 has no single AP binary).
- **2026-11-18 03:30: AP results and report v2.** AP run 88 sessions; 003 --source ap (learners + all). Report v2 adds
  the AP section, a key result and Figure 7 (AP time-resolved coherence). Bug fixed before use: report_lib snapshot
  names copies from the last two folders only, so ap/stats/<pop>/x.csv and spikes/stats/<pop>/x.csv collided and the
  first v2 build read spike tables for AP. Now one snapshot tag per source (<date>_spikes, _ap, _simulation); the
  collided 2026-11-18 snapshot folder was deleted.
- **2026-11-18 09:00: v2 (user decisions).** No MC correction (exploratory), AP = spikes (envelope dropped), MSC focus,
  mouse level, within-area at both levels, 50-ms corr50, jaw-onset alignment. 002 gets --align stim|jaw (v2_<align>/);
  007_exploratory_stats.py. Runs on haas: ~/ssl_spectral_v2_chain.sh (logs 002_v2_stim.log, 002_v2_jaw.log).
- **2026-11-18 09:40: v2 runs done.** stim 89/89 sessions; jaw 86/89 (no dlc_jaw_onset_times.pkl for AB080, AB082,
  MH026). 007 run for both alignments (all mice + learners). Jaw-onset latency does not differ by cohort (checked,
  mouse level), so jaw-aligned differences are not a reaction-time shift.
