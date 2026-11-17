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
