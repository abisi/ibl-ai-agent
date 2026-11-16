# SSL reactivation decoding

Detection of whisker (and, as control, auditory) stimulus-like population events in continuous data with
stimulus-presence decoders. Question and decisions: `question.md`. Results (not in git):
`combined_results_ks4/ssl-reactivation-decoding/` on the NAS.

## TODO
- [ ] Settle the open design points (decoder option first, areas, continuous periods, readout step, pilot sessions).
- [ ] Pilot on a few sessions per stage: training-set build, decoder fit at low FPR, held-out stimulus sensitivity.
- [ ] Continuous readout + both nulls; one threshold per area group, the same for every session and template (004, 2026-10-24); event table.
- [x] Pilot 8 sessions x {clean, spont} (2026-10-24); 004 group thresholds superseded by per-decoder thresholds.
- [x] 006 (2026-10-29): per-decoder threshold (time-shift null <= 5 % of data), whisker + auditory decoders, held-out passive pre / post / both (direct + injected), shuffled template, identity test, coincidence; 8 sessions.
- [x] 007 (2026-10-29): shuffled-template threshold + identity gate + leave-one-unit-out participation; 8 sessions (threshold too strict: no held-out sensitivity left).
- [x] 008 (2026-10-29): held-out-sensitivity threshold, real vs shuffled excess, continuous-detection figures with licks / whisking / jaw, cohort figures; 8 sessions.
- [x] 009 (2026-10-29): full dataset (122 sessions, stable units, whole brain + every area group), threshold-free counts saved for re-thresholding.
- [ ] Next: movement control (event rate vs whisking / licking / jaw), auditory-decoder events vs environmental sounds, leave-one-unit-out on the full dataset.
- [ ] Explore detected events: state (DLC whisking, licks, grooming), task events, passive blocks.
- [ ] Day 0 vs expert, R+ vs R-; auditory control.
- [ ] Report (skills/project-report).
