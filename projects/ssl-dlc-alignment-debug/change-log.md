# change-log

- 2026-11-02: project created (question.md, TODO.md) from the NWB audit of DLC vs camera-timestamp length mismatches.
- 2026-11-02: 006_pairing_evidence.py (lick-bout heatmaps under NWB pairing vs end-alignment, correlation vs assumed offset per third, pulse IPIs) on the constant-offset, camera_flag-0 and control sessions.
- 2026-11-03: 007_top_metrics_pairing.py (every top-view metric, value and speed, vs licks and whisker stimuli; clipped at 5 z). 006 top camera switched to whisker_angle (largest |deviation|).
- 2026-11-03: 008_piecewise_mapping.py (step function of the frame -> pulse offset, steps placed on licks, side lead 5 frames, top on side steps); clauses for MH013 0219, MH017 0302, MH004 1206; AB116 0726 end-aligned + DLC NaN before 2,175 s; all reconverted and checked with 003.
- 2026-11-16: 010a_extract_raw_pulses.py (raw camera pulses + all piezo licks per session via the converter's
  analyze_continuous_log; cache/raw_pulses_<session>.npz) and 010_drift_review_pdf.py (one PDF over the 8 drift
  sessions: offset curves for naive / current NWB / re-fit pairings, jaw distance aligned to trial start, whisker speed
  aligned to the whisker stimulus, jaw opening - first lick per trial; drift_review/drift_review_2026-11-16.pdf).
  Findings: (1) the 2023 AB NWBs keep only response-window piezo licks (~800/session; the NIDQ piezo_licks.txt is
  mains-noise dominated), so 008's 60 s windows never passed the 10-event threshold there -> windows now scaled to
  >= 25 licks; (2) MH017 0302 and MH013 0219 clauses miss a final large side step (+130 -> +552 at frame 453,065;
  +205 -> +684 at frame 1,112,830; both = pulses - frames), the last ~20-25 % of trials are misaligned in NWB_ks4;
  (3) AB092 1205 side: one +813-frame step at frame 654,713 (jaw consistency 54 % -> 91 %); (4) AB095 1212 top is in
  register (stimulus-locked whisker-speed line constant over the session; the 005 drift verdict was a lick-correlation
  artefact); (5) lick-based top re-fits are unreliable (AB116, AB095), and tying the top start offset to the side is
  wrong when the cameras start differently (MH028, AB116); the stimulus-locked whisker response is the usable top check.
