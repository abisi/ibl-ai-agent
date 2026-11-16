# change-log

- 2026-11-02: project created (question.md, TODO.md) from the NWB audit of DLC vs camera-timestamp length mismatches.
- 2026-11-02: 006_pairing_evidence.py (lick-bout heatmaps under NWB pairing vs end-alignment, correlation vs assumed offset per third, pulse IPIs) on the constant-offset, camera_flag-0 and control sessions.
- 2026-11-03: 007_top_metrics_pairing.py (every top-view metric, value and speed, vs licks and whisker stimuli; clipped at 5 z). 006 top camera switched to whisker_angle (largest |deviation|).
- 2026-11-03: 008_piecewise_mapping.py (step function of the frame -> pulse offset, steps placed on licks, side lead 5 frames, top on side steps); clauses for MH013 0219, MH017 0302, MH004 1206; AB116 0726 end-aligned + DLC NaN before 2,175 s; all reconverted and checked with 003.
