# Dead-time screen: results (2026-11-20)

Outputs: `combined_results_ks4/ssl-whisker-artefact-gap/screen/tables/02_dead_time_screen.csv`, `screen/figures/02_dead_time_screen.*`,
`../flagged_probes.csv`.

- 452 probes screened (130 sessions, 95 mice). Not screened: AB155_20250208 (5 probes, merged session).
- Flag: whisker RMS ratio (6-18 ms / baseline) < 0.7, or flat fraction 6-18 ms minus baseline flat fraction > 0.3.
- Flagged (6 probes, 5 in the dataset):

| session | probe | type | ratio 6-18 ms | recovery |
|---|---|---|---|---|
| MH037_20250524_143522 | imec0, imec1, imec2 | NP2.0 | 0.02 (flat) | 34 ms |
| MH037_20250524_143522 | imec3 | NP1.0 | 0.40 | 34 ms |
| MH020_20250304_161732 | imec0 | NP1.0 | 0.22 | 32 ms |
| MH070_20260124_102803 | imec2 | NP2.0 | 0.005 (flat) | none within 60 ms; not in dataset |

- Gap between flagged and unflagged: flagged <= 0.40, unflagged >= 0.71. The 0.7 cut sits at the low edge of a graded
  tail. AB142_20241128 (all 4 probes, 0.72-0.74, 32 ms), AB143_20241126, MH014_20250219 and MH023_20250316 sit in
  0.71-0.85: NP1.0 probes with saturated channels in 0-4 ms and a 15-30 % RMS dip. Their KS4 spike-rate ratios are
  0.6-1.9, so spike detection is not lost. Not flagged.
- Alignment check failed for 32 probes. 26 have the artefact peak at 0.5-2.5 ms (alignment correct, partial event
  match) and ratio ~1. Peak far off: MH038_20250527 imec0, MH064_20260114 imec2, MH064_20260116 imec0 (in the dataset).
  Their KS4 ratios (0.94-2.17) show no dead time.

## Proposed masking rule (NOT applied; needs user decision)

For whisker-aligned analyses, on flagged probes treat 0 to `dead_time_end_ms` (32-34 ms; use 35 ms) after whisker
onset as missing: either NaN those bins, or drop the probe from analyses that read latencies < 35 ms. Other probes:
keep the current -10..+5 ms gap. Auditory trials are unaffected (ratio_a ~1 on all flagged probes).
