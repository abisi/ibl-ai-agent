# Change log

- 2026-11-20: project folder created in the repo (diagnosis scripts copied from the NAS results folder); question.md
  and TODO.md written for the dataset-wide dead-time screen.
- 2026-11-20: full screen done (452 probes, 130 sessions, 95 mice; 5 AB155 probes fail = merged session). Flat rule
  made baseline-relative (flat 6-18 ms minus flat baseline > 0.3): removes AB107 imec2 and AB144 imec4 (dead channels
  in baseline too). 02_summarize.py now writes flagged_probes.csv (per shank) for the pairs analysis.

## 2026-11-23 — raw data after +5 ms on the flagged probes (diagnosis/05_raw_after_5ms.py, read-only)
Outputs: combined_results_ks4/ssl-whisker-artefact-gap/raw_after_5ms/ (one figure per probe + summary_raw_after_5ms.csv).
Probes: MH037_20250524 imec0-3 and MH020_20250304 imec0, with MH031_20250507 imec3 as a normal control.
- **MH037 imec0-2 (NP2.0):** the raw signal is exactly flat (sigma 0.06-0.3 uV, against 95 uV at baseline) until 20-37 ms; the time varies by trial. The flat stretch ends in a large release transient. There is nothing to recover.
- **MH037 imec3 (NP1.0):** a signal is present after 5 ms at about half amplitude (event median 215 against 435 uV). KS4 rate is 0.66x baseline at 5-10 ms and 0.9x at 10-20 ms. A large common transient at about 30 ms, coinciding with the NP2.0 release, needs masking.
- **MH020 imec0 (NP1.0):** the gain is down about 4x (sigma 5 against 19 uV; events 41 against 153 uV). Recovery varies by trial, and KS4 sees 0.26-0.4x baseline up to 30 ms. The local-threshold rate (12-20x) is inflated by pooling trials with different recovery times, so it does not measure neural rate. Panel J is meaningless for flat NP2.0 probes.
- **MH020 imec0, trial x channel map** (`diagnosis/06_mh020_trial_channel_map.py`, all 202 active whisker trials): the gap is set by the trial, not the channel.
  - About the first 35 whisker trials recover within 5-15 ms, apart from a few short dead runs around trials 15-25. Trials ~37-178 are dead (spike-band RMS ~1/8 of baseline), with recovery at 30-40 ms. After trial ~178, recovery shortens steadily to about 5 ms.
  - Channels 335 and above (and channels 164, 190 and 302) are not attenuated. Their location (possibly outside the brain or reference) is still to be checked.
  - 0-4 ms saturation is about the same on every trial and channel (about 16 %) and does not predict the gap (r = 0.05).
  - Implication: a per-trial dead-time mask would keep the fast-recovery trials.
- **Stages split by recovery** (`diagnosis/07_stages_by_recovery.py`): single trials at each stage, fastest 20 vs slowest 20 whisker trials. Outputs: `raw_after_5ms/stages_*.png` and `recovery_per_trial_*.csv`.
  - The gap is the same at every stage: raw, CatGT -gblcar, tcat_corrected, KS4 input, KS4 high-pass and whitening. No step creates it or removes it. gblcar removes only the slow common-mode recovery, and tcat_corrected covers only 0-4 ms.
  - Per-trial recovery, median (IQR): MH020 imec0 34 ms (18-37); MH037 imec3 36 ms (29-42); MH037 imec1 34 ms (29-40, flat from 8 to 53 ms depending on the trial).
  - New finding: when the NP2.0 amplifier releases (MH037 imec1, at 20-50 ms depending on the trial), the step produces a burst of false KS4 spikes. The same release appears as a deflection on all channels of MH037 imec3, with false KS4 bursts at about 27 ms in fast trials and about 50 ms in slow ones. These could be mistaken for late whisker responses.
  - MH020 fast-recovery trials: the spike band is normal from 5 ms, but the KS4 rate stays below baseline until about 40 ms and shows no evoked response. The cause is unresolved.

## 2026-11-23 — are the flagged probes outliers in the current results? (`screen/03_outlier_check.py`, read-only)
The five flagged probes were compared with the 302 other day-0 probes (>= 10 units each). Sources: ROC `ssl-roc-learning-stages/roc_long.parquet` (post window +5..+50 ms), `ssl-zeta-responses/zeta_long.parquet` and `ssl-sensory-spatial-maps/unit_latency.parquet`.
Outputs: `screen/tables/03_outlier_probe.csv`, `03_outlier_area.csv`, `screen/figures/03_outlier_check.png`.
- **Whisker, active trials:** the flagged probes are clear outliers.
  - ROC fraction suppressed is 0.20-0.45, against a median of 0.025 on other probes. All five are at the 99.7-100th percentile, and they are four of the five highest probes.
  - The ZETA fraction suppressed is likewise at the 98.7-100th percentile.
  - Median response latency is 6-7 ms, against a median of 20 ms (0th percentile). This comes from the false spike burst at +2..+5 ms. MH037 imec0 is the exception, at 19 ms.
  - The fraction excited is not an outlier (10th-65th percentile).
- **Auditory, active trials (control):** within the normal range.
- **Whisker, passive pre-session:** MH037 is mildly high (fraction suppressed at the 91st-98th percentile, ZETA down at the 94th-99th). MH020 is normal (27th percentile), consistent with its early-session trials recovering fast.
- **Per area:** every whisker cell of the flagged sessions is at the 100th percentile for fraction suppressed and the 0th for latency, except striatum (37 units), which shows instead the most excitation.
- **Next-highest probes:**
  - MH014 imec2 (31 units) and MH013 imec2 (65 units) have a normal raw spike band, so their suppression is not this artefact (or it is a small-number effect).
  - AB155 imec0 (0.15) was not screened (merged session).
  - Spearman correlation of suppression vs the raw screen ratio over all probes: 0.03. The effect is confined to the flagged probes.

## 2026-11-23 — TENTATIVE gain-recovery pilot on the NP1.0 probes (`diagnosis/08_gain_recovery_pilot.py`, read-only)
Outputs: `gain_recovery_pilot/<tag>_{fig.png, summary.json, synthetic.csv, units.csv, psth.npz}`.
- **Synthetic test:** auditory trials were shrunk by real whisker-trial gain profiles, then corrected and re-detected by template matching.
  - With the TRUE gain, recall is 0.91-0.95 even at gain 0.2-0.4, and precision is 0.88-0.96, on both probes. The spikes are recoverable in principle because signal-to-noise is preserved.
  - Without correction, recall is about 0 below gain 0.4, which reproduces the gap.
- **The bottleneck is the gain estimate.** The noise-based estimate (median |x| per 1-ms bin) is inflated by evoked activity: on auditory trials it reads 1.5-1.7 at about 10 ms instead of 1. As a result:
  - recall with the estimated gain is only 0.29-0.36 at gain 0.2-0.4 and 0.40-0.60 at 0.4-0.7;
  - precision stays about 0.99, so the errors are misses only (conservative).
- **Real whisker trials, recovered:** 5-15 ms 7.8 Hz (KS4 3.8, auditory 47, baseline 21.5) for MH020; 11.5 Hz (KS4 1.2, auditory 61, baseline 29) for MH037 imec3. These values are biased low by the estimate, so not interpretable yet.
- **Detector:** only 49/314 (MH020) and 32/373 (MH037 imec3) units reach F1 >= 0.6 against KS4 on clean data.
- **Next:**
  - Try a gain estimate independent of neural activity: high-frequency noise (>6 kHz) and, on MH037, the amplitude of the ~100 uV common-mode interference before re-referencing.
  - Improve the detector (KS4 templates and whitening).
- **Gain estimators** (`diagnosis/09_gain_estimators.py`): high-frequency noise (6-9.5 kHz of the raw signal) is the least driven by responses.
  - On auditory trials it peaks at 1.21 / 1.25 (MH020 / MH037), against 1.54 / 1.71 for the noise-based estimate.
  - Against spike amplitude, its ratio is 1.04 / 1.16 (well-isolated units).
  - The common-mode estimate is too noisy.
  - On MH037 imec3, the amplitude at KS4 spikes in 5-20 ms is 0.1-0.2, against 0.35 for every noise estimate. Either KS4 spikes there are partly noise, or spikes are attenuated more than noise (signal-to-noise not preserved). Unresolved.
- **Pilot rerun with the high-frequency gain** (`08_gain_recovery_pilot.py <i> hf`, outputs `*_hf_*`):
  - Synthetic recall at gain 0.2-0.4 rose from 0.29-0.36 to 0.65; at 0.4-0.7 from 0.40-0.60 to 0.75-0.83. Precision is 0.90-0.99. Remaining misses are at 5-15 ms, where the response still inflates the estimate.
  - Real whisker data, 5-15 ms:
    - MH020: 7.1 Hz (KS4 3.8, baseline 21.5).
    - MH037 imec3: 12.2 Hz (KS4 1.2, baseline 29), with 24 Hz at 15-30 ms.
  - MH037 overshoots to 38 Hz at 30-50 ms, around the release time: likely false detections on the release deflection, so it needs a mask.
  - Not validated on real data yet: the waveform and amplitude distribution of the recovered events still have to be checked against baseline events.
- **Validation of recovered events** (`diagnosis/10_validate_recovered.py`): dead-window events at +5..+30 ms with high-frequency gain < 0.6 were compared with the same units' baseline events, plus a noise null (the same detector on the sign-inverted signal).
  - **MH037 imec3: recovery looks genuine.**
    - Event rate is 18.2 Hz per unit, against a null of 9.1. The excess (about 9 Hz) is half of the baseline excess (18.6 Hz).
    - Amplitude distribution matches baseline (median 0.93), and amplitude stays flat across gain levels 0.2-0.6 (0.96 / 0.93 / 0.92), so signal-to-noise is preserved.
    - Waveform shape correlation with the template is 0.90 (baseline 0.98; 30 units).
    - So the low KS4 amplitudes in this window came from false KS4 spikes, not from spikes shrinking more than noise.
  - **MH020 imec0: not recoverable on attenuated trials.**
    - Event rate is 2.4 Hz per unit, against a null of 2.65: indistinguishable from noise.
    - Amplitudes are broad (0.9-2.3x) and match the null distribution.
    - Only 6 of 49 units have 10 or more events.
    - Only the fast-recovery trials (early and late in the session) give usable spikes in 5-30 ms on this probe.

## 2026-11-23 - diagnosis/11_recovered_psth.py (TENTATIVE): recovered rasters and PSTHs, NP1.0 probes
- Gain-corrected (HF) detection over -60..+80 ms, whisker + auditory, real + inverted null, KS4 for the same units; trials sorted by recovery time (g_hf >= 0.8).
- Outputs gain_recovery_pilot/<tag>_recovered_{fig.png,fig.pdf,psth.npz,events.csv.gz}.
- 5-30 ms, mean Hz/unit (baseline): MH037 imec3 KS4 3.3, recovered null-corr. 20.0 (29.1), same on fast and slow trials; MH020 imec0 KS4 5.9, recovered 7.1 (21.6); fast trials 12.3 vs KS4 10.4, slow trials 1.0 vs KS4 0.6 (nothing recovered).
- No excitatory whisker response recovered on either probe; dead window is a partial drop below baseline (auditory KS4 peaks 60-90 Hz at 5-20 ms on the same units).
- Caveats: MH037 release (25-50 ms) gives false detections along the recovery line in some units (e.g. 587); MH020 unit 432 events at the artefact edge (3-6 ms); null correction adds a flat offset where both real and null rates are 0 (MH020 slow trials). Usable-trial masking (g < 0.12) almost never triggers with the HF gain.
