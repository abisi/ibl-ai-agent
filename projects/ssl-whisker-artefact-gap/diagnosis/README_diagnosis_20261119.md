# Whisker-artefact spike gap (MH037_20250524, MH020_20250304) — diagnosis

Read-only diagnosis, 2026-11-19. Nothing was reprocessed or overwritten.

## Conclusion
The gap is already in the raw SpikeGLX recording; no processing step creates it. The whisker-coil artefact drives the probe front end into saturation. The amplifiers then take 20–35 ms to recover, and in that time the spike band carries no neural signal.
- **MH037 imec3 and MH020 imec0 (NP1.0):** ~55–60 % of channels sit at the ADC limit (±512 counts) for 2.5–3.7 ms. After that, the spike-band RMS of the whole probe falls to the electronic noise floor (6.9 and 7.8 µV, against a baseline of 17 and 15 µV) until about 20 ms (MH037) and 35 ms (MH020). The ~100 µV common-mode interference in the MH037 raw data is attenuated by the same factor (raw high-passed RMS goes from 107 to 44 µV). That points to a drop in front-end gain rather than silent neurons, because silent neurons would leave non-neural interference untouched. The drop is the same on channels that did and did not reach the ADC limit, so it is probe-wide, as expected if the shared reference / front end saturates.
- **MH037 imec1 (NP2.0, type 2003):** the raw output jumps to a constant value and stays exactly flat (100 % of sample-to-sample differences are 0) for 20–35 ms, depending on the trial. The flat level is below the ADC limit (±2048 counts = ±6.2 mV), so the amplifier is blocked. A slow recovery follows.
- **MH031 imec3 (normal):** the artefact lasts about 3 ms, no channel saturates, and the spike band does not drop. The KS4 hole from 0 to 5 ms comes from `RemoveArtifacts` (−1..+5 ms), which only this session's KS4 run had.
- **The KS4 burst at +2..+3 ms** comes from spikes detected on the artefact edges that are left after the high-pass filter. The 3–4 ms pre-mean replacement in `tcat_corrected` does not cover them in MH037 / MH020.
- **Auditory trials:** clean in all four cases.

## Pipeline found
- **Raw data:** `/mnt/lsens-data/<mouse>/Recording/<session>/Ephys/<run>/<run>_imecN/<run>_t0.imecN.ap.bin`.
- **CatGT outputs (Myriam):** `Myriam_Hamon/data/.../catgt_<run>/<run>_imecN/`.
  - `tcat`: `-gblcar`, no `-gfix`, no `-apfilter`.
  - `tcat_corrected`: lab `run_artifact_correction.py`, which replaces a 3–4 ms window after each TPrime whisker time (`whisker_stim_times_to_imecN.txt`) with the mean of the preceding window.
- **KS4:** `Axel_Bisi/data/.../catgt_<run>/<run>_imecN/{preprocess,kilosort4}`.
  - Chain: SpikeGLX → astype float32 → [MH031 only: `RemoveArtifacts` linear, 1 ms before / 5 ms after] → DREDGE `InterpolateMotion` → KS4.
  - KS4 settings: `do_CAR=False`, highpass 300 Hz, whitening_range 32, `artifact_threshold=inf`, Th 9/8.
  - The KS4 input traces match `tcat_corrected` (RMS difference 0.1–19 µV, against 28–78 µV for `tcat` and 34–2974 µV for raw). The Kuma-CatGT meta now in Axel's folder (no `-gblcar`, dated 2026-08-06) is newer than the sort and is not what was sorted.
  - MH020 is closest to `tcat_corrected`, but its 0.3–3 ms window is not blanked (uncertain why).

## Files
- `01_extract_stages.py`: per-stage peri-stimulus summaries in `cache/`.
- `02_plot_quantify.py`: `quantification_per_stage.csv`, `fig1_summary_saturation_rms_spikes.png`, and `fig2_stages_<case>.png`.
  - `fig2` rows are the stages (raw, `tcat`, `tcat_corrected`, KS4 input, plus the KS4 high-pass, plus whitening, `temp_wh`, raw plus high-pass).
  - Its columns are 8 channels averaged over trials, single trials, and a channel × time heat map of log10 RMS / baseline for whisker and for auditory trials.
- `03_gain_test.py`: `gain_test.csv`, with absolute spike-band RMS, raw flatness and KS4 amplitudes.
- `04_channel_saturation_vs_gap.py`: `fig3_channel_saturation_vs_gap.png` and `channel_saturation_vs_gap.csv`.

## Fix
Reprocessing cannot bring back spikes that were not recorded, so the 20–35 ms windows have to be treated as missing data:
1. **Mask the dead time.** Write an invalid interval for each session and probe (stimulus to recovery), set where the spike-band RMS of the KS4 input stays below 0.6× baseline, or where the raw signal is flat. Drop those windows, or these sessions, from analyses of early whisker responses and latencies.
2. **Re-sort only to remove the false burst.** Add `spre.remove_artifacts` (`mode='zeros'` or linear, about −0.5..+5 ms, or as long as the saturated / flat samples last) to the KS4 chain before motion correction. This is what MH031 already had. It needs a KS4 rerun on Kuma, then bombcell, ibl_format and the NWB reconversion, for 5 probes (MH037 imec0–3, MH020 imec0). That is roughly a day of compute and does not change the gap.
3. **Screen all sessions.** Run the whisker vs auditory spike-band RMS ratio at 6–18 ms with a cheap version of 01/03 to find other affected sessions.
4. **At acquisition (future):** reduce the coil coupling (shielding and grounding of the coil and its cable, coil distance or orientation, reference choice).
