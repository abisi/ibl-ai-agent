# Question

## Original (user, 2026-11-19)
Units in MH037_20250524 and MH020_20250304 show almost no KS4 spikes from ~+5 ms to 20-40 ms after whisker onset.
Where does this gap come from, and which other sessions / probes are affected?

## Current explication
- Diagnosis (2026-11-19, `diagnosis/`): the whisker-coil artefact saturates or blocks the probe front end; the spike band
  carries no neural signal until the amplifiers recover. The gap is in the raw SpikeGLX data; no processing step creates it.
- Screen (2026-11-20, `screen/`): every ephys session x probe with a KS4 sorting and an NWB_ks4 file is screened from the
  raw `.ap.bin` with a cheap spike-band measure, to find all probes with this dead time and propose a masking rule.

## Definitions
- Spike band: raw .ap.bin (AB155: merged CatGT tcat), -0.5..+4 ms after onset linearly bridged, Butterworth-3 300 Hz
  high-pass (zero phase), RMS per trial in 1-ms bins, median over trials, ratio to pre-stimulus (-45..-5 ms) per channel,
  median over channels.
- Trials: up to 40 active whisker and 40 active auditory trials (perf != 6, not passive; evenly spaced through the
  session); NWB trial start -> probe-file time by a linear fit on the TPrime whisker times file.
- Dead-time flag: whisker ratio 6-18 ms < 0.7, or > 50 % flat raw samples (zero sample-to-sample differences) in 6-18 ms.
- Recovery time: first time >= 5 ms after whisker onset with 3 consecutive 1-ms bins >= 0.8 x pre-stimulus.
- Saturation fraction: fraction of channel x samples (and of channels) at the ADC limit 0-4 ms after whisker onset.
