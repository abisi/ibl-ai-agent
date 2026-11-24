# TODO

- [X] Diagnosis of MH037 / MH020 / MH031 (2026-11-19, read-only): `diagnosis/00-04_*.py`, `diagnosis/README_diagnosis_20261119.md`;
      outputs in `combined_results_ks4/ssl-whisker-artefact-gap/` (cache/, CSVs, fig1-3).
- [X] Inventory of all ephys session x probe (KS4 + NWB_ks4): `screen/00_inventory.py` -> `screen/tables/00_inventory.csv`.
- [X] Pilot of the per-probe screen on the 4 diagnosis probes (median-over-trials RMS adopted, mean was outlier-driven):
      `screen/01_screen_probe.py`.
- [X] Full screen on haas (2026-11-20; 96 workers stalled on NAS raw reads, rerun with 24): `screen/run_screen.sh` -> `screen/cache/<session>_<probe>.{json,npz}`.
- [X] Table + summary figure + flag (6 probes; flat rule made baseline-relative): `screen/02_summarize.py` -> `screen/tables/02_dead_time_screen.csv`,
      `screen/figures/02_dead_time_screen.{png,pdf,svg}`.
- [X] Screen results + proposed masking rule in screen/RESULTS.md (not applied). CONSULT USER before applying the rule anywhere.
- [ ] (Optional, user decision) re-sort with remove_artifacts to remove the false +2-3 ms burst.
- [ ] User decision: apply the masking rule (mask whisker 0-35 ms on flagged probes, or drop those probes from early-latency analyses).
