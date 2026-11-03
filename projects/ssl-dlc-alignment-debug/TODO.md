# TODO: ssl-dlc-alignment-debug

- [X] 000 `exploratory-analyses/000_dlc_event_psth.py`: event-aligned DLC PSTHs per trial type and lick, under
      start / end / half-rate alignment, all trials and by session thirds; sharpness summary table.
      Small test first: MH067_20260120 + control MH065_20260116. Outputs in
      `combined_results_ks4/ssl-dlc-alignment-debug/` (figures/<session>_hypotheses.png|pdf,
      figures/<session>_thirds_<hyp>.png|pdf, tables/sharpness.csv, cache/<session>.npz).
- [X] Run all mismatch sessions + controls (same outputs; 8 sessions).
- [ ] Consult user with the figures.
- [ ] Consult user: which alignment holds per session, and what to do with the NWB files.
