# TODO (ssl-spectral-coupling)

Results: `combined_results_ks4/ssl-spectral-coupling/` (NAS). Code: this folder. Compute: haas.
User asked for an autonomous overnight run (2026-11-17): no checkpoints during the night; consult in the morning
before any confirmatory follow-up.

- [X] 000 project docs (question.md, TODO.md, change-log.md) with pre-registered tests
- [X] spec_lib.py: loading (pilot v2 loader + warm-up cut), 5-ms tensors, multitaper spectra, MSC / imag coherency /
      canonical coherence / partial coherence / PSI / spectral Granger, shuffle null
- [X] 001 simulation validation -> simulation/001_simulation.{csv,png,pdf}
- [X] 002 spike run: quick test AB127, then 89 day-0 sessions (45 workers, ~40 min, no errors)
      -> spikes/sessions/*_{bands,spectra}.parquet, _meta.json, spikes/sessions_included.csv, spikes/config.json
- [X] 003 stats + figures (learners + all mice, area group + acronym) -> spikes/stats/<pop>/*.csv,
      spikes/figures/<pop>/fig1-5
- [X] 004 AP-band check on AB127 (clock, artefact latency, envelope vs spikes) -> ap/check/004_*.{txt,png}
- [X] 005 AP envelope caches (88/89; AB138 added by ~/ssl_spectral_ap_chain2.sh, done flag logs/ap_chain2_done) (ap/cache) -> 002 --source ap -> 003 --source ap (chain ~/ssl_spectral_ap_chain.sh on
      haas, logs/005_*, 002_ap.log, 003_ap.log, done flag logs/ap_chain_done)
- [X] report v1 (spikes only): report/build_report.py, render.sh -> report/report.{pdf,md,html}
- [X] report v2 with AP results (report/report.{pdf,md,html})
- [ ] publish to the private reports repo (ask first); push commits to the fork (ask first)
- [ ] Consult user: interpretation of "ap band" (envelope, not LFP), windows, lick-aligned execution, LFP coherence
- [ ] Optional follow-ups (ask): lick-aligned execution windows / lick-rate matching for the AP execution effect; LFP (.lf.bin) coherence; pooled-trial pseudo-sessions for power
