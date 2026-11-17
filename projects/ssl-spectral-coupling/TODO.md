# TODO (ssl-spectral-coupling)

Results: `combined_results_ks4/ssl-spectral-coupling/` (NAS). Code: this folder. Compute: haas.
User asked for an autonomous overnight run (2026-11-17): no checkpoints during the night; consult in the morning
before any confirmatory follow-up.

- [ ] 000 project docs (question.md, TODO.md, change-log.md) with pre-registered tests
- [ ] spec_lib.py: loading (pilot v2 loader + warm-up cut), 5-ms tensors, multitaper spectra, MSC / imag coherency /
      canonical coherence / partial coherence / PSI / spectral Granger, shuffle null
- [ ] 001 simulation validation (known coupling frequency and direction, common input, null)
- [ ] 002 spike run, quick test on one session (timing, sanity), then all day-0 sessions on haas
- [ ] 003 stats + figures (H1-H3, learners + all mice, area group + acronym levels)
- [ ] 004 AP-band feasibility: clock alignment, channel mapping, read speed, envelope check on one session
- [ ] 005 AP-band run (subset or all sessions, time permitting) + stats via 003
- [ ] report (project-report skill) + summary for the user
- [ ] Consult user: interpretation of "ap band", windows, whether to add lick-aligned execution windows
