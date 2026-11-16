# TODO

- [x] 000 units and rates (2026-11-03)
- [x] 001 reference point (2026-11-03; scan of NWB_ks4 in tables/dlc_reference_at_origin.csv)
- [x] 002 cleaning steps (2026-11-03)
- [x] 003 DLC lick times (2026-11-03)
- [x] rig scale for MH sessions whose yaml says setup 2 (004; converter clause `setup2_sessions`, reconverted 2026-11-03)
- [x] cleaning defaults (user 2026-11-04: stored values returned, no likelihood mask by default; pupil running median 0.1 s
      with the pupil QC; cutoffs 0.1 for pupil / whisker / nose when a mask is requested)
- [x] low jaw likelihood sessions (005: jaw visible and well placed; jaw not gated by likelihood)
- [x] top orientation (008; converter rotates upside-down top DLC from the data since 2026-11-04)
- [x] event-locked overview of all main series through the module (010, 011)
- [x] pupil QC cleaning and outlier review (013-015)
- [x] video / DLC audit (018) and failed top-view frames (019)
- [x] restructure into sections + report (2026-11-10)
- [ ] user: failed top-view sessions (MH009 0207 / 0208, MH025 0421 / 0422, MH027 0422): exclude from top-view analyses
      or replace with the merged-model re-inference (ssl-dlc-kuma)
- [ ] jaw angle scale depends on the jaw_ref placement of the DLC model: compare jaw angles only within one model, or
      re-derive with a fixed reference after the DLC rerun decision
- [ ] then switch dependent analyses to dlc_helpers
