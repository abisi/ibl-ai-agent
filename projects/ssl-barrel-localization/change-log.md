
## 2026-11-22 -- final user-picked registration; first report

- 000-005 rerun on ws1-barrel-registration/D2_final_user_picked (85 use_local, 16 use_global brains). use_global brains
  placed with the NWB global registration (global flat position via the Blue Brain flatmap; 99.8 % barrel agreement).
  005 (tissue, C2 test, DiI check, fit QC) on the final fits; use_global brains skipped there.
- Stale outputs removed: MH007 / MH008 (no wS1-targeted probe in the current NWBs), 005 tissue figures of use_global
  brains, the AB156 / AB164 / MH064 pilot figures and tables. Everything before is in _archive_lana_d2_neuropixels_20261111/.
- Report built (report/build_report.py): 4 main figures, 3 supplementary, 5 tables; published to abisi/ibl-ai-agent-reports
  docs/ssl-barrel-localization/.
