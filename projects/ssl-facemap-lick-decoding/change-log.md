# change-log: ssl-facemap-lick-decoding

- 2026-11-03/05: pipeline 001-007 written and run (clips, Facemap on Kuma, decoding, automatic ROIs, ROI vs DLC).
- 2026-11-10: tidy-up. `dlc_kpca.py` added (DLC helpers moved verbatim from `ssl-face-lick-decoding/001_pilot_decoding.py`);
  `003_analysis.py` and `ssl-iti-signal-screen/001_features.py` now import it, so `ssl-face-lick-decoding` is no longer
  imported by this project or the ITI screen. Docstring references in 003 / 006 updated. Added `README.md`,
  `008_report_figures.py` (report figures from existing tables, no recomputation), `report/build_report.py`,
  `report/render.sh`. No scripts superseded (no archive needed); no result table overwritten.
- 2026-11-10: first report build (report/: PDF, md, html; 6 main figures, 5 tables + 4 supplementary tables).
