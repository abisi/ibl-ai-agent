# change-log

- 2026-10-15: project created by moving the passive hit / miss axis analyses (132-135b, 140, 145-147, 149-151, 153, 154, 156,
  158, 160-175) out of ssl-whisker-hitmiss-timeresolved-decoding; results moved on the NAS with symlinks at the old paths
  (one locked file, figures/publication/175_across_days_passive_projection.pdf, left in place); shared modules re-exported;
  figures now written to this project's figures/publication. Script history before this date is in the change-log of
  ssl-whisker-hitmiss-timeresolved-decoding.
- 2026-10-15: first report (report/build_report.py on report_lib; report/ on the NAS): Figure 1 = 158 (within day 0), Figure 2 = 165 hit - miss row (new SSL_165_ROW1=1 with SSL_165_ANGLE=1: angle and projection beyond session time, areas), Figure 3 = new 177_report_across_days (119 row-1 panels + 175 v2 panels with passive-post auditory and the per-area panel). Supplementary: 174 schematic, 160 decoder, 153 exposure, 156 expert control. Not included: 162 and 164 (same quantities as Figures 2e / 1d), 175 v1 / v2 (replaced by 177), Part III figures of the old decoding report (135b, 145, 147-151).
