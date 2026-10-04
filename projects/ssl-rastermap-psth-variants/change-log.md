# change-log

- 2026-09-25: created question.md, TODO.md; test run done; full run launched.
- 2026-09-26: added ROC/metacluster EDA step to TODO; full run relaunched after disk-full failure.
- 2026-09-26: added population_matrix_summary step (user-confirmed ROC category definitions).
- 2026-09-27: lick_time offset found (NWB lick_time = response_window_start + RT, RT counts from stimulus): corrected lick = start_time + lick_time - response_window_start_time; rastermap_utils `correct_lick_time=True`; lickfix reruns (010 RPE pilot v3, 015 rebuilds); skill `ssl-lick-alignment`.
- 2026-09-27: publication matrix made flexible (`fig5_population_matrix_publication(anatomy=, function=, enrichment='share'|'fmk'|None, meta_style='dendrogram+strip'|'dendrogram', row_mode=)`); default = waveform, area, SSp-whisker proj., CC hierarchy | stim., lick-resp., modality, decision, gated/decision | R+ share; user-requested renames and rotated column titles.
- 2026-10-03: pre-lick convergence analyses locked (prelick_convergence_LOCKED.md): distance-based main figures, Fig1h, Fig2f/g (area bars + ANOVA), Fig2S CCF maps, Fig3 i-j, Fig3S LDA plane, Fig5 pseudo-pop rows (064 new readouts), COSYNE v3, article PDFs per variant.
- 2026-10-03: pre-lick convergence analyses (051–071) split into projects/ssl-prelick-convergence (copies; originals kept here until haas jobs finish).
