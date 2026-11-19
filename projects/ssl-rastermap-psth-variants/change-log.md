# change-log

- 2026-09-25: created question.md, TODO.md; test run done; full run launched.
- 2026-09-26: added ROC/metacluster EDA step to TODO; full run relaunched after disk-full failure.
- 2026-09-26: added population_matrix_summary step (user-confirmed ROC category definitions).
- 2026-09-27: lick_time offset found (NWB lick_time = response_window_start + RT, RT counts from stimulus): corrected lick = start_time + lick_time - response_window_start_time; rastermap_utils `correct_lick_time=True`; lickfix reruns (010 RPE pilot v3, 015 rebuilds); skill `ssl-lick-alignment`.
- 2026-09-27: publication matrix made flexible (`fig5_population_matrix_publication(anatomy=, function=, enrichment='share'|'fmk'|None, meta_style='dendrogram+strip'|'dendrogram', row_mode=)`); default = waveform, area, SSp-whisker proj., CC hierarchy | stim., lick-resp., modality, decision, gated/decision | R+ share; user-requested renames and rotated column titles.
- 2026-10-03: pre-lick convergence analyses locked (prelick_convergence_LOCKED.md): distance-based main figures, Fig1h, Fig2f/g (area bars + ANOVA), Fig2S CCF maps, Fig3 i-j, Fig3S LDA plane, Fig5 pseudo-pop rows (064 new readouts), COSYNE v3, article PDFs per variant.
- 2026-10-03: pre-lick convergence analyses (051–071) split into projects/ssl-prelick-convergence (copies; originals kept here until haas jobs finish).
- 2026-11-12: 045_metacluster_description.py (user: Figure 2 metacluster description at A4 width, prototype in _publication_layout_preview): 5 largest metaclusters (contiguous, cut 0.40) of the lickfix publication variant; PSTHs merged into passive / active stimulus-aligned / lick-aligned sub-panels, area-group composition, sampling deviation (log2 vs dataset, Fisher + BH), ROC category fractions vs all neurons (Fisher + BH); table csv with variant + method_dir.
- 2026-11-19: 046 / 047 spectral (Bayesian) GMM on the main variant (Supp. Fig. 4); 045 --name, --cols, --fs, --width-mm (Supp. Fig. 3).
- 2026-11-19: cleanup: README.md (script index, main variant, figure links); lick-timing, reward-free MMD, ROC-rate and RPE / novelty scripts moved to topic subfolders of exploratory-analyses; local population_matrix_summary.py (older copy, not imported) removed.
