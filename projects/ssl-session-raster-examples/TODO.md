# ssl-session-raster-examples TODO

Goal: example whole-session raster figures around a whisker trial, for every session with >= 4 valid probes;
default (black) vs area_group-coloured versions side by side.

- [X] 000 plot_session_raster (unchanged) on MH062_20260113_125836 -> user check
      (combined_results_ks4/MH062/whisker_4/session_raster/session_raster_whisker_trial_*_default_vs_area_group.*)
- [X] 001 publication-style raster (Figma schematic style, nose_side_distance = side nose_distance), pilot MH062
- [X] 001 lick-bout criterion (>= 5 piezo licks within 2 s of lick onset, jaw onset before lick onset); all 74
      sessions on haas: 71 done, 3 without side/top video (AB105_20240314, MH064_20260117, MH065_20260115).
      Figures: combined_results_ks4/<mouse>/whisker_<day>/session_raster/paper_raster_whisker_hit_*;
      ranking: combined_results_ks4/ssl-session-raster-examples/tables/001_session_ranking.csv
- [X] 001 style revisions: allen_utils area order, compact 150 mm layout, window -0.5..2 s, bigger semi-transparent
      spikes, thicker DLC lines; vector formats with the spike layer at 1200 dpi + all-vector _vector.svg
- [X] 001 no piezo lick before lick onset - 30 ms (54 / 71 sessions keep a preferred trial)
- [X] 002 pre-stimulus 8-9 Hz population rhythm in AB149 (MO / FRO / SSp-orof, TH, STR; low pupil, low whisking)
- [X] 001 top 5 preferred trials per session (235 figures; rank 1 all formats, ranks 2-5 png/pdf/svg);
      003 contact sheet combined_results_ks4/ssl-session-raster-examples/example_contact_sheet.pdf
- [ ] lick-bout criterion counts licks up to lick onset + 2 s, beyond the plotted window (stim + 2 s): restrict
- [ ] user picks the example(s); polish (label overlap when jaw/lick onsets are close, area short names)  **consult user**
