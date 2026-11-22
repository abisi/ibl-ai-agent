# ssl-barrel-localization TODO

- [X] 000 inventory: brains with a D2 fit, fit quality, barrels visible (Lana's visual check), SSL ephys sessions and
      probes per brain
- [X] 001 link units -> brain voxel -> D2 barrel / layer / corrected CCF / flat coordinate; test which NWB coordinate set
      (`ccf_atlas_*` or `ccf_*`) matches the registration; QC: residual of the inverse field, distance to the DiI track,
      area label agreement; pilot AB156, AB164, MH064
- [X] 002 functional table: ROC (whisker / auditory passive pre, active, whisker vs auditory), evoked responsiveness and
      latency from the NWB spikes (passive whisker / auditory trials)
- [X] 003 maps: probes and units on the flat barrel map; coronal / depth views per barrel; per-probe barrel and layer table
- [X] pilot AB156, AB164, MH064 (2026-11-11). Link (user): unit global CCF = peak channel in channel_locations.json
      (IBL alignment). The atlas-space and sample-space DiI tracks of brainglobe disagree by 85-176 um (median) under
      the deformation field, so units are placed on the physical DiI track at their aligned arc fraction (track
      transfer, `trk_`); this reproduces Lana's per-track barrel calls. Direct inverse mapping and global labels kept
      for comparison. Latency unsmoothed (user).
- [X] pilot review with the user: run all brains (2026-11-11)
- [X] all barrel-field brains (2026-11-11): 49 of 58 D2 brains have a wS1 probe and are linked (57 probes); MH007, MH020
      have no units on a DiI track. Global vs D2 barrel agree for 17 % of units (median shift 277 um, pilot AB156 was
      the exception). C2 L4 latency 7 ms vs 9 ms in neighbours.
- [X] rerun on the final user-picked D2 fits (ws1-barrel-registration/D2_final_user_picked, 2026-11-22): 000 reads the
      final table (85 use_local, 16 use_global; per mouse the Anatomy brain), 001 reads <brain>/results; use_global
      brains placed with the NWB global registration (glob_barrel / glob_layer, global flat position via the Blue Brain
      flatmap; lookup check 99.8 % barrel agreement). Maps: 41 mice D2 (14934 units), 7 mice global (1160 units).
      MH007, MH008 have no wS1-targeted probe in the current NWBs (dropped). Old outputs in _archive_lana_d2_neuropixels_20261111/.
- [ ] full results to the user  **consult user**; report
