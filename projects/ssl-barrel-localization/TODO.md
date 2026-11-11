# ssl-barrel-localization TODO

- [X] 000 inventory: brains with a D2 fit, fit quality, barrels visible (Lana's visual check), SSL ephys sessions and
      probes per brain
- [X] 001 link units -> brain voxel -> D2 barrel / layer / corrected CCF / flat coordinate; test which NWB coordinate set
      (`ccf_atlas_*` or `ccf_*`) matches the registration; QC: residual of the inverse field, distance to the DiI track,
      area label agreement; pilot AB156, AB164, MH064
- [X] 002 functional table: ROC (whisker / auditory passive pre, active, whisker vs auditory), evoked responsiveness and
      latency from the NWB spikes (passive whisker / auditory trials)
- [X] 003 maps: probes and units on the flat barrel map; coronal / depth views per barrel; per-probe barrel and layer table
- [X] pilot AB156, AB164, MH064 run (2026-11-11). Linking decision pending: track route (DiI track + atlas depth,
      scale from Lana's registration) is primary, as it reproduces Lana's per-track barrel calls; the coordinate route
      fails where the NWB coordinates come from an earlier registration (MH, AB164). Fix before scaling: latency floor
      at 5 ms (2-ms smoothing leaks the response before the window start); flat-map panel cropping and labels.
- [ ] pilot review with the user  **consult user**
- [ ] all barrel-field brains; C2 vs surround vs septa by layer; R+ / R-; registration validation
- [ ] full results to the user  **consult user**; report
