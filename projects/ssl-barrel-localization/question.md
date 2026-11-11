# Question

Where did the Neuropixels probes and their units go in the whisker barrel field (wS1), which barrels and layers, and how do
whisker and auditory responses vary with position relative to the stimulated C2 barrel?

Data
- Lana Smith's local barrel registration (method D2, barrelreg; read only):
  `/raid0/lmsmith/local_barrel_reg/D2_neuropixels/<brain>/` on haas: barrel per brain voxel (`barrels_corrected_full.tif`,
  names in `barrel_index.csv`), corrected CCF per voxel (`ccf_field_{0,1,2}.tif`, mm), flat L4 coordinates
  (`h1/flatcoords.npz`), fit metrics (`h1/metrics.json`); left hemisphere only. Coarse brainreg registration (deformation
  fields, registered atlas) in `/raid0/lmsmith/{AB,MH}_brains/<brain>/fused/registered/`. Figures and summary on the share:
  `Lana_Smith/neuropixels/`, `local_barrel_reg/SUMMARY.md`.
- NWB_ks4 units: CCF position per unit and channel (`ccf_atlas_*`, `ccf_*`), peak channel, depth; ROC results per session
  (`combined_results_ks4/<mouse>/<day>/roc_analysis`).

Link (user decisions 2026-11-11)
- Units -> brain voxel through the inverse of the brainreg deformation field (unit CCF -> nearest brain voxel), then Lana's
  barrel / layer / corrected CCF / flat coordinate at that voxel. Cross-check: distance of the unit voxel to the annotated
  DiI track, and NWB area label vs the registered atlas at the voxel.
- AB mice: one set of DiI tracks per brain, linked to every ephys session of the mouse (same insertion every day).
- Population: all SSL ephys mice with a barrel-field track; all good / mua units, stable units flagged; R+ / R- separately.
- Lana's per-track tables (`tracks_to_barrels.py`, in progress) are compared where they exist; her volumes are read only.

Functional characterisation: whisker and auditory responsiveness (ROC whisker / auditory passive and active vs baseline;
evoked vs baseline rate), latency, whisker vs auditory selectivity; mapped on the flat barrel field and summarised as
C2 / surround barrels / septa by layer. Validation of the registration: with the C2 whisker stimulated, C2 units should
respond earliest and most.
