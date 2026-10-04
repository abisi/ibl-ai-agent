# render.sh: copy the current figures from the NAS (M:) and render the report PDF locally with Quarto (typst)
S=/m/analysis/Axel_Bisi/combined_results_ks4
mkdir -p fig
cp $S/_sensory_spatial_maps/figures/{cortical_flatmaps,projection_zones_coronal,projection_zones_sagittal,colocation_figure}.png fig/
for q in whisker modality latency_whisker; do cp $S/_sensory_spatial_maps/figures/$q/targets_p1.png fig/${q}_targets_p1.png; done
cp $S/_stimulus_arrival/figures/{arrival_summary_area_group,arrival_summary_area_acronym_custom,arrival_main_N200_area_group}.png fig/
"/c/Program Files/Quarto/bin/quarto.exe" render sensory_maps_report.qmd --to typst
cp sensory_maps_report.pdf $S/_sensory_spatial_maps/
