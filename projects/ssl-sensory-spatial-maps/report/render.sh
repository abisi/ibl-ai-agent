# render.sh: build the report source on haas (build_report.py, from the result tables), copy it and its figures from the
# NAS (M:), render the PDF locally with Quarto (typst) and copy it back to the NAS
cd "$(dirname "$0")"
ssh haas 'cd ~/code/unit_spikes_analysis && ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-sensory-spatial-maps/report/build_report.py' 2>&1 | grep -v -i warn | tail -1
R=/m/analysis/Axel_Bisi/combined_results_ks4/_sensory_spatial_maps/report
mkdir -p fig
cp $R/sensory_maps_report.qmd .
while IFS=$'\t' read -r src name; do cp "${src/\/mnt\/lsens-analysis/\/m\/analysis}" "fig/$name"; done < $R/report_figures.txt
"/c/Program Files/Quarto/bin/quarto.exe" render sensory_maps_report.qmd --to typst 2>&1 | tail -1
cp sensory_maps_report.pdf /m/analysis/Axel_Bisi/combined_results_ks4/_sensory_spatial_maps/ 2>/dev/null || cp sensory_maps_report.pdf /m/analysis/Axel_Bisi/combined_results_ks4/_sensory_spatial_maps/sensory_maps_report_$(date +%Y%m%d_%H%M).pdf
ls -la sensory_maps_report.pdf
