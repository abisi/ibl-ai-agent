# render_deck.sh: on haas, write report_numbers.json (../build_report.py, from the result tables) and the slide images
# (../build_deck_assets.py); copy both from the NAS (M:), build the PowerPoint locally (build_deck.py, python-pptx) and copy
# it to the NAS. PYX = folder with python-pptx if it is not installed.
cd "$(dirname "$0")"
NAS=/m/analysis/Axel_Bisi/combined_results_ks4/_sensory_spatial_maps
ssh haas 'cd ~/code/unit_spikes_analysis && ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-sensory-spatial-maps/report/build_report.py && ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-sensory-spatial-maps/report/build_deck_assets.py' > /dev/null
mkdir -p img && cp $NAS/deck/img/*.png img/ && cp $NAS/report/report_numbers.json .
PYTHONPATH="${PYX:-}" python build_deck.py
[ "${NO_NAS:-0}" = 1 ] || cp sensory_deck.pptx $NAS/deck/ 2>/dev/null || cp sensory_deck.pptx $NAS/deck/sensory_deck_$(date +%Y%m%d_%H%M).pptx
