# render_deck.sh: rebuild the slide images on haas (../build_deck_assets.py), copy them from the NAS (M:), build the
# PowerPoint locally (build_deck.py, python-pptx) and copy it to the NAS. PYX = folder with python-pptx if not installed.
cd "$(dirname "$0")"
NAS=/m/analysis/Axel_Bisi/combined_results_ks4/_sensory_spatial_maps/deck
ssh haas 'cd ~/code/unit_spikes_analysis && ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-sensory-spatial-maps/report/build_deck_assets.py' > /dev/null
mkdir -p img && cp $NAS/img/*.png img/
PYTHONPATH="${PYX:-}" python build_deck.py
cp sensory_deck.pptx $NAS/ 2>/dev/null || cp sensory_deck.pptx $NAS/sensory_deck_$(date +%Y%m%d_%H%M).pptx
