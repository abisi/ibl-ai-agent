"""SuppFigure4_rastermap_validation -- split from SuppFigure3_rastermap_clustering (user 2026-11-19). The compute_<key>
functions live in ../SuppFigure3_rastermap_clustering/compute.py (one source); this wrapper runs them for this spec's
panels and writes this figure's snapshot.

Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. .venv/bin/python \
  ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure4_rastermap_validation/compute.py [--only key ...]
then build.py figures/SuppFigure4_rastermap_validation.
"""
import importlib.util
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("sf3_compute", FIG_DIR.parent / "SuppFigure3_rastermap_clustering" / "compute.py")
sf3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sf3)
sf3.SPEC = sf3.load_spec(FIG_DIR)
sf3.PANELS = {p.get("key", p["id"]): p for p in sf3.SPEC["panels"]}

if __name__ == "__main__":
    sf3.main()
