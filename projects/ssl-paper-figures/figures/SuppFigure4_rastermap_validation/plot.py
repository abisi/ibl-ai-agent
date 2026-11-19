"""SuppFigure4_rastermap_validation -- plot_<key> functions shared with SuppFigure3_rastermap_clustering (one source)."""
import importlib.util
from pathlib import Path

_s = importlib.util.spec_from_file_location(
    "sf3_plot", Path(__file__).resolve().parent.parent / "SuppFigure3_rastermap_clustering" / "plot.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
globals().update({k: v for k, v in vars(_m).items() if k.startswith("plot_")})
