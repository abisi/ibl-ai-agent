"""Recompute one row of the Figure 2G cache (G_maps.pkl) without the full compute (iteration helper).
Run (haas): PYTHONPATH=<NAS ephys_utilities>:~/py_extra ~/code/unit_spikes_analysis/.venv/bin/python G_update_row.py barrel
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("g", HERE / "G_response_maps.py")
g = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g)


def main(row):
    with open(g.CACHE, "rb") as f:
        R = pickle.load(f)
    A = g.m3.Atlas(load_zones=False)
    u = g.load_units()
    u = u[A.unit_ids(u) != 0].reset_index(drop=True)
    lut, _ = g.group_lut(A)
    Z, Zs, norm99, P = g.load_projection()
    if row == "barrel":
        R["barrel"] = g.barrel_row(u, A, Z, Zs, norm99, lut)
    elif row == "frontal":
        R["frontal"] = g.frontal_row(u, A, P, Zs, lut)
    else:
        raise ValueError(row)
    with open(g.CACHE.with_suffix(".tmp"), "wb") as f:
        pickle.dump(R, f)
    g.CACHE.with_suffix(".tmp").replace(g.CACHE)
    print("updated", row, len(R[row]["units"]))


if __name__ == "__main__":
    main(sys.argv[1])
