"""Areas decoded in ssl-stimulus-arrival-decoding (user, 2026-10-04): every area group, and the 40 fine areas
(area_acronym_custom) with the most recorded good + mua neurons; epoch switch (active / passive) and output folders."""
import os
import pathlib

import pandas as pd

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
UNITS = RES / "_roc_stage_analysis" / "units.parquet"
EPOCH = os.environ.get("ARRIVAL_EPOCH", "active")                 # "active" (task trials) or "passive"
assert EPOCH in ("active", "passive"), EPOCH
HOME = RES / "ssl-stimulus-arrival-decoding"                     # project results home: active/, passive/, final_n200/
OUT = pathlib.Path(os.environ.get("ARRIVAL_OUT", HOME / EPOCH))
N_FINE = 40
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]


def _lists():
    U = pd.read_parquet(UNITS, columns=KEYS + ["quality_label", "area_group", "area_acronym_custom"])
    U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    groups = U.area_group.dropna().value_counts().index.tolist()
    fine = U.area_acronym_custom.dropna().value_counts().index[:N_FINE].tolist()
    parent = (U[U.area_acronym_custom.isin(fine)].groupby("area_acronym_custom").area_group
              .agg(lambda x: x.value_counts().index[0]).to_dict())
    return groups, fine, parent


COARSE, FINE, FINE_PARENT = _lists()
LEVELS = {"area_group": COARSE, "area_acronym_custom": FINE}


def _allen():
    """area-group palette and membership from ephys_utilities.allen_utils.allen_utils (Axel Bisi, user 2026-10-06):
    get_custom_area_groups_colors() is keyed by the same split group names as the unit table's area_group;
    get_custom_area_groups_from_name() gives each fine area's group"""
    import sys
    sys.path.insert(0, str(pathlib.Path.home() / "code" / "ephys_utilities"))
    from ephys_utilities.allen_utils import allen_utils as au
    return au.get_custom_area_groups_colors(), au.get_custom_area_groups_from_name()


def allen_parent():
    """area -> its area group: groups are their own parent; fine areas from get_custom_area_groups_from_name(), else the
    group holding most of their units in the unit table (FINE_PARENT)"""
    pal, from_name = _allen()
    par = {g: g for g in COARSE}
    for a in FINE:
        par[a] = from_name.get(a) or FINE_PARENT.get(a)
    return par, pal


def shades(base, n):
    """n distinguishable shades of one colour, darker to lighter, base in the middle (never near white)"""
    import numpy as np
    from matplotlib.colors import to_hex, to_rgb
    b = np.array(to_rgb(base))
    if n == 1:
        return [to_hex(b)]
    out = []
    lum = 0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2]
    lo, hi = -0.38, min(0.45, 0.75 * (1 - lum))      # no near-black, and light bases are not pushed to near-white
    for f in np.linspace(lo, hi, n):                  # < 0: towards black, > 0: towards white
        c = b * (1 + f) if f < 0 else b + (1 - b) * f
        out.append(to_hex(np.clip(c, 0, 1)))
    return out


def colors():
    """area -> colour (user 2026-10-06): area groups exactly get_custom_area_groups_colors()[group]; fine areas in
    shades of their group's colour (one shade per area of the group, ordered by number of units: largest darkest)"""
    par, pal = allen_parent()
    missing = [g for g in COARSE if g not in pal]
    if missing:
        raise KeyError(f"area groups without a palette colour: {missing}")
    col = {g: pal[g] for g in COARSE}
    by = {}
    for a in FINE:
        by.setdefault(par.get(a), []).append(a)
    for g, kids in by.items():
        for a, c in zip(kids, shades(pal.get(g, "#888888"), len(kids))):
            col[a] = c
    return col
