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


# allen_utils.get_custom_area_groups_colors() keys for our area_group names (groups without a key take the nearest one)
GROUP_KEY = {"Motor areas": "Motor and frontal areas", "Frontal areas": "Motor and frontal areas",
             "Somatosensory-whisker": "Somatosensory areas-whisker", "Somatosensory-orofacial": "Somatosensory areas-orofacial",
             "Somatosensory-body": "Somatosensory areas-orofacial", "Auditory areas": "Auditory areas",
             "Retrosplenial areas": "Retrosplenial areas", "Posterior parietal areas": "Posterior parietal areas",
             "Visual areas": "Posterior parietal areas", "Insular areas": "Somatosensory areas-orofacial", "Hippocampus": "Hippocampus",
             "Striatum": "Striatum and pallidum", "Pallidum": "Striatum and pallidum", "Lateral septal complex": "Striatum and pallidum",
             "Thalamus": "Thalamus", "Midbrain": "Midbrain", "Olfactory areas": "Olfactory areas",
             "Amygdala and hypothalamus": "Amygdala and hypothalamus", "Cortical subplate": "Cortical subplate",
             "Pons and medulla": "Pons and medulla"}


FINE_EXTRA = {"SSp-body": "Somatosensory areas-orofacial"}   # body areas (SSp-ll/-ul/-tr) are orofacial in allen_utils


def _allen():
    """allen_utils palette (Axel Bisi): {custom group: colour}, {area acronym: custom group}"""
    import sys
    sys.path.insert(0, str(pathlib.Path.home() / "code"))
    from allen_utils import allen_utils as au
    return au.get_custom_area_groups_colors(), au.get_custom_area_groups_from_name()


def allen_parent():
    """decoded area group / fine area -> allen_utils custom group (user 2026-10-06: colours from
    allen_utils.get_custom_area_groups_colors, sub-areas in shades of their group). Fine areas: allen_utils name lookup,
    else their decoded group's parent. Decoded groups: the allen group holding most of their good + mua units (by
    area_acronym_custom) only when GROUP_KEY has no entry."""
    pal, from_name = _allen()
    U = pd.read_parquet(UNITS, columns=KEYS + ["quality_label", "area_group", "area_acronym_custom"])
    U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    U["allen"] = U.area_acronym_custom.map(from_name)
    par = {}
    for g in COARSE:                                  # explicit GROUP_KEY first, else the majority allen_utils group
        v = U[U.area_group == g].allen.dropna().value_counts()
        par[g] = GROUP_KEY.get(g) or (v.index[0] if len(v) else None)
    for a in FINE:                                    # allen_utils name lookup, else FINE_EXTRA, else the decoded group's
        par[a] = from_name.get(a) or FINE_EXTRA.get(a) or par.get(FINE_PARENT.get(a))
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
    """area -> colour (user 2026-10-06). Area groups: exactly allen_utils.get_custom_area_groups_colors() of their
    allen_utils group (allen_parent); groups in the same allen_utils group therefore share its colour. Fine areas: shades
    of their group's colour (one shade per fine area within a colour family, ordered by number of units)."""
    try:
        par, pal = allen_parent()
    except Exception:
        par, pal = {}, {}
    col = {g: pal.get(par.get(g), "#888888") for g in COARSE}
    by = {}
    for a in FINE:
        by.setdefault(pal.get(par.get(a), "#888888"), []).append(a)
    for base, kids in by.items():
        for a, c in zip(kids, shades(base, len(kids))):
            col[a] = c
    return col
