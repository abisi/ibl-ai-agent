"""Areas decoded in ssl-stimulus-arrival-decoding (user, 2026-10-04): every area group, and the 40 fine areas
(area_acronym_custom) with the most recorded good + mua neurons; epoch switch (active / passive) and output folders."""
import os
import pathlib

import pandas as pd

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
UNITS = RES / "_roc_stage_analysis" / "units.parquet"
EPOCH = os.environ.get("ARRIVAL_EPOCH", "active")                 # "active" (task trials) or "passive"
assert EPOCH in ("active", "passive"), EPOCH
OUT = pathlib.Path(os.environ.get("ARRIVAL_OUT", RES / ("_stimulus_arrival" if EPOCH == "active" else "_stimulus_arrival_passive")))
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
             "Somatosensory-body": "Somatosensory areas-whisker", "Auditory areas": "Auditory areas",
             "Retrosplenial areas": "Retrosplenial areas", "Posterior parietal areas": "Posterior parietal areas",
             "Visual areas": "Posterior parietal areas", "Insular areas": "Cortical subplate", "Hippocampus": "Hippocampus",
             "Striatum": "Striatum and pallidum", "Pallidum": "Striatum and pallidum", "Lateral septal complex": "Striatum and pallidum",
             "Thalamus": "Thalamus", "Midbrain": "Midbrain", "Olfactory areas": "Olfactory areas",
             "Amygdala and hypothalamus": "Amygdala and hypothalamus", "Cortical subplate": "Cortical subplate",
             "Pons and medulla": "Pons and medulla"}


def colors():
    """area -> colour: groups from allen_utils (Axel Bisi's palette), fine areas in shades of their parent group"""
    import sys
    import numpy as np
    from matplotlib.colors import to_hex, to_rgb
    try:
        sys.path.insert(0, str(pathlib.Path.home() / "code"))
        from allen_utils import allen_utils as au
        pal = au.get_custom_area_groups_colors()
    except Exception:
        pal = {}
    col = {g: pal.get(GROUP_KEY.get(g, ""), "#888888") for g in COARSE}
    for parent in set(FINE_PARENT.values()):
        kids = [a for a in FINE if FINE_PARENT.get(a) == parent]
        base = np.array(to_rgb(col.get(parent, "#888888")))
        for i, a in enumerate(kids):
            f = np.linspace(-0.45, 0.45, len(kids))[i] if len(kids) > 1 else 0.0
            c = base * (1 - f) if f > 0 else base + (1 - base) * (-f)
            col[a] = to_hex(np.clip(c, 0, 1))
    return col
