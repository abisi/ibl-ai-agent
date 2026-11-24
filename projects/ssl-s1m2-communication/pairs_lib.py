"""Connected-pairs extension of the S1-M2 time-shifted CCA (user 2026-11-20): area definitions, unit selection, pair list
and the optional probe exclusion. Imports s1m2_lib (not edited: another session owns it); results go to OUT/pairs/.

Areas (short name -> area_acronym_custom labels of unit_info_all_days_v2):
  wS1 SSp-bfd; wS2 SSs; A1 AUD (the table has no AUDp / secondary split; A1-targeted probes); PPC; DLS / DMS / TS
  (striatum split by bregma coordinates, allen_utils.create_areas_subdivisions: DLS AP -1.5..+0.5, ML > 2.4 mm; DMS
  ML < 3 mm, AP > -1.5; TS AP < -1.5 mm); SC = SCs + SCm; MO-wM1 / MO-wM2 / MO-ALM / MO-tjM1 (MOs / MOp of probes
  targeted there); MO-FC = MOs / MOp of OFC-targeted probes (allen_utils cc65eac rule, applied here because the v2 table
  predates it); MRN; RSP.
Units: as v3 (002_sessions.stable_units): good + mua, 137_stable_units.stable, no rate threshold, >= 5 per area
(PAIR_CONFIG; v3 used 10).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import s1m2_lib as L

OUT = L.OUT / "pairs"
TABLES, FIGS = OUT / "tables", OUT / "figures"
ROC_UNITS = L.ROOT / "combined_results_ks4" / "ssl-roc-learning-stages" / "units.parquet"   # CCF coordinates
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]

AREAS = {
    "wS1": ["SSp-bfd"], "wS2": ["SSs"], "A1": ["AUD"], "PPC": ["PPC"], "DLS": ["DLS"], "DMS": ["DMS"],
    "SC": ["SCs", "SCm"], "MO-wM1": ["MO-wM1"], "MO-wM2": ["MO-wM2"], "TS": ["TS"], "MO-ALM": ["MO-ALM"],
    "MO-tjM1": ["MO-tjM1"], "MO-FC": ["MO-FC"], "MRN": ["MRN"], "RSP": ["RSP"],
}
SOURCES = ["wS1", "wS2", "A1"]          # areas with extracted projection data (anatomical source = leading at delay > 0)
ALLEN_SRC = {"wS1": "SSp-bfd", "wS2": "SSs", "A1": "AUDp"}   # projection_zones.npz keys
LIU_SRC = {"wS1": "s1", "wS2": "s2"}                          # Liu et al. 2024 injection sites (no A1)

PAIR_CONFIG = {
    "min_units": 5,                                           # stable good + mua units per area (user 2026-11-20: 5,
                                                              # v3 used 10; PCA keeps min(pca_dims, n units) dims)
    "min_sessions": 3,                                        # per cohort x stage (ssl_area_inclusion_within_cohort)
    # probe exclusion (amplifier dead time after whisker onset; ssl-whisker-artefact-gap). CSV with session_id,
    # electrode_group and a boolean column 'flagged'; applied at the statistics stage: a session-pair is dropped when
    # either area has a unit on a flagged probe. None = no exclusion.
    "exclude_probes_csv": str(L.ROOT / "combined_results_ks4" / "ssl-whisker-artefact-gap" / "flagged_probes.csv"),
}


def pair_name(src, tgt):
    return f"{src}__{tgt}"


def all_pairs():
    """Unordered pairs of AREAS, oriented so the extracted source (order wS1 > wS2 > A1) comes first."""
    names = list(AREAS)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in SOURCES or b in SOURCES:
                s = min((a, b), key=lambda x: SOURCES.index(x) if x in SOURCES else 99)
                out.append((s, b if s == a else a, True))
            else:
                out.append((a, b, False))
    return out


def stable_units(with_coords=False):
    """v3 unit selection with a short-name column 'area' (NaN outside AREAS)."""
    u = pd.read_parquet(L.UNIT_TABLE)
    u["cluster_id"] = u.cluster_id.astype(int)
    s = pd.read_parquet(L.STABLE_TABLE, columns=["session_id", "electrode_group", "nwb_cluster_id", "stable"])
    s = s.rename(columns={"nwb_cluster_id": "cluster_id"})
    u = u.merge(s, on=["session_id", "electrode_group", "cluster_id"], how="left")
    u = u[u.quality_label.isin(L.CONFIG["unit_quality"]) & u.stable.fillna(False).astype(bool)
          & (u.firing_rate >= L.CONFIG["min_rate_hz"])].copy()
    u.loc[u.area_acronym_custom.isin(["MOs", "MOp"]) & (u.target_region.astype(str) == "OFC"),
          "area_acronym_custom"] = "MO-FC"
    lab = {a: k for k, v in AREAS.items() for a in v}
    u["area"] = u.area_acronym_custom.map(lab)
    if with_coords:
        c = pd.read_parquet(ROC_UNITS, columns=KEYS + ["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"])
        c = c.drop_duplicates(KEYS)
        c["cluster_id"] = c.cluster_id.astype(int)
        u = u.merge(c, on=KEYS, how="left")
    return u


def reference():
    ref = L.reference_sheet()
    return ref[(ref.exclude != 1) & (ref.exclude_ephys != 1)].set_index("subject_id")


def flagged_probes(path=None):
    """(session_id, electrode_group) pairs flagged for whisker dead time, or None when the table does not exist."""
    p = Path(path or PAIR_CONFIG["exclude_probes_csv"] or "")
    if not p.is_file():
        return None
    f = pd.read_csv(p)
    if "flagged" in f.columns:
        f = f[f.flagged.astype(bool)]
    return set(zip(f.session_id.astype(str), f.electrode_group.astype(str)))


def lib_md5():
    return hashlib.md5(Path(L.__file__).read_bytes()).hexdigest()


def same_probe_frac(units, a, b):
    """fraction of area-b units recorded on a probe that also holds area-a units (shared-probe confound)"""
    pa = set(units[units.area == a].electrode_group)
    ub = units[units.area == b]
    return float(ub.electrode_group.isin(pa).mean()) if len(ub) else np.nan
