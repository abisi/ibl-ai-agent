"""One-off patch: add `area_custom` (Axel's own `area_acronym_custom` labels,
the same nomenclature `tca_pipeline_bis.py` uses for all its area-level
plots) to every already-saved ../artifacts/per_mouse/<session_id>.npz.

Reuses `allen_utils.create_area_custom_column` directly from
M:\\analysis\\Axel_Bisi\\brain_wide_analysis (added to sys.path) rather than
reimplementing its layer-stripping / region-generalizing / SSp-bfd-barrel
logic -- same function the reference pipeline itself calls, so labels are
directly comparable to any of its own area-level figures. Does not refit
TCA; only adds one field per file, alongside the existing `beryl_area`
(kept for cross-checking, not removed).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"M:\analysis\Axel_Bisi\brain_wide_analysis")
import allen_utils as allen  # noqa: E402

from tca_lib import load_session_tables  # noqa: E402

ARTIFACTS_DIR = Path("../artifacts/per_mouse")
_, units, _ = load_session_tables()

n_patched = 0
for f in sorted(ARTIFACTS_DIR.glob("*.npz")):
    session_id = f.stem
    d = dict(np.load(f, allow_pickle=True))
    n_neurons = d["neuron_factors"].shape[0]

    units_session = units[units["session_id"] == session_id].reset_index(drop=True).copy()
    assert len(units_session) == n_neurons, f"{session_id}: units table length {len(units_session)} != neuron_factors {n_neurons}"

    # allen_utils.simplify_per_nomenclature only checks that the
    # ccf_atlas_* *columns* exist, not that their values are non-null (its
    # own raw-NWB loading apparently never leaves them present-but-NaN) --
    # this dataset's ccf_atlas_acronym is only populated for a subset of
    # units (see area_lib.py), so fill NaNs with the non-atlas columns
    # before calling it, preserving its own "prefer atlas fields" intent.
    units_session["ccf_atlas_acronym"] = units_session["ccf_atlas_acronym"].fillna(units_session["ccf_acronym"])
    units_session["ccf_atlas_parent_acronym"] = units_session["ccf_atlas_parent_acronym"].fillna(units_session["ccf_parent_acronym"])

    units_session = allen.create_area_custom_column(units_session)
    d["area_custom"] = units_session["area_acronym_custom"].to_numpy()
    np.savez(f, **d)
    n_patched += 1

print(f"patched {n_patched} files with area_custom")
