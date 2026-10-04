"""One-off patch: recompute `beryl_area` in every already-saved
../artifacts/per_mouse/<session_id>.npz using the corrected, length-
preserving mapper in area_lib.py (the original 005 run used
`iblatlas`'s `acronym2acronym` directly, which silently drops unmapped
rows and desynced `beryl_area` from `neuron_factors` for most mice -- see
area_lib.py's docstring). Does not refit TCA; only rewrites the one field.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions

from tca_lib import load_session_tables
from area_lib import acronym_to_beryl

ARTIFACTS_DIR = Path("../artifacts/per_mouse")
br = BrainRegions()
_, units, _ = load_session_tables()

n_patched = 0
n_unmapped_total = 0
for f in sorted(ARTIFACTS_DIR.glob("*.npz")):
    session_id = f.stem
    d = dict(np.load(f, allow_pickle=True))
    n_neurons = d["neuron_factors"].shape[0]

    units_session = units[units["session_id"] == session_id].reset_index(drop=True)
    assert len(units_session) == n_neurons, f"{session_id}: units table length {len(units_session)} != neuron_factors {n_neurons}"

    beryl_area, n_unmapped = acronym_to_beryl(units_session, br)
    assert len(beryl_area) == n_neurons
    d["beryl_area"] = beryl_area
    np.savez(f, **d)
    n_patched += 1
    n_unmapped_total += n_unmapped

print(f"patched {n_patched} files; {n_unmapped_total} total unmapped units across the cohort")
