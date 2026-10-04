"""Length-preserving Beryl-atlas area mapping.

`iblatlas.regions.BrainRegions.acronym2acronym` silently *drops* rows it
can't map (confirmed: `_find_inds` uses `ismember` and keeps only matched
indices) rather than returning one output per input -- fatal here, since
`beryl_area` must stay row-aligned with `neuron_factors`.

`units.parquet` has two parallel region annotations per unit: `ccf_acronym`/
`ccf_parent_acronym` (Axel's own registration, includes fine-grained
whisker-somatotopy subdivisions like `SSp-bfd-C6-6a` that aren't in
`BrainRegions.acronym` at all) and `ccf_atlas_acronym`/
`ccf_atlas_parent_acronym` (only populated for a subset of units -- a
standardized re-registration, per the ssl_ks2_ephys module docstring's
note about post-AB116 schema differences -- but Beryl-mappable directly
when present). Checked against the full local dataset: preferring the
atlas_ columns and falling back through all four, in that order, leaves
only 109/143604 units (0.08%) unmapped, all units where every one of the
four fields is missing outright. Those, and this residual, are labelled
"unmapped" and counted rather than silently coerced, per
brain_regions_qc.md's "unmapped labels must be counted and reported".
"""
import pandas as pd

FALLBACK_COLUMNS = ["ccf_atlas_acronym", "ccf_atlas_parent_acronym", "ccf_acronym", "ccf_parent_acronym"]


def acronym_to_beryl(units_df, br):
    """Map each unit's region to its Beryl-atlas grouping, preserving row
    order and length.

    Parameters
    ----------
    units_df : DataFrame with the four columns in `FALLBACK_COLUMNS`
    br : iblatlas.regions.BrainRegions instance

    Returns
    -------
    beryl : ndarray of str, shape (len(units_df),), "unmapped" where none
        of the four fallback columns could be mapped
    n_unmapped : int
    """
    lut = dict(zip(br.acronym, br.acronym[br.mappings["Beryl"]]))
    combined = None
    for col in FALLBACK_COLUMNS:
        mapped = units_df[col].map(lut)
        combined = mapped if combined is None else combined.fillna(mapped)
    beryl = combined.fillna("unmapped").to_numpy()
    n_unmapped = int((beryl == "unmapped").sum())
    return beryl, n_unmapped
