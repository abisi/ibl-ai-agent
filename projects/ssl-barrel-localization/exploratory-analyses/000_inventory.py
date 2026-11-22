"""000 -- Inventory: brains of the final user-picked D2 barrel registration (left hemisphere; ws1-barrel-registration
D2_final_user_picked, 2026-11-22: per brain use_local = the picked fit, use_global = no local fit), fit quality, Lana's
visual barrel check, and the SSL ephys sessions / probes of each mouse in NWB_ks4. Per mouse, the brain exported to the
mouse's Anatomy folders (anatomy_export.csv) is the one used; the mouse's other brains are flagged d2_variant.

Output: combined_results_ks4/ssl-barrel-localization/tables/000_inventory.csv (one row per brain x session x probe).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 000_inventory.py
"""
import ast
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402
from ssl_nwb_freshness import session_nwbs  # noqa: E402

FINAL = axel_bisi_root() / "combined_results_ks4" / "ws1-barrel-registration" / "D2_final_user_picked"
LANA_SHARE = axel_bisi_root().parent / "Lana_Smith" / "neuropixels"
OUT = axel_bisi_root() / "combined_results_ks4" / "ssl-barrel-localization" / "tables"


def d2_fit(row):
    """one brain of the final table: decision, picked variant and (use_local) its fit metrics"""
    out = dict(d2_folder=row.brain, d2_decision=row.decision, d2_picked_variant=row.picked_variant or "")
    m = FINAL / row.brain / "results" / "h1" / "metrics.json"
    if row.decision == "use_local" and m.exists():
        met = json.load(open(m))
        out.update(d2_n_matched=met.get("n_matched"), d2_n_atlas=met.get("n_atlas"),
                   d2_contrast=met.get("contrast_after"), d2_naming_ambiguous=met.get("naming_ambiguous"),
                   d2_dice=met.get("dice_after"), d2_l4_bottom_um=met.get("l4_bottom_um"))
    return out


def probes(nwb):
    """Probes of a session NWB with units: electrode group, target area and insertion depth (group attrs)."""
    with h5py.File(nwb, "r") as f:
        if "units" not in f:
            return []
        eph = f["general/extracellular_ephys"]
        rows = []
        for g in sorted(k for k in eph.keys() if k.startswith("imec")):  # units/electrode_group holds object references
            loc = eph[g].attrs.get("location", "{}")
            try:
                loc = ast.literal_eval(loc)
            except (ValueError, SyntaxError):
                loc = {}
            rows.append(dict(session_id=nwb.stem, electrode_group=g, target_area=loc.get("area"),
                             hemisphere=loc.get("hemisphere"), insertion_depth_um=loc.get("depth")))
        return rows


def main():
    final = pd.read_csv(FINAL / "final_registration_table.csv", keep_default_na=False)
    fit = pd.DataFrame([d2_fit(r) for r in final.itertuples()])
    fit["mouse_id"] = fit.d2_folder.str.extract(r"^([A-Z]+\d+)")[0]
    used = pd.read_csv(FINAL / "anatomy_export.csv").groupby("mouse_id").brain.first()  # the mouse's brain in Anatomy
    fit["d2_variant"] = fit.d2_folder != fit.mouse_id.map(used).fillna(fit.mouse_id)  # other brains of the same mouse
    vis = pd.concat([pd.read_csv(LANA_SHARE / f"C2_barrel_coronal_{g}" / "summary.csv").assign(group=g) for g in ("AB", "MH")])
    vis["mouse_id"] = vis.folder.astype(str).str.extract(r"^([A-Z]+\d+)")[0]
    vis = vis.groupby("mouse_id").barrels_visible_visual_check.agg(lambda s: "/".join(sorted(set(map(str, s)))))
    fit["barrels_visible"] = fit.mouse_id.map(vis)

    nwbs = [p for p in session_nwbs() if p.name.split("_")[0] in set(fit.mouse_id)]
    with ThreadPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(probes, nwbs) for r in rs]
    pr = pd.DataFrame(rows)
    pr["mouse_id"] = pr.session_id.str.split("_").str[0]
    inv = fit.merge(pr, on="mouse_id", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    inv.to_csv(OUT / "000_inventory.csv", index=False)
    ws1 = inv[inv.target_area.astype(str).str.contains("S1", na=False)]
    print(fit.d2_decision.value_counts().to_string())
    print(f"D2 brains: {len(fit)} folders ({fit.mouse_id.nunique()} mice); with NWB units: {inv.session_id.notna().groupby(inv.mouse_id).any().sum()} mice, "
          f"{inv.session_id.nunique()} sessions; wS1-targeted probes: {len(ws1)} in {ws1.mouse_id.nunique()} mice")
    print(fit[["d2_folder", "d2_n_matched", "d2_contrast", "d2_naming_ambiguous", "barrels_visible"]].to_string(index=False))


if __name__ == "__main__":
    main()
