"""Coverage table for the full-dataset communication study (user request 2026-11-06; hypothesis: R+ feedforward Go
routing from SSp-w vs R- top-down NoGo signal, auditory hits as the Go reference in both cohorts).

Per whisker day-0 session: good units (firing rate >= 1 Hz) per area set, and trial counts per design cell.
Area sets (area_acronym_custom; hypothesis needs motor and frontal separated):
  wMO (MO-wM1, MO-wM2) | ALM-tjM1 (MO-ALM, MO-tjM1) | Frontal (mPFC, ORB, FRP) | SSp-w (SSp-bfd, SSs) | DLS | DMS |
  Striatum (DLS, DMS, TS, VS) | VPM-PO (VP: VPM, VPL, PO, PoT) | SC (SCm, SCs) | the three pilot v3 coarse groups.
Trials: jaw-onset tables of behaviour_analysis (combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl, day 0; same
columns as the NWB trial table + jaw_dlc_onset in s from trial start), context active, perf != 6, early_lick == 0,
disengaged tail trimmed (rule A1: trials after the last lick are dropped when they hold >= 5 whisker and >= 1 auditory
trials). Cells: W_lick, W_nolick, A_hit, A_miss, NS_cr, NS_fa; pre-movement subsets: lick trials with jaw onset >= 100
(150) ms after the stimulus (W_lick_pm100, ...); no-lick trials without a jaw onset in 0-500 ms (W_nolick_still).
Units: ssl-cross-area-communication full_unit_table_metadata_v3.parquet (2026-08-31; day-0 sessions only; quality
labels with that project's thresholds). Cohort and learner status from the reference sheet (learning_category good /
moderate = learner).
Outputs: OUT/coverage/{session_coverage.csv, pair_coverage.csv, coverage.md}
Run (haas): python projects/ssl-communication-methods-pilot/006_coverage_table.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402
from common import REFERENCE_SHEET  # noqa: E402  (ssl-paper-figures, on sys.path via comm_lib)

OUT = cl.OUT / "coverage"
UNITS = cl.ROOT / "combined_results_ks4" / "ssl-cross-area-communication" / "artifacts" / "full_unit_table_metadata_v3.parquet"
AREA_SETS = {
    "wMO": ["MO-wM1", "MO-wM2"], "ALM-tjM1": ["MO-ALM", "MO-tjM1"], "Frontal": ["mPFC", "ORB", "FRP"],
    "SSp-w": ["SSp-bfd", "SSs"], "DLS": ["DLS"], "DMS": ["DMS"], "Striatum": ["DLS", "DMS", "TS", "VS"],
    "VPM-PO": ["VP"], "SC": ["SCm", "SCs"],
}
COARSE = {"v3 Motor-frontal": "Motor and frontal areas", "v3 Striatum-pallidum": "Striatum and pallidum"}
COARSE_ACR = {"v3 Motor-frontal": ["MO-wM1", "MO-wM2", "MO-ALM", "MO-tjM1", "mPFC", "ORB", "FRP"],
              "v3 Striatum-pallidum": ["DLS", "DMS", "GPe", "GPi", "LS", "PAL", "SF", "TS", "VS"]}
UNIT_FLOORS, TRIAL_FLOORS = (20, 30), (15, 30)
# design cells needed per contrast (hypothesis tests) and per cohort
CONTRASTS = {
    "W_lick & A_hit": ["W_lick", "A_hit"],
    "W_lick & A_hit, jaw >= 100 ms": ["W_lick_pm100", "A_hit_pm100"],
    "W_nolick & A_hit": ["W_nolick", "A_hit"],
    "W_lick & W_nolick": ["W_lick", "W_nolick"],
}


def md(df, index=True):
    """DataFrame -> markdown table (no tabulate on haas)."""
    d = df.reset_index() if index else df
    head = [str(c) for c in d.columns]
    rows = [[str(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def trim_a1(t):
    """Rule A1: drop the tail after the last lick trial when it holds >= 5 whisker and >= 1 auditory trials."""
    lick = np.flatnonzero(t.lick_flag.to_numpy() == 1)
    if not len(lick):
        return t
    tail = t.iloc[lick[-1] + 1:]
    if (tail.trial_type == "whisker_trial").sum() >= 5 and (tail.trial_type == "auditory_trial").sum() >= 1:
        return t.iloc[: lick[-1] + 1]
    return t


def trial_cells(mouse):
    p = cl.ROOT / "combined_results_ks4" / mouse / "dlc_jaw_onset_times.pkl"
    if not p.exists():
        return {}
    d = pd.read_pickle(p)
    d = d[d.day == 0]
    out = {}
    for sid, t in d.groupby("session_id"):
        t = t[(t.context != "passive") & (t.perf != 6) & (t.early_lick == 0)].sort_values("start_time")
        n_before = len(t)
        t = trim_a1(t)
        jaw = pd.to_numeric(t.jaw_dlc_onset, errors="coerce")
        w, a, n = (t.trial_type == f"{k}_trial" for k in ("whisker", "auditory", "no_stim"))
        lk = t.lick_flag == 1
        still = ~((jaw >= 0) & (jaw <= 0.5))
        c = {"W_lick": w & lk, "W_nolick": w & ~lk, "A_hit": a & lk, "A_miss": a & ~lk, "NS_cr": n & ~lk,
             "NS_fa": n & lk, "W_nolick_still": w & ~lk & still, "W_lick_jaw_known": w & lk & jaw.notna()}
        for ms in (100, 150):
            c[f"W_lick_pm{ms}"] = w & lk & (jaw >= ms / 1000)
            c[f"A_hit_pm{ms}"] = a & lk & (jaw >= ms / 1000)
        out[sid] = {k: int(v.sum()) for k, v in c.items()}
        out[sid].update(n_trials_active=n_before, n_trimmed_a1=n_before - len(t),
                        median_jaw_onset_W_lick_ms=float(np.nanmedian(jaw[w & lk]) * 1000) if (w & lk).any() else np.nan,
                        median_jaw_onset_A_hit_ms=float(np.nanmedian(jaw[a & lk]) * 1000) if (a & lk).any() else np.nan)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_excel(REFERENCE_SHEET)
    sheet = sheet.drop_duplicates("mouse_id").set_index("mouse_id")
    u = pd.read_parquet(UNITS, columns=["mouse_id", "session_id", "day", "quality_label", "firing_rate",
                                        "area_acronym_custom", "area_group_coarse_v3"])
    u = u[(u.day == 0) & (u.quality_label == "good") & (u.firing_rate >= cl.CONFIG["min_rate_hz"])]
    rows = []
    for sid, g in u.groupby("session_id"):
        mouse = sid.split("_")[0]
        r = {"session_id": sid, "mouse_id": mouse,
             "cohort": sheet.reward_group.get(mouse, np.nan),
             "learning_category": sheet.learning_category.get(mouse, np.nan),
             "exclude_ephys": sheet.exclude_ephys.get(mouse, np.nan)}
        r["learner"] = r["learning_category"] in ("good", "moderate")
        for name, acr in AREA_SETS.items():
            r[f"units_{name}"] = int(g.area_acronym_custom.isin(acr).sum())
        for name, grp in COARSE.items():
            r[f"units_{name}"] = int((g.area_group_coarse_v3 == grp).sum())
        rows.append(r)
    ses = pd.DataFrame(rows)
    cells = {}
    for mouse in ses.mouse_id.unique():
        cells.update(trial_cells(mouse))
    cdf = pd.DataFrame.from_dict(cells, orient="index")
    ses = ses.merge(cdf, left_on="session_id", right_index=True, how="left")
    ses["has_jaw_table"] = ses.session_id.isin(cdf.index)
    ses.to_csv(OUT / "session_coverage.csv", index=False)

    # pairs x contrast x floors: sessions per cohort meeting unit floor in both areas and trial floor in every cell
    areas = list(AREA_SETS) + list(COARSE)
    prow = []
    base = ses[ses.has_jaw_table & (ses.exclude_ephys != 1)]
    for a, b in itertools.combinations(areas, 2):
        acr = {**AREA_SETS, **COARSE_ACR}
        if set(acr[a]) & set(acr[b]):
            continue  # overlapping sets (e.g. DLS within Striatum)
        for uf, tf in itertools.product(UNIT_FLOORS, TRIAL_FLOORS):
            ok_u = (base[f"units_{a}"] >= uf) & (base[f"units_{b}"] >= uf)
            for cname, need in CONTRASTS.items():
                ok = ok_u & np.logical_and.reduce([base[c] >= tf for c in need])
                for pop, m in (("all", ok), ("learners", ok & base.learner)):
                    prow.append(dict(area_a=a, area_b=b, unit_floor=uf, trial_floor=tf, contrast=cname,
                                     population=pop, n_Rplus=int((m & (base.cohort == "R+")).sum()),
                                     n_Rminus=int((m & (base.cohort == "R-")).sum())))
    pairs = pd.DataFrame(prow)
    pairs.to_csv(OUT / "pair_coverage.csv", index=False)

    # markdown summary
    lines = ["# Coverage for the full communication study (day 0)", "",
             f"Sessions with units: {len(ses)}; with a jaw-onset table: {int(ses.has_jaw_table.sum())}; "
             f"exclude_ephys: {int((ses.exclude_ephys == 1).sum())}.", "",
             "## Trial cells per cohort (median [min-max] over sessions; active, perf != 6, no early lick, A1-trimmed)", ""]
    cols = ["W_lick", "W_lick_pm100", "W_lick_pm150", "W_nolick", "W_nolick_still", "A_hit", "A_hit_pm100", "NS_cr",
            "median_jaw_onset_W_lick_ms", "median_jaw_onset_A_hit_ms"]
    tab = base.groupby("cohort")[cols].agg(lambda x: f"{np.nanmedian(x):.0f} [{np.nanmin(x):.0f}-{np.nanmax(x):.0f}]")
    lines += [md(tab.T), ""]
    lines += ["## Sessions with >= 15 / >= 30 trials per cell, per cohort", ""]
    cnt = []
    for c in cols[:-2]:
        for tf in TRIAL_FLOORS:
            cnt.append(dict(cell=c, floor=tf, **{k: int((g[c] >= tf).sum()) for k, g in base.groupby("cohort")}))
    lines += [md(pd.DataFrame(cnt), index=False), ""]
    key = pairs[(pairs.unit_floor == 20) & (pairs.trial_floor == 15) & (pairs.population == "all")]
    piv = key.pivot_table(index=["area_a", "area_b"], columns="contrast", values=["n_Rplus", "n_Rminus"])
    piv.columns = [f"{c} {v.replace('n_Rplus', 'R+').replace('n_Rminus', 'R-')}" for v, c in piv.columns]
    piv = piv[sorted(piv.columns)]
    lines += ["## Sessions per pair and contrast (>= 20 units per area, >= 15 trials per cell, all mice)", "",
              md(piv), ""]
    (OUT / "coverage.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
