"""Focality-index summary, hit-vs-miss version, per Axel's 2026-09-01
follow-up: "do the same but for a single cohort, compare whisker hit and
misse across areas." Same Gini-coefficient focality index as
`028_focality_summary.py` (see that file's docstring for the full
definition), but:
- Fixed to a SINGLE cohort instead of comparing R+ vs R-.
- The comparison axis is now whisker HIT (`whisker_trial`, lick=1) vs
  MISS (`whisker_trial`, lick=0) instead of cohort.

**Cohort choice**: R+ (not R-). Per this project's task-design memory and
the `wh_reward_reward_group` column name in the loading pipeline, R+ is
the whisker-REWARDED cohort -- so a lick on a whisker trial is a genuine
rewarded "hit" for R+ mice. For R- mice whisker is not the rewarded
modality, so a lick there isn't a "hit" in the behavioral sense (it would
be a non-contingent/spurious response) -- making R+ the only cohort for
which "whisker hit vs miss" is a meaningful behavioral contrast. Flag if
R- was actually intended.

Reads the same cached main-compute-pass results as `027`/`028` (no
recompute).

Figures:
1. `barplot_focality_hitmiss_per_area.png`: per-area Gini coefficient of
   partner-interaction strength, hit vs miss bars, R+ only, one panel per
   variant (A/B).
2. `barplot_focality_hitmiss_overall.png`: whole-network Gini coefficient
   (across all 39 pairs), hit vs miss bars, R+ only, variant A/B side by
   side.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
fig024 = importlib.import_module("024_updated_figures")
driver = importlib.import_module("026_all_pairs_driver")
cov_lib = importlib.import_module("000_coverage_lib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CACHE_DIR = ARTIFACTS_DIR / "cache"
SUMMARY_DIR = RUNS_DIR / "cross_pair_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MIN_UNITS = fig024.MIN_UNITS
N_BOOTSTRAP = 500
COHORT = "R+"
HIT_COND = ("whisker_trial", 1)
MISS_COND = ("whisker_trial", 0)
HITMISS_COLOR = {"hit": "tab:blue", "miss": "tab:red"}

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "figure.titlesize": 15,
})


def _stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def units_fingerprint(session_ids: list[str]) -> str:
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    ref_fp = _stable_hash(ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group"]].astype(str).values.tolist())
    return _stable_hash({"session_ids": sorted(session_ids), "day_to_analyze": "learning",
                          "quality_tiers": ["good", "mua"], "ref_fp": ref_fp})


def load_pair_main_pass(area_a: str, area_b: str, session_ids: list[str]) -> dict | None:
    units_fp = units_fingerprint(session_ids)
    pair_fp = _stable_hash({"units_fp": units_fp, "area_a": area_a, "area_b": area_b, "min_units": MIN_UNITS})
    params_fp = fig024.main_pass_params_fingerprint()
    cache_file = CACHE_DIR / f"main_pass_{pair_fp}_{params_fp}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def dim1_mean(data: dict, cond: tuple, variant: str, cohort: str) -> float:
    d = data[cond][variant]
    rg = np.array(d["reward_group"])
    idx = np.where(rg == cohort)[0]
    if len(idx) == 0:
        return np.nan
    vals = np.abs(np.array([d["dim_corrs"][i][0] for i in idx], dtype=float))
    return float(np.nanmean(vals))


def gini(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or np.all(v == 0):
        return np.nan
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def main() -> None:
    pairs_df, session_lists = driver.derive_session_lists()
    areas = sorted(set(pairs_df["area_a"]) | set(pairs_df["area_b"]))
    pair_order = [(row["area_a"], row["area_b"]) for _, row in pairs_df.iterrows()]
    area_partners = {a: [] for a in areas}
    for a, b in pair_order:
        area_partners[a].append(b)
        area_partners[b].append(a)

    print(f"[029] Loading cached main-pass results for {len(pairs_df)} pairs...")
    pair_data: dict[tuple[str, str], dict] = {}
    for a, b in pair_order:
        d = load_pair_main_pass(a, b, session_lists[(a, b)])
        if d is not None:
            pair_data[(a, b)] = d
    print(f"[029] {len(pair_data)}/{len(pairs_df)} pairs loaded from cache")

    def pair_value(a: str, b: str, cond: tuple, variant: str) -> float:
        data = pair_data.get((a, b)) or pair_data.get((b, a))
        if data is None or cond not in data:
            return np.nan
        return dim1_mean(data, cond, variant, COHORT)

    rng = np.random.default_rng(0)

    # ==== Per-area focality, hit vs miss, R+ only ====
    for variant in ("A", "B"):
        fig, ax = plt.subplots(figsize=(0.6 * len(areas) + 3, 6.5), constrained_layout=True)
        x = np.arange(len(areas))
        for label, cond, offset in (("hit (lick=1)", HIT_COND, -0.18), ("miss (lick=0)", MISS_COND, 0.18)):
            means, sems = [], []
            for area in areas:
                partners = area_partners[area]
                vals = np.array([pair_value(area, p, cond, variant) for p in partners])
                means.append(gini(vals))
                valid = vals[~np.isnan(vals)]
                if len(valid) >= 4:
                    boot = [gini(rng.choice(valid, size=len(valid), replace=True)) for _ in range(N_BOOTSTRAP)]
                    sems.append(np.nanstd(boot))
                else:
                    sems.append(np.nan)
            color = HITMISS_COLOR["hit" if "hit" in label else "miss"]
            ax.bar(x + offset, means, width=0.36, yerr=sems, color=color, label=label, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(areas, rotation=90, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("focality index (Gini)\nof partner-interaction strength\n+/- bootstrap SEM")
        ax.legend(fontsize=9)
        ax.set_title(f"Per-area focality, whisker hit vs miss, {COHORT} only (variant {variant})")
        out = SUMMARY_DIR / f"barplot_focality_hitmiss_per_area_variant{variant}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out.name}")

    # ==== Overall (whole-network) focality, hit vs miss, R+ only ====
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    for ax, variant in zip(axes, ("A", "B")):
        x = np.arange(2)
        means, sems = [], []
        for cond in (HIT_COND, MISS_COND):
            vals = np.array([pair_value(a, b, cond, variant) for a, b in pair_order])
            means.append(gini(vals))
            valid = vals[~np.isnan(vals)]
            boot = [gini(rng.choice(valid, size=len(valid), replace=True)) for _ in range(N_BOOTSTRAP)]
            sems.append(np.nanstd(boot))
        colors = [HITMISS_COLOR["hit"], HITMISS_COLOR["miss"]]
        ax.bar(x, means, yerr=sems, color=colors, capsize=4, width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(["hit\n(lick=1)", "miss\n(lick=0)"])
        ax.set_ylim(0, 1)
        ax.set_title(f"variant {variant}")
        ax.set_ylabel("focality index (Gini)\nof all 39 pairs' interaction strength")
    fig.suptitle(f"Overall network focality, whisker hit vs miss, {COHORT} only")
    out = SUMMARY_DIR / "barplot_focality_hitmiss_overall.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out.name}")

    print(f"\nAll hit/miss focality figures written to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
