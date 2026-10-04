"""Fit TCA separately per reward-group cohort on the megamouse tensor
(`process_single_condition`'s rplus/rminus split in the reference
pipeline, as opposed to `process_combined_condition`'s single shared fit
used in 012/013) -- reusing the already-pooled raw tensor from 011 (just
subsetting the neuron axis per cohort and renormalizing) rather than
rebuilding from spike times again.

Motivation: the combined megamouse (013) produced only 3 activation-type
components and no "late" component at all, structurally unable to show the
single-mouse pipeline's R+/R- "late activation" difference. This tests
whether fitting each cohort's own megamouse *separately* recovers a
late-stage component specifically in R+ (where the single-mouse analysis
found it in 17/37 mice) but not R- (4/31 mice) -- i.e. whether that
difference was washed out by pooling both cohorts into one fit, or would
stay invisible even to a cohort-specific pooled fit.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensortools as tt

sys.path.insert(0, r"M:\analysis\Axel_Bisi\brain_wide_analysis")
import allen_utils as allen  # noqa: E402

from component_semantics_lib import classify_component, STAGE_ORDER, SIGN_ORDER  # noqa: E402
from stats_lib import circular_shift_test  # noqa: E402

RANK = 4
N_REPLICATES = 5
OUT_DIR = Path("../artifacts/megamouse")

d = np.load(OUT_DIR / "pooled_raw.npz", allow_pickle=True)
tensor = d["tensor"]
area_custom, reward_group = d["area_custom"], d["reward_group"]
time_bins = d["time_bins"]
pooled_plick_all, pooled_dprime_all = d["pooled_plick"], d["pooled_dprime"]

results = {}
for rg, label in [(1, "R+"), (0, "R-")]:
    mask = reward_group == rg
    sub_tensor = tensor[:, mask, :]
    print(f"\n{label}: {sub_tensor.shape[1]} neurons")

    tmin = sub_tensor.min(axis=(0, 2), keepdims=True)
    tmax = sub_tensor.max(axis=(0, 2), keepdims=True)
    sub_norm = ((sub_tensor - tmin) / (tmax - tmin + 1e-10)).astype(np.float32)

    t0 = time.time()
    best_model, best_obj = None, np.inf
    for rep in range(N_REPLICATES):
        model = tt.ncp_hals(sub_norm, rank=RANK, verbose=False, random_state=rep)
        if model.obj < best_obj:
            best_obj, best_model = model.obj, model
    print(f"  fit time: {time.time() - t0:.1f}s, best objective={best_obj:.4f}")

    trial_factors, neuron_factors, time_factors = best_model.factors
    results[label] = {
        "trial_factors": trial_factors, "neuron_factors": neuron_factors, "time_factors": time_factors,
        "area_custom": area_custom[mask], "n_neurons": int(mask.sum()),
    }

    rng = np.random.default_rng(0)
    for comp in range(RANK):
        sign, stage, latency_ms = classify_component(time_factors[:, comp], time_bins)
        r_plick, _, p_plick = circular_shift_test(trial_factors[:, comp], pooled_plick_all, n_shifts=2000, rng=rng)
        r_dprime, _, p_dprime = circular_shift_test(trial_factors[:, comp], pooled_dprime_all, n_shifts=2000, rng=rng)
        print(f"  comp {comp + 1}: {stage} {sign} (latency {latency_ms:.0f} ms) | "
              f"P(lick) r={r_plick:+.2f} p={p_plick:.3f} | d-prime r={r_dprime:+.2f} p={p_dprime:.3f}")
        results[label].setdefault("classification", []).append((sign, stage, latency_ms))

np.savez(
    OUT_DIR / "megamouse_by_cohort.npz",
    **{f"{label}_{k}": v for label, r in results.items() for k, v in r.items() if k != "classification"},
)

# --- Figure: time factors per cohort, side by side ---
time_centers = (time_bins[:-1] + time_bins[1:]) / 2
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, label in zip(axes, ["R+", "R-"]):
    ax.plot(time_centers, results[label]["time_factors"], lw=1.8)
    ax.axvline(0, color="0.5", ls="--", lw=0.8)
    ax.set_title(f"{label} megamouse ({results[label]['n_neurons']} neurons)", fontsize=10)
    ax.set_xlabel("time from stim onset (s)")
    ax.legend([f"c{i+1}: {results[label]['classification'][i][1]} {results[label]['classification'][i][0]}"
               for i in range(RANK)], fontsize=6.5)
axes[0].set_ylabel("time factor")
fig.suptitle("Megamouse time factors, fit separately per cohort", y=1.03)
fig.tight_layout()
fig.savefig("015_megamouse_by_cohort_timefactors.png", dpi=150, bbox_inches="tight")

# --- Neuron type proportions per cohort-specific fit ---
STAGE_BASE_COLOR = {"sensory": "#1f6f78", "decision": "#c98a2c", "motor": "#b8433f", "late": "#5b4b8a"}
TYPE_ORDER = [f"{stage}\n{sign}" for stage in STAGE_ORDER for sign in SIGN_ORDER]
TYPE_COLORS = {f"{stage}\n{sign}": (STAGE_BASE_COLOR[stage] if sign == "activation" else STAGE_BASE_COLOR[stage] + "80") for stage in STAGE_ORDER for sign in SIGN_ORDER}

fig, ax = plt.subplots(figsize=(5, 5))
bottoms = np.zeros(2)
group_order = ["R+", "R-"]
for label in group_order:
    dominant = np.argmax(results[label]["neuron_factors"], axis=1)
    ntype = np.array([f"{results[label]['classification'][c][1]}\n{results[label]['classification'][c][0]}" for c in dominant])
    results[label]["proportions"] = pd.Series(ntype).value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0)
for t in TYPE_ORDER:
    vals = np.array([results[g]["proportions"].get(t, 0) for g in group_order])
    ax.bar(group_order, vals, bottom=bottoms, color=TYPE_COLORS[t], label=t.replace("\n", " "), edgecolor="white", linewidth=0.3)
    bottoms += vals
ax.set_ylabel("proportion of neurons")
ax.set_title("Neuron type proportions,\nseparate per-cohort megamouse fits")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("015_megamouse_by_cohort_proportions.png", dpi=150, bbox_inches="tight")

print("\nsaved 015_megamouse_by_cohort_timefactors.png, 015_megamouse_by_cohort_proportions.png")
