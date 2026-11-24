"""Robustness of the reward-free R+ vs R- MMD^2 (016) across the v2 rastermap variants (050).

Reads each variant's <method_dir>/<run-dir>/mmd_cv_summary.csv; one figure: z (mean of CV directions) per PSTH type x
variant, '*' = p < 0.05 in both halves (uncorrected), '**' = BH q < 0.05 across single PSTH types. Variants that share
main's neurons and features (loc0p00, loc0p75, noWhiskerReward: reward-free cuts make the features identical) give
the same statistics and are listed once, as a check.
Outputs -> <VARIANTS_ROOT>/_comparison/mmd_rewardfree/{mmd_variant_summary.csv, mmd_variant_robustness.{png,pdf,svg}}
"""
import argparse
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
d016 = importlib.import_module("016_mmd_rewardfree_cv")

VARIANTS = ["main", "units_all", "units_good", "noFR", "mice_all", "loc0p00", "loc0p75", "noWhiskerReward"]
LABELS = {"main": "Main (stable good+MUA, learners)", "units_all": "All units (stability off)",
          "units_good": "Good units only", "noFR": "No firing-rate threshold", "mice_all": "All mice",
          "loc0p00": "Locality 0", "loc0p75": "Locality 0.75", "noWhiskerReward": "No whisker reward bins"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="mmd_rewardfree")
    a = ap.parse_args()
    out = d016.VARIANTS_ROOT / "_comparison" / "mmd_rewardfree"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in VARIANTS:
        try:
            _, _, md = d016.paths(v)
        except IndexError:
            continue
        f = md / a.run_dir / "mmd_cv_summary.csv"
        if not f.exists():
            print(f"[025] {v}: no 016 result", flush=True)
            continue
        s = pd.read_csv(f)
        s["z"] = s[["mmd2_z__fit_odd_test_even", "mmd2_z__fit_even_test_odd"]].mean(1)
        import json
        cfg = json.load(open(md / a.run_dir / "config.json"))
        s["variant"], s["n_neurons"], s["n_mice"] = v, cfg["n_neurons"], cfg["n_mice"]
        rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "mmd_variant_summary.csv", index=False)
    versions = list(dict.fromkeys(df.feature_version))
    variants = list(dict.fromkeys(df.variant))
    Z = df.pivot(index="variant", columns="feature_version", values="z").loc[variants, versions]
    S = df.pivot(index="variant", columns="feature_version", values="significant_both").loc[variants, versions]
    Q = df.pivot(index="variant", columns="feature_version", values="q_bh_single_types").loc[variants, versions]
    N = df.groupby("variant")[["n_neurons", "n_mice"]].first().loc[variants]

    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "Arial"})
    fig, ax = plt.subplots(figsize=((60 + 8 * len(versions)) / 25.4, (22 + 6 * len(variants)) / 25.4))
    vmax = max(4.0, float(np.nanpercentile(Z.values, 98)))
    im = ax.imshow(Z.values, cmap="magma_r", vmin=0, vmax=vmax, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z.values[i, j]
            mark = "**" if Q.values[i, j] < d016.ALPHA else "*" if S.values[i, j] else ""
            ax.text(j, i, f"{z:.1f}{mark}", ha="center", va="center", fontsize=5.5,
                    color="w" if z > 0.55 * vmax else "k")
    ax.set_xticks(range(len(versions))); ax.set_xticklabels(versions, rotation=50, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([f"{LABELS.get(v, v)} ({int(N.loc[v, 'n_neurons'])} n, {int(N.loc[v, 'n_mice'])} mice)"
                        for v in variants])
    ax.axvline(len([v for v in versions if v not in d016.GROUPS]) - 0.5, color="w", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("MMD² z (mean of CV directions)")
    ax.set_title("Reward-free R+ vs R− MMD² across variants (*: p<0.05 both halves; **: BH q<0.05 across single "
                 "PSTH types)", loc="left", fontsize=6.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"mmd_variant_robustness.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(Z.round(1).to_string())
    print("ALL DONE")


if __name__ == "__main__":
    main()
