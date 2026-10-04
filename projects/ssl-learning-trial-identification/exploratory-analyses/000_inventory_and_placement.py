"""000 -- Inventory of every stored whisker learning curve (day 0) and
diagnostics of where the stored learning_trial (LT) sits.

Inputs: `combined_results_ks4/<mouse>/whisker_0/learning_curve/<mouse>_whisker_0_
whisker_trial_learning_curve_interp.h5` (all files on disk, ephys or not).

Per session:
  - n_trials, n posterior samples, stored LT, lt_source (ported rule,
    `ssl_timeresolved_decoding.reconstruct_learning_trial`), mouse_cat.
  - roughness: median |diff(logit p_mean)| per trial.
  - sustain_20: fraction of the 20 trials from LT on where the LT criterion
    still holds (R+: p_low > p_chance; R-: p_mean <= p_chance, i.e. whisker
    licking not above FA). A well-placed LT should have sustain ~1.
  - contrast_20: whisker lick rate (outcomes) in the 20 trials after LT minus
    the 20 before (R+ expected > 0, R- < 0).
  - cp_mle: maximum-likelihood single change point of the whisker outcome
    sequence (two-segment Bernoulli), constrained to the cohort's learning
    direction (R+ rate up, R- rate down), segments >= MIN_SEG trials;
    cp_llr = log-likelihood gain over one segment. Independent of the
    curve model -> a model-free reference for where behavior changes.
Outputs: artifacts/000_inventory.csv; exploratory-analyses/000_all_curves.png
(every curve: p_mean + 80% CI, p_chance, outcomes, stored LT red, cp_mle blue).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ssl_timeresolved_decoding import reconstruct_learning_trial  # noqa: E402

OUT = Path(__file__).resolve().parent
ART = OUT.parent / "artifacts"
COHORT_COLOR = {1: "#00B400", 0: "#C800C8", 2: "#7fd97f"}
MIN_SEG = 5
SUSTAIN_N = 20


def two_segment_cp(o: np.ndarray, direction: int, min_seg: int = MIN_SEG) -> tuple[float, float]:
    """MLE change point k (first trial of segment 2) for Bernoulli outcomes,
    requiring rate2 - rate1 to have sign `direction`. Returns (k, llr)."""
    n = len(o)

    def ll(x):
        m = len(x)
        if m == 0:
            return 0.0
        p = x.mean()
        if p in (0.0, 1.0):
            return 0.0
        return float(np.sum(x) * np.log(p) + (m - np.sum(x)) * np.log(1 - p))

    base = ll(o)
    best_k, best = np.nan, 0.0
    for k in range(min_seg, n - min_seg + 1):
        a, b = o[:k], o[k:]
        if np.sign(b.mean() - a.mean()) != direction:
            continue
        g = ll(a) + ll(b) - base
        if g > best:
            best, best_k = g, k
    return best_k, best


def main():
    root = axel_bisi_path("combined_results_ks4")
    files = sorted(glob.glob(str(root / "*" / "whisker_0" / "learning_curve" / "*_whisker_0_whisker_trial_learning_curve_interp.h5")))
    rows, curves = [], []
    for f in files:
        r = pd.read_hdf(f).iloc[0]
        pm_, pl, ph, pc = (np.asarray(r[k], float) for k in ("p_mean", "p_low", "p_high", "p_chance"))
        o = np.asarray(r["outcomes"], float)
        rg = int(r["reward_group"]) if not pd.isna(r["reward_group"]) else -1
        lt = r["learning_trial"]
        cat, lt_r, src = reconstruct_learning_trial(r) if rg in (0, 1) else (None, np.nan, None)
        lg = np.log(np.clip(pm_, 1e-4, 1 - 1e-4) / (1 - np.clip(pm_, 1e-4, 1 - 1e-4)))
        row = dict(mouse_id=r["mouse_id"], reward_group=rg, n_trials=len(o), n_samples=np.asarray(r["p_samples"]).shape[0],
                   learning_trial=lt, lt_repro=lt_r, lt_source=src, mouse_cat=r["mouse_cat"],
                   roughness=float(np.median(np.abs(np.diff(lg)))), overall_hit=o.mean(), overall_fa=np.nanmean(pc))
        if rg in (0, 1):
            direction = 1 if rg == 1 else -1
            k, llr = two_segment_cp(o, direction)
            row.update(cp_mle=k, cp_llr=llr)
            if not pd.isna(lt):
                lt = int(lt)
                seg = slice(lt, min(len(o), lt + SUSTAIN_N))
                held = (pl[seg] > pc[seg]) if rg == 1 else (pm_[seg] <= pc[seg])
                row.update(sustain_20=float(held.mean()) if len(held) else np.nan,
                           contrast_20=float(o[lt:lt + SUSTAIN_N].mean() - o[max(0, lt - SUSTAIN_N):lt].mean())
                           if lt > 0 else np.nan,
                           lt_minus_cp=lt - k if not np.isnan(k) else np.nan)
        rows.append(row)
        curves.append((row, pm_, pl, ph, pc, o))
    df = pd.DataFrame(rows)
    ART.mkdir(exist_ok=True)
    df.to_csv(ART / "000_inventory.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.groupby("reward_group")[["n_trials", "n_samples", "roughness", "sustain_20", "contrast_20", "lt_minus_cp"]]
          .describe().T.round(2).to_string())
    print(df.groupby(["reward_group", "lt_source"]).size())
    s = df[df.sustain_20.notna()]
    print("sessions whose LT criterion holds for <50% of the next 20 trials:", (s.sustain_20 < 0.5).sum(), "of", len(s))

    ncol = 10
    nrow = int(np.ceil(len(curves) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.75 * nrow), sharey=True, constrained_layout=True)
    order = sorted(range(len(curves)), key=lambda i: (curves[i][0]["reward_group"], curves[i][0]["mouse_id"]))
    for ax, i in zip(axes.flat, order):
        row, pm_, pl, ph, pc, o = curves[i]
        col = COHORT_COLOR.get(row["reward_group"], "#888888")
        x = np.arange(len(pm_))
        ax.fill_between(x, pl, ph, color=col, alpha=0.25, lw=0)
        ax.plot(x, pm_, color=col, lw=1)
        ax.plot(x, pc, color="#555555", lw=0.8, ls="--")
        ax.scatter(x, np.where(o == 1, 1.06, -0.06), s=1.5, color="k", marker="|")
        if not pd.isna(row["learning_trial"]):
            ax.axvline(row["learning_trial"], color="#d62728", lw=1.2)
        if not pd.isna(row.get("cp_mle", np.nan)):
            ax.axvline(row["cp_mle"], color="#1f77b4", lw=1.0, ls="--")
        sus = row.get("sustain_20", np.nan)
        ax.set_title(f"{row['mouse_id']} {'R+' if row['reward_group'] == 1 else 'R-' if row['reward_group'] == 0 else 'R+p'} "
                     f"LT={row['learning_trial']} sus={sus:.2f}" if not pd.isna(sus) else
                     f"{row['mouse_id']} rg={row['reward_group']} LT={row['learning_trial']}", fontsize=5.5)
        ax.tick_params(labelsize=4.5)
        ax.set_ylim(-0.1, 1.1)
    for ax in list(axes.flat)[len(curves):]:
        ax.axis("off")
    fig.suptitle("All stored whisker learning curves (day 0): p_mean + 80% CI (40 posterior samples), dashed = FA curve; "
                 "red = stored learning_trial, blue dashed = model-free change point (2-segment Bernoulli MLE); "
                 "sus = fraction of next 20 trials where the LT criterion still holds", fontsize=9)
    fig.savefig(OUT / "000_all_curves.png", dpi=170)


if __name__ == "__main__":
    main()
