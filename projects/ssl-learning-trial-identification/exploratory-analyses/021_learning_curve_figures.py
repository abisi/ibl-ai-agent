"""021 -- Publication-quality learning-curve figure per session, saved in the mouse's learning_curve folder (user requests
2026-09-30), in combined_results_ks4/<mouse>/whisker_0/learning_curve/.
Model: the learning-curve state-space model as an HMM solved exactly (lt_lib): hidden state = logit P(lick) on a fine
grid, Gaussian random-walk transitions (sigma = 1, the user's choice), Bernoulli lick emissions, forward-backward ->
exact posterior per trial. Whisker, auditory and no-stim trials are fitted SEPARATELY on their own trials; the auditory
and no-stim (false-alarm) curves are placed at their REAL trial times and interpolated in time onto the whisker-trial
axis (lt_lib.interp_marginals; the corrected FA interpolation). Bands = 80% posterior intervals.
Trial sets: whisker = curve-aligned active whisker trials (as 001/018); auditory and no-stim =
_active_trials_for_curve_untrimmed (same set as the stored no-stim curve; auditory trials of the same set).
Layout (square, 3.0 x 3.0 in, single panel; P(whisker > FA) panel removed (user); no text header, learning trials hidden -- user):
  main panel  P(lick): auditory (blue), whisker (cohort colour), no-stim / false alarm (grey) curves + 80% bands;
              lick trials as ticks ABOVE the curves and no-lick trials BELOW (one row per trial type; whisker hits in the
              rewarded-whisker (R+) colour, misses in the non-rewarded (R-) colour, FA black, CR grey, auditory hits blue, misses light blue;
              non-whisker trials at their time-interpolated whisker-trial position);
  no legend (user); colours: auditory blue, whisker cohort colour, no stim grey.
Sessions: the 88 ephys learning sessions (001) + behaviour-only mice from their day-0 NWB (022).
New files only (existing pipeline outputs untouched): <mouse>_whisker_0_learning_curves_exact_sigma1.{pdf,png,svg}
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/021_learning_curve_figures.py [mouse ...]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import lt_lib as L  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

SIGMA = 1.0
# GROUP_COLORS of ephys_utilities.plotting_utils.plotting_utils (not importable in this venv: needs cmasher) --
# rplus = rewarded whisker, rminus = non-rewarded whisker
GROUP_COLORS = {"rplus": "#00B400", "rminus": "#C800C8"}
COL = {"R+": GROUP_COLORS["rplus"], "R-": GROUP_COLORS["rminus"]}
MISS_LIGHT_BLUE = "#8fd3f5"                     # auditory-miss ticks (user 2026-09-30)
AUD, FA = "#1f5fbf", "0.35"
TICK = {"no stim (FA)": ("k", "0.65"),          # false alarms black, correct rejections grey (user)
        "auditory": (AUD, MISS_LIGHT_BLUE)}     # auditory hits blue, misses light blue (user)
FS_M, FS_S = 8, 7


def fit(y):
    g, _ = L.forward_backward(np.asarray(y, int), SIGMA)
    return g


def band(g):
    lo, hi = L.quantiles(g, [0.1, 0.9])
    return g @ L.P_GRID, lo, hi


def figure(inp, aud, out_dir, label="whisker_0"):
    plt.rcParams.update({"font.family": "Arial", "font.size": FS_M, "axes.labelsize": FS_M, "xtick.labelsize": FS_S,
                         "ytick.labelsize": FS_S, "legend.fontsize": FS_S, "axes.linewidth": 0.9,
                         "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 3,
                         "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    rg = inp["reward_group"]
    yw, tw = np.asarray(inp["w_outcomes"], int), np.asarray(inp["w_start"], float)
    yn, tn = np.asarray(inp["n_outcomes"], int), np.asarray(inp["n_start"], float)
    gw = fit(yw)
    fa_marg = L.interp_marginals(fit(yn), tn, tw)
    curves = [("whisker", COL[rg], gw, xw := np.arange(1, len(yw) + 1), yw),
              ("no stim (FA)", FA, fa_marg, np.interp(tn, tw, xw), yn)]
    if aud is not None and len(aud[0]) >= 3:
        ya, ta = aud
        curves.insert(0, ("auditory", AUD, L.interp_marginals(fit(ya), ta, tw), np.interp(ta, tw, xw), ya))
    fig = plt.figure(figsize=(3.0, 3.0))
    ac = fig.add_axes([0.19, 0.165, 0.795, 0.795])     # square axes box in a square figure (3 x 3 in)
    handles = []
    nrow = len(curves)
    for k, (lab, c, g, x_tr, y_tr) in enumerate(curves):
        m, lo, hi = band(g)
        ac.fill_between(xw, lo, hi, color=c, alpha=0.2, lw=0)
        h, = ac.plot(xw, m, color=c, lw=1.9 if lab == "whisker" else 1.5)
        handles.append((h, lab))
        top = 1.07 + 0.065 * k                     # lick-trial rows above the curves
        bot = -0.07 - 0.065 * k                    # no-lick rows below
        c_lick, c_nolick = TICK[lab] if lab in TICK else (c, c)
        if lab == "whisker":
            c_lick, c_nolick = GROUP_COLORS["rplus"], GROUP_COLORS["rminus"]   # hits: rewarded whisker; misses: non-rewarded (user)
        ac.vlines(x_tr[y_tr == 1], top - 0.025, top + 0.025, color=c_lick, lw=0.9)
        ac.vlines(x_tr[y_tr == 0], bot - 0.025, bot + 0.025, color=c_nolick, lw=0.9)
    ac.set_ylim(-0.08 - 0.065 * nrow, 1.08 + 0.065 * nrow)
    ac.set_yticks([0, 0.5, 1])
    ac.set_ylabel("P(lick)")
    ac.set_xlim(0.5, len(yw) + 0.5)
    ac.set_xlabel("whisker trial")
    ac.text(-0.02, 1.07 + 0.065 * (nrow - 1) / 2, "licks", transform=ac.get_yaxis_transform(), ha="right", va="center",
            fontsize=FS_S, color="0.3")
    ac.text(-0.02, -0.07 - 0.065 * (nrow - 1) / 2, "no licks", transform=ac.get_yaxis_transform(), ha="right",
            va="center", fontsize=FS_S, color="0.3")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{inp['mouse_id']}_{label}_learning_curves_exact_sigma1"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    return out_dir / f"{stem}.pdf"


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed
    tt = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "trials.parquet")
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    inputs.update(pickle.load(open(ART / "022_inputs_behaviour_only.pkl", "rb")))   # + behaviour-only mice (user)
    root = axel_bisi_root() / "combined_results_ks4"
    only = set(sys.argv[1:])
    n = 0
    for sid, inp in inputs.items():
        if only and inp["mouse_id"] not in only:
            continue
        out_dir = root / inp["mouse_id"] / "whisker_0" / "learning_curve"
        if not out_dir.parent.exists():
            print("no whisker_0 folder:", inp["mouse_id"])
            continue
        if "a_outcomes" in inp:                      # behaviour-only mice (022, trials from the NWB)
            aud = (inp["a_outcomes"], inp["a_start"]) if len(inp["a_outcomes"]) else None
        else:
            u = _active_trials_for_curve_untrimmed(sid, tt)
            a = u[u.trial_type == "auditory_trial"]
            aud = (a.lick_flag.to_numpy().astype(int), a.start_time.to_numpy()) if len(a) else None
        print(figure(inp, aud, out_dir), flush=True)
        n += 1
    print(n, "figures")


if __name__ == "__main__":
    main()
