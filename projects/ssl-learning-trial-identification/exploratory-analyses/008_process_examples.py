"""008 -- Step-by-step re-estimation of learning curves and learning trials
on example sessions, annotated and compared with the stored curves (user
request 2026-09-25: "Show more of the process of reestimating learning
curves and learning trials, with more details and annotations and compare
with previous learning curves").

One row per example session (one per outcome type), five panels:
  A  STORED: raw whisker outcomes (ticks), stored curve (40-sample PyMC,
     80% band), stored chance line (FA on an evenly spaced time grid),
     trials counted "above chance" (p_low > p_chance) shaded, stored LT.
  B  SAME MODEL, EXACT: grid forward-backward posterior with the stored
     prior (sigma ~ 1) vs the stored curve -> what the 40-sample sampler
     noise did (MAE and band-width difference annotated).
  C  DATA-CHOSEN SMOOTHNESS + FA FIX: 'eb' posterior (sigma chosen by the
     data; value and log Bayes factor vs the stored prior annotated), FA at
     actual no-stim times (dashed) vs stored even-grid FA (dotted), no-stim
     outcomes as grey ticks.
  D  DISCRIMINATION: P(p_whisker > p_FA) per trial from the eb posteriors,
     with the 0.9 level, and the stored 'above chance' trials for contrast.
  E  LEARNING-TRIAL POSTERIORS: joint whisker+FA change point (blue) and
     whisker-only change point (grey), 90% CI of the joint LT, and the
     final candidates: stored (red), L6 strict (blue), lenient (orange),
     half-way L7 (brown dotted); title = category, clean-separation P.
Output: exploratory-analyses/008_process_examples.png/.pdf
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
sys.path.insert(0, str(HERE))
import lt_lib as L  # noqa: E402

sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from ssl_timeresolved_decoding import reconstruct_learning_trial  # noqa: E402

ART = HERE.parent / "artifacts"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


EXAMPLES = [  # (subject prefix, label, required L6_category or lenient source) -- chosen from the 006/007 review grids
    ("AB119", "R+ learner, LT later (14 -> 49)", "learner"),
    ("MH028", "R+ learns by FA dropping", "learner"),
    ("AB082", "R+ lenient rule finds LT but FAILS clean gate", "L5w_len"),
    ("AB107", "R+ no learning event", "no_clear_transition"),
    ("AB085", "R- gradual: step 141 vs half-way 77", "learner"),
    ("AB139", "R- never licked; stored LT 117 arbitrary", "never_licked"),
]


def pick_examples(df: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for prefix, lab, req in EXAMPLES:
        row = df[df.session_id.str.startswith(prefix)].iloc[0]
        assert req in (row.L6_category, row.lt_lenient_source), (prefix, row.L6_category, row.lt_lenient_source)
        out.append((row.session_id, lab))
    return out


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    cpj = pickle.load(open(ART / "005_cp_posteriors.pkl", "rb"))
    cpw = pickle.load(open(ART / "004_cp_posteriors.pkl", "rb"))
    df = pd.read_csv(ART / "007_learning_trials_v2.csv")
    ex = pick_examples(df)
    print(ex)
    fig, axes = plt.subplots(len(ex), 5, figsize=(28, 3.8 * len(ex)), constrained_layout=True, squeeze=False)
    for r, (sid, lab) in enumerate(ex):
        d, c = inputs[sid], curves[sid]
        row = df[df.session_id == sid].iloc[0]
        rg = d["reward_group"]
        col = COHORT_COLOR[rg]
        o, st = d["w_outcomes"], d["stored"]
        x = np.arange(len(o))
        ticks = lambda ax: ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=3, color="k", marker="|")  # noqa: E731

        # A stored
        ax = axes[r, 0]
        ax.fill_between(x, st["p_low"], st["p_high"], color=col, alpha=0.25, lw=0)
        ax.plot(x, st["p_mean"], color=col, lw=1.2)
        ax.plot(x, st["p_chance"], color="#555555", lw=1, ls=":")
        above = st["p_low"] > st["p_chance"]
        for a0, a1 in L.runs(above):
            ax.axvspan(a0, a1, color="#ffd27f", alpha=0.45, lw=0)
        ticks(ax)
        if not pd.isna(row.L0_stored):
            ax.axvline(row.L0_stored, color="#d62728", lw=1.6)
        _, _, src0 = reconstruct_learning_trial(dict(outcomes=o, p_mean=st["p_mean"], p_low=st["p_low"], p_high=st["p_high"],
                                                     p_chance=st["p_chance"], reward_group=1 if rg == "R+" else 0))
        ax.set_title(f"A. STORED curve (40 samples, 80% band); dotted = FA (even grid)\nyellow = p_low > FA; red = stored LT "
                     f"{row.L0_stored:.0f} [{src0}]".replace("nan", "-"), fontsize=7.5, loc="left")
        ax.set_ylabel(f"{sid[:14]}\n{rg}, {row.learning_category}\n{lab}", fontsize=7.5)

        # B exact same model
        ax = axes[r, 1]
        o_ = c["orig"]
        ax.fill_between(x, o_["p_low80"], o_["p_high80"], color="k", alpha=0.15, lw=0)
        ax.plot(x, o_["p_mean"], color="k", lw=1.3, label="exact, stored prior")
        ax.plot(x, st["p_mean"], color=col, lw=0.9, alpha=0.8, label="stored (40 samples)")
        ax.plot(x, st["p_low"], color=col, lw=0.5, ls="--")
        ax.plot(x, st["p_high"], color=col, lw=0.5, ls="--")
        ticks(ax)
        mae = np.abs(o_["p_mean"] - st["p_mean"]).mean()
        bw = ((st["p_high"] - st["p_low"]) - (o_["p_high80"] - o_["p_low80"])).mean()
        ax.set_title(f"B. SAME MODEL, EXACT (no sampler noise): sigma {o_['sigma']:.2f}\n|exact - stored| = {mae:.3f}; "
                     f"band width diff {bw:+.3f}", fontsize=7.5, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

        # C eb + FA fix
        ax = axes[r, 2]
        e = c["eb"]
        ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.25, lw=0)
        ax.plot(x, e["p_mean"], color=col, lw=1.6, label="re-estimated whisker")
        ax.plot(x, st["p_mean"], color=col, lw=0.5, alpha=0.5, label="stored whisker")
        ax.plot(x, e["fa_time"], color="#333333", lw=1.2, ls="--", label="FA at actual no-stim times")
        ax.plot(x, st["p_chance"], color="#999999", lw=1, ls=":", label="stored FA (even grid)")
        nt_idx = np.searchsorted(d["w_start"], d["n_start"]).clip(0, len(o) - 1)
        ax.scatter(nt_idx, np.where(d["n_outcomes"] == 1, 1.14, -0.14), s=3, color="#888888", marker="|")
        ticks(ax)
        bf = (e["logev"] - o_["logev"]) / np.log(10)
        fa_shift = np.nanmax(np.abs(e["fa_time"] - st["p_chance"]))
        ax.set_title(f"C. DATA-CHOSEN SMOOTHNESS: sigma {e['sigma']:.2f} (prior ~1), log10 BF {bf:+.1f}\n"
                     f"FA at real times (dashed); max FA shift {fa_shift:.2f}; grey = no-stim licks", fontsize=7.5, loc="left")
        ax.legend(fontsize=6, frameon=False, loc="upper right")

        # D discrimination
        ax = axes[r, 3]
        ax.plot(x, e["p_above"], color=col, lw=1.4, label="P(whisker > FA), re-estimated")
        ax.plot(x, above.astype(float) * 0.98, color="#ffb000", lw=0.8, drawstyle="steps-mid", label="stored: p_low > chance")
        ax.axhline(0.9, color="#888888", lw=0.8, ls="--")
        ax.axhline(0.5, color="#cccccc", lw=0.8, ls=":")
        ax.set_title("D. DISCRIMINATION: P(whisker > FA) per trial\n(replaces binary 'p_low above chance' on a noisy band)",
                     fontsize=7.5, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")

        # E change point
        ax = axes[r, 4]
        rj, rw = cpj.get(sid), cpw.get(sid)
        if rw is not None:
            ax.fill_between(x, 0, rw["pk"] / rw["pk"].max(), color="#999999", alpha=0.35, lw=0, step="mid",
                            label=f"whisker-only CP (log10 BF {rw['log10_bf']:+.1f})")
        if rj is not None:
            ax.fill_between(x, 0, rj["pk"] / rj["pk"].max(), color="#1f77b4", alpha=0.45, lw=0, step="mid",
                            label=f"joint whisker+FA CP (log10 BF {rj['log10_bf']:+.1f})")
            if not pd.isna(row.L6):
                ax.axvspan(row.L6_ci05, row.L6_ci95, color="#1f77b4", alpha=0.1, lw=0)
        for val, cc, ls, name in ((row.L0_stored, "#d62728", "-", "stored"), (row.L6, "#1f77b4", "-", "L6 strict"),
                                  (row.lt_lenient, "#ff7f0e", "--", "lenient"), (row.L7, "#8c564b", ":", "L7 half-way")):
            if not pd.isna(val):
                ax.axvline(val, color=cc, lw=1.6, ls=ls, label=f"{name} = {val:.0f}")
        ax.set_ylim(0, 1.05)
        clean = "clean" if row.lt_lenient_sep_p > 0.9 else ("NOT clean" if not pd.isna(row.lt_lenient_sep_p) else "-")
        ax.set_title(f"E. LT POSTERIOR: {row.L6_category}; lenient via {row.lt_lenient_source}\n"
                     f"clean-sep P {row.lt_lenient_sep_p:.2f} ({clean}); hit 20 before/after "
                     f"{row.lt_lenient_hit_pre20:.2f}/{row.lt_lenient_hit_post20:.2f}".replace("nan", "-"), fontsize=7.5, loc="left")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        for ax in axes[r]:
            ax.set_xlim(-1, len(o))
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[r, :3]:
            ax.set_ylim(-0.2, 1.2)
    for ax in axes[-1]:
        ax.set_xlabel("whisker trial (index used by learning_trial)", fontsize=8)
    fig.suptitle("Re-estimating learning curves and learning trials, step by step (A stored -> B exact same model -> "
                 "C data-chosen smoothness + FA at real times -> D discrimination probability -> E change-point learning trial)",
                 fontsize=12)
    fig.savefig(HERE / "008_process_examples.png", dpi=170)
    fig.savefig(HERE / "008_process_examples.pdf")


if __name__ == "__main__":
    main()
