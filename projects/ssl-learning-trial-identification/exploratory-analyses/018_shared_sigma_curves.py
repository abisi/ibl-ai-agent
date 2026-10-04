"""018 -- One smoothness value (random-walk step sigma) for the whole dataset (user request
2026-09-25: per-session data-chosen curves are "way too smooth ... especially when there are few
licks. so choose a sigma that is unique across the dataset").

sigma_shared = argmax over a sigma grid of the SUMMED log evidence log p(licks | sigma) over all
curves (whisker and no-stim, all 88 sessions, both cohorts). Every curve is then refitted with
that single sigma (exact grid posterior, original model), FA placed at real no-stim times and
evaluated at whisker times, P(whisker > FA) per trial. Stored under key "shared" in a copy of the
002 curves dict -> artifacts/018_curves_shared.pkl (keys "orig", "eb", "shared").
Figure: 018_shared_sigma.png -- summed evidence vs sigma (whisker, FA, pooled) and examples
(stored, per-session sigma, shared sigma) for low- and high-lick sessions.
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))
import lt_lib as L  # noqa: E402

SIG = np.exp(np.linspace(np.log(0.02), np.log(1.5), 60))
# SSL_FIXED_SIGMA=1.0 (user decision 2026-09-25: "my stored sigma was better, keep at 1.0") -> curves fitted
# with that fixed sigma, stored under key "sigma1" in artifacts/018_curves_sigma1.pkl.
import os  # noqa: E402
FIXED_SIGMA = float(os.environ.get("SSL_FIXED_SIGMA", "0")) or None
KEY = "shared" if not FIXED_SIGMA else f"sigma{FIXED_SIGMA:g}"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
EXAMPLES = ["AB139", "AB093", "MH018", "AB085", "AB119", "MH028"]


def profile(item):
    sid, d = item
    return sid, {k: np.array([L.forward_backward(np.asarray(y).astype(int), s)[1] for s in SIG])
                 for k, y in (("whisker", d["w_outcomes"]), ("fa", d["n_outcomes"]))}


def fit_shared(args):
    sid, d, sigma = args
    gw, _ = L.forward_backward(np.asarray(d["w_outcomes"]).astype(int), sigma)
    gn, _ = L.forward_backward(np.asarray(d["n_outcomes"]).astype(int), sigma)
    fa_marg = L.interp_marginals(gn, d["n_start"], d["w_start"])
    lo80, hi80 = L.quantiles(gw, [0.1, 0.9])
    return sid, dict(p_mean=gw @ L.P_GRID, p_low80=lo80, p_high80=hi80, fa_time=fa_marg @ L.P_GRID,
                     p_above=L.prob_greater(gw, fa_marg), sigma=sigma)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    with ProcessPoolExecutor(max_workers=24) as ex:
        prof = dict(ex.map(profile, inputs.items()))
    tot = {k: np.sum([prof[s][k] for s in inputs], 0) for k in ("whisker", "fa")}
    tot["pooled"] = tot["whisker"] + tot["fa"]
    best = {k: float(SIG[np.argmax(v)]) for k, v in tot.items()}
    sigma = FIXED_SIGMA if FIXED_SIGMA else best["pooled"]
    print("best shared sigma:", {k: round(v, 3) for k, v in best.items()}, "-> using", round(sigma, 3))
    with ProcessPoolExecutor(max_workers=24) as ex:
        sh = dict(ex.map(fit_shared, [(s, d, sigma) for s, d in inputs.items()]))
    for s in curves:
        curves[s][KEY] = sh[s]
    pickle.dump(curves, open(ART / f"018_curves_{KEY}.pkl", "wb"))

    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[:, 0])
    for k, c in (("whisker", "#2ca02c"), ("fa", "#555555"), ("pooled", "k")):
        ax.plot(SIG, tot[k] - tot[k].max(), color=c, lw=2 if k == "pooled" else 1.3, label=f"{k}: best {best[k]:.2f}")
    ax.axvline(sigma, color="k", ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-150, 5)
    ax.set_xlabel("shared sigma (logit / trial)")
    ax.set_ylabel("summed log evidence, all sessions (0 = best)")
    ax.set_title(f"One sigma for the whole dataset: {sigma:.2f}\n(stored prior ~1; per-session data-chosen median R+ 0.49 / R- 0.14)",
                 fontsize=10)
    ax.legend(frameon=False)
    for i, pfx in enumerate(EXAMPLES):
        sid = next(k for k in inputs if k.startswith(pfx))
        d, c = inputs[sid], curves[sid]
        ax = fig.add_subplot(gs[i // 3, 1 + i % 3])
        o = d["w_outcomes"]
        x = np.arange(len(o))
        col = COHORT_COLOR[d["reward_group"]]
        ax.plot(x, d["stored"]["p_mean"], color="#bbbbbb", lw=0.8, label="stored (sigma ~1)")
        ax.plot(x, c["eb"]["p_mean"], color=col, lw=1, ls=":", label=f"per-session (sigma {c['eb']['sigma']:.2f})")
        s = c[KEY]
        ax.fill_between(x, s["p_low80"], s["p_high80"], color=col, alpha=0.2, lw=0)
        ax.plot(x, s["p_mean"], color=col, lw=1.8, label=f"{KEY} (sigma {sigma:.2f})")
        ax.plot(x, s["fa_time"], color="#444444", lw=1, ls="--", label="FA, shared")
        ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=2, color="k", marker="|")
        ax.set_ylim(-0.12, 1.12)
        ax.set_title(f"{sid[:5]} ({d['reward_group']}), {int(o.sum())} whisker licks / {len(o)} trials", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.suptitle("Learning curves with a single smoothness for the whole dataset", fontsize=12)
    fig.savefig(HERE / f"018_{KEY}_curves.png", dpi=150)


if __name__ == "__main__":
    main()
