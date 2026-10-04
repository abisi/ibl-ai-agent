"""036 -- Joint whisker + FA change point (L6) with a LOWERED evidence threshold, LOWERED checks and NO start gate (user
2026-10-01: "a joint whisker/FA model with lowered Bayes evidence threshold and lowered checks -- what would it look like?";
R+ mice that are high from the start should be able to pass the learner test).
Two model versions, same rule:
  L6     005 joint model (naive -> learned [-> end decline]; R-: generalising -> learned), posteriors from the 028 run;
  L6x    034 extended model (optional LAPSE segment before the learned segment), FA stream, BF from 034.
Rule (one rule, no cascade, no high_from_start / never_licked gates):
  R+ learner: log10 BF > BF_TH and P(whisker > FA) > P_TH in the WIN whisker trials from the LT;
  R- learner: log10 BF > BF_TH and P(whisker > FA) > P_TH before the LT (there was licking to suppress) and the
              discrimination after the LT is at most half of before (as L6).
  P(whisker > FA): Beta(1,1) posteriors of the whisker rate and the FA rate (no-stim trials in the same time span), 4000
  draws (005 Streams.p_above). LT = posterior median of the learned-segment start.
Sweep: BF_TH in {0.5, 0.3, 0.0}, P_TH in {0.95, 0.9, 0.8}, WIN = 20.
Outputs: artifacts/036_joint_cp_relaxed.csv (per session x version x setting), 036_sweep.csv;
figures 036_review_<cohort>.png (every mouse, chosen setting: L6x, BF 0.3, P 0.9; new learners vs L6 strict outlined).
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/036_joint_cp_relaxed.py
"""

from __future__ import annotations

import importlib.util
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
import lt_lib as L  # noqa: E402

_s = importlib.util.spec_from_file_location("j005", HERE / "005_joint_changepoint.py")
J = importlib.util.module_from_spec(_s)
sys.modules["j005"] = J
_s.loader.exec_module(J)
COL = {"R+": "#00B400", "R-": "#C800C8"}
CAT_COL = {"good": "#1a9850", "moderate": "#fee08b", "bad": "#d73027", "NA": "0.75"}
BF_THS, P_THS, WIN = (0.5, 0.3, 0.0), (0.95, 0.9, 0.8), 20
CHOSEN = ("L6x", 0.3, 0.9)


def main():
    inp = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    cp6 = pickle.load(open(ART / "028_chain_all" / "005_cp_posteriors.pkl", "rb"))
    x34 = pd.read_csv(ART / "034_single_cp.csv")
    x34 = x34[x34.stream == "FA"].set_index("session_id")
    t6 = pd.read_csv(ART / "028_chain_all" / "005_learning_trials.csv").set_index("session_id")
    rows = []
    for sid, d in inp.items():
        rg = d["reward_group"]
        o, wt, no, nt = (np.asarray(d[k]) for k in ("w_outcomes", "w_start", "n_outcomes", "n_start"))
        S = J.Streams(o, wt, no, nt)
        n = len(o)
        for ver in ("L6", "L6x"):
            if ver == "L6":
                r = cp6.get(sid)
                bf, lt = (r["log10_bf"], r["median"]) if r is not None else (-np.inf, np.nan)
            else:
                bf, lt = (x34.loc[sid, "log10_bf"], x34.loc[sid, "LT"]) if sid in x34.index else (-np.inf, np.nan)
            if not np.isfinite(lt) or lt <= 0 or lt >= n - 1:
                p_after = p_before = np.nan
                ratio = np.nan
            else:
                k = int(lt)
                p_after = S.p_above(k, min(n, k + WIN))
                p_before = S.p_above(0, k)
                db, da = S.disc(0, k), S.disc(k, n)
                ratio = da / db if db > 0 else np.nan
            for bth in BF_THS:
                for pth in P_THS:
                    if rg == "R+":
                        ok = bf > bth and np.isfinite(p_after) and p_after > pth
                    else:
                        ok = (bf > bth and np.isfinite(p_before) and p_before > pth and np.isfinite(ratio) and ratio <= 0.5)
                    rows.append(dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=rg,
                                     learning_category=d["learning_category"] or "NA", version=ver, bf_th=bth, p_th=pth,
                                     log10_bf=bf, LT=lt, p_after=p_after, p_before=p_before, learner=bool(ok),
                                     L6_strict=pd.notna(t6.loc[sid, "L6"]) if sid in t6.index else False,
                                     L6_category=t6.loc[sid, "L6_category"] if sid in t6.index else None))
    R = pd.DataFrame(rows)
    R.to_csv(ART / "036_joint_cp_relaxed.csv", index=False)
    sw = (R.groupby(["version", "bf_th", "p_th", "reward_group", "learning_category"]).learner.sum().unstack(fill_value=0)
          .assign(total=lambda x: x.sum(1)))
    sw.to_csv(ART / "036_sweep.csv")
    pd.set_option("display.width", 200)
    print(sw.unstack("reward_group").to_string())
    hfs = R[(R.L6_category == "high_from_start")]
    print("former high_from_start R+ mice that become learners:")
    print(hfs[hfs.learner].groupby(["version", "bf_th", "p_th"]).size().unstack(fill_value=0).to_string())
    print(R[R.mouse_id.isin(["AB118", "MH070"]) & (R.bf_th == 0.3) & (R.p_th == 0.9)][
        ["mouse_id", "version", "log10_bf", "LT", "p_after", "learner"]].to_string(index=False))
    review(R, inp)


def review(R, inp):
    plt.rcParams.update({"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False})
    ver, bth, pth = CHOSEN
    G = R[(R.version == ver) & (R.bf_th == bth) & (R.p_th == pth)]
    for rg in ("R+", "R-"):
        g = G[G.reward_group == rg].copy()
        g["grp"] = np.select([g.learner & g.L6_strict, g.learner & ~g.L6_strict, ~g.learner], [0, 1, 2])
        g = g.sort_values(["grp", "log10_bf"], ascending=[True, False])
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.7 * nrow + 0.5), squeeze=False)
        fig.subplots_adjust(left=0.03, right=0.995, top=1 - 0.45 / (1.7 * nrow + 0.5), bottom=0.02, hspace=0.75, wspace=0.15)
        for ax, r in zip(axes.flat, g.itertuples()):
            d = inp[r.session_id]
            tw = np.asarray(d["w_start"], float)
            w = L.forward_backward(np.asarray(d["w_outcomes"], int), 1.0)[0] @ L.P_GRID
            f = L.interp_marginals(L.forward_backward(np.asarray(d["n_outcomes"], int), 1.0)[0],
                                   np.asarray(d["n_start"], float), tw) @ L.P_GRID
            x = np.arange(len(w))
            ax.plot(x, f, color="0.45", lw=0.9)
            ax.plot(x, w, color=COL[rg], lw=1.1)
            if np.isfinite(r.LT):
                ax.axvline(r.LT, color="#1f77b4", lw=1.4 if r.learner else 0.7, ls="-" if r.learner else ":")
            if r.grp == 1:
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color("#ff7f0e")
                    sp.set_linewidth(1.5)
            ax.set_ylim(-0.03, 1.03)
            tag = {0: "learner (also L6)", 1: "NEW learner", 2: "not learner"}[r.grp]
            ax.set_title(f"{r.mouse_id} {r.learning_category} · {tag}\nlog10 BF {r.log10_bf:.2f} · P(w>FA) "
                         f"{(r.p_after if rg == 'R+' else r.p_before):.2f} · LT {'' if not np.isfinite(r.LT) else int(r.LT)}",
                         fontsize=5.8, color="#ff7f0e" if r.grp == 1 else "k")
            ax.tick_params(labelsize=5)
        for ax in axes.flat[len(g):]:
            ax.set_axis_off()
        nl = int(g.learner.sum())
        fig.text(0.03, 0.998, f"{'R+' if rg == 'R+' else 'R−'}: joint whisker + FA change point with lapse (L6x), log10 BF > "
                 f"{bth}, P(whisker > FA) > {pth}, no start gate -> {nl}/{len(g)} learners "
                 f"({int((g.grp == 1).sum())} new vs L6 strict, orange frame). Blue line = LT (solid = learner).",
                 fontsize=7, va="top")
        fig.savefig(HERE / f"036_review_{'Rplus' if rg == 'R+' else 'Rminus'}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
