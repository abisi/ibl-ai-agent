"""094 -- Cohort difference in the time-resolved whisker-hit selectivity of reward-lick neurons (user 2026-10-10: "I want to show a
cohort difference; is this the correct metric?"). Exploratory.
Problem with 093: conditioning on converging neurons (WH > SL significant) removes the cohort difference. Here neurons are selected
on AH vs SL only (reward-lick neurons: pre-lick AH > SL significant), and their WH vs SL selectivity, a dimension not used for the
selection, is followed over time (092: 50-ms windows, 5-ms stride). The AH vs SL curve is shown for reference (circular inside the
selection window [-100, 0) ms).
  a  mean selectivity over time (mean over neurons within session, mean +- s.e.m. over sessions), R+ / R- x day 0 / expert
  b  mean WH vs SL selectivity in [-300, -100) ms (before the selection window)
  c  mean WH vs SL selectivity in [-100, 0) ms
  d  ratio of the WH vs SL to the AH vs SL selectivity in [-100, 0) ms per session (modality generalisation; 1 = whisker hits
     as selective as auditory hits)
Strictly the WH vs SL and AH vs SL selectivities share the SL events, so the selection is not fully independent of WH vs SL
(a small common bias, the same in all groups). Statistics as 088. Learners rule.
Output: across_days/<ref>/timeresolved_roc/ws_in_rl.{png,pdf}, ws_in_rl_stats.csv
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m93 = importlib.import_module("093_timeresolved_roc_figure")
m88, m61, m62 = m93.m88, m93.m61, m93.m62
OUT, GROUPS = m93.OUT, m93.GROUPS


def main():
    plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(OUT / "selectivity_index.parquet"); S = np.load(OUT / "selectivity.npy"); t = np.load(OUT / "t.npy")
    keep = D.index.isin(m61.learner_filter(D).index); D, S = D[keep].reset_index(drop=True), S[keep]
    rl = D.rl.fillna(False).to_numpy()
    rows = []
    for pr in ("WS", "AS"):
        mm = rl & (D.pair == pr).to_numpy()
        for w, (a, b) in (("early", (-0.3, -0.1)), ("late", (-0.1, 0.0))):
            v = S[mm][:, (t >= a) & (t < b)].mean(1)
            rows.append(D[mm][["session_id", "mouse_id", "cohort", "stage"]].assign(pair=pr, win=w, v=v))
    R = pd.concat(rows).groupby(["session_id", "mouse_id", "cohort", "stage", "pair", "win"]).v.mean().unstack(["pair", "win"])
    R.columns = [f"{p}_{w}" for p, w in R.columns]; R = R.reset_index()
    R["ratio"] = R.WS_late / R.AS_late
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.32))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1, 1, 1], wspace=0.6, left=0.06, right=0.99, top=0.78, bottom=0.2)
    m93.insets(fig, gs[0, 0], D, S, t, rl, "a"); fig.text(0.06, 0.9, "a  Reward-lick neurons (selected on AH vs SL only)", fontsize=5.6)
    for k, (col, ylab, ttl) in enumerate((("WS_early", "WH vs SL selectivity", "WH vs SL, −300 to −100 ms"),
                                           ("WS_late", "WH vs SL selectivity", "WH vs SL, −100 to 0 ms"),
                                           ("ratio", "WH vs SL / AH vs SL", "Generalisation ratio\n(−100 to 0 ms)"))):
        ax = fig.add_subplot(gs[0, 1 + k]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, R, col, ylab, rng, f"094 {col}", ttl, ref=0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ws_in_rl.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("094")]); St.to_csv(OUT / "ws_in_rl_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning",
              "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print("ALL DONE")


if __name__ == "__main__":
    main()
