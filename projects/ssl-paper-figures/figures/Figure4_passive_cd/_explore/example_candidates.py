"""Example-session candidates (user 2026-11-24: "find one mouse example and let me decide"). Run (haas, repo root):
.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/example_candidates.py
Example-session candidates for Fig 4 (whole-brain hit-miss axis): typical sessions per cohort (closest to the cohort mean
of the excess angle and projection change, in cohort-SD units) with many units; gallery of time courses."""
import sys
sys.path.insert(0, "projects/ssl-paper-figures")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shared_passive_cd as S
from paperstyle import COLORS
w = S.whole("W")
w["dth"] = S.theta(w.cos_post) - S.theta(w.cos_pre)
w["dth0"] = S.theta(w.cos_post_null) - S.theta(w.cos_pre_null)
w["qth_deg"] = w.dth - w.dth0
rows = []
for c in S.COH:
    g = w[w.cohort == c].copy()
    z = lambda v: (v - v.mean()) / v.std()
    g["dist"] = np.hypot(z(g.q), z(g.qth_deg))
    g["sign_ok"] = (np.sign(g.q) == np.sign(g.q.mean())) & (np.sign(g.qth_deg) == np.sign(g.qth_deg.mean()))
    rows.append(g)
w = pd.concat(rows)
cols = ["session_id", "cohort", "n_units", "n_hit", "n_miss", "n_pre", "n_post", "dprime_loo", "a_pre", "a_post", "q",
        "dth", "qth_deg", "dist", "sign_ok"]
for c in S.COH:
    print(c, "mean q %.3f qth %.2f" % (w[w.cohort == c].q.mean(), w[w.cohort == c].qth_deg.mean()))
    print(w[(w.cohort == c)].sort_values("dist")[cols].head(8).round(3).to_string(index=False))
cand = {c: w[(w.cohort == c) & w.sign_ok & (w.n_units >= 60) & (w.dprime_loo > 0.5)].sort_values("dist").head(4 if c == "R-" else 3)
        for c in S.COH}
t = pd.read_parquet(S.TAB / "179_cd_time.parquet")
t = t[t.area == "All units"]
sel = pd.concat(cand.values())
fig, axs = plt.subplots(2, len(sel), figsize=(2.0 * len(sel), 3.6), gridspec_kw=dict(height_ratios=[1.4, 1]))
for j, r in enumerate(sel.itertuples()):
    a = axs[0, j]
    g = t[t.session_id == r.session_id]
    for cls, col in (("hit", S.W_COL), ("miss", S.MISS_COL), ("pre_W", "0.55"), ("post_W", COLORS[r.cohort])):
        h = g[g.cls == cls].sort_values("t")
        a.plot(h.t, h.value, color=col, lw=0.9, label=cls)
    a.axvspan(0.005, 0.035, color="0.92", zorder=0); a.axhline(0, color="0.6", lw=0.4); a.set_xlim(-0.1, 0.2)
    a.set_title(f"{r.session_id[:14]} {r.cohort}\nn={r.n_units} hit/miss {r.n_hit}/{r.n_miss}", fontsize=6, color=COLORS[r.cohort])
    a.tick_params(labelsize=5)
    if j == 0: a.legend(fontsize=4.5, frameon=False)
    b = axs[1, j]
    b.plot([0, 1], [S.theta(r.cos_pre), S.theta(r.cos_post)], "o-", color=COLORS[r.cohort], ms=3)
    b.plot([0, 1], [S.theta(r.cos_pre_null), S.theta(r.cos_post_null)], "o-", color="0.6", mfc="white", ms=3)
    b.set_xticks([0, 1], ["Pre", "Post"]); b.set_xlim(-0.4, 1.4); b.tick_params(labelsize=5)
    b.set_title(f"angle  Δ excess {r.qth_deg:+.1f}°, Δproj excess {r.q:+.3f}", fontsize=5)
    if j == 0: b.set_ylabel("Angle to axis (°)", fontsize=5)
fig.tight_layout()
fig.savefig(__file__.replace(".py", ".png"), dpi=200)
