"""171 -- Two diagnostic views of the gain / rotation split of 169-170 (user 2026-10-08: "make 1 and 2 as diagnostics").
Learners, learning day; unit = session; values per neuron (169).
View 1, polar state space (one panel per cohort; rows: passive whisker, passive auditory): x = rho cos(theta) = projection on the
hit - miss axis; y = rho sin(theta) = part orthogonal to the axis, sqrt(rho^2 - x^2) with rho cross-validated (clipped at 0).
  faint dots: sessions (grey: pre, colour: post; axes zoomed to the central 20-80 % of sessions and the means); large symbols: cohort means of x and y (pre, post); black arrow: pre -> post on the
  real axis; dashed ray: the pure-gain path through the pre mean; open diamond: the pure-gain prediction at the post response size
  (mean rho_post on the pre angle); dotted arc: rotation at constant size from that point to the post angle; grey: the same means on
  the shifted-label axes (mean over shifts, same rho).
View 2, stacked contributions: real - shifted change along the axis per cohort split into gain part ((rho_post - rho_pre) cos_pre) and
rotation part (rho_post (cos_post - cos_pre)); positive and negative parts stacked separately; black marker: total +- s.e.m.; numbers:
each part's mean +- s.e.m.; whisker and auditory.
Output: figures/171_gain_rotation_views.{png,pdf}. Run (haas, repo root): python .../171_gain_rotation_views.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
H = importlib.import_module("143_lt_split_windows_figures")
M170 = importlib.import_module("170_along_orthogonal_links")
COL, COH = H.COL, H.COH
EV = {"W": "#f7b519", "A": "#2c2cdb"}


def xy(d, ep, null=False):
    a = d[f"a_{ep}" + ("_null" if null else "")].to_numpy(float); r = d[f"rho_{ep}"].to_numpy(float)
    return a, np.sqrt(np.clip(r ** 2 - a ** 2, 0, None))


def polar(ax, d, c, s_, small=False):
    """small: compact version for 165 (short labels, smaller symbols)"""
    for ep, fc in (("pre", "0.65"), ("post", COL[c])):
        x, y = xy(d, ep); ax.plot(x, y, "o", ms=1.4 if small else 2, color=fc, alpha=0.35, mew=0)
    m = {ep: np.array([np.nanmean(v) for v in xy(d, ep)]) for ep in ("pre", "post")}
    mn = {ep: np.array([np.nanmean(v) for v in xy(d, ep, True)]) for ep in ("pre", "post")}
    th_pre = np.arctan2(m["pre"][1], m["pre"][0]); th_post = np.arctan2(m["post"][1], m["post"][0])
    rp = np.hypot(*m["post"])
    L = 1.15 * max(np.hypot(*m["pre"]), rp)
    ax.plot([0, L * np.cos(th_pre)], [0, L * np.sin(th_pre)], color="0.5", lw=0.7, ls=(0, (4, 2)))          # pure-gain path
    gpt = rp * np.r_[np.cos(th_pre), np.sin(th_pre)]
    ax.plot(*gpt, "D", ms=3.5 if small else 4.5, mfc="white", mec="0.2", mew=0.8, zorder=6)
    tt = np.linspace(th_pre, th_post, 30); ax.plot(rp * np.cos(tt), rp * np.sin(tt), color="0.2", lw=0.9, ls=":", zorder=6)   # rotation
    ax.annotate("", mn["post"], mn["pre"], arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.1, ls=(0, (2, 1.2)), shrinkA=2, shrinkB=2))
    for ep, fc in (("pre", "0.65"), ("post", COL[c])):
        ax.plot(*m[ep], "o", ms=4.5 if small else 6, mfc=fc, mec="black", mew=0.8, zorder=7)
        ax.plot(*mn[ep], "o", ms=2.8 if small else 3.5, mfc="white", mec="0.6", mew=0.8, zorder=5)
    ax.annotate("", m["post"], m["pre"], arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, shrinkA=3, shrinkB=3), zorder=8)
    ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
    P = np.vstack([np.c_[xy(d, ep)] for ep in ("pre", "post")]); P = P[np.isfinite(P).all(1)]
    M = np.vstack([m["pre"], m["post"], mn["pre"], mn["post"], gpt])
    lo = np.minimum(np.percentile(P, 20, 0), M.min(0)); hi = np.maximum(np.percentile(P, 80, 0), M.max(0))   # zoom: central sessions
    pad = 0.12 * (hi - lo).max(); lo, hi = lo - pad, hi + pad; span = (hi - lo).max()
    ctr = (lo + hi) / 2; ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2); ax.set_ylim(ctr[1] - span / 2, ctr[1] + span / 2)
    ax.set_aspect("equal", adjustable="box")
    if small:
        ax.set_xlabel("along axis ρ cos θ (→ hit)", fontsize=5); ax.set_ylabel("orthogonal ρ sin θ", fontsize=5)
        ax.text(0.03, 0.97, f"{c.replace('-', '−')}  θ {np.degrees(th_pre):.0f}°→{np.degrees(th_post):.0f}°\nρ {np.hypot(*m['pre']):.2f}→{rp:.2f}",
                transform=ax.transAxes, va="top", fontsize=4.6, color=COL[c])
        return
    ax.set_xlabel("along hit − miss axis  ρ cos θ  (→ hit)", fontsize=6)
    ax.set_ylabel("orthogonal to the axis  ρ sin θ", fontsize=6)
    ax.set_title(f"{c.replace('-', '−')}, passive {'whisker' if s_ == 'W' else 'auditory'} (n = {len(d)})\n"
                 f"θ {np.degrees(th_pre):.0f}° → {np.degrees(th_post):.0f}°, ρ {np.hypot(*m['pre']):.2f} → {rp:.2f}",
                 color=COL[c], fontsize=6)


def stacked(ax, D, small=False):
    xs, labs = [], []
    from scipy import stats as st_
    for i, s_ in enumerate(("W", "A")):
        for j, c in enumerate(COH):
            d = D[(D.stim == s_) & (D.reward_group == c)]
            x = i * 2.6 + j; xs.append(x)
            labs.append(c.replace('-', '−') if small else f"{c.replace('-', '−')}\n{'whisker' if s_ == 'W' else 'auditory'}")
            pos = neg = 0.0
            for q, col, hatch in (("gain_exc", "0.55", None), ("rot_exc", COL[c], None)):
                v = np.nanmean(d[q]); se = H.sem(d[q].to_numpy(float))
                base = pos if v >= 0 else neg
                ax.bar(x, v, bottom=base, width=0.6, color=col, alpha=0.85 if q == "rot_exc" else 0.6, edgecolor="black", lw=0.5)
                if not small:
                    ax.text(x + 0.33, base + v / 2, f"{'gain' if q == 'gain_exc' else 'rot.'} {v:+.3f}\n±{se:.3f}", fontsize=4.6, va="center")
                if v >= 0:
                    pos += v
                else:
                    neg += v
            t = d.d_along_exc.to_numpy(float)
            ax.errorbar(x, np.nanmean(t), H.sem(t), fmt="o", color="black", ms=3.5, capsize=0, zorder=5)
    ax.axhline(0, color="0.5", lw=0.6)
    if small:                                                      # p of R+ vs R- per part (Mann-Whitney), per modality
        for i, s_ in enumerate(("W", "A")):
            d = D[D.stim == s_]
            pp = {q: st_.mannwhitneyu(*(d[d.reward_group == c][q].dropna() for c in COH)).pvalue for q in ("gain_exc", "rot_exc")}
            f = lambda p_: "n.s." if p_ >= 0.05 else f"p = {p_:.3f}"
            ax.text(i * 2.6 + 0.5, 1.0, f"{'whisker' if s_ == 'W' else 'auditory'}\ngain {f(pp['gain_exc'])}\nrot. {f(pp['rot_exc'])}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.5,
                    color=EV[s_] if s_ == "A" else "#b07e00")
        ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=5); ax.set_xlim(-0.6, xs[-1] + 0.6)
        for t, c in zip(ax.get_xticklabels(), list(COH) * 2):
            t.set_color(COL[c])
        ax.set_ylabel("Δ along axis, real − shifted\n(gain grey + rotation colour)", fontsize=5)
        return
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=5.5); ax.set_xlim(-0.6, xs[-1] + 1.2)
    ax.set_ylabel("Δ along the hit − miss axis, real − shifted\n(post − pre, per neuron)", fontsize=6)
    ax.set_title("gain (grey) + rotation (colour) = total (black ± s.e.m.)", fontsize=6)


def main():
    H.setup()
    D = pd.read_parquet(EA / "169_along_orthogonal.parquet")
    D = M170.part_a(D[H.mouse_of(D).isin(H.learners())])
    fig = plt.figure(figsize=(9.5, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], wspace=0.45, hspace=0.45)
    for r, s_ in enumerate(("W", "A")):
        for k, c in enumerate(COH):
            polar(fig.add_subplot(gs[r, k]), D[(D.stim == s_) & (D.reward_group == c)], c, s_)
    stacked(fig.add_subplot(gs[:, 2]), D)
    fig.text(0.01, 0.005, "black arrow: real axis pre → post (means); grey dashed arrow / open circles: shifted-label axes; dashed ray: pure gain path; "
             "diamond: pure-gain prediction; dotted arc: rotation", fontsize=5.5, color="0.35")
    (EA / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(EA / "figures" / f"171_gain_rotation_views.{ext}", dpi=200, bbox_inches="tight")
    print("ALL DONE", EA / "figures" / "171_gain_rotation_views.png")


if __name__ == "__main__":
    main()
