"""010 -- Abstract figure (COSYNE; user 2026-09-29): 6 square panels, no schematic; learning day, pseudo-populations
(20 mice x 10 neurons per draw), area groups coloured with allen_utils; areas as in the comparisons (005 EXCLUDE: no
Amygdala and hypothalamus, Cortical subplate, Visual areas, Pons and medulla, Somatosensory-body, Olfactory areas).
  a  whisker vs auditory, stimulus-aligned, -10..50 ms: real - shift null per area (R+, mean over iterations);
  b  onset latency of whisker vs auditory decoding per area, R+ vs R- (onset = first above-chance bin with >= 4 of 5
     above, searched after the stimulus; above chance = 5th percentile of real - null > 0);
  c  hit vs miss, whole brain, R+ vs R- (mean +- SD across iterations), -50..300 ms, windows 5-50 and 5-100 ms shaded;
  d  hit vs miss window effect 5-50 ms per area, R+ vs R- (mean +- SD across iterations);
  e  same, 5-100 ms;
  f  whisker vs auditory before the first lick (-100..0 ms) per area, R+ vs R-.
Cohort tests per area and window: mouse-level cohort-label permutation with shifts (009) when its 200 permutations are
available for that area/window ("perm"), otherwise the two bootstrap tests of 006 (overlap and z, both BH q < 0.05;
"boot"); BH across areas within window; * q < 0.05, ** q < 0.01. Stats: ../artifacts/010_abstract_stats.csv.
Output: figures/010_abstract_figure.pdf/.png
Run (repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/010_abstract_figure.py [n_iter]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S003 = _load("s003", "003_summarize_pseudopop.py")
S005 = _load("s005", "005_overlay_areas.py")
COL = {"R+": "#00B400", "R-": "#C800C8"}
WIN = {("hitmiss", "5-50ms"): (0.005, 0.05), ("hitmiss", "5-100ms"): (0.005, 0.1),
       ("modality_stim", "5-50ms"): (0.005, 0.05), ("modality_lick", "-100-0ms"): (-0.10, 0.0)}
N_PERM = 200                                  # user 2026-09-30: permutation cut to 200 x 10 iterations, whole brain


def two_tests(a, b):
    gt, lt = (a[:, None] > b[None, :]).mean(), (a[:, None] < b[None, :]).mean()
    pb = float(np.clip(2 * min(gt, lt), 1 / min(len(a), len(b)), 1))
    z = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) + b.var(ddof=1))
    return pb, float(2 * stats.norm.sf(abs(z)))


def stars(q):
    return "**" if q < 0.01 else "*" if q < 0.05 else ""


def load(target, n_iter, order, T):
    align = "lick" if target == "modality_lick" else "stim"
    t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
    out = {}
    for c in ("R+", "R-"):
        d = pd.read_parquet(ART / f"002_pseudo_{target}_{c}.parquet")
        d = d[d.skipped_reason.isna()]
        res = {}
        for a in ["All units"] + [x for x in order if x not in S005.EXCLUDE]:
            g = d[d.area == a].sort_values("rep")
            if len(g) < n_iter:
                continue
            g = g.head(n_iter)
            D = np.stack(g.curve.map(np.asarray).to_numpy()) - np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
            search = t > 0 if align == "stim" else np.ones(len(t), bool)
            res[a] = dict(D=D, on=S003.onset((np.nanpercentile(D, 5, 0) > 0)[search], t[search]) * 1000,
                          mice=int(g.n_eligible_mice.iloc[0]))
        out[c] = res
    return t * 1000, out


def perm_p(target, window, area):
    fs = sorted(ART.glob("009_cohort_perm_windows*.parquet"))
    if not fs:
        return np.nan, 0
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.target == target) & (d.window == window) & (d.area == area)].dropna(subset=["d_a", "d_b"])
    obs = d[d.perm == 0]
    null = d[d.perm > 0]
    if obs.empty or len(null) < N_PERM:
        return np.nan, len(null)
    s0 = abs(obs.d_a.iloc[0] - obs.d_b.iloc[0])
    return (1 + int((np.abs(null.d_a - null.d_b) >= s0).sum())) / (1 + len(null)), len(null)


def window_table(target, wname, x, data, areas):
    w0, w1 = (v * 1000 for v in WIN[(target, wname)])
    m = (x >= w0 - 1e-6) & (x <= w1 + 1e-6)
    rows = []
    for a in areas:
        A, B = np.nanmean(data["R+"][a]["D"][:, m], 1), np.nanmean(data["R-"][a]["D"][:, m], 1)
        pb, pz = two_tests(A, B)
        pp, nperm = perm_p(target, wname, a)
        rows.append(dict(target=target, window=wname, area=a, n_mice_rplus=data["R+"][a]["mice"],
                         n_mice_rminus=data["R-"][a]["mice"], d_rplus=A.mean(), sd_rplus=A.std(ddof=1), d_rminus=B.mean(),
                         sd_rminus=B.std(ddof=1), diff=A.mean() - B.mean(), p_boot=pb, p_z=pz, p_perm=pp, n_perm=nperm))
    W = pd.DataFrame(rows)
    fam = W.area != "All units"                       # BH across area groups; whole brain reported on its own
    for col in ("p_boot", "p_z", "p_perm"):
        W[col.replace("p_", "q_")] = W[col]
        ok = fam & W[col].notna()
        if ok.any():
            W.loc[ok, col.replace("p_", "q_")] = S005.bh(W.loc[ok, col].to_numpy())
    use_perm = W.p_perm.notna().all()
    W["test"] = "perm" if use_perm else "boot"
    W["q_used"] = W.q_perm if use_perm else np.maximum(W.q_boot, W.q_z)
    return W


def dumbbell(ax, W, colors, xlabel, order_by=None):
    W = W.set_index("area")
    ar = [a for a in W.index if a != "All units"]
    ar = sorted(ar, key=lambda a: W.loc[a, "d_rplus"] if order_by is None else order_by[a]) + ["All units"]
    for k, a in enumerate(ar):
        r = W.loc[a]
        ax.plot([r.d_rplus, r.d_rminus], [k, k], color="0.8", lw=0.8, zorder=1)
        for c, m, s, dy in (("R+", r.d_rplus, r.sd_rplus, 0.14), ("R-", r.d_rminus, r.sd_rminus, -0.14)):
            ax.errorbar(m, k + dy, xerr=s, fmt="o", ms=2.4, color=COL[c], elinewidth=0.5, capsize=0, zorder=2)
        st = stars(r.q_used)
        if st:
            ax.text(1.0, k, st, transform=ax.get_yaxis_transform(), va="center", fontsize=6.5)
    ax.set_yticks(range(len(ar)))
    ax.set_yticklabels(["Whole brain" if a == "All units" else S005.ABBR.get(a, a) for a in ar], fontsize=5)
    for lab, a in zip(ax.get_yticklabels(), ar):
        lab.set_color("k" if a == "All units" else colors.get(a, "0.3"))
        if a == "All units":
            lab.set_fontweight("bold")
    ax.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.set_ylim(-0.7, len(ar) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    return ar


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    colors, order = S005.allen()
    xs, MS = load("modality_stim", n_iter, order, T)
    xh, HM = load("hitmiss", n_iter, order, T)
    xl, ML = load("modality_lick", n_iter, order, T)
    both = lambda D: [a for a in D["R+"] if a in D["R-"]]
    tabs = [window_table("hitmiss", "5-50ms", xh, HM, both(HM)), window_table("hitmiss", "5-100ms", xh, HM, both(HM)),
            window_table("modality_stim", "5-50ms", xs, MS, both(MS)), window_table("modality_lick", "-100-0ms", xl, ML, both(ML))]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1, hspace=0.6, wspace=0.62)
    for ax in axes.flat:
        ax.set_box_aspect(1)
    (a_, b_, c_), (d_, e_, f_) = axes
    # a: whisker vs auditory, stimulus-aligned, areas (R+)
    z = (xs >= -10) & (xs <= 50)
    for a in [x for x in MS["R+"] if x != "All units"]:
        a_.plot(xs[z], MS["R+"][a]["D"].mean(0)[z], color=colors.get(a, "0.6"), lw=0.7)
    a_.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    a_.axvline(0, color="k", lw=0.5)
    a_.set_xlim(-10, 50)
    a_.set_xlabel("time from stimulus (ms)")
    a_.set_ylabel("whisker vs auditory\nbalanced accuracy - null")
    a_.set_title("stimulus identity (R+)", pad=6)
    # b: onsets R+ vs R- per area
    ar = [a for a in both(MS) if a != "All units"]
    ar = sorted(ar, key=lambda a: (-(MS["R+"][a]["on"] if np.isfinite(MS["R+"][a]["on"]) else 1e9)))
    for k, a in enumerate(ar):
        op, om = MS["R+"][a]["on"], MS["R-"][a]["on"]
        b_.plot([op, om], [k, k], color="0.8", lw=0.8, zorder=1)
        b_.scatter([op], [k + 0.14], s=8, color=COL["R+"], zorder=2)
        b_.scatter([om], [k - 0.14], s=8, color=COL["R-"], zorder=2)
    b_.set_yticks(range(len(ar)))
    b_.set_yticklabels([S005.ABBR.get(a, a) for a in ar], fontsize=5)
    for lab, a in zip(b_.get_yticklabels(), ar):
        lab.set_color(colors.get(a, "0.3"))
    b_.tick_params(axis="y", length=0)
    b_.set_ylim(-0.7, len(ar) - 0.3)
    b_.set_xlim(0, max(max(MS[c][a]["on"] for c in MS for a in ar if np.isfinite(MS[c][a]["on"])) + 5, 45))
    b_.set_xlabel("onset from stimulus (ms)")
    b_.set_title("whisker vs auditory onset", pad=6)
    # c: hit vs miss whole brain R+ vs R-
    z = (xh >= -50) & (xh <= 300)
    c_.axvspan(5, 100, color="0.93", lw=0, zorder=0)
    c_.axvspan(5, 50, color="0.85", lw=0, zorder=0)
    for c in ("R+", "R-"):
        D = HM[c]["All units"]["D"]
        mu, sd = D.mean(0), D.std(0, ddof=1)
        c_.fill_between(xh[z], (mu - sd)[z], (mu + sd)[z], color=COL[c], alpha=0.2, lw=0)
        c_.plot(xh[z], mu[z], color=COL[c], lw=0.9, label=f"{c} ({HM[c]['All units']['mice']} mice)")
    c_.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    c_.axvline(0, color="k", lw=0.5)
    c_.set_xlim(-50, 300)
    c_.set_xlabel("time from stimulus (ms)")
    c_.set_ylabel("hit vs miss, whole brain\nbalanced accuracy - null")
    c_.legend(loc="upper left", fontsize=5.5, handlelength=1.2)
    c_.set_title("choice (hit vs miss)", pad=6)
    # d, e, f: cohort comparison per area in windows
    Wd, We, Wf = tabs[0], tabs[1], tabs[3]
    dumbbell(d_, Wd, colors, "hit vs miss, real - null\n5 to 50 ms")
    dumbbell(e_, We, colors, "hit vs miss, real - null\n5 to 100 ms")
    dumbbell(f_, Wf, colors, "whisker vs auditory, real - null\n-150 to 0 ms from first lick")
    d_.set_title("R+ vs R-, 5-50 ms", pad=6)
    e_.set_title("R+ vs R-, 5-100 ms", pad=6)
    f_.set_title("R+ vs R-, before lick", pad=6)
    for ax, L in zip(axes.flat, "abcdef"):
        ax.text(-0.42 if ax in (b_, d_, e_, f_) else -0.3, 1.1, L, transform=ax.transAxes, fontweight="bold", fontsize=9)
    test = {W.test.iloc[0] for W in (Wd, We, Wf)}
    fig.suptitle("Learning day, pseudo-populations (20 mice x 10 neurons per draw, "
                 f"{n_iter} iterations x 10 linear shifts{', PILOT' if n_iter < 1000 else ''}); real - null = balanced accuracy "
                 "minus paired linear-shift null\ncohort tests (d-f): "
                 + ("mouse-level cohort-label permutation (200) " if test == {"perm"} else
                    "bootstrap overlap and z tests (both) -- permutation pending; ")
                 + "BH across areas, * q < 0.05, ** q < 0.01; mean ± SD across iterations", fontsize=6, y=0.995)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"010_abstract_figure.{ext}", dpi=300)
    S = pd.concat(tabs, ignore_index=True)
    S.to_csv(ART / "010_abstract_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["target", "window", "area", "n_mice_rplus", "n_mice_rminus", "d_rplus", "d_rminus", "diff", "p_boot", "p_z",
             "q_boot", "q_z", "p_perm", "test", "q_used"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
