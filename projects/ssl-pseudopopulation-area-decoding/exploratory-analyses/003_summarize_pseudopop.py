"""003 -- Summary and publication-style figure of the paired pseudo-population decoding (002), per target x cohort
(rules in ../question.md, user design 2026-09-29).

Per area and iteration i: real_i (real-label curve), null_i (mean of the N_SHIFTS paired linear-shift curves of the
SAME draw), d_i = real_i - null_i. Across iterations:
  - figure (user 2026-09-29): square panels, one column per area; row 1 = balanced accuracy, real and paired shift
    null, mean +- SD across iterations; row 2 = paired difference d, mean +- SD; black bar = above chance, dotted line =
    onset (both rows);
  - a bin is above chance when the 5th percentile of d_i across iterations is > 0 (one-sided bootstrap);
  - onset latency = earliest bin t that is itself above chance and has at least 4 of the 5 bins starting at t above
    chance (bin itself required: user 2026-09-29); for stimulus-aligned targets searched after the stimulus only
    (bin end > 0), for the lick-aligned target over the whole window;
  - windows: mean d, its 5th percentile and the one-sided bootstrap p = fraction of iterations with window d <= 0.
Iterations (100) and N_SHIFTS (10) are PILOT values.
Outputs: ../artifacts/003_summary_<target>_<cohort>[suffix].csv, figures/003_pseudo_<target>_<cohort>[suffix].pdf/.png
Run (repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/003_summarize_pseudopop.py <target> <cohort> [suffix]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"
COL = {"R+": "#00B400", "R-": "#C800C8"}
NULL_COL = "#8c8c8c"
WIN = {"stim": (-0.2, 0.6), "lick": (-0.6, 0.2)}
WINDOWS = {"hitmiss": {"baseline": (-0.2, -0.01), "sensory": (0.005, 0.05)}, "modality_stim": {"sensory": (0.005, 0.05)},
           "modality_lick": {"pre-lick": (-0.10, 0.0), "post-lick": (0.005, 0.2)}}
TITLES = {"hitmiss": "Hit vs miss (stimulus-aligned)", "modality_stim": "Whisker vs auditory (stimulus-aligned)",
          "modality_lick": "Whisker vs auditory (first-lick-aligned)"}


def style():
    from matplotlib import font_manager
    for f in list(Path.home().glob(".local/share/fonts/arial*.ttf")) + list(Path("C:/Windows/Fonts").glob("arial*.ttf")):
        font_manager.fontManager.addfont(str(f))
    family = "Arial" if "Arial" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    })


def onset(above, t, k=5, m=4):
    """earliest bin that is itself above chance and has >= m of the k bins starting at it above (user 2026-09-29)"""
    for i in range(len(above) - k + 1):
        if above[i] and above[i:i + k].sum() >= m:
            return t[i]
    return np.nan


BANDS = __import__("os").environ.get("SSL_PSEUDO_BANDS", "sd")   # "sd" (mean +- SD) or "pct" (percentile bands)


def band(ax, x, M, color, label, pct=(2.5, 97.5)):
    """Mean with +- SD (BANDS "sd") or the given percentile band across iterations (BANDS "pct")."""
    mu = np.nanmean(M, 0)
    if BANDS == "pct":
        lo, hi = np.nanpercentile(M, pct[0], 0), np.nanpercentile(M, pct[1], 0)
    else:
        sd = np.nanstd(M, 0)
        lo, hi = mu - sd, mu + sd
    ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(x, mu, color=color, lw=1.1, label=label)


def main():
    import ssl_timeresolved_decoding as T
    style()
    target, cohort = sys.argv[1], sys.argv[2]
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    align = "lick" if target == "modality_lick" else "stim"
    t = np.array([e[1] for e in T.causal_bin_edges(WIN[align], bin_width=0.05, stride=0.005)])
    x = t * 1000
    d = pd.read_parquet(ART / f"002_pseudo_{target}_{cohort}{suffix}.parquet")
    d = d[d.skipped_reason.isna()]
    areas = sorted(d.area.unique(), key=lambda a: (a != "All units", a))
    n = len(areas)
    ncol = min(n, 5)                                   # areas wrap into blocks of <= 5 columns (2 rows per block)
    nblk = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(2 * nblk, ncol, figsize=(2.3 * ncol + 0.4, 2.45 * 2 * nblk + 0.5), squeeze=False)
    rows = []
    for j, a in enumerate(areas):
        g = d[d.area == a]
        R = np.stack(g.curve.map(np.asarray).to_numpy())
        N = np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
        D = R - N
        above = np.nanpercentile(D, 5, 0) > 0
        search = t > 0 if align == "stim" else np.ones(len(t), bool)   # stim-aligned: after the stimulus only (user)
        on = onset(above[search], t[search])
        ax, ax2 = axes[2 * (j // ncol), j % ncol], axes[2 * (j // ncol) + 1, j % ncol]
        band(ax, x, N, NULL_COL, "shift null")
        band(ax, x, R, COL[cohort], "real")
        band(ax2, x, D, "k", "real - null", pct=(5, 95))     # 5-95%: its lower edge is the test (5th percentile > 0)
        for a_, top in ((ax, 1.02), (ax2, None)):
            if top is None:
                top = a_.get_ylim()[1]
            a_.plot(np.where(above, x, np.nan), np.full(len(x), top), color="k", lw=2.2, solid_capstyle="butt", clip_on=False)
            a_.axvline(0, color="k", lw=0.5)
            if np.isfinite(on):
                a_.axvline(on * 1000, color="k", ls=":", lw=0.8)
            a_.set_box_aspect(1)
            a_.set_xlim(x[0], x[-1])
        ax.axhline(0.5, color="#bbbbbb", lw=0.5, ls="--")
        ax.set_ylim(0.35, 1.05)
        ax2.axhline(0, color="#bbbbbb", lw=0.5, ls="--")
        mice = int(g.n_eligible_mice.iloc[0])
        ax.set_title(f"{'Whole brain' if a == 'All units' else a}\n({mice} mice)")
        ax2.set_xlabel(f"time from {'stimulus' if align == 'stim' else 'first lick'} (ms)")
        if j % ncol == 0:
            ax.set_ylabel("balanced accuracy")
            ax2.set_ylabel("real - paired null")
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.2)
        if np.isfinite(on):
            ax.text(0.98, 0.04, f"onset {on * 1000:.0f} ms", transform=ax.transAxes, ha="right", fontsize=6.5)
        mu = np.nanmean(R, 0)
        row = dict(target=target, cohort=cohort, area=a, n_eligible_sessions=int(g.n_eligible.iloc[0]), n_eligible_mice=mice,
                   n_iterations=len(R), n_shifts=int(g.n_null.median()), n_neurons=int(g.n_neurons.iloc[0]),
                   t_train=int(g.t_train.iloc[0]), t_test=int(g.t_test.iloc[0]), peak=np.nanmax(mu),
                   t_peak_ms=t[np.nanargmax(mu)] * 1000, onset_ms=on * 1000, frac_bins_above=above.mean(), null_mean=np.nanmean(N))
        for w, (a0, a1) in WINDOWS[target].items():
            m = (t >= a0) & (t <= a1)
            dw = np.nanmean(D[:, m], 1)
            row.update({f"real_{w}": np.nanmean(R[:, m]), f"null_{w}": np.nanmean(N[:, m]), f"d_{w}": dw.mean(),
                        f"d_sd_{w}": dw.std(), f"d5_{w}": np.percentile(dw, 5), f"p_{w}": float((dw <= 0).mean())})
        rows.append(row)
    g0 = d.iloc[0]
    fig.suptitle(f"{TITLES[target]}, {cohort}, learning day -- pseudo-populations of {int(g0.n_mice)} mice x {int(g0.n_neurons)} "
                 f"neurons, {int(len(d) / n)} iterations (PILOT)\n"
                 + ("mean with 2.5-97.5% (accuracy) and 5-95% (difference) bands across iterations"
                    if BANDS == "pct" else "mean +- SD across iterations")
                 + "\nbar: 5th percentile of real - paired null > 0; dotted: onset (4 of 5 bins)", fontsize=7, y=0.995)
    for k in range(n, nblk * ncol):
        axes[2 * (k // ncol), k % ncol].set_axis_off()
        axes[2 * (k // ncol) + 1, k % ncol].set_axis_off()
    fig.subplots_adjust(left=0.09, right=0.99, top=1 - 0.55 / fig.get_figheight(), bottom=0.35 / fig.get_figheight(),
                        hspace=0.3, wspace=0.35)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"003_pseudo_{target}_{cohort}{suffix}{'_pctbands' if BANDS == 'pct' else ''}.{ext}", dpi=300)
    S = pd.DataFrame(rows)
    S.to_csv(ART / f"003_summary_{target}_{cohort}{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
