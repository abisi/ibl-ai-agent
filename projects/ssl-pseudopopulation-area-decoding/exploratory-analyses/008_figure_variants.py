"""008 -- Closely related publication-style figure variants of the pseudo-population decoding per cohort (user request
2026-09-29: "several types of figures all squared, with stat bars for significance, barplots for significance and
latency, area comparison (anova), full window and zoomed in ... several closely related examples").
All variants: one row per decoding type (hit vs miss; whisker vs auditory stimulus-aligned; whisker vs auditory
first-lick-aligned), square panels, area groups coloured with allen_utils, whole brain / Amygdala and hypothalamus /
Cortical subplate / Visual areas / Pons and medulla not shown (as 005), no error bands on overlaid curves.

Quantities per cohort x target x area (100 iterations of 002; PILOT counts):
  d_i(t) = real_i - paired shift null_i; curve = mean over iterations;
  above chance = 5th percentile of d_i > 0; onset = first above-chance bin with >= 4 of the 5 bins from it above
  (003.onset), stimulus-aligned targets searched after the stimulus only;
  onset 95% CI = bootstrap over iterations (500 resamples of the 100 iterations, onset recomputed each time);
  window effect = mean d in the window (hit/miss 5..600 ms, whisker vs auditory stimulus 5..50 ms, lick -100..0 ms),
  one-sided bootstrap p = fraction of iterations with window d <= 0, BH across areas; * q < 0.05, ** q < 0.01;
  time above chance = % of bins after the alignment (stimulus-aligned: 0..600 ms; lick-aligned: -600..0 ms) above chance;
  area comparison ("ANOVA"): pseudo-populations have no per-mouse values, so areas are compared with their bootstrap
  distributions: omnibus Wald chi-square test of equal means (inverse-variance weights from the bootstrap SDs; the
  ANOVA analogue with known variances), pairwise: bootstrap-overlap test and z test (both BH across pairs; a pair is
  marked when both q < 0.05).

Variants (files figures/008_<variant>_<cohort>.pdf/.png):
  A_full       time course (full window) with stacked above-chance bars | onset latency bars (+ CI) | window effect bars (stars)
  B_zoom       as A, zoomed (hit/miss -50..250 ms, stimulus -10..50 ms, lick -350..50 ms)
  C_clean_full clean time course (full) | onset latency bars (+ CI) | % time above chance bars
  D_clean_zoom as C, zoomed
  E_areas      area comparison: window effect +- SD (omnibus Wald p) | pairwise difference matrix (dots: both tests
               q < 0.05) | onset latency +- bootstrap SD. No omnibus test on onsets: bootstrapped onsets move in 5-ms
               steps and are often constant (SD 0), which makes a Wald test meaningless (chi2 in the thousands); the
               onset Wald values stay in the csv for reference only.
Stats: ../artifacts/008_stats_<cohort>.csv (per area), ../artifacts/008_pairwise_<cohort>.csv (pairs).
Run (repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/008_figure_variants.py <cohort> [n_iter]
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
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
S012 = _load("s012", "012_rt_band.py")          # reaction-time band
TARGETS = ["hitmiss", "modality_stim", "modality_lick"]
ROWLAB = {"hitmiss": "Hit vs miss", "modality_stim": "Whisker vs auditory\n(stimulus-aligned)",
          "modality_lick": "Whisker vs auditory\n(first-lick-aligned)"}
WINDOW = {"hitmiss": (0.005, 0.6), "modality_stim": (0.005, 0.05), "modality_lick": (-0.10, 0.0)}
AFTER = {"hitmiss": (0.0, 0.6), "modality_stim": (0.0, 0.6), "modality_lick": (-0.6, 0.0)}
FULL = {"hitmiss": (-200, 600), "modality_stim": (-200, 600), "modality_lick": (-600, 200)}
ZOOM = {"hitmiss": (-50, 250), "modality_stim": (-10, 50), "modality_lick": (-350, 50)}
N_BOOT = 500
CURVE_FRAC = 0.55


def stars(q):
    return "**" if q < 0.01 else "*" if q < 0.05 else ""


def two_tests(a, b):
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    pb = float(np.clip(2 * min(gt, lt), 1 / min(len(a), len(b)), 1))
    z = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) + b.var(ddof=1))
    return pb, float(2 * stats.norm.sf(abs(z)))


def wald(m, s):
    w = 1 / np.maximum(np.asarray(s, float) ** 2, 1e-12)
    m = np.asarray(m, float)
    chi = float((w * (m - (w * m).sum() / w.sum()) ** 2).sum())
    return chi, len(m) - 1, float(stats.chi2.sf(chi, len(m) - 1))


def compute(cohort, n_iter, order, T):
    out = {}
    for tg in TARGETS:
        align = "lick" if tg == "modality_lick" else "stim"
        t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
        d = pd.read_parquet(ART / f"002_pseudo_{tg}_{cohort}.parquet")
        d = d[d.skipped_reason.isna()]
        done = d.groupby("area").size()
        areas = [a for a in order if a not in S005.EXCLUDE and done.get(a, 0) >= n_iter]
        search = t > 0 if align == "stim" else np.ones(len(t), bool)
        wm = (t >= WINDOW[tg][0]) & (t <= WINDOW[tg][1])
        am = (t >= AFTER[tg][0]) & (t <= AFTER[tg][1])
        rng = np.random.default_rng(0)
        res = {}
        for a in areas:
            g = d[d.area == a].sort_values("rep").head(n_iter)
            D = np.stack(g.curve.map(np.asarray).to_numpy()) - np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
            above = np.nanpercentile(D, 5, 0) > 0
            on = S003.onset(above[search], t[search]) * 1000
            bo = []
            for _ in range(N_BOOT):
                Db = D[rng.integers(0, len(D), len(D))]
                bo.append(S003.onset((np.nanpercentile(Db, 5, 0) > 0)[search], t[search]) * 1000)
            bo = np.asarray(bo)
            dw = np.nanmean(D[:, wm], 1)
            res[a] = dict(D=D, curve=D.mean(0), above=above, on=on, on_boot=bo,
                          on_lo=np.nanpercentile(bo, 2.5) if np.isfinite(bo).any() else np.nan,
                          on_hi=np.nanpercentile(bo, 97.5) if np.isfinite(bo).any() else np.nan,
                          on_sd=np.nanstd(bo) if np.isfinite(bo).sum() > 1 else np.nan,
                          on_frac_found=float(np.isfinite(bo).mean()), dw=dw, p=max(float((dw <= 0).mean()), 1 / len(dw)),
                          frac=float(above[am].mean()) * 100, mice=int(g.n_eligible_mice.iloc[0]))
        q = S005.bh([res[a]["p"] for a in areas])
        for a, qa in zip(areas, q):
            res[a]["q"] = qa
        # area comparison
        chi_w = wald([res[a]["dw"].mean() for a in areas], [res[a]["dw"].std(ddof=1) for a in areas])
        oa = [a for a in areas if np.isfinite(res[a]["on"]) and np.isfinite(res[a]["on_sd"]) and res[a]["on_sd"] > 0]
        chi_o = wald([res[a]["on"] for a in oa], [res[a]["on_sd"] for a in oa]) if len(oa) > 1 else (np.nan, 0, np.nan)
        pairs = []
        for a, b in itertools.combinations(areas, 2):
            pb, pz = two_tests(res[a]["dw"], res[b]["dw"])
            pairs.append(dict(target=tg, area_a=a, area_b=b, diff=res[a]["dw"].mean() - res[b]["dw"].mean(), p_boot=pb, p_z=pz))
        P = pd.DataFrame(pairs)
        if len(P):
            P["q_boot_bh"], P["q_z_bh"] = S005.bh(P.p_boot), S005.bh(P.p_z)
            P["sig_both"] = (P.q_boot_bh < 0.05) & (P.q_z_bh < 0.05)
        out[tg] = dict(t=t * 1000, areas=areas, res=res, wald_window=chi_w, wald_onset=chi_o, pairs=P, d=d)
    return out


# ------------------------------------------------------------------------------------------------ panels
def tc_panel(ax, x, areas, res, colors, view, bars):
    z = (x >= view[0]) & (x <= view[1])
    for a in areas:
        ax.plot(x[z], res[a]["curve"][z], color=colors.get(a, "0.6"), lw=0.7)
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(*view)
    if not bars:
        return
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    lo = min(np.nanmin(res[a]["curve"][z]) for a in areas)
    hi = max(np.nanmax(res[a]["curve"][z]) for a in areas)
    ylo = lo - 0.05 * (hi - lo)
    ax.set_ylim(ylo, ylo + (hi - ylo) / CURVE_FRAC)
    dy = (1 - CURVE_FRAC - 0.04) / len(areas)
    step = np.median(np.diff(x)) / 2
    srt = sorted(areas, key=lambda a: (not np.isfinite(res[a]["on"]), res[a]["on"] if np.isfinite(res[a]["on"]) else 0))
    for k, a in enumerate(srt):
        y = 1 - (k + 0.5) * dy
        m = np.r_[False, res[a]["above"] & z, False].astype(int)
        st, en = np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0] - 1
        ax.hlines(np.full(len(st), y), np.maximum(x[st] - step, view[0]), np.minimum(x[en] + step, view[1]),
                  color=colors.get(a, "0.6"), lw=1.6, transform=tr)
        on = res[a]["on"]
        if np.isfinite(on) and view[0] <= on <= view[1]:
            ax.plot([on - step] * 2, [y - 0.45 * dy, y + 0.45 * dy], color="k", lw=0.5, transform=tr)
    ax.set_yticks([v for v in ax.get_yticks() if ylo - 1e-9 <= v <= hi])
    ax.spines["left"].set_bounds(ylo, hi)


def hbar_labels(ax, ordered, colors):
    n = len(ordered)
    ax.set_yticks(range(n))
    ax.set_yticklabels([S005.ABBR.get(a, a) for a in ordered[::-1]], fontsize=5)
    for lab, a in zip(ax.get_yticklabels(), ordered[::-1]):
        lab.set_color(colors.get(a, "0.3"))
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0)


def latency_panel(ax, areas, res, colors, ref, err="ci"):
    has = sorted([a for a in areas if np.isfinite(res[a]["on"])], key=lambda a: res[a]["on"])
    ordered = has + [a for a in areas if not np.isfinite(res[a]["on"])]
    n = len(ordered)
    for k, a in enumerate(ordered):
        y = n - 1 - k
        r = res[a]
        if a in has:
            ax.barh(y, r["on"], color=colors.get(a, "0.6"), height=0.72, lw=0)
            if err == "ci" and np.isfinite(r["on_lo"]):
                ax.plot([r["on_lo"], r["on_hi"]], [y, y], color="0.2", lw=0.6)
            elif err == "sd" and np.isfinite(r["on_sd"]):
                ax.plot([r["on"] - r["on_sd"], r["on"] + r["on_sd"]], [y, y], color="0.2", lw=0.6)
        else:
            ax.text(0, y, " none", va="center", ha="left", fontsize=4.5, color="0.4")
    hbar_labels(ax, ordered, colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(f"onset from {ref} (ms)")


def effect_panel(ax, areas, res, colors, xlabel, key="dw", star=True, sd=False):
    val = (lambda a: res[a][key].mean()) if key == "dw" else (lambda a: res[a][key])
    ordered = sorted(areas, key=lambda a: -val(a))
    n = len(ordered)
    vmax = max(val(a) for a in ordered)
    for k, a in enumerate(ordered):
        y = n - 1 - k
        v = val(a)
        ax.barh(y, v, color=colors.get(a, "0.6"), height=0.72, lw=0)
        e = res[a][key].std(ddof=1) if (sd and key == "dw") else 0
        if e:
            ax.plot([v - e, v + e], [y, y], color="0.2", lw=0.6)
        if star:
            s = stars(res[a]["q"])
            if s:
                ax.text(max(v + e, 0) + 0.02 * vmax, y, s, va="center", fontsize=6)
    hbar_labels(ax, ordered, colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(xlabel)


def pair_matrix(ax, areas, res, P, colors):
    ordered = sorted(areas, key=lambda a: -res[a]["dw"].mean())
    n = len(ordered)
    M = np.full((n, n), np.nan)
    S = np.zeros((n, n), bool)
    idx = {a: i for i, a in enumerate(ordered)}
    for r in P.itertuples():
        i, j = idx[r.area_a], idx[r.area_b]
        M[i, j], M[j, i] = r.diff, -r.diff
        S[i, j] = S[j, i] = r.sig_both
    Ml = np.where(np.tril(np.ones((n, n), bool), -1), np.abs(M), np.nan)
    im = ax.imshow(Ml, cmap="Purples", vmin=0, vmax=np.nanmax(Ml), interpolation="nearest")
    ii, jj = np.where(np.tril(S, -1))
    ax.scatter(jj, ii, s=3, color="k", lw=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labs = [S005.ABBR.get(a, a) for a in ordered]
    ax.set_xticklabels(labs, rotation=90, fontsize=4.2)
    ax.set_yticklabels(labs, fontsize=4.2)
    for lab, a in zip(ax.get_yticklabels(), ordered):
        lab.set_color(colors.get(a, "0.3"))
    for lab, a in zip(ax.get_xticklabels(), ordered):
        lab.set_color(colors.get(a, "0.3"))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=4.5, length=1.5)
    cb.set_label("|difference| in window effect", fontsize=5)


# ------------------------------------------------------------------------------------------------ figures
def figure(variant, cohort, R, colors, n_iter):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.07, hspace=0.62, wspace=0.62)
    for ax in axes.flat:
        ax.set_box_aspect(1)
    titles = {"A_full": ("time course, above chance", "onset latency (95% CI)", "window effect (BH)"),
              "B_zoom": ("time course (zoom), above chance", "onset latency (95% CI)", "window effect (BH)"),
              "C_clean_full": ("time course", "onset latency (95% CI)", "time above chance"),
              "D_clean_zoom": ("time course (zoom)", "onset latency (95% CI)", "time above chance"),
              "E_areas": ("window effect ± SD", "pairwise differences", "onset latency ± SD")}[variant]
    for r, tg in enumerate(TARGETS):
        X = R[tg]
        x, areas, res = X["t"], X["areas"], X["res"]
        ref = "first lick" if tg == "modality_lick" else "stimulus"
        w0, w1 = (v * 1000 for v in WINDOW[tg])
        a0, a1, a2 = axes[r]
        if variant == "E_areas":
            chi, df, p = X["wald_window"]
            effect_panel(a0, areas, res, colors, f"real - null, {w0:.0f} to {w1:.0f} ms\n"
                         f"omnibus Wald χ²({df}) = {chi:.0f}, p = {p:.1e}", star=False, sd=True)
            pair_matrix(a1, areas, res, X["pairs"], colors)
            latency_panel(a2, areas, res, colors, ref, err="sd")
            a0.set_ylabel(ROWLAB[tg], fontsize=6.5)
        else:
            view = (FULL if variant in ("A_full", "C_clean_full") else ZOOM)[tg]
            tc_panel(a0, x, areas, res, colors, view, bars=variant in ("A_full", "B_zoom"))
            S012.add_rt_band(a0, S012.rt_values(tg, cohort), view)
            a0.set_xlabel(f"time from {ref} (ms)")
            a0.set_ylabel(f"{ROWLAB[tg]}\nbalanced accuracy - null")
            latency_panel(a1, areas, res, colors, ref, err="ci")
            if variant in ("A_full", "B_zoom"):
                effect_panel(a2, areas, res, colors, f"real - null, {w0:.0f} to {w1:.0f} ms", star=True)
            else:
                span = "0 to 600 ms" if tg != "modality_lick" else "-600 to 0 ms"
                effect_panel(a2, areas, res, colors, f"% of bins above chance\n({span})", key="frac", star=False)
        if r == 0:
            for ax, lab, L in zip(axes[r], titles, "abc"):
                ax.set_title(lab, pad=7)
                ax.text(-0.42, 1.12, L, transform=ax.transAxes, fontweight="bold", fontsize=9)
    d0 = R["hitmiss"]["d"]
    fig.suptitle(f"{cohort}, learning day: pseudo-population decoding ({int(d0.n_mice.iloc[0])} mice x {int(d0.n_neurons.iloc[0])} "
                 f"neurons, {n_iter} iterations x {int(d0.n_null.median())} shifts, PILOT counts) -- variant {variant}\n"
                 "curves = means across iterations; above chance = 5th percentile of real - null > 0; onset = first above-"
                 "chance bin with >= 4 of 5 above (stimulus-aligned: after the stimulus)\n"
                 + ("areas compared with bootstrap distributions (Wald χ² omnibus; pairs: bootstrap-overlap and z "
                    "tests, dots = both BH q < 0.05)" if variant == "E_areas" else
                    "window effect: one-sided bootstrap, BH across areas (* q < 0.05, ** q < 0.01); onset CI: bootstrap "
                    "over iterations"), fontsize=5.8, y=0.995)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"008_{variant}_{cohort}.{ext}", dpi=300)
    plt.close(fig)


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    cohort = sys.argv[1]
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    colors, order = S005.allen()
    R = compute(cohort, n_iter, order, T)
    for v in ("A_full", "B_zoom", "C_clean_full", "D_clean_zoom", "E_areas"):
        figure(v, cohort, R, colors, n_iter)
    rows = []
    for tg in TARGETS:
        X = R[tg]
        for a in X["areas"]:
            r = X["res"][a]
            rows.append(dict(cohort=cohort, target=tg, area=a, n_eligible_mice=r["mice"], onset_ms=r["on"],
                             onset_ci_lo=r["on_lo"], onset_ci_hi=r["on_hi"], onset_boot_sd=r["on_sd"],
                             onset_boot_found=r["on_frac_found"], window_ms=f"{WINDOW[tg][0] * 1000:.0f}..{WINDOW[tg][1] * 1000:.0f}",
                             d_window=r["dw"].mean(), d_window_sd=r["dw"].std(ddof=1), p_window=r["p"], q_window_bh=r["q"],
                             pct_time_above=r["frac"], wald_window_chi2=X["wald_window"][0], wald_window_df=X["wald_window"][1],
                             wald_window_p=X["wald_window"][2], wald_onset_chi2=X["wald_onset"][0],
                             wald_onset_df=X["wald_onset"][1], wald_onset_p=X["wald_onset"][2]))
    pd.DataFrame(rows).to_csv(ART / f"008_stats_{cohort}.csv", index=False)
    pd.concat([R[tg]["pairs"].assign(cohort=cohort) for tg in TARGETS]).to_csv(ART / f"008_pairwise_{cohort}.csv", index=False)
    print("done", cohort)


if __name__ == "__main__":
    main()
