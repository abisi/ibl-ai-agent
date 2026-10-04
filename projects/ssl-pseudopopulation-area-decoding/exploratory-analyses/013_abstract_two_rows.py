"""013 -- COSYNE abstract figure, two decoding types that change over time (user 2026-09-30): row 1 = hit vs miss
(stimulus-aligned), row 2 = whisker vs auditory (first-lick-aligned). Five square panels per row:
  a  area overlay time course (pseudo-populations, real - paired shift null, mean over iterations; row 1 R+, row 2 R+
     and R- averaged -- no cohort difference before the lick), grey band = reaction-time distribution (012);
  b  onset latency per area, vertical bars fast -> slow, no error bars (011 rule: first above-chance bin with >= 4 of 5
     above; stimulus-aligned searched after the stimulus);
  c  whole brain R+ vs R- (pseudo-populations, mean +- SD across iterations), one reaction-time histogram per cohort;
     black bar = bins where the cohorts differ (bootstrap overlap and z test, both p < 0.05, uncorrected, as 006);
  d  within-session change, whole brain: 2nd - 1st session half (single-session decoding with a separate decoder per
     half, 024 "half" condition of ssl-whisker-hitmiss-timeresolved-decoding; accuracy - mean linear-shift null),
     R+ vs R- (mean +- SEM across sessions with both halves);
  e  whole brain window value, 1st vs 2nd half per cohort (mean +- SEM; within cohort paired Wilcoxon | paired t; change
     R+ vs R-: Mann-Whitney | Welch). Windows: hit vs miss 5-100 ms, lick-aligned -100-0 ms.
Areas as the comparisons (005 EXCLUDE). Pseudo-population counts are the pilot 100 iterations x 10 shifts.
Outputs: figures/013_abstract_two_rows.pdf/.png, figures/013_panels/*, ../artifacts/013_stats.csv
Run (haas, repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/013_abstract_two_rows.py [n_iter]
"""

from __future__ import annotations

import importlib.util
import json
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
HALVES = ROOT_REPO / "projects" / "ssl-whisker-hitmiss-timeresolved-decoding" / "exploratory-analyses"



def perm_text(target):
    """Whole-brain mouse-level cohort-label permutation p per window (009, 200 permutations x 10 iterations per group,
    with the linear-shift null; artifacts/009_perm_wb_abstract_pvalues.csv), or None."""
    f = ART / "009_perm_wb_abstract_pvalues.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.target == target]
    if d.empty:
        return None
    return "cohort permutation (whole brain):\n" + "\n".join(f"{w.replace('ms', ' ms')}: p = {p:.2f}" for w, p in zip(d.window, d.p_perm))

def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S003 = _load("s003", "003_summarize_pseudopop.py")
S005 = _load("s005", "005_overlay_areas.py")
S011 = _load("s011", "011_overlay_latency.py")
S012 = _load("s012", "012_rt_band.py")
COL = {"R+": "#00B400", "R-": "#C800C8"}
ROWS = [dict(target="hitmiss", overlay="R+", tag="hitmiss_stim", ref="stimulus", win=(5, 100),
             label="Hit vs miss\n(stimulus-aligned)", view=(-200, 600)),
        dict(target="modality_lick", overlay="avg", tag="modality_lick", ref="first lick", win=(-100, 0),
             label="Whisker vs auditory\n(first-lick-aligned)", view=(-600, 200))]
STATS = []
RECORD = {"on": True}                     # single-panel redraws must not add stats rows


def sem(a, axis=0):
    a = np.asarray(a, float)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(a), axis=axis))


def two_tests(A, B):
    gt, lt = (A[:, None, :] > B[None, :, :]).mean((0, 1)), (A[:, None, :] < B[None, :, :]).mean((0, 1))
    pb = np.clip(2 * np.minimum(gt, lt), 1 / min(len(A), len(B)), 1)
    z = (A.mean(0) - B.mean(0)) / np.sqrt(A.var(0, ddof=1) + B.var(0, ddof=1))
    return pb, 2 * stats.norm.sf(np.abs(z))


def pseudo_wholebrain(target, n_iter):
    out = {}
    for c in ("R+", "R-"):
        d = pd.read_parquet(ART / f"002_pseudo_{target}_{c}.parquet")
        g = d[d.skipped_reason.isna() & (d.area == "All units")].sort_values("rep").head(n_iter)
        out[c] = (np.stack(g.curve.map(np.asarray).to_numpy()) - np.stack(g.null_mean_curve.map(np.asarray).to_numpy()),
                  int(g.n_eligible_mice.iloc[0]))
    return out


def halves(tag):
    """Whole-brain session halves: dict cohort -> (first, second, session ids), t in ms."""
    d = pd.read_parquet(HALVES / f"024_master_results_{tag}_whole_brain.parquet",
                        columns=["session_id", "reward_group", "condition_type", "condition_value", "skipped_reason",
                                 "real_curve", "shift_null_curves"])
    d = d[d.skipped_reason.isna() & (d.condition_type == "half")].copy()
    t = np.array([b[1] for b in json.load(open(HALVES / f"024_bin_edges_{tag}_whole_brain.json"))]) * 1000
    d["dc"] = [np.asarray(r, float) - np.nanmean(np.stack([np.asarray(x, float) for x in L]), 0)
               for r, L in zip(d.real_curve, d.shift_null_curves)]
    out = {}
    for rg in ("R+", "R-"):
        g = d[d.reward_group == rg]
        f = g[g.condition_value == "first"].set_index("session_id").dc
        s = g[g.condition_value == "second"].set_index("session_id").dc
        ids = sorted(set(f.index) & set(s.index))
        out[rg] = (np.stack([f[i] for i in ids]), np.stack([s[i] for i in ids]), ids)
    return out, t


def panel_overlay(ax, X, colors, row, rt):
    S011.tc(ax, X, colors, row["view"], row["ref"], f"{row['label']}\nbalanced accuracy - null", rt)


def panel_cohorts(ax, x, W, row):
    view = row["view"]
    z = (x >= view[0]) & (x <= view[1])
    for c in ("R+", "R-"):
        D, n = W[c]
        mu, sd = D.mean(0), D.std(0, ddof=1)
        ax.fill_between(x[z], (mu - sd)[z], (mu + sd)[z], color=COL[c], alpha=0.2, lw=0)
        ax.plot(x[z], mu[z], color=COL[c], lw=0.9, label=f"{c} ({n} mice)")
    pb, pz = two_tests(W["R+"][0], W["R-"][0])
    sig = (pb < 0.05) & (pz < 0.05)
    top = ax.get_ylim()[1]
    ax.plot(np.where(sig & z, x, np.nan), np.full(len(x), top), color="k", lw=2, solid_capstyle="butt")
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(*view)
    ax.set_xlabel(f"time from {row['ref']} (ms)")
    ax.set_ylabel("whole brain, real - null")
    ax.legend(loc="upper left", fontsize=5.3, handlelength=1.2)
    S012.add_rt_band(ax, [(S012.rt_values(row["target"], c), COL[c]) for c in ("R+", "R-")], view)
    pt = perm_text(row["target"])
    if pt:
        ax.text(0.98, 0.3, pt, transform=ax.transAxes, ha="right", va="bottom", fontsize=4.3, color="0.2")
    RECORD["on"] and STATS.append(dict(row=row["target"], panel="c", test="pseudo-pop whole brain R+ vs R- per bin (uncorrected)",
                      n_sig_bins=int(sig[z].sum()), sig_bins_ms=";".join(f"{v:.0f}" for v in x[sig & z])))


def panel_change(ax, H, t, row):
    view = row["view"]
    z = (t >= view[0]) & (t <= view[1])
    for rg in ("R+", "R-"):
        F, S, ids = H[rg]
        D = S - F
        mu, se = np.nanmean(D, 0), sem(D)
        ax.fill_between(t[z], (mu - se)[z], (mu + se)[z], color=COL[rg], alpha=0.22, lw=0)
        ax.plot(t[z], mu[z], color=COL[rg], lw=0.9, label=f"{rg} (n = {len(ids)})")
    ax.axvspan(*row["win"], color="0.9", lw=0, zorder=0)
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(*view)
    ax.set_xlabel(f"time from {row['ref']} (ms)")
    ax.set_ylabel("change, 2nd - 1st half")
    ax.legend(loc="upper left", fontsize=5.3, handlelength=1.2)


def panel_halves(ax, H, t, row):
    m = (t >= row["win"][0]) & (t <= row["win"][1])
    chg = {}
    for k, rg in enumerate(("R+", "R-")):
        F, S, ids = H[rg]
        f, s = np.nanmean(F[:, m], 1), np.nanmean(S[:, m], 1)
        ok = np.isfinite(f) & np.isfinite(s)                 # half without a usable shift null in the window -> pair dropped
        f, s, ids = f[ok], s[ok], [i for i, k_ in zip(ids, ok) if k_]
        chg[rg] = s - f
        for j, v in enumerate((f, s)):
            ax.bar(k * 2.3 + j, v.mean(), yerr=sem(v), color=COL[rg], alpha=0.45 if j == 0 else 0.9, width=0.8, lw=0,
                   error_kw=dict(lw=0.6, capsize=0))
        pw, pt = stats.wilcoxon(s, f).pvalue, stats.ttest_rel(s, f).pvalue
        ax.text(k * 2.3 + 0.5, max(f.mean() + sem(f), s.mean() + sem(s)) * 1.04 + 0.005,
                f"W {pw:.2g}\nt {pt:.2g}", ha="center", va="bottom", fontsize=4.8)
        RECORD["on"] and STATS.append(dict(row=row["target"], panel="e", test=f"1st vs 2nd half, whole brain {row['win']} ms", cohort=rg,
                          n=len(ids), mean_first=f.mean(), mean_second=s.mean(), p_wilcoxon=pw, p_paired_t=pt))
    pm = stats.mannwhitneyu(chg["R+"], chg["R-"]).pvalue
    pwl = stats.ttest_ind(chg["R+"], chg["R-"], equal_var=False).pvalue
    RECORD["on"] and STATS.append(dict(row=row["target"], panel="e", test=f"change R+ vs R-, whole brain {row['win']} ms",
                      n_rplus=len(chg["R+"]), n_rminus=len(chg["R-"]), mean_change_rplus=chg["R+"].mean(),
                      mean_change_rminus=chg["R-"].mean(), p_mannwhitney=pm, p_welch=pwl))
    ax.set_xticks([0, 1, 2.3, 3.3], ["1st", "2nd", "1st", "2nd"])
    ax.set_xlabel(f"R+          R−\nchange R+ vs R−:\nMW p={pm:.2g} | Welch p={pwl:.2g}", fontsize=5.8)
    ax.set_ylabel(f"real - null, {row['win'][0]:.0f} to {row['win'][1]:.0f} ms")
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.3)
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    colors, order = S005.allen()
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 3.9))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.22, hspace=0.95, wspace=0.62)
    for ax in axes.flat:
        ax.set_box_aspect(1)
    pan = FIG / "013_panels"
    pan.mkdir(parents=True, exist_ok=True)
    drawn = set()
    for r, row in enumerate(ROWS):
        X = S011.compute(row["overlay"], n_iter, order, T)[row["target"]]
        rt = S012.rt_values(row["target"], row["overlay"])
        W = pseudo_wholebrain(row["target"], n_iter)
        H, th = halves(row["tag"])
        a, b, c, d, e = axes[r]
        panel_overlay(a, X, colors, row, rt)
        drawn |= set(S011.latency(b, X, colors, row["ref"], legend=False))
        panel_cohorts(c, X["x"], W, row)
        panel_change(d, H, th, row)
        panel_halves(e, H, th, row)
        a.set_title("R+" if row["overlay"] == "R+" else "R+ and R− averaged", fontsize=6.3, pad=4)
        RECORD["on"] = False                          # single-panel copies below: no extra stats rows
        for ax, lab in zip((a, b, c, d, e), ("overlay", "latency", "cohorts", "halfchange", "halves")):
            f1, ax1 = plt.subplots(figsize=(2.2, 2.2))
            ax1.set_box_aspect(1)
            if lab == "overlay":
                panel_overlay(ax1, X, colors, row, rt)
            elif lab == "latency":
                S011.latency(ax1, X, colors, row["ref"], legend=True)
                f1.set_size_inches(3.0, 2.2)
            elif lab == "cohorts":
                panel_cohorts(ax1, X["x"], W, row)
            elif lab == "halfchange":
                panel_change(ax1, H, th, row)
            else:
                panel_halves(ax1, H, th, row)
            f1.tight_layout()
            for ext in ("pdf", "png"):
                f1.savefig(pan / f"013_{lab}_{row['target']}.{ext}", dpi=300)
            plt.close(f1)
        RECORD["on"] = True
    for ax, lab, L in zip(axes[0], ("areas (pseudo-populations)", "onset latency", "R+ vs R−, whole brain",
                                    "within-session change", "session halves"), "abcde"):
        ax.text(0.5, 1.3, lab, transform=ax.transAxes, ha="center", fontsize=6.8)
        ax.text(-0.45, 1.3, L, transform=ax.transAxes, fontweight="bold", fontsize=9)
    from matplotlib.patches import Patch
    names = [g for g in order if g in drawn]
    fig.legend(handles=[Patch(color=colors.get(g, "0.6"), label=g) for g in names] + [Patch(color="#b0b0b0", label="reaction times")],
               loc="lower center", ncol=6, fontsize=5, handlelength=0.9, handleheight=0.7, columnspacing=0.9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Learning day. a-c: pseudo-populations (20 mice x 10 neurons, {n_iter} iterations x 10 shifts"
                 f"{', PILOT' if n_iter < 1000 else ''});\nd-e: single sessions, separate decoder per session half "
                 "(mean ± SEM across sessions); all values = balanced accuracy - linear-shift null",
                 fontsize=5.8, y=0.995)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"013_abstract_two_rows.{ext}", dpi=300)
    pd.DataFrame(STATS).assign(halves_source="024_master_results_<tag>_whole_brain.parquet (half)").to_csv(
        ART / "013_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(STATS).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
