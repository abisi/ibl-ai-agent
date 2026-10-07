"""075 -- Supplementary figures (user review 2026-10-07): absolute selectivity with the sign breakdown, and angles.

FigS_abs_selectivity_<pop>: single-neuron pre-lick ROC (051; selectivity s = 2 AUC - 1, significance by 1,000 label
  permutations, p < 0.05), good + mua units with mean raw pre-lick rate >= 0.1 Hz, sessions with >= 10 tested units.
  Comparisons and sign convention (positive = second class higher):
    WH vs ref   whisker_hit_vs_fa_prelick  (+: WH > ref)
    WH vs AH    wh_vs_aud_hit_prelick      (+: AH > WH)
    AH vs ref   auditory_hit_vs_fa_prelick (+: AH > ref)
  Rows (per session, one dot per session): (1) mean |s| over all tested units; (2) fraction of tested units significant
  with s > 0; (3) fraction significant with s < 0; (4) mean |s| over the significant units (effect size of the
  selective units). Statistics: 062 dots_panel (MWU learning vs expert per cohort, cohort contrast in experts, learning x
  cohort interaction by mouse-level cohort permutation).
FigS_angles_<pop>: from the cross-validated distances of 057 (whole brain, variant all; d per unit, z-scored units):
  along = (WH - ref).(AH - ref) = [d(WH,ref) + d(AH,ref) - d(WH,AH)] / 2,
  cos theta = along / sqrt(d(WH,ref) d(AH,ref)),  length ratio = sqrt(d(WH,ref) / d(AH,ref)),
  lambda = along / d(AH,ref) = length ratio x cos theta.
  Sessions with d(WH,ref) > 0 and an axis passing 057's rule (axis_ok); cos theta clipped to [-1, 1].
Output: across_days/<ref>/publication/<pop>/FigS_abs_selectivity.*, FigS_angles.*, supp_selectivity_angles_<pop>.csv
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
COMP = [("whisker_hit_vs_fa_prelick", "WH vs {r}", "+: WH > {r}"), ("wh_vs_aud_hit_prelick", "WH vs AH", "+: AH > WH"),
        ("auditory_hit_vs_fa_prelick", "AH vs {r}", "+: AH > {r}")]
MIN_UNITS = 10


def session_selectivity(W):
    rows = []
    for sid, g in W.groupby("session_id"):
        r = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0])
        for key, _, _ in COMP:
            s, sig = g[f"sel:{key}@all"], g[f"sig:{key}@all"]
            ok = s.notna() & sig.notna()
            s, sig = s[ok].astype(float), sig[ok].astype(float) == 1
            if len(s) < MIN_UNITS:
                continue
            r[f"abs_{key}"] = s.abs().mean()
            r[f"pos_{key}"] = (sig & (s > 0)).mean()
            r[f"neg_{key}"] = (sig & (s < 0)).mean()
            r[f"abs_sig_{key}"] = s[sig].abs().mean() if sig.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def angles(L):
    d = L[(L.level == "all") & (L.variant == "all")].copy()
    d["along"] = (d.d_WH_FA + d.d_AH_FA - d.d_WH_AH) / 2
    ok = (d.d_WH_FA > 0) & (d.d_AH_FA > 0) & d.axis_ok.astype(bool)
    d["cos"] = np.where(ok, np.clip(d.along / np.sqrt(d.d_WH_FA.clip(lower=1e-12) * d.d_AH_FA.clip(lower=1e-12)), -1, 1), np.nan)
    d["ratio"] = np.where(ok, np.sqrt(d.d_WH_FA.clip(lower=0) / d.d_AH_FA.clip(lower=1e-12)), np.nan)
    d["lam_from_parts"] = d.ratio * d.cos
    return d


def main(a):
    plt = m62.setup()
    RA = "FA" if m51.REF == "fa" else "SL"
    out = m51.OUTROOT / "publication" / a.population
    out.mkdir(parents=True, exist_ok=True)
    keep = m61.learner_filter if a.population == "learners" else (lambda df: df)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(m62.UNIT_SET)]
    for key, _, _ in COMP:                              # mean raw pre-lick rate >= MIN_FR, as every 051 analysis
        fr = W.get(f"fr_window:{key}@all")
        if fr is not None:
            W.loc[fr < m51.MIN_FR, [f"sel:{key}@all", f"sig:{key}@all"]] = np.nan
    S = keep(session_selectivity(W))
    rng = np.random.default_rng(0)
    # --- absolute selectivity + sign breakdown
    rows = [("abs", "Mean |selectivity|\n(all tested units)"), ("pos", "Fraction significant,\nselectivity > 0"),
            ("neg", "Fraction significant,\nselectivity < 0"), ("abs_sig", "Mean |selectivity|\n(significant units)")]
    fig = plt.figure(figsize=(m62.W_IN, 8.6))
    gs = fig.add_gridspec(len(rows), 3, hspace=1.0, wspace=0.6, left=0.1, right=0.98, top=0.92, bottom=0.05)
    axes = []
    for i, (pre, ylab) in enumerate(rows):
        for j, (key, lab, sign) in enumerate(COMP):
            ax = fig.add_subplot(gs[i, j]); axes.append(ax)
            ttl = f"{lab.format(r=RA)} ({sign.format(r=RA)})" if i == 0 else lab.format(r=RA)
            m62.dots_panel(ax, S, f"{pre}_{key}", ylab if j == 0 else "", rng, f"S-sel {pre} {key}", ttl)
            ax.set_title(ax.get_title(), fontsize=5.3)
    m62.letter_row(fig, axes[0:3], "abc"); m62.letter_row(fig, axes[3:6], "def"); m62.letter_row(fig, axes[6:9], "ghi")
    m62.letter_row(fig, axes[9:12], "jkl")
    fig.suptitle(f"Figure 2—supplement | Absolute selectivity and its sign ({RA} reference, {a.population}; one dot per session)",
                 x=0.02, y=0.99, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "FigS_abs_selectivity"); plt.close(fig)
    # --- angles
    L = keep(pd.read_csv(m51.OUTROOT / "lambda" / "lambda_sessions.csv"))
    A = angles(L)
    fig = plt.figure(figsize=(m62.W_IN, 2.9))
    gs = fig.add_gridspec(1, 4, wspace=0.65, left=0.07, right=0.98, top=0.78, bottom=0.2)
    axs = [fig.add_subplot(gs[0, k]) for k in range(4)]
    m62.dots_panel(axs[0], A, "cos", f"cos θ (WH−{RA}, AH−{RA})", rng, "S-ang cos", "Angle: cos θ", ref=[(0, "0.6"), (1, "0.8")])
    m62.dots_panel(axs[1], A, "ratio", f"|WH−{RA}| / |AH−{RA}|", rng, "S-ang ratio", "Length ratio", ref=[(1, "0.8")])
    m62.dots_panel(axs[2], A.assign(lam_=A.lam_from_parts), "lam_", "λ = ratio × cos θ", rng, "S-ang lam", "λ", ref=[(0, "0.6"), (1, "0.8")])
    ax = axs[3]
    for k in m62.GROUPS:
        q = A[(A.cohort == k[0]) & (A.stage == k[1])]
        ax.scatter(q.cos, q.ratio, s=7, facecolor="white" if k[1] == "learning" else m62.COH[k[0]], edgecolor=m62.COH[k[0]],
                   lw=0.6, label=m62.GLAB[k])
    cc = np.linspace(0.05, 1, 50)
    for lv in (0.25, 0.5, 1.0):                         # iso-lambda lines: ratio = lambda / cos
        ax.plot(cc, lv / cc, color="0.75", lw=0.5, ls=(0, (2, 2)))
        ax.text(1.0, lv, f"λ = {lv:g}", fontsize=4.2, color="0.5", va="center")
    ax.set_xlim(-0.2, 1.1); ax.set_ylim(0, max(2.0, np.nanpercentile(A.ratio, 98)))
    ax.set_xlabel("cos θ"); ax.set_ylabel("Length ratio"); ax.legend(frameon=False, fontsize=4.2, loc="upper left")
    ax.set_title("Sessions (dashed: iso-λ)", fontsize=5.6)
    for ax in axs[:3]:
        ax.set_title(ax.get_title(), fontsize=5.6)
    m62.letter_row(fig, axs, "abcd")
    fig.suptitle(f"Figure 3—supplement | Angle and length behind λ ({RA} reference, {a.population}; whole brain, cross-validated)",
                 x=0.02, y=0.99, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "FigS_angles"); plt.close(fig)
    S.merge(A[["session_id", "cos", "ratio", "lam", "lam_from_parts"]], on="session_id", how="outer").to_csv(
        out / f"supp_selectivity_angles_{a.population}.csv", index=False)
    pd.DataFrame(m62.STATS).to_csv(out / f"stats_supp_selectivity_angles_{a.population}.csv", index=False)
    print("ALL DONE", out, "| lambda vs ratio x cos: max |diff| =",
          float(np.nanmax(np.abs(A.lam_from_parts - A.lam.where(A.cos.notna())))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
