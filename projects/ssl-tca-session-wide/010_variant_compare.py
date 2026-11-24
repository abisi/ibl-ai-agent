"""Compare fit variants against CP degeneracy (cancelling component pairs), 6-mouse pilot (user 2026-11-22).

Variants (006g): signed CP (""), signed CP with ridge (tensorly l2_reg: "_l2-1", "_l2-10", "_l2-100", "_l2-1000"),
split non-negative CP ("_splitnn": increase / decrease channels, cannot cancel), semi-non-negative CP ("_seminn":
trial and time weights >= 0, unit weights signed). Per alignment, median over mice
(thin lines = mice) vs rank: error on the signed baseline-subtracted data (fitted trials), mean restart similarity,
most negative component overlap (signed Tucker congruence; < -0.5 = cancelling pair), and the number of mice with a
cancelling pair. Table at a reference rank.

Run: python 010_variant_compare.py --sessions <sid> ... [--alignments stim_long jaw] [--ref-rank 7]
     -> OUT/figures/aligned_gpu/variant_compare.png|pdf, tables/variant_compare.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tca_lib as L  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
VARIANTS = [("", "signed CP", "k"), ("_l2-10", "ridge 10", "#6baed6"), ("_l2-100", "ridge 100", "#2171b5"),
            ("_splitnn", "split non-negative", "#e6550d"), ("_seminn", "semi-non-negative", "#31a354")]
D = L.OUT / "artifacts" / "tca_aligned_gpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--alignments", nargs="*", default=["stim_long", "jaw"])
    ap.add_argument("--norm", default="zscore")
    ap.add_argument("--ref-rank", type=int, default=7)
    a = ap.parse_args()
    rows = []
    for sid in a.sessions:
        for al in a.alignments:
            for tag, lab, _ in VARIANTS:
                f = D / sid / f"{al}_{a.norm}{tag}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                for r in range(1, 13):
                    if f"r{r}_obj" not in d.files:
                        continue
                    rows.append(dict(session_id=sid, alignment=al, variant=lab, rank=r,
                                     err=float(d[f"r{r}_obj"].min()),
                                     sim=float(d[f"r{r}_sim"].mean()) if len(d[f"r{r}_sim"]) else 1.0,
                                     mincong=float(d[f"r{r}_mincong"]) if f"r{r}_mincong" in d.files else np.nan))
    df = pd.DataFrame(rows)
    # signed CP without ridge was fitted before mincong was stored: compute it from its factors
    miss = df.variant.eq("signed CP") & df.mincong.isna() & (df["rank"] > 1)
    for i in np.flatnonzero(miss):
        r_ = df.iloc[i]
        d = np.load(D / r_.session_id / f"{r_.alignment}_{a.norm}.npz")
        F = [d[f"r{r_['rank']}_f{m}"] for m in range(3)]
        un = [f / np.linalg.norm(f, axis=0, keepdims=True) for f in F]
        C = (un[0].T @ un[0]) * (un[1].T @ un[1]) * (un[2].T @ un[2])
        df.iloc[i, df.columns.get_loc("mincong")] = C[~np.eye(len(C), dtype=bool)].min()
    df["cancelling"] = df.mincong < -0.5
    fig, ax = plt.subplots(4, len(a.alignments), figsize=(3.3 * len(a.alignments) + 0.6, 6.2), squeeze=False)
    for j, al in enumerate(a.alignments):
        for tag, lab, col in VARIANTS:
            x = df[(df.alignment == al) & (df.variant == lab)]
            if x.empty:
                continue
            for sid, g in x.groupby("session_id"):
                ax[0, j].plot(g["rank"], g.err, color=col, lw=0.3, alpha=0.5)
            m = x.groupby("rank").agg(err=("err", "median"), sim=("sim", "median"), mincong=("mincong", "median"),
                                      canc=("cancelling", "sum"))
            ax[0, j].plot(m.index, m.err, color=col, lw=1.1, label=lab)
            ax[1, j].plot(m.index, m.sim, color=col, lw=1.1)
            ax[2, j].plot(m.index, m.mincong, color=col, lw=1.1)
            ax[3, j].plot(m.index, m.canc, color=col, lw=1.1, marker="o", ms=1.8)
        ax[1, j].axhline(0.85, color="0.5", lw=0.4, ls=":")
        ax[2, j].axhline(-0.5, color="tab:red", lw=0.4, ls=":")
        ax[0, j].set_title(al, fontsize=6.5)
        for i in range(4):
            ax[i, j].set_xticks([1, 4, 8, 12])
        ax[3, j].set_xlabel("Rank")
    for i, lab in enumerate(["Relative error\n(signed data, fitted trials)", "Restart similarity (median)",
                             "Most negative component\noverlap (median)", "Mice with a\ncancelling pair"]):
        ax[i, 0].set_ylabel(lab)
    ax[0, 0].legend(frameon=False, fontsize=5)
    fig.align_ylabels(ax[:, 0])
    fig.tight_layout()
    out = L.OUT / "figures" / "aligned_gpu"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"variant_compare.{ext}", dpi=230)
    df.to_csv(L.OUT / "tables" / "variant_compare.csv", index=False)
    t = df[df["rank"] == a.ref_rank].groupby(["alignment", "variant"]).agg(
        err=("err", "median"), sim=("sim", "median"), mincong=("mincong", "median"), mice_cancelling=("cancelling", "sum"),
        n=("session_id", "nunique"))
    print(f"rank {a.ref_rank}, median over mice:")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
