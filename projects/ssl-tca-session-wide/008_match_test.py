"""Test: matching TCA components between mice with tensorly's permutation method (user 2026-11-22).

Fits: tensorly parafac (ALS, CPU), same input as 006 (baseline-subtracted rates, chosen normalisation), fixed rank,
10 random restarts, best kept; sign convention as 006 (unit and time factors positive sum).
Matching between two mice: tensorly.metrics.factors.congruence_coefficient (cosine similarity of every component pair,
best one-to-one permutation by the Hungarian algorithm; absolute value, since CP signs are arbitrary). Units and trials
differ between mice, so components are described by features that live on a shared axis:
  time     time factor (same bins in every mouse)
  stacked  [time factor | area profile | trial-class profile], each block scaled to unit norm per component:
           area profile = mean unit weight per area group (groups present in both mice),
           class profile = mean trial weight per trial class (context x trial type x lick; classes in both mice).
Reference (what a perfect match looks like): congruence between the best and the second-best restart of the same mouse.

Run: python 008_match_test.py --sessions AB123_20240806_110231 MH032_20250507_143741 AB144_20241205_111557
     AB092_20231205_140109 [--alignments stim_long jaw] [--norm soft] [--rank 6]
     -> OUT/figures/aligned/match_test/, tables/match_test_<alignment>_<norm>_r<rank>.csv
"""
import argparse
import importlib
import itertools
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorly as tl  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tensorly.decomposition import parafac  # noqa: E402
from tensorly.metrics.factors import congruence_coefficient  # noqa: E402

import tca_lib as L  # noqa: E402

A = importlib.import_module("005_build_aligned")
T = importlib.import_module("006_tca_aligned")
P = importlib.import_module("004_pilot_figures")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
FIG = L.OUT / "figures" / "aligned" / "match_test"
N_REP = 10


def fit(sid, align, norm, rank):
    Y, obs, keep_u, cls, d = T.prepare(sid, align, norm)
    sols = []
    for s in range(N_REP):
        cp, err = parafac(Y, rank, n_iter_max=T.MAX_ITER, tol=T.TOL, init="random", random_state=s,
                          return_errors=True)
        w, (a, b, c) = cp
        a = a * w[None]
        for k in range(rank):   # sign convention as 006
            for f in (b, c):
                sg = np.sign(f[:, k].sum()) or 1.0
                f[:, k] *= sg
                a[:, k] *= sg
        sols.append((float(err[-1]), a, b, c))
    sols.sort(key=lambda x: x[0])
    u = pd.read_parquet(A.OUT_DIR / sid / "units.parquet").iloc[np.flatnonzero(keep_u)]
    _, _, grp = P.area_colours(u.area_acronym_custom.str.strip().tolist())
    return dict(sid=sid, err=sols[0][0], A=sols[0][1], B=sols[0][2], C=sols[0][3], C2=sols[1][3], B2=sols[1][2],
                groups=np.array(grp), classes=cls, t=d["t_centres"])


def unit_norm(M):
    return M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)


def features(f, groups, classes, which):
    """features x components (columns) on axes shared between the two mice"""
    blocks = [unit_norm(f["C"])]
    if which == "stacked":
        blocks.append(unit_norm(np.array([f["B"][f["groups"] == g].mean(0) for g in groups])))
        blocks.append(unit_norm(np.array([f["A"][f["classes"] == c].mean(0) for c in classes])))
    return np.vstack(blocks)


def match(f1, f2, which):
    groups = sorted(set(f1["groups"]) & set(f2["groups"]))
    classes = sorted(c for c in set(f1["classes"]) & set(f2["classes"])
                     if (f1["classes"] == c).sum() >= 3 and (f2["classes"] == c).sum() >= 3)
    X1, X2 = features(f1, groups, classes, which), features(f2, groups, classes, which)
    score, perm = congruence_coefficient(X1, X2, absolute_value=True)
    per = np.abs((unit_norm(X1) * unit_norm(X2[:, perm])).sum(0))
    return float(score), np.asarray(perm), per, len(groups), len(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--alignments", nargs="*", default=["stim_long", "jaw"])
    ap.add_argument("--norm", default="soft")
    ap.add_argument("--rank", type=int, default=6)
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv").set_index("session_id")
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = [(sid, al) for al in a.alignments for sid in a.sessions]
    fits = Parallel(n_jobs=len(jobs), backend="loky")(delayed(fit)(sid, al, a.norm, a.rank) for sid, al in jobs)
    fits = {(sid, al): f for (sid, al), f in zip(jobs, fits)}
    rows = []
    for al in a.alignments:
        for sid in a.sessions:   # reference: best vs second-best restart, same mouse
            f = fits[(sid, al)]
            sc, perm = congruence_coefficient(unit_norm(f["C"]), unit_norm(f["C2"]), absolute_value=True)
            rows.append(dict(alignment=al, mouse_a=sid, mouse_b=sid, which="time (restart reference)", score=sc,
                             cohort_a=s.loc[sid, "cohort"], cohort_b=s.loc[sid, "cohort"]))
        for s1, s2 in itertools.permutations(a.sessions, 2):
            for which in ("time", "stacked"):
                sc, perm, per, ng, nc = match(fits[(s1, al)], fits[(s2, al)], which)
                rows.append(dict(alignment=al, mouse_a=s1, mouse_b=s2, which=which, score=sc, perm=perm.tolist(),
                                 per_component=np.round(per, 3).tolist(), n_groups=ng, n_classes=nc,
                                 cohort_a=s.loc[s1, "cohort"], cohort_b=s.loc[s2, "cohort"]))
        # figure: reference mouse = first session; its components and the matched ones of the others
        ref = a.sessions[0]
        fr = fits[(ref, al)]
        fig, ax = plt.subplots(3, a.rank, figsize=(7.4, 3.6), squeeze=False)
        tms = fr["t"] * 1e3
        for k in range(a.rank):
            ax[0, k].plot(tms, unit_norm(fr["C"])[:, k], color="k", lw=1.2, label=ref.split("_")[0])
        for sid in a.sessions[1:]:
            sc, perm, per, _, _ = match(fr, fits[(sid, al)], "time")
            f2 = fits[(sid, al)]
            col = COH_COL[s.loc[sid, "cohort"]]
            for k in range(a.rank):
                v = unit_norm(f2["C"])[:, perm[k]]
                v = v * np.sign((v * unit_norm(fr["C"])[:, k]).sum())
                ax[0, k].plot(tms, v, color=col, lw=0.7, alpha=0.8, label=f"{sid.split('_')[0]} ({per[k]:.2f})")
        for k in range(a.rank):
            ax[0, k].set_title(f"{ref.split('_')[0]} C{k + 1}", fontsize=6)
            ax[0, k].legend(frameon=False, fontsize=3.8)
            ax[0, k].set_xlabel(f"Time from {A.ALIGN[al]['event']} (ms)")
        ax[0, 0].set_ylabel("Time factor (unit norm)")
        df = pd.DataFrame([r for r in rows if r["alignment"] == al and r["mouse_a"] != r["mouse_b"]])
        for i, which in enumerate(("time", "stacked")):
            M = df[df.which == which].pivot(index="mouse_a", columns="mouse_b", values="score").reindex(
                index=a.sessions, columns=a.sessions)
            axm = ax[1 + i, 0]
            for kk in range(1, a.rank):
                ax[1 + i, kk].axis("off")
            im = axm.imshow(M.to_numpy(float), vmin=0, vmax=1, cmap="magma")
            labs = [f"{x.split('_')[0]} {s.loc[x, 'cohort']}" for x in a.sessions]
            axm.set_xticks(range(len(labs)))
            axm.set_xticklabels(labs, rotation=60, fontsize=4.5)
            axm.set_yticks(range(len(labs)))
            axm.set_yticklabels(labs, fontsize=4.5)
            axm.set_title(f"mean matched congruence, {which}", fontsize=5.5, loc="left")
            for (ii, jj), v in np.ndenumerate(M.to_numpy(float)):
                if np.isfinite(v):
                    axm.text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=4, color="w" if v < 0.6 else "k")
            ref_sc = np.mean([r["score"] for r in rows if r["alignment"] == al and r["mouse_a"] == r["mouse_b"]])
            ax[1 + i, 1].text(0, 0.5, f"reference (two restarts, same mouse,\ntime factors): mean {ref_sc:.2f}",
                              fontsize=5.5, va="center")
        fig.suptitle(f"Component matching between mice (tensorly congruence + permutation), {al}, {a.norm}, "
                     f"rank {a.rank}", fontsize=6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"match_test_{al}_{a.norm}_r{a.rank}.{ext}", dpi=250)
        plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(L.OUT / "tables" / f"match_test_{a.norm}_r{a.rank}.csv", index=False)
    print(out.groupby(["alignment", "which"]).score.describe().round(3).to_string())


if __name__ == "__main__":
    main()
