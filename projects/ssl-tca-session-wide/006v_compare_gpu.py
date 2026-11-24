"""Validate the Kuma GPU port (006g, tensorly float32) against the haas CPU fits (006, tensortools) on one session.

Per TCA type: same input (units kept, trials, shape), error curve difference (best restart, ranks 1..20), restart
similarity curves, chosen rank, and similarity of the best components between engines (tensortools kruskal_align rule)
at ranks 1..20 and at the chosen rank. Pass criteria (set before looking): input identical; |error difference| <= 0.002
at every rank; chosen rank within +-1; component similarity >= 0.9 at the chosen rank.

Run: python 006v_compare_gpu.py --session AB123_20240806_110231 -> tables/gpu_validation_<sid>.csv, figure
"""
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tca_lib as L  # noqa: E402

T = importlib.import_module("006_tca_aligned")
G = importlib.import_module("006g_tca_gpu")
GPU_DIR = L.OUT / "artifacts" / "tca_aligned_gpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    sid = a.session
    R = np.array(list(T.RANKS))
    rows = []
    fig, ax = plt.subplots(3, 12, figsize=(16, 4.6), squeeze=False)
    for j, (al, n) in enumerate([(al, n) for al in G.ALIGNMENTS for n in G.NORMS]):
        c = dict(np.load(T.OUT_DIR / sid / f"{al}_{n}.npz"))
        g = dict(np.load(GPU_DIR / sid / f"{al}_{n}.npz"))
        same_input = (np.array_equal(c["units_kept"], g["units_kept"]) and np.array_equal(c["trial_index"], g["trial_index"])
                      and np.array_equal(c["shape"], g["shape"]))
        ec = np.array([c[f"r{r}_obj"].min() for r in R])
        eg = np.array([g[f"r{r}_obj"].min() for r in R])
        sc = np.array([c[f"r{r}_sim"].mean() for r in R[1:]])
        sg = np.array([g[f"r{r}_sim"].mean() for r in R[1:]])
        comp = np.array([G.tt_similarity([c[f"r{r}_f{m}"] for m in range(3)], [g[f"r{r}_f{m}"] for m in range(3)])
                         for r in R])
        rcc, rcg = int(c["rank_chosen"]), int(g["rank_chosen"])
        row = dict(session_id=sid, alignment=al, norm=n, same_input=same_input,
                   max_abs_err_diff=float(np.abs(ec - eg).max()), mean_err_diff_gpu_minus_cpu=float((eg - ec).mean()),
                   rank_cpu=rcc, rank_gpu=rcg, comp_sim_at_cpu_rank=float(comp[rcc - 1]),
                   mean_restart_sim_cpu=float(sc.mean()), mean_restart_sim_gpu=float(sg.mean()),
                   secs_cpu=float(sum(c[f"r{r}_secs"] for r in R)), secs_gpu=float(sum(g[f"r{r}_secs"] for r in R)))
        row["pass"] = bool(same_input and row["max_abs_err_diff"] <= 0.002 and abs(rcc - rcg) <= 1
                           and row["comp_sim_at_cpu_rank"] >= 0.9)
        rows.append(row)
        ax[0, j].plot(R, ec, color="k", lw=1, label="tensortools CPU")
        ax[0, j].plot(R, eg, color="tab:orange", lw=1, ls="--", label="tensorly GPU")
        ax[1, j].plot(R[1:], sc, color="k", lw=1)
        ax[1, j].plot(R[1:], sg, color="tab:orange", lw=1, ls="--")
        ax[2, j].plot(R, comp, color="tab:purple", lw=1)
        ax[2, j].set_ylim(0, 1.02)
        for i, rc_, col in ((0, rcc, "k"), (0, rcg, "tab:orange")):
            ax[i, j].axvline(rc_, color=col, lw=0.5, ls=":")
        ax[0, j].set_title(f"{al}\n{n}: rank {rcc} / {rcg}\n{'PASS' if row['pass'] else 'FAIL'}", fontsize=5)
        ax[2, j].set_xlabel("Rank")
    for i, lab in enumerate(["Relative error (best)", "Restart similarity", "CPU vs GPU components"]):
        ax[i, 0].set_ylabel(lab)
    ax[0, 0].legend(frameon=False, fontsize=4.5)
    fig.tight_layout()
    out = L.OUT / "figures" / "aligned" / sid
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"gpu_validation_{sid}.png", dpi=220)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "tables" / f"gpu_validation_{sid}.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
