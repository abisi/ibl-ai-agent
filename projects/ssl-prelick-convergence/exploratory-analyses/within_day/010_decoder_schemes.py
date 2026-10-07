"""Does the decoder result depend on how many neurons the single-session decoder uses?

Compares the within-day, across-day and carry-over changes of the decoder readout (003, 4 events per class) between
unit-sampling schemes: one draw of 150 units (original), 10 draws of 150 units (averaged), 10 draws of 50 and of 100 units,
5 draws of 400 units, and all units of each session (one fit; unit counts then differ between sessions).
Output: combined_results_ks4/ssl-prelick-convergence/within_day/<ref>/cosyne/decoder_schemes.{png,pdf,svg} + decoder_schemes.csv
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent                                         # across-day scripts (projects merged 2026-10-07)
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
BASE = m51.WITHIN
SCHEMES = [("epochs_n4_u50x10", "50 × 10"), ("epochs_n4_u100x10", "100 × 10"), ("epochs_n4", "150 × 1"),
           ("epochs_n4_u150x10", "150 × 10"), ("epochs_n4_u400x5", "400 × 5"), ("epochs_n4_uall", "all")]
NAMES = [("within-day", "Within day 0\n(late − early)"), ("across-day (early)", "Across days\n(expert − day 0, early halves)"),
         ("carry-over", "Carry-over\n(expert early − day-0 late)")]


def main():
    plt = m62.setup()
    rows = []
    for d, lab in SCHEMES:
        f = BASE / d / "all" / "epoch_contrasts.csv"
        if not f.exists():
            continue
        E = pd.read_csv(f)
        E = E[(E.measure == "dec") & (E.kind == "contrast")]
        for _, r in E.iterrows():
            rows.append(dict(scheme=lab, cohort=r.cohort, name=r["name"], value=r.value, lo=r.get("lo"), hi=r.get("hi"),
                             p_boot=r.get("p_boot"), p_perm=r.get("p_perm")))
    T = pd.DataFrame(rows)
    T.to_csv(BASE / "cosyne" / "decoder_schemes.csv", index=False)
    labs = [l for d, l in SCHEMES if l in set(T.scheme)]
    fig, axs = plt.subplots(1, 3, figsize=(m62.W_IN, 2.3), gridspec_kw=dict(wspace=0.35))
    for ax, (nm, ttl) in zip(axs, NAMES):
        for k_, c in enumerate(["R+", "R-"]):
            q = T[(T.cohort == c) & (T.name == nm)].set_index("scheme").reindex(labs)
            x = np.arange(len(labs)) + (k_ - 0.5) * 0.25
            ax.errorbar(x, q.value, [q.value - q.lo, q.hi - q.value], fmt="o", ms=3, color=m62.COH[c], lw=0.8, capsize=0,
                        label=c.replace("-", "−"))
        pp = T[(T.cohort == "R+ - R-") & (T.name == nm)].set_index("scheme").reindex(labs)
        for i, l in enumerate(labs):
            ax.text(i, 1.01, m62.fmt_p(pp.loc[l, "p_perm"]).replace("p = ", "").replace("p < ", "<"),
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=4.3)
        ax.axhline(0, color="0.3", lw=0.4)
        ax.set_xticks(range(len(labs)), labs, fontsize=4.8, rotation=30)
        ax.set_xlabel("Units per decoder × number of draws", fontsize=5)
        ax.set_title(ttl, fontsize=5.4, pad=9)
        if ax is axs[0]:
            ax.set_ylabel("Change in decoder readout\n(P(AH|WH) − P(AH|SL) − chance)", fontsize=5.2)
            ax.legend(frameon=False, fontsize=4.8)
    fig.suptitle("Single-session decoder: results by number of neurons (95% CI; top: R+ vs R−, mouse permutation)",
                 x=0.06, y=1.02, ha="left", fontsize=6.6, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.27)
    m62.save(fig, BASE / "cosyne", "decoder_schemes"); plt.close(fig)
    print("ALL DONE", BASE / "cosyne")


if __name__ == "__main__":
    main()
