"""Recap of all iterations of the pre-lick convergence analyses (same statistics recomputed for every iteration).

Iterations (session tables as saved by each version; archived versions under _roc_prelick/_archive_*):
  v1      min raw pre-lick rate 1 Hz, all units (no quality filter); lambda >= 10 units, AH-FA axis >= 0.02;
          lambda_LDA >= 10 units, d' >= 0.5; decoder >= 10 units, bacc >= 0.6 (uncorrected)
  v2      0.1 Hz, good + mua; lambda >= 10 units, axis >= 0.02; lambda_LDA >= 5 units, d' >= 0.5; decoder >= 5 units,
          bacc >= 0.55 (uncorrected)
  v3      as v2 but lambda >= 5 units and no axis-reliability threshold (lambda and lambda_LDA)
  v4 FA   as v3 but axis >= 0.01 (lambda), d' >= 0.3 (lambda_LDA); decoders chance-corrected by linear shift (069)
  v4 SL   as v4 FA with spontaneous licks as the unrewarded-lick reference
Populations: all mice; learners (current rule: day-0 sessions of non-learner mice removed, their expert sessions kept;
applied to every iteration so they are comparable).
Measures (whole brain, session = unit): lambda (population similarity of whisker hits to auditory hits),
lambda_LDA, ROC transfer (fraction of reward-lick neurons also WH vs reference, same sign), decoder transfer
(uncorrected normalised transfer, v1-v4) and chance-corrected probability numerator (v4 only); plus the pseudo-
population (current runs). Statistics: Mann-Whitney U (R+ learning vs expert; expert R+ vs R-), learning x cohort
interaction by mouse-level cohort permutation (062 group_stats, 10,000 permutations).
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/recap/ (recap.csv, recap.md, recap.png)
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m62 = importlib.import_module("062_pub_convergence_figures")
m61 = importlib.import_module("061_roc_prelick_learners")
R = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
P0 = R / "ssl-prelick-convergence" / "across_days" / "fa"
OUT = P0 / "recap"
ITER = [("v1", P0 / "_archive_v1_minfr1Hz_allunits", "FA"), ("v2", P0 / "_archive_v2_minunits10_axisthr", "FA"),
        ("v3", P0 / "_archive_v3_minunits5_noaxisthr", "FA"), ("v4 FA", P0, "FA"), ("v4 SL", R / "ssl-prelick-convergence" / "across_days" / "sl", "SL")]
MEAS = [("lambda", "lambda/lambda_sessions.csv", "lam"), ("lambda_LDA", "lambda_lda/lambda_lda_sessions.csv", "lam_lda"),
        ("ROC transfer", "transfer/transfer_sessions.csv", "frac_transfer"),
        ("decoder transfer (raw)", "transfer/transfer_sessions.csv", "transfer_n"),
        ("decoder, chance-corrected", "decoder_transfer_shift/sessions.csv", "num_prob_corrected")]


def table(path, col):
    f = path
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if "variant" in d:
        d = d[d.variant == "all"]
    if "level" in d:
        d = d[d.level == "all"]
    return d if col in d and d[col].notna().any() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for it, base, ref in ITER:
        for mname, rel, col in MEAS:
            src = base / rel
            d = table(src, col)
            if d is None:
                continue
            for pop, D in [("all mice", d), ("learners", m61.learner_filter(d))]:
                S = m62.group_stats(D, col, f"{it} {mname} {pop}", rng, "recap")
                rows.append(dict(iteration=it, reference=ref, measure=mname, population=pop,
                                 **{f"mean {m62.GLAB[k]}": S["means"][k] for k in m62.GROUPS},
                                 **{f"n {m62.GLAB[k]}": S["n"][k] for k in m62.GROUPS},
                                 p_Rplus_change=S.get("R+ L vs E", (np.nan,))[0],
                                 p_expert_Rplus_vs_Rminus=S.get("expert R+ vs R-", (np.nan,))[0],
                                 interaction=S["interaction"][0], p_interaction=S["interaction"][1]))
    # pseudo-population (current runs only; M = 2000, chance-corrected readouts)
    for ref, base in [("FA", P0), ("SL", R / "ssl-prelick-convergence" / "across_days" / "sl")]:
        for pop, folder in [("all mice", "all"), ("learners", "learners")]:
            f = base / "pseudopop" / folder / "pseudopop_stats.csv"
            if not f.exists():
                continue
            T = pd.read_csv(f)
            for ro, lab in [("transfer", "pseudo-population decoder transfer"), ("lambda", "pseudo-population lambda")]:
                t = T[(T.region == "all") & (T.M == 2000) & (T.readout == ro)]
                if len(t):
                    t = t.iloc[0]
                    rows.append(dict(iteration=f"v4 {ref}", reference=ref, measure=lab, population=pop,
                                     **{f"mean {m62.GLAB[k]}": t[f"mean {k[0]} {k[1]}"] for k in m62.GROUPS},
                                     p_Rplus_change=t.get("dR+ p_boot", np.nan), p_expert_Rplus_vs_Rminus=np.nan,
                                     interaction=t["interaction"], p_interaction=t["perm_p_interaction"]))
    T = pd.DataFrame(rows); T.to_csv(OUT / "recap.csv", index=False)
    fmt = lambda p: "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}")
    lines = ["| iteration | reference | measure | population | R+ L → E | R− L → E | p R+ change | p expert R+ vs R− | p interaction |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in T.to_dict("records"):
        lines.append(f"| {r['iteration']} | {r['reference']} | {r['measure']} | {r['population']} | "
                     f"{r['mean R+ learning']:.2f} → {r['mean R+ expert']:.2f} | "
                     f"{r['mean R− learning']:.2f} → {r['mean R− expert']:.2f} | {fmt(r['p_Rplus_change'])} | "
                     f"{fmt(r['p_expert_Rplus_vs_Rminus'])} | {fmt(r['p_interaction'])} |")
    (OUT / "recap.md").write_text("\n".join(lines), encoding="utf-8")
    # figure: -log10 p of the interaction per measure x iteration, both populations
    plt = m62.setup()
    meas = [m for m, _, _ in MEAS] + ["pseudo-population decoder transfer", "pseudo-population lambda"]
    its = [i for i, _, _ in ITER]
    fig, axs = plt.subplots(1, 3, figsize=(m62.W_IN, 3.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (col, ttl) in zip(axs, [("p_Rplus_change", "R+ learning vs expert"),
                                    ("p_expert_Rplus_vs_Rminus", "Expert R+ vs R−"),
                                    ("p_interaction", "Learning × cohort interaction")]):
        for k, (pop, mk, dx) in enumerate([("all mice", "o", -0.15), ("learners", "s", 0.15)]):
            for i, m in enumerate(meas):
                for j, it in enumerate(its):
                    t = T[(T.measure == m) & (T.iteration == it) & (T.population == pop)]
                    if not len(t) or not np.isfinite(t[col].iloc[0]):
                        continue
                    v = -np.log10(max(t[col].iloc[0], 1e-4))
                    ax.scatter(v, i + dx + (j - 2) * 0.06, marker=mk, s=10,
                               color=["#bdbdbd", "#969696", "#636363", "#00B400", "#2c2cdb"][j],
                               edgecolor="none", label=f"{it} ({pop})" if i == 0 and ax is axs[0] else None)
        ax.axvline(-np.log10(0.05), color="k", lw=0.6, ls=(0, (2, 2)))
        ax.set_title(ttl, fontsize=6); ax.set_xlabel("−log10 p")
    axs[0].set_yticks(range(len(meas)), meas, fontsize=5); axs[0].invert_yaxis()
    axs[0].legend(frameon=False, fontsize=4.2, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.suptitle("Recap of iterations (whole brain; circles: all mice, squares: learners; dashed: p = 0.05)",
                 x=0.02, y=0.995, ha="left", fontsize=6.8, weight="bold")
    fig.subplots_adjust(left=0.27, right=0.98, top=0.7, bottom=0.14)
    m62.save(fig, OUT, "recap"); plt.close(fig)
    pd.set_option("display.width", 250)
    print(T[["iteration", "measure", "population", "p_Rplus_change", "p_expert_Rplus_vs_Rminus", "p_interaction"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
