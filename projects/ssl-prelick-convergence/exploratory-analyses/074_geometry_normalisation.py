"""074 -- Does the unit normalisation change the population-geometry results? (user 2026-10-07)
Same events, units and estimator as 057 (cross-validated split-half distances and lambda, 50 splits, whole brain, variant
"all", good + mua units with mean raw pre-lick rate >= 0.1 Hz, >= 5 units, >= 4 events per class), with three
scalings of each unit's per-event pre-lick rates (baseline-subtracted, as 051) before the class means are built:
  raw      spikes/s, centred only (high-rate / high-variance units weigh more)
  pooled   z-score over all used events pooled (current; the SD contains the class differences and is dominated by the
           most frequent class)
  within   centred, divided by the pooled within-class SD with equal weight per class, sqrt(mean_c var_c) (diagonal
           noise normalisation, as the diagonal cross-validated Mahalanobis / "crossnobis" distance)
Per session: d(WH,SL), d(WH,AH), d(AH,SL), delta d = d(WH,SL) - d(WH,AH), lambda (clipped to [-1, 2]; sessions with
d(AH,SL) <= 0 excluded; no axis-length threshold, as it is scale-dependent). lambda_LDA is not recomputed: LDA whitens
by the within-class covariance and is invariant to per-unit scaling (up to its shrinkage).
Statistics as the report (062 group_stats: MWU / Welch learning vs expert per cohort, cohort contrast per stage,
learning x cohort interaction by mouse-level cohort permutation); populations all mice and learners.
Output: combined_results_ks4/ssl-prelick-convergence/across_days/<ref>/normalisation/: sessions.csv, stats.csv,
  geometry_normalisation.{png,pdf,svg}
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m57 = importlib.import_module("057_roc_prelick_lambda")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
OUT = m51.OUTROOT / "normalisation"
NORMS = ["raw", "pooled", "within"]
NLAB = {"raw": "raw (spikes/s)", "pooled": "pooled SD (current)", "within": "within-class SD"}
METRICS = [("dd", "Δd = d(WH,SL) − d(WH,AH)"), ("lam", "λ"), ("d_WH_FA", "d(WH, SL)"), ("d_WH_AH", "d(WH, AH)"),
           ("d_AH_FA", "d(AH, SL)")]


def scale(X, lab, kind):
    """X (units, events) -> scaled copy; units with zero scale return None in the mask"""
    mu = X.mean(1, keepdims=True)
    if kind == "raw":
        return X - mu, np.ones(len(X), bool)
    if kind == "pooled":
        sd = X.std(1)
    else:
        sd = np.sqrt(np.mean([X[:, lab == c].var(1) for c in m51.CLASSES], axis=0))
    ok = sd > 0
    return (X - mu) / np.where(ok, sd, 1)[:, None], ok


def compute():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    rows = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        info = W[W.session_id == r.session_id][["electrode_group", "cluster_id", "cohort", "stage", "mouse_id", "quality_label"]]
        K = K.merge(info, on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        meta = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                    stage=K.stage.dropna().iloc[0])
        X, raw, lab, rt = z["rates"].astype(float), z["raw"].astype(float), z["cls"], z["rt"]
        idx = m51.trial_set(lab, rt, m51.CLASSES, "all", np.random.default_rng(0))
        if len(idx) < 12:
            continue
        l = lab[idx]
        ok0 = (raw[:, idx].mean(1) >= m51.MIN_FR) & K.quality_label.isin(m57.UNIT_SET).to_numpy()
        for kind in NORMS:
            Z, ok = scale(X[:, idx], l, kind)
            u = ok0 & ok
            if u.sum() < m57.MIN_UNITS:
                continue
            res = m57.lam(Z[u], l, np.random.default_rng(0))            # same splits for every scaling
            if res:
                lam_ = res["lam"] if res["d_AH_FA"] > 0 else np.nan
                rows.append(dict(meta, norm=kind, n_units=int(u.sum()), d_WH_FA=res["d_WH_FA"], d_WH_AH=res["d_WH_AH"],
                                 d_AH_FA=res["d_AH_FA"], dd=res["d_WH_FA"] - res["d_WH_AH"],
                                 lam=np.clip(lam_, *m57.LAM_CLIP) if np.isfinite(lam_) else np.nan))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "sessions.csv"
    D = pd.read_csv(f) if (f.exists() and "--recompute" not in sys.argv) else compute()
    D.to_csv(f, index=False)
    rows = []
    plt = m62.setup()
    pops = {"all": D, "learners": m61.learner_filter(D)}
    for pop, P in pops.items():
        fig = plt.figure(figsize=(m62.W_IN, 1.75 * len(METRICS) + 0.8))
        gs = fig.add_gridspec(len(METRICS), len(NORMS) + 1, wspace=0.45, hspace=0.95, left=0.08, right=0.98, top=0.94, bottom=0.04)
        for i, (col, lab) in enumerate(METRICS):
            for j, kind in enumerate(NORMS):
                rng = np.random.default_rng(0)
                d = P[P.norm == kind]
                ax = fig.add_subplot(gs[i, j])
                S = m62.dots_panel(ax, d, col, lab if j == 0 else "", rng, f"{col}|{kind}", f"{NLAB[kind]}")
                r = dict(population=pop, metric=col, norm=kind, interaction=S["interaction"][0], p_interaction=S["interaction"][1],
                         **{f"mean {m62.GLAB[k]}": S["means"][k] for k in m62.GROUPS}, **{f"n {m62.GLAB[k]}": S["n"][k] for k in m62.GROUPS})
                for name in ["R+ L vs E", "R- L vs E", "expert R+ vs R-", "learning R+ vs R-"]:
                    if name in S:
                        r[f"{name} p_MWU"], r[f"{name} diff"] = S[name][0], S[name][2]
                rows.append(r)
            # agreement of the session values across scalings
            ax = fig.add_subplot(gs[i, len(NORMS)])
            pv = P.pivot_table(index="session_id", columns="norm", values=col)
            for kind, c in (("raw", "0.55"), ("within", "k")):
                if {"pooled", kind} <= set(pv.columns):
                    q = pv[["pooled", kind]].dropna()
                    rho = q.corr(method="spearman").iloc[0, 1]
                    zs = (q - q.mean()) / q.std()                       # z-scored across sessions: scales differ
                    ax.scatter(zs["pooled"], zs[kind], s=5, color=c, lw=0, alpha=0.7, label=f"{kind}: ρ = {rho:.2f}")
                    rows.append(dict(population=pop, metric=col, norm=f"spearman pooled vs {kind}", interaction=rho))
            ax.plot([-3, 3], [-3, 3], color="0.7", lw=0.5, ls="--")
            ax.set_xlabel("pooled SD (current), z across sessions", fontsize=5); ax.set_ylabel("other scaling, z across sessions", fontsize=5)
            ax.legend(frameon=False, fontsize=4.4, loc="upper left"); ax.set_title("Agreement (sessions)", loc="left", fontsize=5.6)
        fig.suptitle(f"Population geometry under three unit scalings ({m51.REF.upper()} reference, {pop}; whole brain; raw: spikes²/s² "
                     "per unit, others: SD² per unit)", x=0.02, y=0.99, ha="left", va="top", fontsize=7, weight="bold")
        m62.save(fig, OUT, f"geometry_normalisation_{pop}")
        plt.close(fig)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "stats.csv", index=False)
    cols = ["population", "metric", "norm", "interaction", "p_interaction", "R+ L vs E p_MWU", "R- L vs E p_MWU", "expert R+ vs R- p_MWU"]
    print(T[[c for c in cols if c in T]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
