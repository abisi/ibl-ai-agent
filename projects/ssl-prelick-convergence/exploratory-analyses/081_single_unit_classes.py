"""081 -- Single-neuron counterparts of the population distances with all three comparisons, event-count-matched
fractions, and convergence within stimulus-defined neuron classes (user 2026-10-07).

1. Three comparisons, cross-validated. Per unit, ROC selectivity s = 2 AUC - 1 (Eq. 2) for WH vs SL (WS), AH vs SL (AS) and
   WH vs AH (WA; > 0: WH > AH), each on random halves a / b of every class's events (N_SPLIT = 50); squared selectivities
   are taken across halves (s^a s^b), so their mean over units is unbiased by the number of events -- the single-neuron
   counterparts of d(WH, SL), d(AH, SL), d(WH, AH) (Eq. 5). Per unit, (WH - SL) - (AH - SL) = WH - AH, so
   d(WS) + d(AS) - d(WA) = 2 a: Delta-d counterpart = C_WS - C_WA, along counterpart = cross product C_X (080).
2. Event-count-matched fractions. Fraction selective depends on the number of events (power), and the number of whisker hits
   differs strongly between cohorts. Per session, N_MATCH = 8 events per class are drawn (N_DRAW = 10 draws); s per unit
   and pair, significance by N_PERM = 200 label permutations (two-sided, p < 0.05); fraction selective per pair averaged
   over draws; Delta-fraction = frac_WS - frac_WA.
3. Neuron classes from the existing stimulus-aligned ROC analyses (_roc_stage_analysis roc_long.parquet, 045):
     sensory      significant in any of whisker_passive_pre, whisker_passive_post, auditory_passive_pre,
                  auditory_passive_post (units of sessions without passive data: class "no passive", left out of 3)
     lick-gated   not sensory, whisker_choice significant (whisker hit vs whisker miss)
     non-sensory  neither
   Within each class: converging fraction among AH > SL neurons (051 pre-lick ROC on all events: WH > SL significant), mean
   AH-likeness c (Eq. 3; class means of the pre-lick rates) of AH > SL neurons, mean per-unit cross product wa_u (080);
   sessions with >= 3 AH > SL neurons in the class.
Units: good / mua (stable run: stable only), mean raw pre-lick rate >= 0.1 Hz; sessions >= 8 events per class, >= 10 units.
Statistics (unit = session): learning vs expert per cohort (MWU), expert R+ vs R-, learning x cohort interaction (Eq. 15;
10,000 mouse-level cohort permutations; 062 dots_panel).
Output: across_days/<ref>[_units]/single_unit_classes/: units.parquet (join keys mouse_id, session_id, electrode_group,
cluster_id), sessions.csv, classes.csv; <pop>/single_unit_classes.{png,pdf,svg}, stats.csv.
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m80 = importlib.import_module("080_single_unit_geometry")
OUT = m51.OUTROOT / "single_unit_classes"
N_SPLIT, N_MATCH, N_DRAW, N_PERM, MIN_EV, MIN_UNITS, MIN_CLASS = 50, 8, 10, 200, 8, 10, 3
ROC_LONG = m51.RES / "_roc_stage_analysis" / "roc_long.parquet"
PASSIVE = ["whisker_passive_pre", "whisker_passive_post", "auditory_passive_pre", "auditory_passive_post"]
CLASSES = ["sensory", "lick-gated", "non-sensory"]
PAIRS = {"WS": ("WH", "FA"), "AS": ("AH", "FA"), "WA": ("WH", "AH")}
SEL_WS, SEL_AS = m80.SEL_WS, m80.SEL_AS


def neuron_classes():
    L = pd.read_parquet(ROC_LONG, columns=["session_id", "electrode_group", "cluster_id", "analysis_type", "sig"])
    L = L[L.analysis_type.isin(PASSIVE + ["whisker_choice"])]
    L["cluster_id"] = L.cluster_id.astype(str); L["electrode_group"] = L.electrode_group.astype(str)
    P = L.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type", values="sig",
                      aggfunc="first").reset_index()
    sig = lambda c: P[c].astype(str).str.lower().isin(["true", "1", "1.0"]) if c in P else pd.Series(False, index=P.index)
    has_passive = P[[c for c in PASSIVE if c in P]].notna().any(axis=1)
    sens = np.logical_or.reduce([sig(c) for c in PASSIVE])
    gated = ~sens & sig("whisker_choice")
    P["nclass"] = np.select([~has_passive, sens, gated], ["no passive", "sensory", "lick-gated"], "non-sensory")
    return P[["session_id", "electrode_group", "cluster_id", "nclass"]]


def session(args):
    f, meta, K, seed = args
    rng = np.random.default_rng(seed)
    z = np.load(f, allow_pickle=True)
    X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
    idx = {c: np.where(lab == c)[0] for c in ("WH", "AH", "FA")}
    if min(len(v) for v in idx.values()) < MIN_EV:
        return None
    ok = (raw.mean(1) >= m51.MIN_FR) & K.quality_label.isin(m80.UNIT_SET).to_numpy() & (X.std(1) > 0)
    if ok.sum() < MIN_UNITS:
        return None
    X, K = X[ok], K[ok].reset_index(drop=True)
    # 1. cross-validated squared selectivities for the three pairs (+ the WS x AS cross product)
    acc = {k: np.zeros(len(X)) for k in ("WS", "AS", "WA", "X")}
    for _ in range(N_SPLIT):
        h = {}
        for c, v in idx.items():
            p = rng.permutation(v); k = len(p) // 2
            h[c] = (p[:k], p[k:])
        s = {(pr, hh): m80.auc_sel(X, h[a][hh], h[b][hh]) for pr, (a, b) in PAIRS.items() for hh in (0, 1)}
        for pr in PAIRS:
            acc[pr] += s[(pr, 0)] * s[(pr, 1)]
        acc["X"] += (s[("WS", 0)] * s[("AS", 1)] + s[("WS", 1)] * s[("AS", 0)]) / 2
    # 2. event-count-matched significance (N_MATCH events per class, label permutations)
    fr = {pr: np.zeros(len(X)) for pr in PAIRS}
    for _ in range(N_DRAW):
        sub = {c: rng.choice(v, N_MATCH, replace=False) for c, v in idx.items()}
        for pr, (a, b) in PAIRS.items():
            ib, ia = sub[a], sub[b]
            obs = m80.auc_sel(X, ib, ia)
            both = np.r_[ib, ia]
            null = np.empty((N_PERM, len(X)))
            for q in range(N_PERM):
                p = rng.permutation(both)
                null[q] = m80.auc_sel(X, p[:N_MATCH], p[N_MATCH:])
            pv = (1 + (np.abs(null) >= np.abs(obs)[None]).sum(0)) / (1 + N_PERM)
            fr[pr] += (pv < 0.05)
    m = {c: X[:, v].mean(1) for c, v in idx.items()}
    den = np.abs(m["AH"] - m["FA"])
    like = np.where(den > 0, (np.abs(m["WH"] - m["FA"]) - np.abs(m["WH"] - m["AH"])) / np.where(den > 0, den, 1), np.nan)
    U = K.assign(**meta, **{f"cv_{k}": v / N_SPLIT for k, v in acc.items()},
                 **{f"msig_{pr}": v / N_DRAW for pr, v in fr.items()}, like=like,
                 n_WH=len(idx["WH"]), n_AH=len(idx["AH"]), n_SL=len(idx["FA"]))
    return U


def compute(n_proc):
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    W = W.merge(neuron_classes(), on=["session_id", "electrode_group", "cluster_id"], how="left")
    W["nclass"] = W.nclass.fillna("no passive")
    cols = ["mouse_id", "electrode_group", "cluster_id", "quality_label", "area_group", "nclass",
            f"sel:{SEL_WS}", f"sel:{SEL_AS}", f"sig:{SEL_WS}", f"sig:{SEL_AS}"]
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        Ws = W[W.session_id == r.session_id]
        K = K.merge(Ws[cols].drop_duplicates(["electrode_group", "cluster_id"]), on=["electrode_group", "cluster_id"], how="left")
        K["mouse_id"] = Ws.mouse_id.iloc[0]
        jobs.append((f, dict(session_id=r.session_id, cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0]), K, 1000 + i))
    out = []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for res in pool.imap_unordered(session, jobs):
            if res is not None:
                out.append(res)
    return pd.concat(out, ignore_index=True)


def summarise(U):
    sig = lambda c: U[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U = U.assign(ahsl_pos=sig(f"sig:{SEL_AS}") & (U[f"sel:{SEL_AS}"] > 0),
                 whsl_pos=sig(f"sig:{SEL_WS}") & (U[f"sel:{SEL_WS}"] > 0))
    rows, crow = [], []
    for sid, d in U.groupby("session_id"):
        meta = d.iloc[0][["session_id", "mouse_id", "cohort", "stage"]].to_dict()
        r = dict(meta, n_units=len(d))
        for k in ("WS", "AS", "WA", "X"):
            r[f"C_{k}"] = d[f"cv_{k}"].mean()
        r["dd_counterpart"] = r["C_WS"] - r["C_WA"]
        for pr in PAIRS:
            r[f"frac_{pr}"] = d[f"msig_{pr}"].mean()
        r["dfrac"] = r["frac_WS"] - r["frac_WA"]
        for c in CLASSES:
            q = d[d.nclass == c]
            r[f"n_{c}"] = len(q)
            r[f"prop_{c}"] = len(q) / len(d)
            rl = q[q.ahsl_pos]
            ok = len(rl) >= MIN_CLASS
            r[f"conv_{c}"] = rl.whsl_pos.mean() if ok else np.nan
            r[f"like_{c}"] = rl.like.mean() if ok else np.nan
            r[f"wa_{c}"] = q.cv_X.mean() if len(q) >= MIN_CLASS else np.nan
            crow.append(dict(meta, nclass=c, n=len(q), n_ahsl_pos=len(rl), conv=r[f"conv_{c}"], like=r[f"like_{c}"],
                             wa=r[f"wa_{c}"]))
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(crow)


def figure(plt, S, out, pop):
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(m62.W_IN, 8.4))
    gs = fig.add_gridspec(4, 4, hspace=0.95, wspace=0.6, left=0.08, right=0.98, top=0.93, bottom=0.05)
    rows = [
        [("C_WS", "WH vs SL: mean cv s²\n(d(WH, SL) counterpart)"), ("C_AS", "AH vs SL: mean cv s²\n(d(AH, SL) counterpart)"),
         ("C_WA", "WH vs AH: mean cv s²\n(d(WH, AH) counterpart)"), ("dd_counterpart", "C_WS − C_WA\n(Δd counterpart)")],
        [("frac_WS", f"WH vs SL selective\n({N_MATCH} events / class)"), ("frac_AS", f"AH vs SL selective\n({N_MATCH} events / class)"),
         ("frac_WA", f"WH vs AH selective\n({N_MATCH} events / class)"), ("dfrac", "frac WS − frac WA")],
        [(f"conv_{c}", f"Converging fraction:\n{c} neurons") for c in CLASSES] + [("prop_non-sensory", "Proportion of\nnon-sensory neurons")],
        [(f"like_{c}", f"AH-likeness:\n{c} neurons") for c in CLASSES] + [("wa_non-sensory", "Per-unit cross product:\nnon-sensory")],
    ]
    ylab = ["Cross-validated, per unit", "Fraction (matched counts)", "Among AH > SL neurons", "AH-likeness c (Eq. 3)"]
    axs = []
    for i, rr in enumerate(rows):
        ra = []
        for j, (col, ttl) in enumerate(rr):
            ax = fig.add_subplot(gs[i, j]); ra.append(ax); ax.set_box_aspect(1)
            m62.dots_panel(ax, S, col, ylab[i] if j == 0 else "", rng, f"081 {col}", ttl,
                           ref=[(0, "0.6")] if col in ("dd_counterpart", "dfrac") or col.startswith("like") else [])
            ax.set_title(ax.get_title(), fontsize=4.9)
        axs.append(ra)
    for ra, letters in zip(axs, ("abcd", "efgh", "ijkl", "mnop")):
        m62.letter_row(fig, ra, letters)
    fig.suptitle(f"Single neurons: three comparisons, matched-count fractions, convergence by stimulus class ({pop})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold")
    m62.save(fig, out, "single_unit_classes"); plt.close(fig)


def main(a):
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "units.parquet"
    if a.replot and f.exists():
        U0 = pd.read_parquet(f)
    else:
        U0 = compute(a.n_proc); U0.to_parquet(f, index=False)
    print(U0.nclass.value_counts().to_dict(), flush=True)
    S0, C0 = summarise(U0)
    S0.to_csv(OUT / "sessions.csv", index=False); C0.to_csv(OUT / "classes.csv", index=False)
    for pop in ("learners", "all"):
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        S = flt(S0)
        out = OUT / pop; out.mkdir(exist_ok=True)
        m62.STATS.clear()
        figure(plt, S, out, pop)
        T = pd.DataFrame(m62.STATS); T.to_csv(out / "stats.csv", index=False)
        cols = [c for c in T.columns if c in ("measure", "interaction_p_perm") or c.startswith("mean_") or c.startswith("n_")]
        with pd.option_context("display.width", 250):
            print(pop); print(T[cols].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
