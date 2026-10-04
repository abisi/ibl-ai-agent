"""Decoder accuracy above chance with a linear-shift null (pre-lick window; auditory hit vs unrewarded-lick reference).

Events of a session (WH, AH and the reference class "FA" = false alarms or, with PRELICK_REF=sl, spontaneous licks)
are ordered in time (051 per-event table). Decoded classes: AH vs reference. Trial types and motivation are
autocorrelated, so an i.i.d. label shuffle underestimates chance; instead the label sequence is kept fixed and the
neural pre-lick vectors are shifted by k events (no wrap-around): label of event i is paired with the activity of event
i + k on the overlapping events only. The decoder is refitted for every shift.
Decoder: L2 logistic regression (C = 0.05, balanced class weights), stratified k-fold (k = min(5, smallest class)),
units z-scored on the training folds; balanced accuracy on held-out events. Units: good + mua, mean raw pre-lick rate
>= 0.1 Hz over all events, >= 5 units per session x area.
Shifts: K_SHIFT shifts, |k| from MIN_SHIFT to N_events // 3, half positive, half negative.
Per session x area: bacc (k = 0), null median and 95th percentile, accuracy above chance = bacc - null median,
p = (1 + #null >= bacc) / (1 + n_shifts). Statistics (unit = session): Mann-Whitney / Welch, learning x cohort
interaction by mouse-level permutation (062 group_stats); areas: >= 3 sessions at both stages of a cohort.
The same linear-shift null is used for both references (false alarms and spontaneous licks; user 2026-10-03), so
their results are comparable; label permutation is used only for the single-neuron ROC (051).
Output: <OUTROOT>/decoder_shift/: sessions.csv, stats.csv, figures
"""
import argparse
import importlib
import json
import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
NULL = "shift"   # decoders: linear-shift null for both references (user 2026-10-03; label permutation reserved for ROC)
OUT = m51.OUTROOT / f"decoder_{NULL}"
UNIT_SET = ("good", "mua")
MIN_UNITS = 5
MIN_EVENTS = 4            # per decoded class on the overlap
K_SHIFT = 40
MIN_SHIFT = 5
C_REG = 0.05
DATA = {}


def cv_bacc(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 2:
        return np.nan
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((X[te] - mu) / sd)
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def decode_shift(Xall, lab, seed=0):
    """Xall (events x units) in time order; lab event labels. Returns bacc, null array"""
    n = len(lab)
    m0 = np.isin(lab, ["AH", "FA"])
    y0 = (lab[m0] == "AH").astype(int)
    if min(y0.sum(), (1 - y0).sum()) < MIN_EVENTS:
        return np.nan, np.array([])
    obs = cv_bacc(Xall[m0], y0, seed)
    if NULL == "perm":                                            # FA: trials randomised -> label permutation null
        rng = np.random.default_rng(seed)
        return obs, np.array([cv_bacc(Xall[m0], rng.permutation(y0), seed + j + 1) for j in range(K_SHIFT)])
    kmax = n // 3
    if kmax <= MIN_SHIFT:
        return obs, np.array([])
    ks = np.unique(np.r_[np.linspace(MIN_SHIFT, kmax, K_SHIFT // 2).astype(int),
                         -np.linspace(MIN_SHIFT, kmax, K_SHIFT // 2).astype(int)])
    null = []
    for k in ks:
        i = np.arange(max(0, -k), min(n, n - k))                 # label index; activity index i + k
        li = lab[i]; m = np.isin(li, ["AH", "FA"]); y = (li[m] == "AH").astype(int)
        if min(y.sum(), (1 - y).sum()) < MIN_EVENTS:
            continue
        null.append(cv_bacc(Xall[i + k][m], y, seed + int(k) + 1000))
    return obs, np.array(null)


def job(args):
    sid, f, meta, groups = args
    z = np.load(f, allow_pickle=True)
    order = np.argsort(z["trial_start"])
    X, raw, lab = z["rates"].astype(float)[:, order], z["raw"].astype(float)[:, order], z["cls"][order]
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(groups, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    rows = []
    regs = {"all": np.ones(len(K), bool)}
    regs.update({a: (K.area_group == a).to_numpy() for a in K.area_group.dropna().unique()})
    for reg, mreg in regs.items():
        u = ok & mreg
        if u.sum() < MIN_UNITS:
            continue
        obs, null = decode_shift(X[u].T, lab)
        if not np.isfinite(obs) or len(null) < 10:
            continue
        rows.append(dict(meta, level="all" if reg == "all" else "area_group", region=reg, n_units=int(u.sum()),
                         n_AH=int((lab == "AH").sum()), n_ref=int((lab == "FA").sum()), bacc=obs,
                         null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
                         above_chance=obs - float(np.median(null)),
                         p_shift=(1 + np.sum(null >= obs)) / (1 + len(null)), n_shifts=len(null)))
    return rows


def main(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        g = W[W.session_id == r.session_id]
        if not f.exists() or g.empty:
            continue
        meta = dict(session_id=r.session_id, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0])
        jobs.append((r.session_id, f, meta, g[["electrode_group", "cluster_id", "quality_label", "area_group"]]))
    rows = []
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        for i, rr in enumerate(pool.imap_unordered(job, jobs)):
            rows += rr
            if i % 20 == 0:
                print(f"[068] {i + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    S = pd.DataFrame(rows); S.to_csv(OUT / "sessions.csv", index=False)
    figure_and_stats(S)
    json.dump(dict(script="068_decoder_linear_shift.py", reference=m51.REF, null=NULL, n_null=K_SHIFT, min_shift=MIN_SHIFT,
                   max_shift="n_events // 3", min_units=MIN_UNITS, min_events_per_class=MIN_EVENTS, C=C_REG,
                   unit_set=UNIT_SET, min_fr=m51.MIN_FR, unit_of_analysis="session",
                   runtime_min=round((time.time() - t0) / 60, 1)), open(OUT / "provenance.json", "w"), indent=1)


def figure_and_stats(S):
    import ephys_utilities.allen_utils.allen_utils as au
    m62 = importlib.import_module("062_pub_convergence_figures")
    m61 = importlib.import_module("061_roc_prelick_learners")
    plt = m62.setup(); rng = np.random.default_rng(0)
    RN = "false alarms" if m51.REF == "fa" else "spontaneous licks"
    NLAB = "linear-shift" if NULL == "shift" else "label-permutation"
    st = []
    for pop, D in [("all", S), ("learners", m61.learner_filter(S))]:
        wb = D[D.level == "all"]
        fig = plt.figure(figsize=(m62.W_IN, 4.6))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.95, wspace=0.6, left=0.08, right=0.98, top=0.88,
                              bottom=0.1)
        ax = fig.add_subplot(gs[0, 0])
        for k_ in m62.GROUPS:
            d = wb[(wb.cohort == k_[0]) & (wb.stage == k_[1])]
            ax.scatter(d.null_median, d.bacc, s=7, color=m62.COH[k_[0]], alpha=0.8, lw=0.5,
                       facecolor="white" if k_[1] == "learning" else m62.COH[k_[0]], edgecolor=m62.COH[k_[0]])
        ax.plot([0.3, 1], [0.3, 1], color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xlabel(f"Chance level (median of\n{NLAB} null)"); ax.set_ylabel("Balanced accuracy (observed)")
        ax.set_title(f"Sessions: {int((wb.p_shift < 0.05).sum())}/{len(wb)} above\nchance (p < 0.05)", fontsize=5.8)
        axs = [ax]
        for j, (col, lab, ttl) in enumerate([("above_chance", "Balanced accuracy − chance", "Accuracy above chance"),
                                             ("null_median", f"Chance level ({NLAB} null)", "Chance level"),
                                             ("bacc", "Balanced accuracy", "Raw balanced accuracy")]):
            ax = fig.add_subplot(gs[0, 1 + j]); axs.append(ax)
            S_ = m62.mean_panel(ax, wb, col, lab, rng, f"shift {pop} {col}", f"{ttl}\n(AH vs {RN})",
                                ref=[(0, "0.5")] if col == "above_chance" else None)
            st.append(dict(population=pop, measure=col, level="all", region="all",
                           **{f"mean {m62.GLAB[k]}": S_["means"][k] for k in m62.GROUPS},
                           **{f"n {m62.GLAB[k]}": S_["n"][k] for k in m62.GROUPS},
                           **{f"{nm} p_MWU": S_[nm][0] for nm in ["R+ L vs E", "R- L vs E", "expert R+ vs R-"] if nm in S_},
                           interaction_p=S_["interaction"][1]))
        ax = fig.add_subplot(gs[1, :]); axs.append(ax)
        A = D[D.level == "area_group"]
        ok = {}
        for c in ["R+", "R-"]:
            n = A[A.cohort == c].groupby(["region", "stage"]).size().unstack().fillna(0)
            ok[c] = {g for g in n.index if n.loc[g].get("learning", 0) >= 3 and n.loc[g].get("expert", 0) >= 3}
        order = [g for g in au.get_area_group_custom_order() if g in ok["R+"] | ok["R-"]]
        from scipy import stats as sst
        for k, c in enumerate(["R+", "R-"]):
            for i, g in enumerate(order):
                if g not in ok[c]:
                    continue
                d = A[(A.region == g) & (A.cohort == c)]
                for s_, dx in [("learning", -0.3), ("expert", -0.1)]:
                    v = d[d.stage == s_].above_chance
                    x = i + dx + 0.4 * k
                    ax.errorbar(x, v.mean(), v.sem(), fmt="o", ms=3.2, color=m62.COH[c], lw=0.8, capsize=0,
                                mfc="white" if s_ == "learning" else m62.COH[c])
                a_, b_ = d[d.stage == "learning"].above_chance, d[d.stage == "expert"].above_chance
                p = sst.mannwhitneyu(a_, b_).pvalue
                st.append(dict(population=pop, measure="above_chance", level="area_group", region=g, cohort=c,
                               mean_learning=a_.mean(), mean_expert=b_.mean(), n_learning=len(a_), n_expert=len(b_),
                               p_MWU=p))
                if p < 0.05:
                    ax.text(i - 0.2 + 0.4 * k, ax.get_ylim()[1] if False else max(a_.mean(), b_.mean()) + 0.06, "*",
                            ha="center", color=m62.COH[c], fontsize=8)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xticks(range(len(order)), [g.replace(" areas", "").replace("Somatosensory-", "SS-") for g in order],
                      rotation=30, ha="right")
        ax.set_ylabel("Balanced accuracy − chance\n(mean ± s.e.m. over sessions)")
        ax.set_title("Per area group: open = learning, filled = expert; left R+, right R−; * learning vs expert p < 0.05 "
                     "(≥ 3 sessions per stage)", fontsize=5.8, loc="left")
        m62.letter_row(fig, axs[:4], "abcd"); m62.letter_row(fig, axs[4:], "e")
        fig.suptitle(f"Decoding auditory hits vs {RN} before the lick: accuracy above {NLAB} chance "
                     f"({'all mice' if pop == 'all' else 'learners'})", x=0.08, y=0.985, ha="left", fontsize=7, weight="bold")
        m62.save(fig, OUT, f"decoder_{NULL}_{pop}"); plt.close(fig)
    pd.DataFrame(st).to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250)
    T = pd.DataFrame(st); print(T[T.level == "all"].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    if a.stats_only:
        figure_and_stats(pd.read_csv(OUT / "sessions.csv"))
    else:
        main(a)
