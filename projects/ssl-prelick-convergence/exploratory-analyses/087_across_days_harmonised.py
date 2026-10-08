"""087 -- Across days with the within-day metric definitions (user 2026-10-08: "port our metric standardisation (definition, unit
norm, ...) and the methods used here to the across-days analysis, for both coding directions").
Reward-lick CD, per session (learning day and expert), same events, units and axis as within_day 026: per-event baseline-corrected
pre-lick rates (051, [-100, 0) ms), stable units with mean pre-lick rate >= 0.1 Hz, >= 4 WH, >= 8 AH, >= 8 SL; units z-scored over
the AH + SL events (label-free); 5-fold CV over AH + SL (026 folds and seeds): u = unit(mean AH - mean SL) of the training folds, the
held-out AH and SL events and every WH event projected on it (WH averaged over folds). All projections per neuron (/ sqrt(N)).
  p_WH, p_SL, p_AH        class means of the projections
  d_WS = p_WH - p_SL      native position of whisker hits relative to spontaneous licks on the CD (per neuron)
  d_AS = p_AH - p_SL      held-out separation of the CD
  lam  = d_WS / d_AS      position between SL (0) and AH (1); reported only where d_AS > 0
  rho_v                   |mean WH - mean SL| per neuron, cross-validated (product of the differences of random halves, 50 splits)
  theta = arccos(d_WS / rho_v)   angle between the WH - SL difference and the CD (degrees; undefined where rho_v = 0)
Levels: whole brain and area groups (>= 5 units and held-out AH-vs-SL d' >= 0.3, as 015 / 026).
Statistics (learners rule, m61.learner_filter; unit = session): per cohort day 0 vs expert (Mann-Whitney U and Welch t), cohort at each
stage (MWU, Welch), cohort x stage interaction [E - L](R+) - [E - L](R-) with cohort labels permuted across mice (10,000; a mouse keeps
its sessions). No shifted-label control: across days the comparison is between sessions (different neurons), not a drift within a
session; within-session slopes at both stages are in 026.
Output: across_days/<ref>/harmonised/{sessions_areas.csv, tests.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../087_across_days_harmonised.py [--n-proc 40]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m51 = importlib.import_module("051_roc_prelick")
m002 = importlib.import_module("002_trial_slopes")
OUT = m51.OUTROOT / "harmonised"
N_HALF, N_PERM = 50, 10000


def metrics(X, lab, seed):
    """X units x events (baseline-corrected pre-lick rates); returns the per-neuron metrics or None"""
    from sklearn.model_selection import StratifiedKFold
    y_idx = np.where(np.isin(lab, ["AH", "FA"]))[0]; y = (lab[y_idx] == "AH").astype(int); w_idx = np.where(lab == "WH")[0]
    A = X[:, y_idx].T; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
    Z = ((X.T - mu) / sd)                                              # events x units, z over AH + SL (label-free)
    N = Z.shape[1]; sN = np.sqrt(N)
    hold = np.full(len(y_idx), np.nan); pw = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(y_idx, y):
        d = Z[y_idx[tr][y[tr] == 1]].mean(0) - Z[y_idx[tr][y[tr] == 0]].mean(0)
        u = d / max(np.linalg.norm(d), 1e-12)
        hold[te] = Z[y_idx[te]] @ u; pw.append(Z[w_idx] @ u)
    pAH, pSL, pWH = hold[y == 1].mean() / sN, hold[y == 0].mean() / sN, np.mean(pw, 0).mean() / sN
    a, b = hold[y == 1], hold[y == 0]; spd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    rng = np.random.default_rng(seed); ww, ss = Z[w_idx], Z[y_idx[y == 0]]; prod = []
    for _ in range(N_HALF):
        pi, pj = rng.permutation(len(ww)), rng.permutation(len(ss))
        vA = ww[pi[: len(ww) // 2]].mean(0) - ss[pj[: len(ss) // 2]].mean(0)
        vB = ww[pi[len(ww) // 2:]].mean(0) - ss[pj[len(ss) // 2:]].mean(0)
        prod.append(vA @ vB / N)
    rho = np.sqrt(max(np.mean(prod), 0)); dWS = pWH - pSL; dAS = pAH - pSL
    cos_ = dWS / rho if rho > 0 else np.nan
    return dict(n_units=N, p_WH=pWH, p_SL=pSL, p_AH=pAH, d_WS=dWS, d_AS=dAS, dprime=(a.mean() - b.mean()) / spd if spd > 0 else np.nan,
                lam=dWS / dAS if dAS > 0 else np.nan, rho_v=rho,
                theta=np.degrees(np.arccos(np.clip(cos_, -1, 1))) if np.isfinite(cos_) else np.nan)


def session(args):
    sid, f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o]
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return []
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    rows = []
    groups = [("whole_brain", ok)] + [(g, ok & (K.area_group == g).to_numpy()) for g in sorted(K.area_group.dropna().unique())]
    for area, m in groups:
        if m.sum() < m002.MIN_UNITS:
            continue
        r = metrics(X[m], lab, seed)
        if area != "whole_brain" and not (np.isfinite(r["dprime"]) and r["dprime"] >= m002.MIN_DPRIME):
            continue
        rows.append(dict(meta, area=area, n_WH=int((lab == "WH").sum()), n_AH=int((lab == "AH").sum()), n_SL=int((lab == "FA").sum()), **r))
    return rows


def two(a, b):
    a, b = (np.asarray(v, float) for v in (a, b)); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def interaction(D, q):
    d = D.dropna(subset=[q]).sort_values("session_id")
    rng = np.random.default_rng(zlib.crc32(f"087|{q}".encode()))
    mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy(); idx = pd.Index(mice).get_indexer(d.mouse_id)
    st, v = d.stage.to_numpy(), d[q].to_numpy(float)

    def stat(coh):
        f = lambda c, s_: v[(coh == c) & (st == s_)].mean()
        return (f("R+", "expert") - f("R+", "learning")) - (f("R-", "expert") - f("R-", "learning"))
    obs = stat(d.cohort.to_numpy())
    null = np.array([stat(rng.permutation(mc)[idx]) for _ in range(N_PERM)])
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + N_PERM)


def tests(D):
    rows = []
    for q in ("d_WS", "lam", "theta", "rho_v", "d_AS", "p_WH"):
        for c in ("R+", "R-"):
            L, E = (D[(D.cohort == c) & (D.stage == s_)][q] for s_ in ("learning", "expert"))
            pm, pt = two(E, L)
            rows.append(dict(measure=q, comparison=f"{c} expert vs learning", mean_learning=L.mean(), sem_learning=L.sem(), n_learning=L.notna().sum(),
                             mean_expert=E.mean(), sem_expert=E.sem(), n_expert=E.notna().sum(), p_mwu=pm, p_welch=pt))
        for s_ in ("learning", "expert"):
            P, M = (D[(D.cohort == c) & (D.stage == s_)][q] for c in ("R+", "R-"))
            pm, pt = two(P, M)
            rows.append(dict(measure=q, comparison=f"R+ vs R- at {s_}", mean_Rp=P.mean(), mean_Rm=M.mean(), p_mwu=pm, p_welch=pt))
        obs, pi = interaction(D, q)
        rows.append(dict(measure=q, comparison="cohort x stage (mouse permutation)", diff=obs, p_perm=pi))
    return pd.DataFrame(rows)


def main(a):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    m61 = importlib.import_module("061_roc_prelick_learners")
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):                         # same enumeration (and seeds) as 026
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0]),
                         Ws[["electrode_group", "cluster_id", "quality_label", "area_group"]], i))
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        rows = [x for r in pool.imap_unordered(session, jobs) for x in r]
    DA = pd.DataFrame(rows); DA.to_csv(OUT / "sessions_areas.csv", index=False)
    D = m61.learner_filter(DA[DA.area == "whole_brain"])
    T = tests(D); T.to_csv(OUT / "tests.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(4).to_string())
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    main(ap.parse_args())
