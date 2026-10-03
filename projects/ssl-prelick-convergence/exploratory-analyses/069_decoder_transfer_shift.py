"""Chance-corrected single-session decoder readouts with a linear-shift null, identical for both unrewarded-lick
references (false alarms, PRELICK_REF=fa; spontaneous licks, PRELICK_REF=sl; user 2026-10-03).

Per session x area (good + mua units, mean raw pre-lick rate >= 0.1 Hz, >= 5 units): events (WH, AH, spontaneous licks
"SL", stored as class "FA") ordered in time. Decoder: L2 logistic regression (C = 0.05, balanced classes) trained on AH vs
SL with stratified k-fold CV (units z-scored on training folds); every WH event (never used for training) is classified
by each fold's decoder (averaged over folds). Readouts:
  bacc            held-out balanced accuracy (AH vs SL)
  transfer_bin    num / den with num = fraction of WH decoded as AH - FPR, den = TPR - FPR (TPR / FPR = held-out AH /
                  SL decoded as AH); 0 = WH classified like SL, 1 = like AH
  transfer_prob   num / den with num = mean P(AH | WH) - mean P(AH | SL_test), den = mean P(AH | AH_test) - mean
                  P(AH | SL_test) (continuous-probability version)
  Chance correction of the transfers is component-wise: num and den each minus the median of its own shift null,
  then the ratio (den_corrected > 0.1 for bin, > 0.02 for prob); a shift-null median of the ratio itself is undefined
  (num and den ~0 under the null).
Linear-shift null (event types and activity are autocorrelated, e.g. lick bouts and engagement): labels kept in time order,
activity shifted by k events (no wrap-around; label i paired with activity i + k on the overlap), decoder refitted and
re-applied for every shift; 40 shifts, |k| from 5 to N / 3. For every readout: null median, corrected = observed - null
median, p = (1 + #null >= observed) / (1 + n_null). Transfers clipped to [-1, 2] before correction.
Statistics (unit = session) as 062 (MWU / Welch; mouse-level permutation interaction); both populations.
Output: <OUTROOT>/decoder_transfer_shift/
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
OUT = m51.OUTROOT / f"decoder_transfer_{NULL}"
UNIT_SET = ("good", "mua")
MIN_UNITS, MIN_EVENTS, K_SHIFT, MIN_SHIFT, C_REG = 5, 4, 40, 5, 0.05
READOUTS = ["bacc", "num_bin", "den_bin", "num_prob", "den_prob"]


def fit_apply(X, lab, seed):
    """X events x units (time order), lab labels; train AH vs FA(SL), apply to WH"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"
    y = (lab[m] == "AH").astype(int)
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 2 or min(y.sum(), (1 - y).sum()) < MIN_EVENTS or w.sum() < 3:
        return None
    Xa, Xw = X[m], X[w]
    pred = np.zeros(len(y)); prob = np.zeros(len(y)); wb, wp = [], []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xa, y):
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
        clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit((Xa[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((Xa[te] - mu) / sd); prob[te] = clf.predict_proba((Xa[te] - mu) / sd)[:, 1]
        wb.append(clf.predict((Xw - mu) / sd).mean()); wp.append(clf.predict_proba((Xw - mu) / sd)[:, 1].mean())
    tpr, fpr = pred[y == 1].mean(), pred[y == 0].mean()
    pa, pf = prob[y == 1].mean(), prob[y == 0].mean()
    return dict(bacc=0.5 * (tpr + 1 - fpr), num_bin=np.mean(wb) - fpr, den_bin=tpr - fpr,
                num_prob=np.mean(wp) - pf, den_prob=pa - pf)


def session_area(X, lab, seed=0):
    obs = fit_apply(X, lab, seed)
    if obs is None:
        return None
    n = len(lab); kmax = n // 3
    if kmax <= MIN_SHIFT:
        return None
    ks = np.unique(np.r_[np.linspace(MIN_SHIFT, kmax, K_SHIFT // 2).astype(int),
                         -np.linspace(MIN_SHIFT, kmax, K_SHIFT // 2).astype(int)])
    null = {r: [] for r in READOUTS}
    if NULL == "perm":                     # FA trials sit in the randomised trial sequence: permute AH / FA labels
        rng = np.random.default_rng(seed)
        m = np.isin(lab, ["AH", "FA"])
        for j in range(K_SHIFT):
            lp = lab.copy(); lp[m] = rng.permutation(lab[m])
            res = fit_apply(X, lp, seed + j + 1000)
            if res:
                for r in READOUTS:
                    null[r].append(res[r])
    for k in (ks if NULL == "shift" else []):
        i = np.arange(max(0, -k), min(n, n - k))
        res = fit_apply(X[i + k], lab[i], seed + int(k) + 1000)
        if res:
            for r in READOUTS:
                null[r].append(res[r])
    out = {}
    for r in READOUTS:
        nl = np.array([v for v in null[r] if np.isfinite(v)])
        o = obs[r]
        out[r] = o
        if len(nl) >= 10 and np.isfinite(o):
            out[f"{r}_null_median"] = float(np.median(nl))
            out[f"{r}_corrected"] = o - float(np.median(nl))
            out[f"{r}_p"] = (1 + np.sum(nl >= o)) / (1 + len(nl))
        out[f"{r}_n_null"] = len(nl)
    # transfers: raw ratio, and component-wise chance correction (numerator and denominator each minus its
    # shift-null median, then ratio; the ratio itself is undefined under the null because both terms are ~0)
    for v in ["bin", "prob"]:
        n_, d_ = out[f"num_{v}"], out[f"den_{v}"]
        out[f"transfer_{v}"] = float(np.clip(n_ / d_, -1, 2)) if d_ > (0.2 if v == "bin" else 0) else np.nan
        if f"num_{v}_corrected" in out and f"den_{v}_corrected" in out:
            dc = out[f"den_{v}_corrected"]
            out[f"transfer_{v}_corrected"] = float(np.clip(out[f"num_{v}_corrected"] / dc, -1, 2)) \
                if dc > (0.1 if v == "bin" else 0.02) else np.nan
    return out


def job(args):
    sid, f, meta, groups = args
    z = np.load(f, allow_pickle=True)
    order = np.argsort(z["trial_start"])
    X, raw, lab = z["rates"].astype(float)[:, order], z["raw"].astype(float)[:, order], z["cls"][order]
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(groups, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    regs = {"all": np.ones(len(K), bool)}
    regs.update({a: (K.area_group == a).to_numpy() for a in K.area_group.dropna().unique()})
    rows = []
    for reg, mreg in regs.items():
        u = ok & mreg
        if u.sum() < MIN_UNITS:
            continue
        res = session_area(X[u].T, lab)
        if res:
            rows.append(dict(meta, level="all" if reg == "all" else "area_group", region=reg, n_units=int(u.sum()), **res))
    return rows


def stats_fig(S):
    m62 = importlib.import_module("062_pub_convergence_figures")
    m61 = importlib.import_module("061_roc_prelick_learners")
    plt = m62.setup(); rng = np.random.default_rng(0)
    RA = "SL" if m51.REF == "sl" else "FA"
    panels = [("bacc_corrected", "Balanced accuracy − chance", f"AH vs {RA} decoding\nabove chance"),
              ("transfer_bin_corrected", f"Transfer − chance\n(0 = like {RA}, 1 = like AH)", "Whisker hits decoded as AH\n(yes/no), chance-corrected"),
              ("transfer_prob_corrected", f"Transfer − chance\n(0 = like {RA}, 1 = like AH)", "Whisker hits decoded as AH\n(probability), chance-corrected"),
              ("num_prob_corrected", f"P(AH|WH) − P(AH|{RA}) − chance", f"Whisker hits more AH-like\nthan {RA}, chance-corrected")]
    st = []
    for pop, D in [("all", S), ("learners", m61.learner_filter(S))]:
        wb = D[D.level == "all"]
        fig, axs = plt.subplots(1, 4, figsize=(m62.W_IN, 2.7), gridspec_kw=dict(wspace=0.75))
        for ax, (col, yl, ttl) in zip(axs, panels):
            S_ = m62.mean_panel(ax, wb, col, yl, rng, f"{pop} {col}", ttl, ref=[(0, "0.5")])
            st.append(dict(population=pop, readout=col, **{f"mean {m62.GLAB[k]}": S_["means"][k] for k in m62.GROUPS},
                           **{f"n {m62.GLAB[k]}": S_["n"][k] for k in m62.GROUPS},
                           **{f"{nm} p_MWU": S_[nm][0] for nm in ["R+ L vs E", "R- L vs E", "expert R+ vs R-"] if nm in S_},
                           interaction_p=S_["interaction"][1]))
        fig.suptitle(f"Single-session decoders, {'spontaneous-lick' if m51.REF == 'sl' else 'false-alarm'} reference, {'linear-shift' if NULL == 'shift' else 'label-permutation'} chance correction "
                     f"({'all mice' if pop == 'all' else 'learners'}; mean ± s.e.m. over sessions)",
                     x=0.08, y=0.995, ha="left", fontsize=6.8, weight="bold")
        fig.subplots_adjust(left=0.08, right=0.97, top=0.66, bottom=0.18)
        m62.letter_row(fig, list(axs), "abcd", dy_in=0.5)         # after the layout is final
        m62.save(fig, OUT, f"decoder_transfer_{NULL}_{pop}"); plt.close(fig)
    T = pd.DataFrame(st); T.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(4).to_string(index=False))


def main(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.stats_only:
        W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
        W = W[W.cohort.isin(["R+", "R-"])]
        W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
        st26 = importlib.import_module("026_roc_rates_all_sessions")
        ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
        jobs = []
        for r in ss.itertuples():
            f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
            g = W[W.session_id == r.session_id]
            if f.exists() and not g.empty:
                jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=g.mouse_id.iloc[0],
                                                   cohort=g.cohort.iloc[0], stage=g.stage.iloc[0]),
                             g[["electrode_group", "cluster_id", "quality_label", "area_group"]]))
        rows = []
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            for i, rr in enumerate(pool.imap_unordered(job, jobs)):
                rows += rr
                if i % 20 == 0:
                    print(f"[069] {i + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "sessions.csv", index=False)
        json.dump(dict(script="069_decoder_transfer_shift.py", reference=m51.REF, null="linear shift" if NULL == "shift" else "AH / FA label permutation", n_null=K_SHIFT,
                       min_shift=MIN_SHIFT, max_shift="n_events // 3", C=C_REG, min_units=MIN_UNITS,
                       unit_of_analysis="session", decoder="single session",
                       runtime_min=round((time.time() - t0) / 60, 1)), open(OUT / "provenance.json", "w"), indent=1)
    stats_fig(pd.read_csv(OUT / "sessions.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--stats-only", action="store_true")
    main(ap.parse_args())
