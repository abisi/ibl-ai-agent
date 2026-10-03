"""Within-session remapping: do whisker hits move relative to auditory hits and spontaneous licks between the first and
the second half of a session, and does this differ between cohorts on the learning day (hypothesis: R- mice remap
within day 0, away from auditory hits toward spontaneous licks; R+ mice do not yet) and not within expert sessions?

Events (051, spontaneous-lick reference, PRELICK_REF=sl): WH, AH, SL (stored as class "FA"), 100 ms before the corrected
first lick; trials active, perf != 6, warm-up cut, A1 end-of-session trim (the session end is trimmed). Units: good + mua,
mean raw pre-lick rate >= 0.1 Hz over all events, whole brain.

Split ("time"): events ordered by time; the split point is the midpoint between the two middle auditory hits, so both
halves contain the same number of auditory hits; WH and SL events are assigned to the half they fall in.
Null split ("oddeven"): events alternately assigned to the two halves in time order within each class (no temporal
structure); a within-session change should not appear with this split.
Count matching: per class, both halves are subsampled to the smaller count (>= MIN_EVENTS of WH, AH and SL per half);
N_SUB repeated subsamples, results averaged over subsamples.

Measures per half (units z-scored over the subsampled events of both halves together, so halves share one scale):
  geometry   cross-validated squared distances per unit (057: 50 random half-splits of each class), distance difference
             dd = d(WH, SL) - d(WH, AH) (> 0: WH closer to AH), lambda (projection of WH on the SL -> AH axis)
  decoder, whole session ("ws"): L2 logistic regression (C = 0.05, balanced) trained on AH vs SL of the whole session
             with stratified 5-fold CV; held-out AH / SL predictions and fold-averaged WH predictions are then evaluated
             separately per half (count-matched): balanced accuracy (AH vs SL) and the probability numerator
             num = mean P(AH | WH) - mean P(AH | SL). One decoder for the session: a change between halves means WH
             (and AH / SL) move on a fixed axis.
  decoder, cross-half ("xh"): trained on the AH vs SL events of one half (count-matched subsample) and applied to the
             AH, SL and WH events of the other half (early -> late gives the late-half values, late -> early the
             early-half values). The training data never overlap with the evaluated events, and each half is read out
             with the axis of the other half: a change means WH moves relative to an axis learnt at another time.
  chance     linear-shift null (40 shifts, |k| in [5, n/3], no wrap-around): ws = activity of the whole session shifted
             against its time-ordered labels, decoder refitted, per-half readouts recomputed; xh = training half shifted,
             test half intact. Reported: observed - null median (bacc_c, num_c).
Controls per half: d(AH, SL) (stability of the reward-lick axis), event rates (WH, AH, SL per minute), median RT of WH and
AH, mean raw pre-lick rate per class, half duration.
lambda clipped to [-1, 2] as in 057. Statistic: per session the change late - early (Δ). Learning day: R+ vs R- (MWU / Welch; mouse-level cohort permutation);
expert: Δ vs 0 per cohort (Wilcoxon signed-rank and one-sample t); cohort x stage x half = learning x cohort interaction
on Δ ([ΔE - ΔL](R+) - [ΔE - ΔL](R-), mouse-level cohort permutation). Populations: all mice and learners.
Output: combined_results_ks4/_within_day<TAG>/halves/ (project ssl-within-day-remapping; was 072 in ssl-prelick-convergence)
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
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"   # data loaders and pre-lick events (051, 057)
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m57 = importlib.import_module("057_roc_prelick_lambda")
OUT = m51.RES / f"_within_day{m51.TAG}" / "halves"
UNIT_SET = ("good", "mua")
MIN_UNITS, MIN_EVENTS, N_SUB, K_SHIFT, MIN_SHIFT, C_REG = 5, 4, 20, 40, 5, 0.05
SPLITS = ["time", "oddeven"]
GEOM = ["dd", "lam", "d_WH_FA", "d_WH_AH", "d_AH_FA"]
DEC = ["ws_bacc", "ws_num", "xh_bacc", "xh_num"]


# ------------------------------------------------------------------ splits and subsampling
def split_halves(lab, t, kind):
    """boolean masks (early, late) over time-ordered events"""
    if kind == "time":
        ta = np.sort(t[lab == "AH"])
        if len(ta) < 2 * MIN_EVENTS:
            return None
        h = len(ta) // 2
        cut = 0.5 * (ta[h - 1] + ta[h])
        return t < cut, t >= cut
    early = np.zeros(len(lab), bool)
    for c in m51.CLASSES:
        i = np.where(lab == c)[0]
        early[i[::2]] = True
    return early, ~early


def matched(lab, e, l, rng):
    """count-matched subsample: per class the same number of events in both halves; returns (idx_early, idx_late)"""
    ie, il = [], []
    for c in m51.CLASSES:
        a, b = np.where(e & (lab == c))[0], np.where(l & (lab == c))[0]
        n = min(len(a), len(b))
        if n < MIN_EVENTS:
            return None
        ie.append(rng.choice(a, n, replace=False)); il.append(rng.choice(b, n, replace=False))
    return np.sort(np.concatenate(ie)), np.sort(np.concatenate(il))


# ------------------------------------------------------------------ decoders
def _clf():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000)


def ws_predict(X, lab, seed):
    """whole-session CV decoder: held-out P(AH) for AH / SL events, fold-averaged P(AH) for WH events (nan otherwise)"""
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"
    y = (lab[m] == "AH").astype(int)
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 2:
        return None
    P = np.full(len(lab), np.nan); Pw = np.zeros(w.sum()); Xa, Xw = X[m], X[w]
    ia = np.where(m)[0]
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xa, y):
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
        c = _clf().fit((Xa[tr] - mu) / sd, y[tr])
        P[ia[te]] = c.predict_proba((Xa[te] - mu) / sd)[:, 1]
        Pw += c.predict_proba((Xw - mu) / sd)[:, 1] / k
    P[w] = Pw
    return P


def readout(P, lab, idx):
    """balanced accuracy (AH vs SL) and probability numerator on the events idx"""
    l, p = lab[idx], P[idx]
    pa, pf, pw = p[l == "AH"], p[l == "FA"], p[l == "WH"]
    if min(len(pa), len(pf), len(pw)) == 0:
        return np.nan, np.nan
    return 0.5 * ((pa > 0.5).mean() + (pf <= 0.5).mean()), pw.mean() - pf.mean()


def xh_predict(Xtr, ltr, Xte):
    """train AH vs SL on one half, P(AH) for all events of the other half"""
    m = np.isin(ltr, ["AH", "FA"]); y = (ltr[m] == "AH").astype(int)
    if y.sum() < 2 or (1 - y).sum() < 2:
        return None
    mu, sd = Xtr[m].mean(0), Xtr[m].std(0); sd[sd == 0] = 1
    return _clf().fit((Xtr[m] - mu) / sd, y).predict_proba((Xte - mu) / sd)[:, 1]


def shifts(n):
    kmax = n // 3
    if kmax <= MIN_SHIFT:
        return []
    return np.unique(np.r_[np.linspace(MIN_SHIFT, kmax, K_SHIFT // 2).astype(int),
                           -np.linspace(MIN_SHIFT, kmax, K_SHIFT // 2).astype(int)])


def shifted(X, lab, k):
    """label i paired with activity i + k (no wrap-around): returns (X', lab', original indices of the labels)"""
    n = len(lab); i = np.arange(max(0, -k), min(n, n - k))
    return X[i + k], lab[i], i


# ------------------------------------------------------------------ one session
def session(args):
    sid, f, meta, W_s, seed = args
    rng = np.random.default_rng(seed)
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, rt, t = (z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o],
                          z["rt"].astype(float)[o], z["trial_start"].astype(float)[o])
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    if ok.sum() < MIN_UNITS:
        return []
    Xu = X[ok].T                                     # events x units
    rows = []
    for kind in SPLITS:
        sp = split_halves(lab, t, kind)
        if sp is None:
            continue
        e, l = sp
        subs = [matched(lab, e, l, rng) for _ in range(N_SUB)]
        subs = [s_ for s_ in subs if s_ is not None]
        if not subs:
            continue
        acc = {h: {k_: [] for k_ in GEOM} for h in ("early", "late")}
        for ie, il in subs:
            both = np.r_[ie, il]
            Z = Xu[both]; mu, sd = Z.mean(0), Z.std(0); good = sd > 0
            for h, ii in (("early", ie), ("late", il)):
                Zh = ((Xu[ii][:, good] - mu[good]) / sd[good]).T          # units x events
                r = m57.lam(Zh, lab[ii], rng)
                if r:
                    acc[h]["dd"].append(r["d_WH_FA"] - r["d_WH_AH"]); acc[h]["lam"].append(r["lam"])
                    for k_ in ("d_WH_FA", "d_WH_AH", "d_AH_FA"):
                        acc[h][k_].append(r[k_])
        row = dict(meta, split=kind, n_units=int(ok.sum()), n_sub=len(subs),
                   **{f"n_{c}_half": int(np.sum(lab[subs[0][0]] == c)) for c in m51.CLASSES})
        for h in ("early", "late"):
            for k_ in GEOM:
                v = np.array(acc[h][k_], float)
                row[f"{k_}_{h}"] = np.nanmean(v) if len(v) else np.nan
        # whole-session decoder: observed and shift null, evaluated on count-matched halves
        P = ws_predict(Xu, lab, seed)
        if P is not None:
            obs = {h: np.array([readout(P, lab, s_[j]) for s_ in subs]).mean(0) for j, h in enumerate(("early", "late"))}
            null = {h: [] for h in ("early", "late")}
            for k in shifts(len(lab)):
                Xs, ls, i = shifted(Xu, lab, int(k))
                Ps = ws_predict(Xs, ls, seed + int(k) + 1000)
                if Ps is None:
                    continue
                Pf = np.full(len(lab), np.nan); Pf[i] = Ps       # back on the original event index (labels = lab[i])
                for j, h in enumerate(("early", "late")):
                    vals = [readout(Pf, lab, s_[j][np.isin(s_[j], i)]) for s_ in subs[:5]]
                    null[h].append(np.nanmean(vals, 0))
            for h in ("early", "late"):
                nl = np.array(null[h], float)
                row[f"ws_bacc_{h}"], row[f"ws_num_{h}"] = obs[h]
                if len(nl) >= 10:
                    med = np.nanmedian(nl, 0)
                    row[f"ws_bacc_c_{h}"], row[f"ws_num_c_{h}"] = obs[h] - med
        # cross-half decoder: train on one count-matched half, read out the other; null = training half shifted
        xh = {h: [] for h in ("early", "late")}; xh0 = {h: [] for h in ("early", "late")}
        for s_ in subs[:5]:
            for tr_i, te_i, h in ((s_[0], s_[1], "late"), (s_[1], s_[0], "early")):
                p = xh_predict(Xu[tr_i], lab[tr_i], Xu[te_i])
                if p is None:
                    continue
                Pf = np.full(len(lab), np.nan); Pf[te_i] = p
                xh[h].append(readout(Pf, lab, te_i))
                nl = []
                for k in shifts(len(tr_i)):
                    Xs, ls, _ = shifted(Xu[tr_i], lab[tr_i], int(k))
                    p0 = xh_predict(Xs, ls, Xu[te_i])
                    if p0 is not None:
                        P0 = np.full(len(lab), np.nan); P0[te_i] = p0
                        nl.append(readout(P0, lab, te_i))
                if len(nl) >= 10:
                    xh0[h].append(np.nanmedian(np.array(nl, float), 0))
        for h in ("early", "late"):
            if xh[h]:
                row[f"xh_bacc_{h}"], row[f"xh_num_{h}"] = np.nanmean(xh[h], 0)
            if xh0[h]:
                row[f"xh_bacc_c_{h}"], row[f"xh_num_c_{h}"] = np.nanmean(xh[h], 0) - np.nanmean(xh0[h], 0)
        # behavioural / stability controls (all events of each half, not subsampled)
        for h, msk in (("early", e), ("late", l)):
            dur = (t[msk].max() - t[msk].min()) / 60 if msk.sum() > 1 else np.nan
            row[f"dur_min_{h}"] = dur
            for c in m51.CLASSES:
                mc = msk & (lab == c)
                row[f"rate_{c}_{h}"] = mc.sum() / dur if dur and dur > 0 else np.nan
                row[f"fr_{c}_{h}"] = raw[ok][:, mc].mean() if mc.any() else np.nan
            for c in ("WH", "AH"):
                v = rt[msk & (lab == c)]; v = v[np.isfinite(v)]
                row[f"rt_{c}_{h}"] = np.median(v) * 1e3 if len(v) else np.nan
        rows.append(row)
    return rows


# ------------------------------------------------------------------ statistics
def perm_interaction(d, col, n_perm=10000, seed=0, kind="stage"):
    """mouse-level cohort permutation: kind='stage' -> [mean E - mean L](R+) - [..](R-); kind='learning' -> R+ - R- on day 0"""
    rng = np.random.default_rng(seed)
    d = d.dropna(subset=[col])
    mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    mi = pd.Index(mice).get_indexer(d.mouse_id); st = d.stage.to_numpy(); v = d[col].to_numpy()

    def stat(coh):
        g = lambda c, s: v[(coh == c) & (st == s)].mean() if ((coh == c) & (st == s)).any() else np.nan
        if kind == "learning":
            return g("R+", "learning") - g("R-", "learning")
        return (g("R+", "expert") - g("R+", "learning")) - (g("R-", "expert") - g("R-", "learning"))
    obs = stat(d.cohort.to_numpy())
    null = np.array([stat(rng.permutation(mc)[mi]) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))


def tests(D):
    rows = []
    for split in SPLITS:
        for meas in GEOM + ["ws_bacc_c", "ws_num_c", "xh_bacc_c", "xh_num_c", "ws_bacc", "ws_num", "xh_bacc", "xh_num",
                            "rate_WH", "rate_AH", "rate_FA", "rt_WH", "rt_AH", "fr_WH", "fr_AH", "fr_FA"]:
            d = D[D.split == split].copy()
            if f"{meas}_early" not in d:
                continue
            d["delta"] = d[f"{meas}_late"] - d[f"{meas}_early"]
            r = dict(split=split, measure=meas)
            for c in ("R+", "R-"):
                for s_ in ("learning", "expert"):
                    v = d[(d.cohort == c) & (d.stage == s_)].delta.dropna()
                    g = f"{c} {s_}"
                    r[f"n {g}"] = len(v); r[f"mean Δ {g}"] = v.mean(); r[f"sem Δ {g}"] = v.sem()
                    r[f"early {g}"] = d[(d.cohort == c) & (d.stage == s_)][f"{meas}_early"].mean()
                    r[f"late {g}"] = d[(d.cohort == c) & (d.stage == s_)][f"{meas}_late"].mean()
                    if len(v) >= 3:
                        r[f"p_wilcoxon {g}"] = stats.wilcoxon(v).pvalue if (v != 0).any() else np.nan
                        r[f"p_t {g}"] = stats.ttest_1samp(v, 0).pvalue
            a = d[(d.cohort == "R+") & (d.stage == "learning")].delta.dropna()
            b = d[(d.cohort == "R-") & (d.stage == "learning")].delta.dropna()
            if len(a) >= 3 and len(b) >= 3:
                r["day0 R+ vs R- p_MWU"] = stats.mannwhitneyu(a, b).pvalue
                r["day0 R+ vs R- p_Welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
                r["day0 R+ - R-"], r["day0 p_perm"] = perm_interaction(d, "delta", kind="learning")
            r["cohort x stage x half"], r["cxsxh p_perm"] = perm_interaction(d, "delta", kind="stage")
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def figures(D, T, out, pop):
    m62 = importlib.import_module("062_pub_convergence_figures")
    plt = m62.setup()
    COH, GROUPS, GLAB = m62.COH, m62.GROUPS, m62.GLAB
    XS = {GROUPS[0]: 0, GROUPS[1]: 1, GROUPS[2]: 2.5, GROUPS[3]: 3.5}
    MEAS = [("dd", "Δd = d(WH,SL) − d(WH,AH)", "Distance difference"),
            ("lam", "λ (0 = SL, 1 = AH)", "λ"),
            ("ws_num_c", "P(AH|WH) − P(AH|SL) − chance", "Decoder, whole session"),
            ("xh_num_c", "P(AH|WH) − P(AH|SL) − chance", "Decoder, cross-half"),
            ("ws_bacc_c", "Balanced accuracy − chance", "Accuracy, whole session"),
            ("xh_bacc_c", "Balanced accuracy − chance", "Accuracy, cross-half")]
    for split in SPLITS:
        d = D[D.split == split]
        fig, axs = plt.subplots(2, len(MEAS), figsize=(m62.W_IN, 4.3), gridspec_kw=dict(hspace=0.9, wspace=0.75))
        for j, (col, yl, ttl) in enumerate(MEAS):
            ax = axs[0, j]
            for k in GROUPS:
                g = d[(d.cohort == k[0]) & (d.stage == k[1])]
                x0 = XS[k]
                for _, r in g.iterrows():
                    ax.plot([x0 - 0.18, x0 + 0.18], [r[f"{col}_early"], r[f"{col}_late"]], color=COH[k[0]], lw=0.35, alpha=0.35)
                for dx, h in ((-0.18, "early"), (0.18, "late")):
                    v = g[f"{col}_{h}"].dropna()
                    if len(v):
                        ax.errorbar(x0 + dx, v.mean(), v.sem(), fmt="o", ms=3.2, color=COH[k[0]],
                                    mfc="white" if h == "early" else COH[k[0]], mew=0.8, lw=0.9, capsize=0, zorder=3)
            ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
            ax.set_xticks(list(XS.values()), ["L", "E", "L", "E"])
            for x_, c in ((0.5, "R+"), (3.0, "R-")):
                ax.annotate(c.replace("-", "−"), xy=(x_, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -11),
                            textcoords="offset points", ha="center", va="top", fontsize=6, color=COH[c], weight="bold")
            ax.set_ylabel(yl, fontsize=5.2); ax.set_title(f"{ttl}\nearly (open) → late (filled)", fontsize=5.3)
            ax = axs[1, j]
            dd_ = d.assign(delta=d[f"{col}_late"] - d[f"{col}_early"])
            for k in GROUPS:
                v = dd_[(dd_.cohort == k[0]) & (dd_.stage == k[1])].delta.dropna()
                ax.scatter(XS[k] + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=5,
                           facecolor="white" if k[1] == "learning" else COH[k[0]], edgecolor=COH[k[0]], lw=0.5)
                if len(v) > 1:
                    ax.errorbar(XS[k] + 0.25, v.mean(), v.sem(), fmt="o", ms=3.5, color=COH[k[0]],
                                mfc="white" if k[1] == "learning" else COH[k[0]], lw=0.9, capsize=0)
            ax.axhline(0, color="0.3", lw=0.5)
            ax.set_xticks(list(XS.values()), ["L", "E", "L", "E"])
            r = T[(T.split == split) & (T.measure == col)]
            if len(r):
                r = r.iloc[0]
                ax.set_title(f"Δ late − early\nday 0 R+ vs R− {m62.fmt_p(r.get('day0 p_perm', np.nan))}\n"
                             f"× stage {m62.fmt_p(r['cxsxh p_perm'])}", fontsize=4.9)
            ax.set_ylabel("Δ (late − early)", fontsize=5.2)
        m62.letter_row(fig, list(axs[0]), "abcdef"); m62.letter_row(fig, list(axs[1]), "ghijkl")
        fig.suptitle(f"Within-session change ({'split at the median auditory hit' if split == 'time' else 'odd / even null split'}; "
                     f"count-matched halves; reference: spontaneous licks; {pop})", x=0.06, y=0.995, ha="left", fontsize=7,
                     weight="bold")
        fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.08)
        m62.save(fig, out, f"within_session_{split}"); plt.close(fig)
    # controls
    d = D[D.split == "time"]
    CTRL = [("d_AH_FA", "d(AH, SL) per unit", "Reward-lick axis length"), ("rate_WH", "Events / min", "WH rate"),
            ("rate_AH", "Events / min", "AH rate"), ("rate_FA", "Events / min", "SL rate"), ("rt_WH", "ms", "WH RT"),
            ("rt_AH", "ms", "AH RT"), ("fr_WH", "Hz", "WH raw pre-lick rate"), ("fr_FA", "Hz", "SL raw pre-lick rate")]
    fig, axs = plt.subplots(2, 4, figsize=(m62.W_IN, 4.0), gridspec_kw=dict(hspace=0.75, wspace=0.6))
    for ax, (col, yl, ttl) in zip(axs.ravel(), CTRL):
        for k in GROUPS:
            g = d[(d.cohort == k[0]) & (d.stage == k[1])]
            for dx, h in ((-0.18, "early"), (0.18, "late")):
                v = g[f"{col}_{h}"].dropna()
                if len(v):
                    ax.errorbar(XS[k] + dx, v.mean(), v.sem(), fmt="o", ms=3.2, color=COH[k[0]],
                                mfc="white" if h == "early" else COH[k[0]], mew=0.8, lw=0.9, capsize=0)
        ax.set_xticks(list(XS.values()), ["L", "E", "L", "E"]); ax.set_ylabel(yl, fontsize=5.2)
        r = T[(T.split == "time") & (T.measure == col)]
        pp = f"\nday 0 R+ vs R− Δ {m62.fmt_p(r.iloc[0].get('day0 p_perm', np.nan))}" if len(r) else ""
        ax.set_title(ttl + pp, fontsize=5.2)
    fig.suptitle(f"Controls per half (early open, late filled; mean ± s.e.m. over sessions; {pop})", x=0.06, y=0.995,
                 ha="left", fontsize=7, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.08)
    m62.save(fig, out, "within_session_controls"); plt.close(fig)


# ------------------------------------------------------------------ main
def main(a):
    t0 = time.time()
    m61 = importlib.import_module("061_roc_prelick_learners")
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        Ws = W[W.session_id == r.session_id]
        meta = dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0])
        jobs.append((r.session_id, f, meta, Ws[["electrode_group", "cluster_id", "quality_label"]], 1000 + i))
    if a.test:
        jobs = jobs[:a.test]
    if a.replot:
        D = pd.read_csv(OUT / "within_session_sessions.csv")
    else:
        rows = []
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            for k, res in enumerate(pool.imap_unordered(session, jobs)):
                rows += res
                if k % 10 == 0:
                    print(f"[072] {k + 1}/{len(jobs)} sessions ({(time.time() - t0) / 60:.1f} min)", flush=True)
        D = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        D.to_csv(OUT / "within_session_sessions.csv", index=False)
    for h in ("early", "late"):                       # lambda clipped as in 057 (a ratio; per-half axes can be short)
        D[f"lam_{h}"] = D[f"lam_{h}"].clip(*m57.LAM_CLIP)
    for pop in ["all", "learners"]:
        Dp = m61.learner_filter(D) if pop == "learners" else D
        out = OUT / pop; out.mkdir(exist_ok=True)
        T = tests(Dp); T.to_csv(out / "within_session_tests.csv", index=False)
        figures(Dp, T, out, pop)
    json.dump(dict(script="001_within_session_halves.py", ref=m51.REF, n_sub=N_SUB, k_shift=K_SHIFT, min_events=MIN_EVENTS,
                   splits=SPLITS, unit_set=UNIT_SET, min_fr=m51.MIN_FR, C=C_REG, n_sessions=int(D.session_id.nunique()),
                   runtime_min=round((time.time() - t0) / 60, 1)), open(OUT / "provenance.json", "w"), indent=1)
    print("ALL DONE", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=60)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--replot", action="store_true", help="recompute tests and figures from the saved session table")
    main(ap.parse_args())
