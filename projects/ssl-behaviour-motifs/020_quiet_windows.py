"""020 -- behaviour in quiet windows (no licking): R+ vs R- and early vs late in the session (haas, full run).

Usage: BM_SET=full python 020_quiet_windows.py compute   (per session -> cache/full/020/<sid>.parquet)
       BM_SET=full python 020_quiet_windows.py stats     (tables/020_*.csv, figures/020_*.png|pdf|svg)

Quiet frames: farther than Q s from any lick event, where lick events = piezo licks + frames with the tongue out
(tongue likelihood > 0.6; catches licks that miss the spout). Q = 2 s (main), 0.5 s and 5 s (robustness). Only the task
period is used (first kept trial - 2 s to last kept trial + 6.5 s).
Epochs (bm_lib.trial_windows, user 2026-11-20): pre (2 s), resp, post (response end -> +5 s), iti (+5 s -> next
quiet window), other (rest of the task period).
Session thirds: early = before the start of kept trial n/3, late = from kept trial 2n/3 on.
Per session x Q x epoch (all / pre / resp / iti / other) x third (all / early / late), on quiet frames:
- usage (fraction of quiet frames) of every state of every state set: lightning-action classes (still / whisking / jaw /
  background), kpms meta-states (side, top), kpms syllables (side, top), AR-HMM states (side, top);
- transition rate per set (state changes per s between consecutive quiet frames);
- median bout duration of lightning-action still and whisking bouts that lie fully inside quiet frames;
- mean whisking amplitude, jaw opening, nose speed (side, top).
Stats (unit = session = mouse; Q = 2 s, epoch all, third all unless stated):
- R+ vs R-: Mann-Whitney + Welch per measure, uncorrected (exploratory) + Benjamini-Hochberg q within each state set;
- learning: late - early per session; within cohort sign-flip (10000), R+ vs R- MW + Welch;
- cohort decoding from each set's usage vector: L2 logistic regression, leave-one-session-out, balanced accuracy,
  label-permutation null (500).
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bm_lib as L  # noqa: E402

_sp = importlib.util.spec_from_file_location("m18", HERE / "018_report_examples.py")
M18 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(M18)

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
OUT = L.CACHE / "full" / "020"
QS = (0.5, 2.0, 5.0)
QMAIN = 2.0
EPN = {0: "other", 1: "pre", 2: "resp", 3: "post", 4: "iti"}
LA = M18.LA
FEAT_IDX = {"whisking_amp": 1, "jaw_open": 2, "nose_speed": 4, "top_nose_speed": 5}
rng = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------ per session
def one(sid):
    S = M18.load_session(sid)
    t, Fm, Z, k = S["t"], S["feats"], S["Z"], S["trials"]
    ep = M18.epoch_codes(t, k, sid)
    tong = np.isfinite(Fm[:, 3]) & (Fm[:, 3] > 0.5)
    ev = np.sort(np.r_[S["licks"], t[tong]])
    j = np.clip(np.searchsorted(ev, t), 1, len(ev) - 1)
    dl = np.minimum(np.abs(t - ev[j - 1]), np.abs(ev[j] - t))
    starts = k.start_time.to_numpy()
    task = (t >= starts[0] - 2) & (t <= starts[-1] + 6.5)
    n = len(starts)
    third = np.zeros(len(t), np.int8)
    third[t < starts[n // 3]] = 1
    third[t >= starts[(2 * n) // 3]] = 3
    rows = []
    for Q in QS:
        quiet = task & (dl > Q)
        for epn in ("all", "pre", "resp", "post", "iti", "other"):
            em = quiet if epn == "all" else quiet & (ep == {v: c for c, v in EPN.items()}[epn])
            for th, tm in (("all", em), ("early", em & (third == 1)), ("late", em & (third == 3))):
                base = dict(session_id=sid, Q=Q, epoch=epn, third=th)
                rows.append({**base, "set": "n", "state": "quiet_s", "value": tm.sum() / L.FS_DS})
                if tm.sum() < 200:  # < 1 s of quiet frames
                    continue
                for name, z in Z.items():
                    zz = z[tm]
                    v = zz >= 0
                    if v.sum() < 200:
                        continue
                    ns = M18.nstates(name, z)
                    u = np.bincount(zz[v], minlength=ns)[:ns] / v.sum()
                    rows += [{**base, "set": name, "state": int(s), "value": float(u[s])} for s in range(ns)]
                    a, b = z[:-1], z[1:]
                    both = tm[:-1] & tm[1:] & (a >= 0) & (b >= 0)
                    rows.append({**base, "set": name, "state": "trans_per_s",
                                 "value": float(np.sum(both & (a != b)) / (both.sum() / L.FS_DS))})
                if "la" in Z:
                    z = Z["la"]
                    on, ln, st = M18.bouts(z)
                    inside = np.array([tm[o:o + l_].all() for o, l_ in zip(on, ln)]) if len(on) else np.zeros(0, bool)
                    for c in (1, 2):
                        d = ln[(st == c) & inside] / L.FS_DS
                        rows.append({**base, "set": "la_bout", "state": LA[c], "value": float(np.median(d)) if len(d) >= 5 else np.nan})
                for fn, fi in FEAT_IDX.items():
                    rows.append({**base, "set": "feature", "state": fn, "value": float(np.nanmean(Fm[tm, fi]))})
    D = pd.DataFrame(rows)
    D["state"] = D.state.astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / f"{sid}.parquet")
    return sid


def compute():
    sids = list(M18.sessions().session_id)
    with ProcessPoolExecutor(int(os.environ.get("BM_WORKERS", 12))) as ex:
        for s in ex.map(one, sids):
            print("done", s, flush=True)


# ------------------------------------------------------------------------------------------------ stats
def two(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3 or (np.std(a) == 0 and np.std(b) == 0):
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def signflip(x, nflip=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    s = rng.choice([-1, 1], size=(nflip, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= abs(x.mean())) + 1) / (nflip + 1))


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pp = p[ok]
        o = np.argsort(pp)
        r = pp[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty(ok.sum())
        qq[o] = np.minimum(r, 1)
        q[ok] = qq
    return q


def label(setname, state, mn):
    if setname == "la" and state.isdigit():
        return f"action {LA[int(state)]}"
    if setname.startswith("meta") and state.isdigit():
        return f"{setname.replace('meta_', 'meta ')} {mn[int(state)] if int(state) < len(mn) else state}"
    if state.isdigit():
        return f"{setname.replace('_', ' ')} {'S' if setname.startswith('kpms') else 'A'}{state}"
    return f"{setname} {state}"


def decode(U, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    def acc(yy):
        pred = np.empty(len(yy))
        for i in range(len(yy)):
            tr = np.arange(len(yy)) != i
            sc = StandardScaler().fit(U[tr])
            m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(U[tr]), yy[tr])
            pred[i] = m.predict(sc.transform(U[i:i + 1]))[0]
        return balanced_accuracy_score(yy, pred)
    obs = acc(y)
    null = np.array([acc(rng.permutation(y)) for _ in range(int(os.environ.get("BM_NPERM", 500)))])
    return obs, null


def run_stats():
    S = M18.sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    D = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("*.parquet"))], ignore_index=True)
    D = D[D.session_id.isin(coh)]
    D["cohort"] = D.session_id.map(coh)
    mn = list(pd.read_csv(T / "full_009_metastates.csv").name)
    # usage in shared states only for kpms / AR-HMM (others flagged); keep all for la / meta
    Pp = pd.read_csv(T / "full_009_fingerprints_pooled.csv")
    shared = {(f"{r.method}_{r.view}", str(r.state)) for r in Pp[Pp.shared].itertuples()}
    keep = ~(D.set.str.match(r"^(kpms|arhmm)_") & D.state.str.isdigit()) | \
        pd.Series([(a, b) in shared for a, b in zip(D.set, D.state)], index=D.index)
    D = D[keep & ~((D.set == "la") & (D.state == "0"))]
    # ---- R+ vs R-
    res = []
    for (Q, epn, setn, st), g in D[D.third == "all"].groupby(["Q", "epoch", "set", "state"]):
        a, b = g[g.cohort == "R+"].value, g[g.cohort == "R-"].value
        pm, pw = two(a, b)
        sd = np.sqrt((a.var() + b.var()) / 2)
        res.append(dict(Q=Q, epoch=epn, set=setn, state=st, label=label(setn, st, mn), n_Rplus=a.notna().sum(),
                        n_Rminus=b.notna().sum(), mean_Rplus=a.mean(), mean_Rminus=b.mean(),
                        cohen_d=(a.mean() - b.mean()) / sd if sd > 0 else np.nan, p_mannwhitney=pm, p_welch=pw))
    C = pd.DataFrame(res)
    C["p_max"] = C[["p_mannwhitney", "p_welch"]].max(axis=1)
    C["q_bh_within_set"] = C.groupby(["Q", "epoch", "set"]).p_max.transform(bh)
    C.to_csv(T / "020_quiet_cohort.csv", index=False)
    # ---- learning: late - early
    W = D[D.third.isin(["early", "late"])].pivot_table(index=["session_id", "cohort", "Q", "epoch", "set", "state"],
                                                       columns="third", values="value").reset_index()
    W["delta"] = W.late - W.early
    lr = []
    for (Q, epn, setn, st), g in W.groupby(["Q", "epoch", "set", "state"]):
        a, b = g[g.cohort == "R+"].delta, g[g.cohort == "R-"].delta
        pm, pw = two(a, b)
        lr.append(dict(Q=Q, epoch=epn, set=setn, state=st, label=label(setn, st, mn), delta_Rplus=a.mean(), delta_Rminus=b.mean(),
                       p_signflip_Rplus=signflip(a) if setn in ("la", "meta_side", "meta_top", "feature", "la_bout") or st == "trans_per_s" else np.nan,
                       p_signflip_Rminus=signflip(b) if setn in ("la", "meta_side", "meta_top", "feature", "la_bout") or st == "trans_per_s" else np.nan,
                       p_mannwhitney=pm, p_welch=pw))
    Lr = pd.DataFrame(lr)
    Lr.to_csv(T / "020_quiet_learning.csv", index=False)
    # ---- decoding
    dec = []
    U0 = D[(D.Q == QMAIN) & (D.epoch == "all") & (D.third == "all") & D.state.str.isdigit()]
    for setn in ("la", "meta_side", "meta_top", "kpms_side", "kpms_top", "arhmm_side", "arhmm_top", "all states"):
        g = U0 if setn == "all states" else U0[U0.set == setn]
        M = g.pivot_table(index="session_id", columns=["set", "state"], values="value").dropna(axis=1, thresh=int(0.9 * g.session_id.nunique()))
        M = M.fillna(M.mean())
        y = np.array([coh[s] == "R+" for s in M.index]).astype(int)
        obs, null = decode(M.to_numpy(float), y)
        dec.append(dict(set=setn, n_features=M.shape[1], n_sessions=len(y), balanced_accuracy=obs, null_mean=null.mean(),
                        null_95=np.quantile(null, 0.95), p_perm=(np.sum(null >= obs) + 1) / (len(null) + 1)))
        print(dec[-1], flush=True)
    Dc = pd.DataFrame(dec)
    Dc.to_csv(T / "020_quiet_decoding.csv", index=False)
    figure(D, C, Lr, Dc, mn)
    json.dump(dict(script="020_quiet_windows.py", Q=list(QS), Q_main=QMAIN, n_sessions=int(D.session_id.nunique()),
                   lick_events="piezo licks + tongue out > 0.5", versions=L.versions()),
              open(L.RES / "provenance" / "020.json", "w"), indent=1)


def strip(ax, g, title, ylabel, cohort_col="cohort"):
    for h, cg in enumerate(("R+", "R-")):
        v = g[g[cohort_col] == cg].value.dropna()
        x = h + np.random.default_rng(1).uniform(-0.17, 0.17, len(v))
        ax.scatter(x, v, s=3, color=L.COH_COL[cg], lw=0, alpha=0.7)
        ax.plot([h - 0.25, h + 0.25], [v.mean()] * 2, color="k", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R+", "R−"])
    ax.set_title(title, fontsize=5.5)
    ax.set_ylabel(ylabel, fontsize=5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def pstar(ax, c):
    if len(c):
        r = c.iloc[0]
        ax.text(0.5, 0.98, f"MW p {r.p_mannwhitney:.2g}\nWelch p {r.p_welch:.2g}", transform=ax.transAxes, ha="center",
                va="top", fontsize=4.3)


def figure(D, C, Lr, Dc, mn):
    fig = plt.figure(figsize=(7.4, 9.4))
    gs = fig.add_gridspec(4, 6, hspace=0.75, wspace=0.75, left=0.07, right=0.98, top=0.95, bottom=0.05)
    m = D[(D.Q == QMAIN) & (D.third == "all")]
    cm = C[C.Q == QMAIN]
    # row 0: amount of quiet time + LA usage (all epochs)
    items = [("n", "quiet_s", "quiet time (s)"), ("la", "1", "P(still)"), ("la", "2", "P(whisking)"), ("la", "3", "P(jaw)"),
             ("la", "trans_per_s", "action switches / s"), ("la_bout", "whisking", "whisking bout (s)")]
    for j, (setn, st, lab) in enumerate(items):
        ax = fig.add_subplot(gs[0, j])
        g = m[(m.epoch == "all") & (m.set == setn) & (m.state == st)]
        strip(ax, g, lab, "per mouse")
        pstar(ax, cm[(cm.epoch == "all") & (cm.set == setn) & (cm.state == st)])
        if j == 0:
            ax.text(-0.5, 1.25, "A", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # row 1: per epoch usage of still / whisking + meta-states, R+ vs R-
    ax = fig.add_subplot(gs[1, :3])
    epochs = ["pre", "resp", "post", "iti", "other"]
    meas = [("la", "1", "still"), ("la", "2", "whisking"), ("la", "3", "jaw")] + \
           [("meta_side", str(k), mn[k].split(" ", 1)[1][:16]) for k in range(len(mn))]
    for i, (setn, st, lab) in enumerate(meas):
        for e, epn in enumerate(epochs):
            for h, cg in enumerate(("R+", "R-")):
                v = m[(m.epoch == epn) & (m.set == setn) & (m.state == st) & (m.cohort == cg)].value.dropna()
                x = i * 6 + e + (h - 0.5) * 0.4
                ax.bar(x, v.mean(), 0.38, yerr=v.sem(), color=L.COH_COL[cg], lw=0, error_kw=dict(lw=0.4), alpha=0.85)
            c = cm[(cm.epoch == epn) & (cm.set == setn) & (cm.state == st)]
            if len(c) and c.p_max.iloc[0] < 0.05:
                ax.text(i * 5 + e, ax.get_ylim()[1] * 0.95 if False else 0.0, "", fontsize=6)
    ax.set_xticks([i * 6 + 2 for i in range(len(meas))])
    ax.set_xticklabels([lab for _, _, lab in meas], rotation=30, ha="right", fontsize=4.8)
    ax.set_ylabel("fraction of quiet frames")
    ax.set_title("usage in quiet frames by epoch (bars per item: pre / resp / post / ITI / other)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.12, 1.1, "B", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # stars on top after limits are known
    ymax = ax.get_ylim()[1]
    for i, (setn, st, lab) in enumerate(meas):
        for e, epn in enumerate(epochs):
            c = cm[(cm.epoch == epn) & (cm.set == setn) & (cm.state == st)]
            if len(c) and c.p_max.iloc[0] < 0.05:
                ax.text(i * 6 + e, ymax * 0.97, "*", ha="center", fontsize=7)
    # volcano: every state of every set, epoch all
    ax = fig.add_subplot(gs[1, 3:])
    sets = {"la": "#d62728", "meta_side": "#ff7f0e", "meta_top": "#bcbd22", "kpms_side": "#1f77b4", "kpms_top": "#17becf",
            "arhmm_side": "#9467bd", "arhmm_top": "#e377c2", "feature": "#000000"}
    g = cm[(cm.epoch == "all") & cm.set.isin(sets)]
    for setn, col in sets.items():
        q = g[g.set == setn]
        ax.scatter(q.cohen_d, -np.log10(q.p_max), s=6, color=col, lw=0, alpha=0.8, label=setn.replace("_", " "))
    ax.axhline(-np.log10(0.05), color="k", lw=0.4, ls=":")
    top = g[g.q_bh_within_set < 0.05].sort_values("p_max").head(8)
    for r in top.itertuples():
        ax.text(r.cohen_d, -np.log10(r.p_max), " " + r.label, fontsize=3.8, va="center")
    ax.set_xlabel("Cohen's d, R+ − R− (usage in quiet frames)", fontsize=5)
    ax.set_ylabel("−log10 p (max of MW, Welch)", fontsize=5)
    ax.legend(frameon=False, fontsize=4, ncol=2, loc="upper left", markerscale=1.5)
    ax.set_title("every state / measure, all quiet frames (labels: BH q < 0.05 within set)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.12, 1.1, "C", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # row 2: features + decoding
    for j, fn in enumerate(FEAT_IDX):
        ax = fig.add_subplot(gs[2, j])
        g2 = m[(m.epoch == "all") & (m.set == "feature") & (m.state == fn)]
        strip(ax, g2, fn.replace("_", " "), "mean in quiet frames")
        pstar(ax, cm[(cm.epoch == "all") & (cm.set == "feature") & (cm.state == fn)])
        if j == 0:
            ax.text(-0.5, 1.25, "D", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = fig.add_subplot(gs[2, 4:])
    x = np.arange(len(Dc))
    ax.bar(x, Dc.balanced_accuracy, 0.6, color="#777777", lw=0)
    ax.scatter(x, Dc.null_95, marker="_", s=40, color="r", label="null 95th pct")
    ax.axhline(0.5, color="k", lw=0.4, ls=":")
    for i, r in Dc.iterrows():
        ax.text(i, r.balanced_accuracy + 0.02, f"p {r.p_perm:.2g}", ha="center", fontsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(Dc.set.str.replace("_", " "), rotation=40, ha="right", fontsize=4.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cohort decoding, balanced acc.", fontsize=5)
    ax.legend(frameon=False, fontsize=4.5)
    ax.set_title("R+ vs R− from quiet-frame usage (leave-one-mouse-out)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.2, 1.1, "E", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # row 3: learning (late - early) for LA / meta / transitions, and Q robustness
    ax = fig.add_subplot(gs[3, :3])
    lm = Lr[(Lr.Q == QMAIN) & (Lr.epoch == "all")]
    W = D[(D.Q == QMAIN) & (D.epoch == "all") & D.third.isin(["early", "late"])].pivot_table(
        index=["session_id", "cohort", "set", "state"], columns="third", values="value").reset_index()
    W["delta"] = W.late - W.early
    lm_items = [("la", "1", "still"), ("la", "2", "whisking"), ("la", "3", "jaw"), ("la", "trans_per_s", "switches/s ÷10")] + \
               [("meta_side", str(k), mn[k].split(" ", 1)[1][:14]) for k in range(len(mn))]
    for i, (setn, st, lab) in enumerate(lm_items):
        for h, cg in enumerate(("R+", "R-")):
            v = W[(W.set == setn) & (W.state == st) & (W.cohort == cg)].delta.dropna() / (10 if st == "trans_per_s" else 1)
            xx = i + (h - 0.5) * 0.36 + np.random.default_rng(2).uniform(-0.07, 0.07, len(v))
            ax.scatter(xx, v, s=2, color=L.COH_COL[cg], lw=0, alpha=0.6)
            ax.plot([i + (h - 0.5) * 0.36 - 0.12, i + (h - 0.5) * 0.36 + 0.12], [v.mean()] * 2, color="k", lw=0.9)
            q = lm[(lm.set == setn) & (lm.state == st)]
            if len(q):
                p = q[f"p_signflip_{'Rplus' if cg == 'R+' else 'Rminus'}"].iloc[0]
                if np.isfinite(p) and p < 0.05:
                    ax.text(i + (h - 0.5) * 0.36, 0.5, "*", ha="center", fontsize=7, color=L.COH_COL[cg],
                            transform=ax.get_xaxis_transform())
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(range(len(lm_items)))
    ax.set_xticklabels([lab for _, _, lab in lm_items], rotation=30, ha="right", fontsize=4.8)
    ax.set_ylabel("late − early third (per mouse)", fontsize=5)
    ax.set_title("change over the session in quiet frames (* sign-flip p < 0.05 within cohort)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.12, 1.1, "F", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = fig.add_subplot(gs[3, 3:])
    rob = [("la", "1", "P(still)"), ("la", "2", "P(whisking)"), ("la", "3", "P(jaw)"), ("feature", "whisking_amp", "whisking amp."),
           ("feature", "jaw_open", "jaw opening"), ("feature", "nose_speed", "nose speed")]
    for i, (setn, st, lab) in enumerate(rob):
        for qi, Q in enumerate(QS):
            for e, (epn, mk) in enumerate((("all", "o"), ("iti", "s"), ("pre", "^"))):
                c = C[(C.Q == Q) & (C.epoch == epn) & (C.set == setn) & (C.state == st)]
                if len(c):
                    xx = i + (qi - 1) * 0.22 + (e - 1) * 0.06
                    ax.scatter(xx, c.cohen_d.iloc[0], marker=mk, s=8, color=["#bbbbbb", "#555555", "#000000"][qi],
                               lw=0, label=f"Q {Q:g} s, {epn}" if i == 0 else None)
                    if c.p_max.iloc[0] < 0.05:
                        ax.scatter(xx, c.cohen_d.iloc[0], marker=mk, s=22, facecolor="none", edgecolor="r", lw=0.5)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(range(len(rob)))
    ax.set_xticklabels([lab for _, _, lab in rob], rotation=30, ha="right", fontsize=4.8)
    ax.set_ylabel("Cohen's d, R+ − R−", fontsize=5)
    ax.legend(frameon=False, fontsize=3.8, ncol=3, loc="upper left")
    ax.set_title("robustness: quiet threshold Q and epoch (red ring: both p < 0.05)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.12, 1.1, "G", transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.suptitle("Quiet windows (> 2 s from any piezo lick or tongue-out frame, task period): R+ (green) vs R− (magenta), "
                 "one point per mouse; tests Mann–Whitney + Welch, uncorrected", fontsize=6, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"020_quiet_windows.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    {"compute": compute, "stats": run_stats}[sys.argv[1]]()
