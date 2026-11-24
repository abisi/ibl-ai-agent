"""022 -- R+ vs R- outside reward consumption (haas, full run, BM_SET=full).

Usage: BM_SET=full python 022_nonreward_cohort.py compute   (per session -> cache/full/022/<sid>.pkl)
       BM_SET=full python 022_nonreward_cohort.py stats     (tables/022_*.csv, figures/022_*.png|pdf|svg)

Reward consumption is removed by time since the last reward, not by trial structure:
- reward times = corrected first lick (bm_lib first_lick_time) of every active trial with reward_available = 1 and a
  lick (R+ whisker hits, auditory hits of both cohorts; kept or not, passive excluded);
- non-reward frames = task-period frames (first kept trial - 2 s to last kept trial + 6.5 s) more than T s after the last
  reward (T = 10 s main; 5 and 20 s robustness) -- unrewarded licks (R- whisker hits, false alarms) stay in;
- no-lick variant: also > 0.5 s from any piezo lick or tongue-out frame.
Per session and mask: usage of every state of every set, transition rate per set, kpms side / top transition
probabilities between shared states (row-normalised), median bout duration of every lightning-action class and meta-state,
mean and SD of the features, whisking frequency (Welch PSD of the whisker angle in non-reward whisking stretches
>= 0.5 s, peak 5-25 Hz) and the fraction of session time that is non-reward.
Unrewarded-trial time courses: kept whisker misses, auditory misses, false alarms, correct rejections whose stimulus
is > 10 s after the last reward (018 peri cache, 25 ms bins).
Stats (unit = session = mouse): R+ vs R- Mann-Whitney + Welch, BH within set (exploratory); cohort decoding (L2
logistic, leave-one-session-out, balanced accuracy, 500 label permutations) from usage, features, transitions and all
combined; time courses: cluster-mass permutation (Welch t, |t| > 2.0, 1000 cohort-label permutations).
"""
import importlib.util
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bm_lib as L  # noqa: E402

_sp = importlib.util.spec_from_file_location("m18", HERE / "018_report_examples.py")
M18 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(M18)

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
OUT = L.CACHE / "full" / "022"
TS = (5.0, 10.0, 20.0)
TMAIN = 10.0
NOLICK_S = 0.5
FEAT = {"whisker_angle_dev": 0, "whisking_amp": 1, "jaw_open": 2, "nose_speed": 4, "top_nose_speed": 5}
SETS = ["la", "meta_side", "meta_top", "kpms_side", "kpms_top", "arhmm_side", "arhmm_top"]
FREQ_RANGE = (5.0, 25.0)
rng = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------ compute
def reward_times(sid):
    tr = pd.read_parquet(M18.INP / f"{sid}_trials.parquet")
    r = tr[(tr.context != "passive") & (tr.reward_available == 1) & (tr.lick_flag == 1) & np.isfinite(tr.first_lick_time)]
    return np.sort(r.first_lick_time.to_numpy())


def one(sid):
    S = M18.load_session(sid)
    t, Fm, Z, k = S["t"], S["feats"], S["Z"], S["trials"]
    rt = reward_times(sid)
    j = np.searchsorted(rt, t, side="right") - 1
    since = np.where(j >= 0, t - rt[np.clip(j, 0, None)], np.inf)
    tong = np.isfinite(Fm[:, 3]) & (Fm[:, 3] > 0.5)
    ev = np.sort(np.r_[S["licks"], t[tong]])
    jj = np.clip(np.searchsorted(ev, t), 1, len(ev) - 1)
    nl = np.minimum(np.abs(t - ev[jj - 1]), np.abs(ev[jj] - t)) > NOLICK_S
    starts = k.start_time.to_numpy()
    task = (t >= starts[0] - 2) & (t <= starts[-1] + 6.5)
    out = dict(session_id=sid, n_rewards=len(rt), task_s=task.sum() / L.FS_DS, res={})
    for Tm in TS:
        for var, extra in (("all", np.ones(len(t), bool)), ("nl", nl)):
            m = task & (since > Tm) & extra
            r = dict(frac_time=float(m.sum() / max(task.sum(), 1)), seconds=float(m.sum() / L.FS_DS))
            if m.sum() < 2000:  # < 10 s
                out["res"][(Tm, var)] = r
                continue
            for name, z in Z.items():
                ns = M18.nstates(name, z)
                zz = z[m]
                v = zz >= 0
                r[f"use:{name}"] = np.bincount(zz[v], minlength=ns)[:ns] / max(v.sum(), 1)
                a, b = z[:-1], z[1:]
                both = m[:-1] & m[1:] & (a >= 0) & (b >= 0)
                r[f"trans_rate:{name}"] = float(np.sum(both & (a != b)) / max(both.sum() / L.FS_DS, 1e-9))
                if name.startswith("kpms"):
                    C = np.zeros((ns, ns))
                    ch = both & (a != b)
                    np.add.at(C, (a[ch], b[ch]), 1)
                    r[f"trans_mat:{name}"] = C
                if name in ("la", "meta_side", "meta_top"):
                    on, ln, st = M18.bouts(z)
                    inside = np.array([m[o:o + l_].all() for o, l_ in zip(on, ln)]) if len(on) else np.zeros(0, bool)
                    r[f"bout:{name}"] = np.array([np.median(ln[(st == s) & inside]) / L.FS_DS if np.sum((st == s) & inside) >= 5
                                                  else np.nan for s in range(ns)])
            for f, idx in FEAT.items():
                x = Fm[m, idx]
                r[f"mean:{f}"] = float(np.nanmean(x))
                r[f"sd:{f}"] = float(np.nanstd(x))
            # whisking frequency: whisking stretches (lightning-action whisking) >= 0.5 s inside the mask
            if "la" in Z:
                w = m & (Z["la"] == 2) & np.isfinite(Fm[:, 0])
                on, ln, st = M18.bouts(w.astype(int))
                segs = [(o, l_) for o, l_, s in zip(on, ln, st) if s == 1 and l_ >= 100]
                P, fr = [], None
                for o, l_ in segs:
                    x = Fm[o:o + l_, 0].astype(float)
                    fr, p = signal.welch(x - x.mean(), fs=L.FS_DS, nperseg=100)
                    P.append(p * l_)
                if P:
                    psd = np.sum(P, 0) / np.sum([l_ for _, l_ in segs])
                    band = (fr >= FREQ_RANGE[0]) & (fr <= FREQ_RANGE[1])
                    r["whisk_psd"] = psd
                    r["whisk_freq"] = fr
                    r["whisk_peak_hz"] = float(fr[band][np.argmax(psd[band])])
                    r["whisk_power_5_25"] = float(np.trapezoid(psd[band], fr[band]))
                    r["whisk_seconds"] = float(np.sum([l_ for _, l_ in segs]) / L.FS_DS)
            out["res"][(Tm, var)] = r
    # unrewarded trials far from any reward (for the 018 time courses)
    st0 = k.start_time.to_numpy()
    jr = np.searchsorted(rt, st0, side="right") - 1
    out["trial_since_reward"] = np.where(jr >= 0, st0 - rt[np.clip(jr, 0, None)], np.inf)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{sid}.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
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


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    n = ok.sum()
    if n:
        pp = p[ok]
        o = np.argsort(pp)
        r = pp[o] * n / (np.arange(n) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty(n)
        qq[o] = np.minimum(r, 1)
        q[ok] = qq
    return q


def decode(U, y, nperm=500):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    def acc(yy):
        pred = np.empty(len(yy))
        for i in range(len(yy)):
            tr = np.arange(len(yy)) != i
            sc = StandardScaler().fit(U[tr])
            mdl = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(U[tr]), yy[tr])
            pred[i] = mdl.predict(sc.transform(U[i:i + 1]))[0]
        return balanced_accuracy_score(yy, pred)
    obs = acc(y)
    null = np.array([acc(rng.permutation(y)) for _ in range(nperm)])
    return obs, null


def cluster_perm(XA, XB, thr=2.0, nperm=1000):
    """cluster-mass permutation over time bins (Welch t), cohort labels shuffled across mice"""
    X = np.vstack([XA, XB])
    na = len(XA)

    def tstat(ia):
        A, B = X[ia], X[~ia]
        with np.errstate(all="ignore"):
            t = (np.nanmean(A, 0) - np.nanmean(B, 0)) / np.sqrt(np.nanvar(A, 0, ddof=1) / np.sum(np.isfinite(A), 0) +
                                                                   np.nanvar(B, 0, ddof=1) / np.sum(np.isfinite(B), 0))
        return np.nan_to_num(t)

    def clusters(t):
        out = []
        for sgn in (1, -1):
            above = sgn * t > thr
            on, ln, st = M18.bouts(above.astype(int))
            for o, l_, s in zip(on, ln, st):
                if s == 1:
                    out.append((o, o + l_, float(np.sum(t[o:o + l_]))))
        return out
    lab = np.zeros(len(X), bool)
    lab[:na] = True
    t0 = tstat(lab)
    obs = clusters(t0)
    mx = np.array([max([abs(c[2]) for c in clusters(tstat(rng.permutation(lab)))] or [0]) for _ in range(nperm)])
    return t0, [(a, b, m, float((np.sum(mx >= abs(m)) + 1) / (nperm + 1))) for a, b, m in obs]


def run_stats():
    S = M18.sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    D = {}
    for f in sorted(OUT.glob("*.pkl")):
        if f.stem in coh:
            D[f.stem] = pickle.load(open(f, "rb"))
    Pp = pd.read_csv(T / "full_009_fingerprints_pooled.csv")
    shared = {f"{r.method}_{r.view}": sorted(int(s) for s in Pp[(Pp.method == r.method) & (Pp.view == r.view) & Pp.shared].state)
              for r in Pp.drop_duplicates(["method", "view"]).itertuples()}
    mn = list(pd.read_csv(T / "full_009_metastates.csv").name)

    def states_of(name, ns):
        if name == "la":
            return [1, 2, 3, 4]
        if name.startswith("meta"):
            return list(range(min(ns, len(mn))))
        return shared.get(name, list(range(ns)))

    def slabel(name, s):
        if name == "la":
            return f"action {M18.LA[s]}"
        if name.startswith("meta"):
            return f"{name.replace('_', ' ')} {mn[s]}"
        return f"{name.replace('_', ' ')} {'S' if name.startswith('kpms') else 'A'}{s}"
    # long table of scalar measures
    rows = []
    for sid, d in D.items():
        for (Tm, var), r in d["res"].items():
            base = dict(session_id=sid, cohort=coh[sid], T=Tm, variant=var)
            rows.append({**base, "group": "time", "measure": "non-reward fraction of task time", "value": r["frac_time"]})
            rows.append({**base, "group": "time", "measure": "rewards", "value": d["n_rewards"]})
            for key, val in r.items():
                kind, _, name = key.partition(":")
                if kind == "use":
                    for s in states_of(name, len(val)):
                        rows.append({**base, "group": f"usage {name}", "measure": slabel(name, s), "value": float(val[s])})
                elif kind == "trans_rate":
                    rows.append({**base, "group": "transitions", "measure": f"{name} switches/s", "value": val})
                elif kind == "trans_mat":
                    sh = shared.get(name, [])
                    C = val[np.ix_(sh, sh)]
                    Pm = C / np.maximum(C.sum(1, keepdims=True), 1)
                    for i_, a in enumerate(sh):
                        for j_, b in enumerate(sh):
                            if a != b:
                                rows.append({**base, "group": f"transmat {name}", "measure": f"{name} {a}->{b}",
                                             "value": float(Pm[i_, j_]) if C[i_].sum() >= 5 else np.nan})
                elif kind == "bout":
                    for s in states_of(name, len(val)):
                        rows.append({**base, "group": "bout duration", "measure": f"{slabel(name, s)} bout (s)", "value": float(val[s])})
                elif kind in ("mean", "sd"):
                    rows.append({**base, "group": "features", "measure": f"{kind} {name}", "value": val})
            for key in ("whisk_peak_hz", "whisk_power_5_25", "whisk_seconds"):
                if key in r:
                    rows.append({**base, "group": "whisking", "measure": key, "value": r[key]})
    V = pd.DataFrame(rows)
    V.to_csv(T / "022_nonreward_per_session.csv", index=False)
    res = []
    for (Tm, var, grp, ms), g in V.groupby(["T", "variant", "group", "measure"]):
        a, b = g[g.cohort == "R+"].value, g[g.cohort == "R-"].value
        pm, pw = two(a, b)
        sd = np.sqrt((a.var() + b.var()) / 2)
        res.append(dict(T=Tm, variant=var, group=grp, measure=ms, n_Rplus=a.notna().sum(), n_Rminus=b.notna().sum(),
                        mean_Rplus=a.mean(), mean_Rminus=b.mean(), cohen_d=(a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                        p_mannwhitney=pm, p_welch=pw))
    C = pd.DataFrame(res)
    C["p_max"] = C[["p_mannwhitney", "p_welch"]].max(axis=1)
    C["q_bh_within_group"] = C.groupby(["T", "variant", "group"]).p_max.transform(bh)
    C.to_csv(T / "022_nonreward_cohort.csv", index=False)
    # decoding (T main, both variants)
    dec = []
    for var in ("all", "nl"):
        g0 = V[(V["T"] == TMAIN) & (V.variant == var) & (V.group != "time")]
        for gname, sel in (("usage (all sets)", g0.group.str.startswith("usage")), ("features + whisking", g0.group.isin(["features", "whisking"])),
                           ("transitions", g0.group.str.startswith("trans")), ("bout durations", g0.group == "bout duration"),
                           ("everything", np.ones(len(g0), bool))):
            g = g0[sel]
            M = g.pivot_table(index="session_id", columns="measure", values="value")
            M = M.loc[:, M.notna().mean() >= 0.9]
            M = M.fillna(M.mean())
            M = M.loc[:, M.std() > 0]
            y = np.array([coh[s] == "R+" for s in M.index]).astype(int)
            obs, null = decode(M.to_numpy(float), y)
            dec.append(dict(variant=var, features=gname, n_features=M.shape[1], n_sessions=len(y), balanced_accuracy=obs,
                            null_mean=null.mean(), null_95=float(np.quantile(null, 0.95)),
                            p_perm=float((np.sum(null >= obs) + 1) / (len(null) + 1))))
            print(dec[-1], flush=True)
    Dc = pd.DataFrame(dec)
    Dc.to_csv(T / "022_nonreward_decoding.csv", index=False)
    # unrewarded-trial time courses (018 cache), stimulus > 10 s after the last reward
    D18 = M18.load_all()
    tb = M18.PERI[0] + M18.BIN * (np.arange(M18.NB) + 0.5)
    chans = ["la:1", "la:2", "la:3", "meta_side:1", "whisking_amp", "jaw_open", "nose_speed", "whisker_angle_dev"]
    conds = {"WM": ("whisker", 0), "AM": ("auditory", 0), "FA": ("no_stim", 1), "CR": ("no_stim", 0)}
    TC, clus = {}, []
    for sid, d in D18.items():
        if sid not in D or sid not in coh:
            continue
        tr = d["trials"]
        far = D[sid]["trial_since_reward"] > TMAIN
        if len(far) != len(tr):
            continue
        for cn, (st, lk) in conds.items():
            msk = far & (tr.stim.to_numpy() == st) & (tr.lick_flag.to_numpy() == lk)
            if msk.sum() < 3:
                continue
            for ch in chans:
                if ch in d["peri_channels"]:
                    k = d["peri_channels"].index(ch)
                    with np.errstate(all="ignore"):
                        mu = np.nanmean(d["peri"][msk, k].astype(np.float32), 0)
                    mu[np.sum(np.isfinite(d["peri"][msk, k].astype(np.float32)), 0) < 3] = np.nan
                    TC.setdefault((cn, ch, coh[sid]), []).append(mu)
    win = (tb >= -2) & (tb < 8)
    for (cn, ch) in sorted({(a, b) for a, b, _ in TC}):
        A, B = np.array(TC.get((cn, ch, "R+"), [])), np.array(TC.get((cn, ch, "R-"), []))
        if len(A) < 5 or len(B) < 5:
            continue
        t0, cl = cluster_perm(A[:, win], B[:, win])
        for a, b, m, p in cl:
            clus.append(dict(condition=cn, channel=ch, t_start=float(tb[win][a]), t_stop=float(tb[win][b - 1]), mass=m, p_cluster=p,
                             n_Rplus=len(A), n_Rminus=len(B)))
    Cl = pd.DataFrame(clus)
    Cl.to_csv(T / "022_unrewarded_timecourse_clusters.csv", index=False)
    figure(C, Dc, TC, Cl, tb, win, chans, conds, D)
    json.dump(dict(script="022_nonreward_cohort.py", T=list(TS), T_main=TMAIN, n_sessions=len(D), versions=L.versions()),
              open(L.RES / "provenance" / "022.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    c = C[(C["T"] == TMAIN)]
    print("tests:", c.groupby("variant").size().to_dict())
    print(c[c.p_max < 0.05].sort_values(["variant", "p_max"])[["variant", "group", "measure", "n_Rplus", "n_Rminus", "mean_Rplus",
                                                                  "mean_Rminus", "cohen_d", "p_mannwhitney", "p_welch",
                                                                  "q_bh_within_group"]].round(4).to_string())
    print(Dc.round(3).to_string())
    print(Cl[Cl.p_cluster < 0.05].round(3).to_string() if len(Cl) else "no clusters")


def figure(C, Dc, TC, Cl, tb, win, chans, conds, D):
    fig = plt.figure(figsize=(7.4, 9.6))
    gs = fig.add_gridspec(4, 4, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.94, bottom=0.06)
    # A: volcano (T main, both variants)
    cols = {"usage la": "#d62728", "usage meta_side": "#ff7f0e", "usage meta_top": "#bcbd22", "usage kpms_side": "#1f77b4",
            "usage kpms_top": "#17becf", "usage arhmm_side": "#9467bd", "usage arhmm_top": "#e377c2", "features": "#000000",
            "whisking": "#8c564b", "transitions": "#7f7f7f", "bout duration": "#2ca02c", "transmat kpms_side": "#aec7e8",
            "transmat kpms_top": "#9edae5"}
    for j, var in enumerate(("all", "nl")):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        g = C[(C["T"] == TMAIN) & (C.variant == var) & (C.group != "time")]
        for grp, col in cols.items():
            q = g[g.group == grp]
            ax.scatter(q.cohen_d, -np.log10(q.p_max), s=5, color=col, lw=0, alpha=0.8, label=grp)
        ax.axhline(-np.log10(0.05), color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4)
        for r in g[g.q_bh_within_group < 0.05].sort_values("p_max").head(8).itertuples():
            ax.text(r.cohen_d, -np.log10(r.p_max), " " + r.measure, fontsize=3.6, va="center")
        ax.set_xlabel("Cohen's d, R+ − R−", fontsize=5)
        ax.set_ylabel("−log10 p (max of MW, Welch)", fontsize=5)
        ax.set_title(f"> {TMAIN:g} s after any reward{'' if var == 'all' else ', > 0.5 s from licks'} (labels: BH q < 0.05)", fontsize=5.5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if j == 0:
            ax.legend(frameon=False, fontsize=3.5, ncol=2, loc="upper left", markerscale=1.2)
        ax.text(-0.1, 1.08, "AB"[j], transform=ax.transAxes, fontsize=8, fontweight="bold")
    # C: decoding
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(Dc))
    ax.bar(x, Dc.balanced_accuracy, 0.6, color=["#777777" if v == "all" else "#bbbbbb" for v in Dc.variant], lw=0)
    ax.scatter(x, Dc.null_95, marker="_", s=40, color="r")
    for i, r in Dc.iterrows():
        ax.text(i, r.balanced_accuracy + 0.02, f"p {r.p_perm:.2g}", ha="center", fontsize=4)
    ax.axhline(0.5, color="k", lw=0.4, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} ({'all' if v == 'all' else 'no-lick'})" for f, v in zip(Dc.features, Dc.variant)], rotation=50,
                       ha="right", fontsize=4.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cohort decoding, balanced acc.", fontsize=5)
    ax.set_title("R+ vs R− from non-reward behaviour (leave-one-mouse-out; red = null 95th pct)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.1, 1.08, "C", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # D: whisking PSD
    ax = fig.add_subplot(gs[1, 2])
    for cg in ("R+", "R-"):
        P = [d["res"][(TMAIN, "nl")]["whisk_psd"] for sid, d in D.items() if "whisk_psd" in d["res"].get((TMAIN, "nl"), {})
             and d.get("cohort", None) is None and sid in TC_COH and TC_COH[sid] == cg]
        if P:
            fr = next(d["res"][(TMAIN, "nl")]["whisk_freq"] for d in D.values() if "whisk_freq" in d["res"].get((TMAIN, "nl"), {}))
            P = np.array([p / np.trapezoid(p[(fr >= 1) & (fr <= 40)], fr[(fr >= 1) & (fr <= 40)]) for p in P])
            m, e = P.mean(0), P.std(0) / np.sqrt(len(P))
            ax.fill_between(fr, m - e, m + e, color=L.COH_COL[cg], alpha=0.2, lw=0)
            ax.plot(fr, m, color=L.COH_COL[cg], lw=0.8, label=f"{cg} ({len(P)})")
    ax.set_xlim(1, 40)
    ax.set_xlabel("Hz", fontsize=5)
    ax.set_ylabel("normalised PSD, whisker angle", fontsize=5)
    ax.set_title("whisking spectrum (non-reward, no-lick)", fontsize=5.5)
    ax.legend(frameon=False, fontsize=4.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.25, 1.08, "D", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # E: key strip plots
    ax = fig.add_subplot(gs[1, 3])
    g = C[(C["T"] == TMAIN) & (C.variant == "nl") & (C.measure.isin(["action still", "action whisking", "mean jaw_open",
                                                                      "whisk_peak_hz", "mean whisking_amp", "kpms_side switches/s"]))]
    ax.barh(range(len(g)), g.cohen_d, color=["k" if p < 0.05 else "#bbbbbb" for p in g.p_max], lw=0)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.measure, fontsize=4.3)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("Cohen's d, R+ − R− (black: both p < 0.05)", fontsize=5)
    ax.set_title("key measures (no-lick)", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.text(-0.6, 1.08, "E", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # F: unrewarded-trial time courses
    labs = {"la:1": "P(still)", "la:2": "P(whisking)", "la:3": "P(jaw)", "meta_side:1": "P(M1 nose + whisk)",
            "whisking_amp": "whisking amp. (deg)", "jaw_open": "jaw opening (mm)", "nose_speed": "nose speed (mm/s)",
            "whisker_angle_dev": "whisker angle (deg)"}
    show = [("WM", "la:1"), ("WM", "whisking_amp"), ("CR", "la:1"), ("CR", "whisking_amp"),
            ("FA", "la:1"), ("FA", "whisking_amp"), ("AM", "la:1"), ("WM", "jaw_open")]
    for i, (cn, ch) in enumerate(show):
        ax = fig.add_subplot(gs[2 + i // 4, i % 4])
        for cg in ("R+", "R-"):
            X = np.array(TC.get((cn, ch, cg), []))
            if not len(X):
                continue
            with np.errstate(all="ignore"):
                m = np.nanmean(X, 0)
                e = np.nanstd(X, 0) / np.sqrt(np.sum(np.isfinite(X), 0).clip(1))
            ax.fill_between(tb[win], (m - e)[win], (m + e)[win], color=L.COH_COL[cg], alpha=0.2, lw=0)
            ax.plot(tb[win], m[win], color=L.COH_COL[cg], lw=0.7, label=f"{cg} ({len(X)})")
        q = Cl[(Cl.condition == cn) & (Cl.channel == ch) & (Cl.p_cluster < 0.05)] if len(Cl) else []
        for r in (q.itertuples() if len(q) else []):
            ax.axvspan(r.t_start, r.t_stop, color="#ffd400", alpha=0.35, lw=0)
        for x in (0, 1.1, 6.1):
            ax.axvline(x, color="k", lw=0.3, ls=":")
        ax.set_title(f"{cn} (> {TMAIN:g} s after reward): {labs[ch]}", fontsize=5)
        ax.set_xlabel("s from stimulus", fontsize=5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if i == 0:
            ax.legend(frameon=False, fontsize=4.3)
            ax.text(-0.3, 1.12, "F", transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.suptitle("R+ vs R− outside reward consumption (frames > 10 s after any reward; unrewarded licks kept in A, removed in B); "
                 "F: unrewarded trials, yellow = cluster p < 0.05", fontsize=6, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"022_nonreward_cohort.{ext}", dpi=300)
    plt.close(fig)


TC_COH = dict(zip(M18.sessions().session_id, M18.sessions().reward_group))

if __name__ == "__main__":
    {"compute": compute, "stats": run_stats}[sys.argv[1]]()
