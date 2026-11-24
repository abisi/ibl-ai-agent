"""021 -- (A) motifs / actions associated with engagement (local P(lick)) and (B) behaviour around the first whisker hit
(haas, full run, BM_SET=full).

Usage: BM_SET=full python 021_engagement_first_hit.py compute   (per session -> cache/full/021/<sid>.npz)
       BM_SET=full python 021_engagement_first_hit.py stats     (tables/021_*.csv, figures/021_*.png|pdf|svg)

Per kept trial x epoch (bm_lib.trial_windows: pre 2 s, resp, post = response end -> +5 s, iti = +5 s -> next quiet
window) x frame filter (all frames / no-lick = > 0.5 s from any piezo lick or tongue-out frame): usage (fraction of
frames) of every state of every set (lightning-action classes, kpms meta-states side / top, kpms syllables side / top,
AR-HMM states side / top) and the mean of the features (whisker angle from the session median, whisking amplitude, jaw
opening, nose speed side / top). States: shared states only (>= 3 mice, 009) in the stats.

A. Engagement. Per session, two engagement curves over the kept trials: whisker P(lick) (Gaussian sigma 5 whisker trials)
   and auditory P(lick) (sigma 5 auditory trials), each interpolated to every kept trial by trial index. Per session and
   measure, OLS over trials: y ~ z(engagement) + z(trial position) -> beta (change in usage per SD of engagement,
   controlled for time in session); also high - low tercile difference (not time-controlled, for display).
   Trials: pre = every trial (the stimulus is not yet known); post / iti = trials WITHOUT a lick (no reward, no lick
   tail); no-lick frames only. Sessions with >= 30 usable trials. Group: sign-flip of beta (10000) for all mice and per
   cohort; R+ vs R- Mann-Whitney + Welch; BH within set x epoch x engagement x scope.
B. First whisker hit (FH: first kept whisker trial with a lick). Per session: FH minus the mean of (i) the whisker misses
   before it (last <= 5), (ii) the next whisker hits (<= 5), (iii) the auditory hits before it (last <= 5; rewarded in both
   cohorts). Epochs pre (no-lick frames) and resp / post / iti (all frames). Group sign-flip per cohort, R+ vs R- MW +
   Welch, uncorrected (exploratory). Time courses: 018 peri cache (25 ms), FH vs references, per cohort.
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
OUT = L.CACHE / "full" / "021"
EPOCHS = ["pre", "resp", "post", "iti"]
FILTERS = ["all", "nl"]
NOLICK_S = 0.5
FEAT = {"whisker_angle_dev": 0, "whisking_amp": 1, "jaw_open": 2, "nose_speed": 4, "top_nose_speed": 5}
SETS = ["la", "meta_side", "meta_top", "kpms_side", "kpms_top", "arhmm_side", "arhmm_top"]
rng = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------ compute
def one(sid):
    S = M18.load_session(sid)
    t, Fm, Z, k = S["t"], S["feats"], S["Z"], S["trials"]
    tong = np.isfinite(Fm[:, 3]) & (Fm[:, 3] > 0.5)
    ev = np.sort(np.r_[S["licks"], t[tong]])
    j = np.clip(np.searchsorted(ev, t), 1, len(ev) - 1)
    nl = np.minimum(np.abs(t - ev[j - 1]), np.abs(ev[j] - t)) > NOLICK_S
    W = L.trial_windows(sid, k)
    cols = []
    for name in SETS:
        if name in Z:
            cols += [(name, s) for s in range(M18.nstates(name, Z[name]))]
    cols += [("feature", f) for f in FEAT]
    X = np.full((len(k), len(EPOCHS), 2, len(cols)), np.nan, np.float32)
    for i in range(len(k)):
        for e, ep in enumerate(EPOCHS):
            a, b = W[f"{ep}_start"].iloc[i], W[f"{ep}_stop"].iloc[i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
            if i1 - i0 < 20:
                continue
            for fi, msk in enumerate((np.ones(i1 - i0, bool), nl[i0:i1])):
                if msk.sum() < 20:
                    continue
                c = 0
                for name in SETS:
                    if name not in Z:
                        continue
                    z = Z[name][i0:i1][msk]
                    z = z[z >= 0]
                    ns = M18.nstates(name, Z[name])
                    if len(z) >= 20:
                        X[i, e, fi, c:c + ns] = np.bincount(z, minlength=ns)[:ns] / len(z)
                    c += ns
                for f, idx in FEAT.items():
                    X[i, e, fi, c] = np.nanmean(Fm[i0:i1, idx][msk])
                    c += 1
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{sid}.npz", X=X, cols=np.array([f"{a}:{b}" for a, b in cols]),
                        stim=k.stim.to_numpy().astype(str), lick=k.lick_flag.to_numpy().astype(int),
                        start=k.start_time.to_numpy())
    return sid


def compute():
    sids = list(M18.sessions().session_id)
    with ProcessPoolExecutor(int(os.environ.get("BM_WORKERS", 12))) as ex:
        for s in ex.map(one, sids):
            print("done", s, flush=True)


# ------------------------------------------------------------------------------------------------ stats helpers
def signflip(x, nflip=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    s = rng.choice([-1, 1], size=(nflip, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= abs(x.mean())) + 1) / (nflip + 1))


def two(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
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


def curve(lick, sel, sig=5):
    idx = np.flatnonzero(sel)
    if len(idx) < 10:
        return None
    c = M18.smooth_nan(lick[idx].astype(float), sig)
    return np.interp(np.arange(len(lick)), idx, c)


def zs(x):
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.full(len(x), np.nan)


def labels(mn):
    def lab(col):
        setn, st = col.split(":")
        if setn == "la":
            return f"action {M18.LA[int(st)]}"
        if setn.startswith("meta"):
            return f"{setn.replace('_', ' ')} {mn[int(st)] if int(st) < len(mn) else st}"
        if setn == "feature":
            return st.replace("_", " ")
        return f"{setn.replace('_', ' ')} {'S' if setn.startswith('kpms') else 'A'}{st}"
    return lab


def shared_cols(cols):
    Pp = pd.read_csv(T / "full_009_fingerprints_pooled.csv")
    sh = {f"{r.method}_{r.view}:{r.state}" for r in Pp[Pp.shared].itertuples()}
    mn = list(pd.read_csv(T / "full_009_metastates.csv").name)
    keep = []
    for c in cols:
        setn, st = c.split(":")
        if setn in ("kpms_side", "kpms_top", "arhmm_side", "arhmm_top"):
            if c in sh:
                keep.append(c)
        elif setn == "la":
            if st in ("1", "2", "3"):  # still / whisking / jaw (lick and background excluded: no-lick frames)
                keep.append(c)
        elif setn.startswith("meta"):
            if int(st) < len(mn):
                keep.append(c)
        else:
            keep.append(c)
    return keep, mn


# ------------------------------------------------------------------------------------------------ A. engagement
def engagement(D, coh):
    rows = []
    for sid, d in D.items():
        X, cols = d["X"], list(d["cols"])
        lick, stim = d["lick"], d["stim"]
        n = len(lick)
        pos = np.arange(n) / max(n - 1, 1)
        for eng, sel in (("whisker P(lick)", stim == "whisker"), ("auditory P(lick)", stim == "auditory")):
            cv = curve(lick, sel)
            if cv is None or np.nanstd(cv) < 0.02:
                continue
            for e, ep in enumerate(EPOCHS):
                if ep == "resp":
                    continue
                tsel = np.ones(n, bool) if ep == "pre" else (lick == 0)
                for c, col in enumerate(cols):
                    if col not in d["keep"]:
                        continue
                    y = X[:, e, 1, c].astype(float)
                    ok = tsel & np.isfinite(y) & np.isfinite(cv)
                    if ok.sum() < 30 or np.std(y[ok]) == 0:
                        continue
                    A = np.c_[np.ones(ok.sum()), zs(cv[ok]), zs(pos[ok])]
                    if not np.isfinite(A).all() or abs(np.corrcoef(cv[ok], pos[ok])[0, 1]) > 0.9:
                        continue
                    beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]
                    q1, q2 = np.nanquantile(cv[ok], [1 / 3, 2 / 3])
                    hi, lo = y[ok][cv[ok] >= q2], y[ok][cv[ok] <= q1]
                    rows.append(dict(session_id=sid, cohort=coh[sid], engagement=eng, epoch=ep, measure=col,
                                     beta=float(beta[1]), beta_time=float(beta[2]),
                                     high_minus_low=float(np.mean(hi) - np.mean(lo)) if len(hi) and len(lo) else np.nan,
                                     n_trials=int(ok.sum())))
    Ps = pd.DataFrame(rows)
    Ps.to_csv(T / "021_engagement_per_session.csv", index=False)
    res = []
    for (eng, ep, col), g in Ps.groupby(["engagement", "epoch", "measure"]):
        a, b = g[g.cohort == "R+"].beta, g[g.cohort == "R-"].beta
        pm, pw = two(a, b)
        res.append(dict(engagement=eng, epoch=ep, measure=col, set=col.split(":")[0], n=len(g), beta_all=g.beta.mean(),
                        p_all=signflip(g.beta), beta_Rplus=a.mean(), p_Rplus=signflip(a), beta_Rminus=b.mean(),
                        p_Rminus=signflip(b), p_cohort_mw=pm, p_cohort_welch=pw, hl_all=g.high_minus_low.mean(),
                        hl_Rplus=g[g.cohort == "R+"].high_minus_low.mean(), hl_Rminus=g[g.cohort == "R-"].high_minus_low.mean()))
    G = pd.DataFrame(res)
    for sc in ("all", "Rplus", "Rminus"):
        G[f"q_{sc}"] = G.groupby(["engagement", "epoch", "set"])[f"p_{sc}"].transform(bh)
    G.to_csv(T / "021_engagement_group.csv", index=False)
    return Ps, G


# ------------------------------------------------------------------------------------------------ B. first whisker hit
def first_hit(D, coh):
    rows, info = [], []
    for sid, d in D.items():
        X, cols = d["X"], list(d["cols"])
        lick, stim = d["lick"], d["stim"]
        wh = np.flatnonzero((stim == "whisker") & (lick == 1))
        if not len(wh):
            continue
        f = wh[0]
        wtr = np.flatnonzero(stim == "whisker")
        info.append(dict(session_id=sid, cohort=coh[sid], first_hit_trial=int(f),
                         first_hit_whisker_index=int(np.searchsorted(wtr, f)),
                         first_hit_time_min=float((d["start"][f] - d["start"][0]) / 60)))
        refs = {"prior whisker misses": np.flatnonzero((stim == "whisker") & (lick == 0) & (np.arange(len(lick)) < f))[-5:],
                "next whisker hits": wh[1:6],
                "prior auditory hits": np.flatnonzero((stim == "auditory") & (lick == 1) & (np.arange(len(lick)) < f))[-5:]}
        for e, ep in enumerate(EPOCHS):
            fi = 1 if ep == "pre" else 0
            for rname, ri in refs.items():
                if len(ri) < 1:
                    continue
                for c, col in enumerate(cols):
                    if col not in d["keep"]:
                        continue
                    v, r = X[f, e, fi, c], np.nanmean(X[ri, e, fi, c]) if len(ri) else np.nan
                    if np.isfinite(v) and np.isfinite(r):
                        rows.append(dict(session_id=sid, cohort=coh[sid], epoch=ep, reference=rname, measure=col,
                                         first_hit=float(v), ref=float(r), diff=float(v - r)))
    Ps = pd.DataFrame(rows)
    I = pd.DataFrame(info)
    Ps.to_csv(T / "021_first_hit_per_session.csv", index=False)
    I.to_csv(T / "021_first_hit_trials.csv", index=False)
    res = []
    for (ep, rn, col), g in Ps.groupby(["epoch", "reference", "measure"]):
        a, b = g[g.cohort == "R+"]["diff"], g[g.cohort == "R-"]["diff"]
        pm, pw = two(a, b)
        res.append(dict(epoch=ep, reference=rn, measure=col, set=col.split(":")[0], n_Rplus=len(a), n_Rminus=len(b),
                        diff_all=g["diff"].mean(), p_all=signflip(g["diff"]), diff_Rplus=a.mean(), p_Rplus=signflip(a),
                        diff_Rminus=b.mean(), p_Rminus=signflip(b), p_cohort_mw=pm, p_cohort_welch=pw))
    G = pd.DataFrame(res)
    for sc in ("all", "Rplus", "Rminus"):
        G[f"q_{sc}"] = G.groupby(["epoch", "reference", "set"])[f"p_{sc}"].transform(bh)
    G.to_csv(T / "021_first_hit_group.csv", index=False)
    return Ps, G, I


# ------------------------------------------------------------------------------------------------ figures
def fig_engagement(G, lab):
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.45, left=0.08, right=0.98, top=0.93, bottom=0.08)
    sets = {"la": "#d62728", "meta_side": "#ff7f0e", "meta_top": "#bcbd22", "kpms_side": "#1f77b4", "kpms_top": "#17becf",
            "arhmm_side": "#9467bd", "arhmm_top": "#e377c2", "feature": "#000000"}
    for i, eng in enumerate(("whisker P(lick)", "auditory P(lick)")):
        for j, ep in enumerate(("pre", "post", "iti")):
            ax = fig.add_subplot(gs[i, j])
            g = G[(G.engagement == eng) & (G.epoch == ep)]
            for setn, col in sets.items():
                q = g[g.set == setn]
                if setn == "feature":
                    continue  # different units; features in the bottom row
                ax.scatter(q.beta_all, -np.log10(q.p_all), s=7, color=col, lw=0, alpha=0.85, label=setn.replace("_", " "))
            ax.axhline(-np.log10(0.05), color="k", lw=0.4, ls=":")
            ax.axvline(0, color="k", lw=0.4)
            top = g[(g.set != "feature") & (g.q_all < 0.05)].sort_values("p_all").head(6)
            for r in top.itertuples():
                ax.text(r.beta_all, -np.log10(r.p_all), " " + lab(r.measure), fontsize=3.8, va="center")
            ax.set_title(f"{eng} — {ep}{'' if ep == 'pre' else ' (no-lick trials)'}", fontsize=5.5)
            ax.set_xlabel("β (usage per SD engagement, time-controlled)", fontsize=5)
            ax.set_ylabel("−log10 p (sign-flip, all mice)", fontsize=5)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=3.8, ncol=2, loc="upper left", markerscale=1.3)
    # bottom row: LA / meta / features, beta per cohort (whisker and auditory engagement, pre)
    for j, (eng, ep) in enumerate((("whisker P(lick)", "pre"), ("auditory P(lick)", "pre"), ("whisker P(lick)", "iti"))):
        ax = fig.add_subplot(gs[2, j])
        g = G[(G.engagement == eng) & (G.epoch == ep) & G.set.isin(["la", "meta_side", "feature"])]
        y = np.arange(len(g))
        for h, (sc, cg) in enumerate((("Rplus", "R+"), ("Rminus", "R-"))):
            v = g[f"beta_{sc}"].to_numpy()
            sd = np.nanmax(np.abs(g[["beta_Rplus", "beta_Rminus"]].to_numpy()), 1)
            ax.barh(y + (h - 0.5) * 0.38, v / np.where(sd > 0, sd, 1), 0.36, color=L.COH_COL[cg], lw=0, label=cg)
            for yy, p in zip(y, g[f"p_{sc}"]):
                if p < 0.05:
                    ax.text(1.05, yy + (h - 0.5) * 0.38, "*", color=L.COH_COL[cg], fontsize=6, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([lab(m) for m in g.measure], fontsize=4.3)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_xlim(-1.2, 1.2)
        ax.set_xlabel("β, scaled per row (* sign-flip p < 0.05)", fontsize=5)
        ax.set_title(f"{eng} — {ep}", fontsize=5.5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if j == 0:
            ax.legend(frameon=False, fontsize=4.5)
    fig.suptitle("Engagement: within-session regression of state usage (no-lick frames) on local P(lick), controlled for "
                 "trial position; volcano = every shared state, all mice (labels: BH q < 0.05)", fontsize=6, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"021_engagement.{ext}", dpi=300)
    plt.close(fig)


def fig_first_hit(G, I, D18, coh, lab):
    fig = plt.figure(figsize=(7.4, 8.8))
    gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.5, left=0.08, right=0.98, top=0.93, bottom=0.07)
    # time courses from 018 cache: FH vs next WH vs prior AH vs prior WM
    chans = [("la:1", "P(still)"), ("la:2", "P(whisking)"), ("la:4", "P(lick)"), ("jaw_open", "jaw opening (mm)")]
    tb = M18.PERI[0] + M18.BIN * (np.arange(M18.NB) + 0.5)
    Imap = dict(zip(I.session_id, I.first_hit_trial))
    for ci, (ch, lab_) in enumerate(chans):
        for h, cg in enumerate(("R+", "R-")):
            ax = fig.add_subplot(gs[h, ci])
            curves = {"first whisker hit": [], "next whisker hits": [], "prior auditory hits": [], "prior whisker misses": []}
            for sid, d in D18.items():
                if coh.get(sid) != cg or sid not in Imap or ch not in d["peri_channels"]:
                    continue
                k = d["peri_channels"].index(ch)
                tr = d["trials"]
                st, lk = tr.stim.to_numpy(), tr.lick_flag.to_numpy()
                f = Imap[sid]
                P = d["peri"][:, k].astype(np.float32)
                ar = np.arange(len(st))
                sel = {"first whisker hit": [f], "next whisker hits": np.flatnonzero((st == "whisker") & (lk == 1))[1:6],
                       "prior auditory hits": np.flatnonzero((st == "auditory") & (lk == 1) & (ar < f))[-5:],
                       "prior whisker misses": np.flatnonzero((st == "whisker") & (lk == 0) & (ar < f))[-5:]}
                for nm, ii in sel.items():
                    if len(ii):
                        with np.errstate(all="ignore"):
                            curves[nm].append(np.nanmean(P[ii], 0))
            for nm, col, ls in (("first whisker hit", "k", "-"), ("next whisker hits", L.TASK_COL["whisker"], "-"),
                                ("prior auditory hits", L.TASK_COL["auditory"], "-"), ("prior whisker misses", "#999999", "--")):
                X = np.array(curves[nm])
                if not len(X):
                    continue
                with np.errstate(all="ignore"):
                    m = np.nanmean(X, 0)
                    e = np.nanstd(X, 0) / np.sqrt(np.sum(np.isfinite(X), 0).clip(1))
                m[np.sum(np.isfinite(X), 0) < 5] = np.nan
                ax.fill_between(tb, m - e, m + e, color=col, alpha=0.15, lw=0)
                ax.plot(tb, m, ls, color=col, lw=0.7, label=f"{nm} ({len(X)})")
            for x in (0, 1.1, 6.1):
                ax.axvline(x, color="k", lw=0.3, ls=":")
            ax.set_xlim(-2, 12)
            ax.set_title(f"{cg}: {lab_}", fontsize=5.5, color=L.COH_COL[cg])
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if h == 1:
                ax.set_xlabel("s from stimulus", fontsize=5)
            if ci == 0 and h == 0:
                ax.legend(frameon=False, fontsize=3.8, loc="lower right")
    # bottom: effect summary (FH - next WH, FH - prior AH, FH - prior WM pre) for LA / meta / features
    for j, (rn, eps) in enumerate((("next whisker hits", ("resp", "post", "iti")), ("prior auditory hits", ("resp", "post", "iti")),
                                   ("prior whisker misses", ("pre",)))):
        ax = fig.add_subplot(gs[2, j])
        g = G[(G.reference == rn) & G.epoch.isin(eps) & G.set.isin(["la", "meta_side", "feature"])]
        g = g.assign(key=g.measure + " | " + g.epoch)
        y = np.arange(len(g))
        for h, (sc, cg) in enumerate((("Rplus", "R+"), ("Rminus", "R-"))):
            v = g[f"diff_{sc}"].to_numpy()
            sd = np.nanmax(np.abs(g[["diff_Rplus", "diff_Rminus"]].to_numpy()), 1)
            ax.barh(y + (h - 0.5) * 0.38, v / np.where(sd > 0, sd, 1), 0.36, color=L.COH_COL[cg], lw=0)
            for yy, p in zip(y, g[f"p_{sc}"]):
                if p < 0.05:
                    ax.text(1.05, yy + (h - 0.5) * 0.38, "*", color=L.COH_COL[cg], fontsize=6, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{lab(m)} | {e}" for m, e in zip(g.measure, g.epoch)], fontsize=3.6)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_xlim(-1.2, 1.2)
        ax.set_title(f"first hit − {rn}", fontsize=5.5)
        ax.set_xlabel("difference, scaled per row (* p < 0.05)", fontsize=5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax = fig.add_subplot(gs[2, 3])
    for h, cg in enumerate(("R+", "R-")):
        v = I[I.cohort == cg].first_hit_whisker_index
        ax.hist(v, bins=np.arange(0, 41, 2), color=L.COH_COL[cg], alpha=0.6, label=f"{cg} (median {v.median():.0f})")
    ax.set_xlabel("whisker trials before the first hit", fontsize=5)
    ax.set_ylabel("mice", fontsize=5)
    ax.legend(frameon=False, fontsize=4.5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    fig.suptitle("First whisker hit vs the next whisker hits, the preceding auditory hits and the preceding whisker misses "
                 "(time courses: mean ± s.e.m. over mice; dotted = stimulus, response end, +5 s)", fontsize=6, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"021_first_hit.{ext}", dpi=300)
    plt.close(fig)


def run_stats():
    S = M18.sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    D = {}
    keep = None
    for f in sorted(OUT.glob("*.npz")):
        if f.stem not in coh:
            continue
        z = np.load(f, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        if keep is None:
            keep, mn = shared_cols(list(d["cols"]))
        d["keep"] = set(keep)
        D[f.stem] = d
    lab = labels(mn)
    Pe, Ge = engagement(D, coh)
    Pf, Gf, I = first_hit(D, coh)
    D18 = M18.load_all()
    fig_engagement(Ge, lab)
    fig_first_hit(Gf, I, D18, coh, lab)
    json.dump(dict(script="021_engagement_first_hit.py", n_sessions=len(D), nolick_s=NOLICK_S, versions=L.versions()),
              open(L.RES / "provenance" / "021.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    for sc in ("all", "Rplus", "Rminus"):
        g = Ge[Ge[f"q_{sc}"] < 0.05]
        print(f"ENGAGEMENT q<0.05 ({sc}):", len(g), "of", len(Ge))
        print(g.sort_values(f"p_{sc}")[["engagement", "epoch", "measure", f"beta_{sc}", f"p_{sc}", f"q_{sc}"]].head(25).round(4).to_string())
    for sc in ("all", "Rplus", "Rminus"):
        g = Gf[Gf[f"q_{sc}"] < 0.05]
        print(f"FIRST HIT q<0.05 ({sc}):", len(g), "of", len(Gf))
        print(g.sort_values(f"p_{sc}")[["epoch", "reference", "measure", f"diff_{sc}", f"p_{sc}", f"q_{sc}"]].head(25).round(4).to_string())
    print(I.groupby("cohort")[["first_hit_whisker_index", "first_hit_time_min"]].median())


if __name__ == "__main__":
    {"compute": compute, "stats": run_stats}[sys.argv[1]]()
