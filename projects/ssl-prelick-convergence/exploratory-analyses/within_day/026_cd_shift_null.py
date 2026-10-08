"""026 -- Within-session whisker-hit drift on the reward-lick CD vs CDs rebuilt from linearly shifted labels (user 2026-10-08:
"compare my effects with those of CD obtained from linear shifted versions ... before looking at effects relative to another
condition"; native scale, unit-norm CD, with and without the per-unit scaling).
Per session (002 events and units; >= 4 WH, >= 8 AH, >= 8 SL; stable units, mean pre-lick rate >= 0.1 Hz):
  real: 5-fold CV over the AH + SL events, units z-scored on the training folds, unit-norm axis u = mean AH - mean SL; every WH
  event projected on each fold's u and averaged over folds (WH never train the axis); beta_WH = OLS slope of the WH projections
  against normalised session time tau (all events, 0 -> 1). Native scale: z units ('unit'); per unit: / sqrt(N) ('pu').
  null: the AH / SL labels shifted against the time-ordered AH + SL events by k = 10-50 % of them (non-wrapping, both directions,
  N_SHIFT shifts keeping >= 8 events per class; as the 135 shift null of the hit / miss project), the axis rebuilt the same way,
  the same WH projected; beta_WH^(k). Stored: null mean / SD, excess = real - null mean, one-sided percentile.
  Reference classes are not used here (raw WH drift on the axis); the axis separation (held-out AH - SL along u) is stored too.
Statistics (learners; unit = session): real vs null mean (Wilcoxon, paired), excess vs 0 (Wilcoxon), R+ vs R- (Mann-Whitney).
Extension (2026-10-08): the held-out SL projections give SL slopes on the real and shifted CDs; contrast WH - SL; binned (5 tau bins)
WH and SL projections on the real CD and on the shifted CDs (mean over shifts); the same per area group (CD from the group's units,
>= 5 units, held-out AH-vs-SL d' >= 0.3, as 015).
Outputs: within_day/<ref>/cd_shift_null/{sessions.csv, sessions_areas.csv, bins.csv, tests.csv, cd_shift_null_learners.{png,pdf,svg}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../026_cd_shift_null.py [--n-proc 40]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
m51 = importlib.import_module("051_roc_prelick")
m002 = importlib.import_module("002_trial_slopes")
OUT = m51.WITHIN / "cd_shift_null"
N_SHIFT, FRAC, N_BINS = 50, (0.10, 0.50), 5           # 50 shifts (as 135; was 20 until 2026-10-08)
EXAMPLES = ("AB138", "MH008")                                  # example sessions (learning day): per-trial projections saved
COH = {"R+": "#00B400", "R-": "#C800C8"}


def cv_projection(X, y_idx, y, w_idx, seed=0):
    """X (events, units); axis from events y_idx with labels y (1 = AH). Returns WH projections (mean over folds), held-out
    projections of the y_idx events (each from the fold that held it out), held-out separation and d'"""
    from sklearn.model_selection import StratifiedKFold
    pw, sep = [], []
    ph = np.full(len(y_idx), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(y_idx, y):
        A = X[y_idx[tr]]; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
        z = lambda q: (q - mu) / sd
        d = z(A[y[tr] == 1]).mean(0) - z(A[y[tr] == 0]).mean(0)
        u = d / max(np.linalg.norm(d), 1e-12)
        pw.append(z(X[w_idx]) @ u)
        ph[te] = z(X[y_idx[te]]) @ u
    a, b = ph[y == 1], ph[y == 0]
    spd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return np.mean(pw, 0), ph, float(a.mean() - b.mean()), float((a.mean() - b.mean()) / spd) if spd > 0 else np.nan


def binned(tau, v, nb=N_BINS):
    b = np.minimum((tau * nb).astype(int), nb - 1)
    return np.array([v[b == i].mean() if (b == i).any() else np.nan for i in range(nb)])


def one_axis(Xe, lab, tau, seed):
    """real and shifted-label CDs for one unit set; slopes of WH and held-out SL, contrast, binned projections (z units)"""
    w_idx = np.where(lab == "WH")[0]
    y_idx = np.where(np.isin(lab, ["AH", "FA"]))[0]
    y = (lab[y_idx] == "AH").astype(int)
    pw, ph, sep, dp = cv_projection(Xe, y_idx, y, w_idx, seed)
    sl = y_idx[y == 0]
    r = dict(sep_unit=sep, dprime=dp, bWH=np.polyfit(tau[w_idx], pw, 1)[0], bSL=np.polyfit(tau[sl], ph[y == 0], 1)[0],
             binsWH=binned(tau[w_idx], pw), binsSL=binned(tau[sl], ph[y == 0]),
             trWH=(tau[w_idx], pw), trSL=(tau[sl], ph[y == 0], sl), trAH=(tau[y_idx[y == 1]], ph[y == 1], y_idx[y == 1]))
    nsum_w, nsum_s, ncnt_s = np.zeros(len(w_idx)), {}, {}                 # per-event sums over shifts (SL and AH events)
    n = len(y_idx); ks = np.unique(np.linspace(FRAC[0] * n, FRAC[1] * n, N_SHIFT // 2).astype(int))
    nWH, nSL, bWH, bSL = [], [], [], []
    for k in np.r_[ks, -ks]:
        i = np.arange(max(0, -k), min(n, n - k))                    # label i paired with activity i + k
        ys, xi = y[i], y_idx[i + k]
        if min(ys.sum(), (1 - ys).sum()) < m002.MIN_EV:
            continue
        pk, phk, _, _ = cv_projection(Xe, xi, ys, w_idx, seed)
        slk = lab[xi] == "FA"                                       # real spontaneous licks among the shifted-axis events
        nWH.append(np.polyfit(tau[w_idx], pk, 1)[0]); nSL.append(np.polyfit(tau[xi[slk]], phk[slk], 1)[0])
        nsum_w += pk
        for ev, v in zip(xi, phk):
            nsum_s[ev] = nsum_s.get(ev, 0.0) + v; ncnt_s[ev] = ncnt_s.get(ev, 0) + 1
        bWH.append(binned(tau[w_idx], pk)); bSL.append(binned(tau[xi[slk]], phk[slk]))
    if len(nWH) < 5:
        return None
    r.update(nWH=np.asarray(nWH), nSL=np.asarray(nSL), nbinsWH=np.nanmean(bWH, 0), nbinsSL=np.nanmean(bSL, 0),
             trWH_null=nsum_w / len(nWH), trSL_null=np.array([nsum_s[e] / ncnt_s[e] if e in ncnt_s else np.nan for e in r["trSL"][2]]),
             trAH_null=np.array([nsum_s[e] / ncnt_s[e] if e in ncnt_s else np.nan for e in r["trAH"][2]]))
    return r


def summarise(r, N):
    s = np.sqrt(N)
    nC = r["nWH"] - r["nSL"]
    out = dict(n_units=N, sep_unit=r["sep_unit"], dprime=r["dprime"], n_null=len(r["nWH"]),
               beta_unit=r["bWH"], null_unit=r["nWH"].mean(), null_sd_unit=r["nWH"].std(), excess_unit=r["bWH"] - r["nWH"].mean(),
               pct=float((r["nWH"] < r["bWH"]).mean()))
    for key, real, null in (("", r["bWH"], r["nWH"]), ("SL", r["bSL"], r["nSL"]), ("C", r["bWH"] - r["bSL"], nC)):
        out[f"beta{key}_pu"], out[f"null{key}_pu"] = real / s, null.mean() / s
        out[f"excess{key}_pu"] = (real - null.mean()) / s
    return out


def session(args):
    sid, f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["trial_start"].astype(float)[o]
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return None
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    if ok.sum() < m002.MIN_UNITS:
        return None
    tau = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    rows, bins, trials = [], [], []
    groups = [("whole_brain", ok)] + [(g, ok & (K.area_group == g).to_numpy()) for g in sorted(K.area_group.dropna().unique())]
    for area, m in groups:
        if m.sum() < m002.MIN_UNITS:
            continue
        r = one_axis(X[m].T, lab, tau, seed)
        if r is None or (area != "whole_brain" and not (np.isfinite(r["dprime"]) and r["dprime"] >= m002.MIN_DPRIME)):
            continue
        rows.append(dict(meta, area=area, n_WH=int((lab == "WH").sum()), n_AH=int((lab == "AH").sum()), n_SL=int((lab == "FA").sum()),
                         **summarise(r, int(m.sum()))))
        if area == "whole_brain":
            s_ = np.sqrt(m.sum())
            if meta["mouse_id"] in EXAMPLES and meta["stage"] == "learning":
                for c, (tt, real), null in (("WH", r["trWH"], r["trWH_null"]), ("SL", r["trSL"][:2], r["trSL_null"]),
                                            ("AH", r["trAH"][:2], r["trAH_null"])):
                    trials += [dict(meta, cls=c, tau=a_, real=b_ / s_, null=c_ / s_) for a_, b_, c_ in zip(tt, real, null)]
            for c in ("WH", "SL"):
                for i in range(N_BINS):
                    bins.append(dict(meta, cls=c, bin=i, real=r[f"bins{c}"][i] / s_, null=r[f"nbins{c}"][i] / s_))
    return (rows, bins, trials) if rows and rows[0]["area"] == "whole_brain" else None


def tests(D):
    rows = []
    for st in ("learning", "expert"):
        for sc in ("unit", "pu"):
            g = {c: D[(D.cohort == c) & (D.stage == st)] for c in COH}
            for c in COH:
                d = g[c]
                rows.append(dict(stage=st, scale=sc, cohort=c, n=len(d), real=d[f"beta_{sc}"].mean(), null=d[f"null_{sc}"].mean(),
                                 excess=d[f"excess_{sc}"].mean(), sem_excess=d[f"excess_{sc}"].sem(),
                                 p_real_vs0=stats.wilcoxon(d[f"beta_{sc}"]).pvalue if len(d) > 5 else np.nan,
                                 p_real_vs_null=stats.wilcoxon(d[f"beta_{sc}"], d[f"null_{sc}"]).pvalue if len(d) > 5 else np.nan,
                                 p_excess_vs0=stats.wilcoxon(d[f"excess_{sc}"]).pvalue if len(d) > 5 else np.nan))
            a, b = g["R+"], g["R-"]
            rows.append(dict(stage=st, scale=sc, cohort="R+ vs R-",
                             p_real=stats.mannwhitneyu(a[f"beta_{sc}"], b[f"beta_{sc}"]).pvalue,
                             p_null=stats.mannwhitneyu(a[f"null_{sc}"], b[f"null_{sc}"]).pvalue,
                             p_excess=stats.mannwhitneyu(a[f"excess_{sc}"], b[f"excess_{sc}"]).pvalue))
    return pd.DataFrame(rows)


def figure(D, T, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    d0 = D[D.stage == "learning"]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9), gridspec_kw=dict(wspace=0.7))
    for j, sc in enumerate(("pu", "unit")):
        ax = axs[2 * j]
        for k, c in enumerate(COH):
            d = d0[d0.cohort == c]; x0 = k * 2.2
            for xx, col in ((x0, f"null_{sc}"), (x0 + 1, f"beta_{sc}")):
                ax.errorbar(xx, d[col].mean(), d[col].sem(), fmt="o", ms=3.5, color=COH[c], mfc="white" if "null" in col else COH[c], lw=1, capsize=0)
            ax.plot([x0, x0 + 1], [d[f"null_{sc}"].mean(), d[f"beta_{sc}"].mean()], color=COH[c], lw=0.8)
            r = T[(T.stage == "learning") & (T.scale == sc) & (T.cohort == c)].iloc[0]
            ax.text(x0 + 0.5, 1.0, f"p = {r.p_real_vs_null:.3g}", color=COH[c], transform=ax.get_xaxis_transform(), ha="center", fontsize=5)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(["shifted", "real"] * 2, fontsize=5); ax.set_xlim(-0.5, 3.7)
        ax.set_ylabel("WH slope on the CD\n" + ("(SD per neuron / session)" if sc == "pu" else "(z units / session)"))
        ax.set_title("Real vs shifted-label CDs\n" + ("per neuron (÷ √N)" if sc == "pu" else "unit-norm CD, no ÷ √N"), pad=12)
        ax = axs[2 * j + 1]
        for k, c in enumerate(COH):
            d = d0[d0.cohort == c][f"excess_{sc}"]
            r = T[(T.stage == "learning") & (T.scale == sc) & (T.cohort == c)].iloc[0]
            ax.errorbar(k, d.mean(), d.sem(), fmt="o", ms=3.8, color=COH[c], mfc=COH[c] if r.p_excess_vs0 < 0.05 else "white", lw=1.1, capsize=0)
        rr = T[(T.stage == "learning") & (T.scale == sc) & (T.cohort == "R+ vs R-")].iloc[0]
        ax.text(0.5, 1.0, f"R+ vs R− p = {rr.p_excess:.3g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=5)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2))); ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R−"]); ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("excess (real − shifted)"); ax.set_title("Beyond session time\n(filled: ≠ 0)", pad=12)
    fig.suptitle("Axis 1 (pre-lick, day 0, learners): whisker-hit drift on the reward-lick CD vs CDs from linearly shifted AH / SL labels", fontsize=6.5, y=1.22)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cd_shift_null_learners.{ext}", dpi=300, bbox_inches="tight")


def main(a):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    m61 = importlib.import_module("061_roc_prelick_learners")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.replot:
        D = pd.read_csv(OUT / "sessions.csv")
    else:
        W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
        W = W[W.cohort.isin(["R+", "R-"])].copy()               # area_group carried for the per-area axes
        W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
        ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
        jobs = []
        for i, r in enumerate(ss.itertuples()):
            f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
            if f.exists():
                Ws = W[W.session_id == r.session_id]
                jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0],
                                                   stage=Ws.stage.iloc[0]), Ws[["electrode_group", "cluster_id", "quality_label", "area_group"]], i))
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            res = [r for r in pool.imap_unordered(session, jobs) if r]
        DA = pd.DataFrame([x for r in res for x in r[0]]); DA.to_csv(OUT / "sessions_areas.csv", index=False)
        pd.DataFrame([x for r in res for x in r[1]]).to_csv(OUT / "bins.csv", index=False)
        pd.DataFrame([x for r in res for x in r[2]]).to_csv(OUT / "example_trials.csv", index=False)
        D = DA[DA.area == "whole_brain"].copy(); D.to_csv(OUT / "sessions.csv", index=False)
    D = m61.learner_filter(D)
    T = tests(D); T.to_csv(OUT / "tests.csv", index=False)
    print(T.round(4).to_string())
    figure(D, T, OUT)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
