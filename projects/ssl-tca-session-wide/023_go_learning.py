"""Does the go component change with learning? (user 2026-11-23). Exploratory, uncorrected; per mouse; MW + Welch.

Components: jaw_all clusters (022 / 013 _go tables): main fit cl. 69 and 73, reference fit cl. 49 and 48 (go-like:
licks >> still / jaw moves without lick, present on false alarms). Per session the member components' trial weights,
each in SD units of the active lick trials, 0 = mean of the active still trials, averaged over members.

Within day 0 (active block, position 0..1):
- fifths of the active block per class: whisker lick (hit), auditory lick (hit), false alarm, jaw move without lick;
- slopes over position per class and the contrast d = slope(whisker lick) - slope(auditory lick) (whisker association
  learned on day 0, auditory already known; a general drift / arousal change gives d ~ 0);
- per-mouse OLS on lick trials: z ~ position + whisker + FA + position x whisker + position x FA + lick latency
  (standardised position / latency; the position x whisker term = change of the whisker-lick go signal relative to
  auditory licks over the session, at equal latency);
- whisker licks binned by P(lick | whisker) (learning curve);
- model-free time course: projection of the data on each member's neuron vector per 10-ms bin (no time factor),
  whisker licks of the first vs the last third of the active block, and auditory licks (does the signal rise earlier?);
  onset = first bin >= 50 % of the peak of the mean projection, per mouse.
Expert sessions: each expert component compared with the day-0 members (013 similarity, threshold = 10th percentile of
the members' leave-one-out similarity); matched sessions give whisker / auditory lick ratios and slopes.

Run (haas): python 023_go_learning.py [--n-jobs 40] [--reuse]
     -> tables/go_learning_sessions.parquet, tables/go_learning_stats.csv, figures/aligned_gpu/go2/go_learning_<fit>_cl<k>.png|pdf
"""
import argparse
import importlib
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

C12 = importlib.import_module("012_component_table")
C13 = importlib.import_module("013_recurring_components")
A5 = importlib.import_module("005_build_aligned")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
WH, AU, FA_C, MV = "#f7b519", "#2c2cdb", "0.35", "0.65"
AL = "jaw_all"
TARGETS = [("_splitnn", 69), ("_splitnn", 73), ("_nn", 49), ("_nn", 48)]
NO_JAW = ("AB080", "AB082", "MH026")
FITN = {"_splitnn": "main fit", "_nn": "reference fit"}
NB, MIN_N = 5, 8
CL = {"W": ("lick", "whisker_trial"), "A": ("lick", "auditory_trial"), "FA": ("lick", "no_stim_trial"),
      "move": ("move", None)}


def feat(r):
    return dict(tf=np.array(json.loads(r.time_factor)), ap=json.loads(r.area_profile),
                cp=np.array([getattr(r, k) for k in C13.CLS], float))


def match_expert(comp, members):
    M = [feat(r) for r in members.itertuples()]
    ms = members.session_id.to_numpy()
    loo = [np.mean([C13.similarity(M[i], M[j]) for j in range(len(M)) if ms[j] != ms[i]]) for i in range(len(M))]
    thr = float(np.percentile(loo, 10))
    g = comp[(comp.method == members.method.iloc[0]) & (comp.alignment == AL) & (comp.stage == "expert")].copy()
    g["sim"] = [np.mean([C13.similarity(feat(r), m) for m in M]) for r in g.itertuples()]
    best = g.loc[g.groupby("session_id").sim.idxmax()]
    return best[best.sim >= thr], thr


def session(sid, meta, comps):
    t = pd.read_parquet(A5.OUT_DIR / sid / "trials.parquet").set_index("trial_index")
    d = np.load(A5.OUT_DIR / sid / f"{AL}.npz")
    R, base = d["rate"], d["baseline"]
    keep = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= 0.5
    X = R[:, keep] - base[:, keep, None]
    X = np.nan_to_num(X / (np.nanstd(X, (0, 2), keepdims=True) + 1e-9))
    t = t.loc[d["trial_index"]].reset_index()
    f = np.load(C12.D / sid / f"{AL}_zscore{meta['method']}.npz")
    J = len(f["units_kept"])
    act = (t.epoch == "active").to_numpy()
    jc, ty = t.jaw_class.to_numpy(), t.trial_type.to_numpy()
    cls = np.full(len(t), "", object)
    for k, (j_, ty_) in CL.items():
        cls[act & (jc == j_) & ((ty == ty_) if ty_ else True)] = k
    still = act & (jc == "still")
    lick = np.isin(cls, ["W", "A", "FA"])
    zs, projs = [], []
    for rk, c in comps:
        w = f[f"r{rk}_f0"][:, c - 1].astype(float)
        Fu = f[f"r{rk}_f1"][:, c - 1]
        net = Fu[:J] - Fu[J:] if Fu.shape[0] == 2 * J else Fu
        sd = np.std(w[lick])
        if sd <= 0 or still.sum() < 3:
            continue
        zs.append((w - w[still].mean()) / sd)
        P = np.einsum("u,kut->kt", net, X)                       # trials x bins, model-free time course
        projs.append(P / max(np.abs(P[lick].mean(0)).max(), 1e-12))
    if not zs:
        return None
    z, P = np.mean(zs, 0), np.mean(projs, 0)
    pos = np.full(len(t), np.nan)
    pos[act] = np.arange(act.sum()) / max(act.sum() - 1, 1)
    out = dict(**meta, n_W=int((cls == "W").sum()), n_A=int((cls == "A").sum()), n_FA=int((cls == "FA").sum()))
    for k in CL:
        m = cls == k
        out[f"slope_{k}"] = float(np.polyfit(pos[m], z[m], 1)[0]) if m.sum() >= MIN_N else np.nan
        b = np.minimum((pos[m] * NB).astype(int), NB - 1)
        out[f"f_{k}"] = [float(z[m][b == i].mean()) if (b == i).sum() >= 2 else np.nan for i in range(NB)]
        out[f"mean_{k}"] = float(z[m].mean()) if m.sum() >= 3 else np.nan
        for th, sel in (("first", pos < 1 / 3), ("last", pos > 2 / 3)):
            out[f"{th}_{k}"] = float(z[m & sel].mean()) if (m & sel).sum() >= 3 else np.nan
    out["d"] = out["slope_W"] - out["slope_A"]
    out["d_FA"] = out["slope_FA"] - out["slope_A"]
    for th in ("first", "last"):
        out[f"ratio_{th}"] = (out[f"{th}_W"] / out[f"{th}_A"]) if out[f"{th}_A"] and out[f"{th}_A"] > 0 else np.nan
    # OLS on lick trials
    m = lick & np.isfinite(t.lick_latency.to_numpy(float))
    if m.sum() >= 30 and (cls[m] == "W").sum() >= MIN_N and (cls[m] == "A").sum() >= MIN_N:
        p_ = pos[m]
        p_ = (p_ - p_.mean()) / max(p_.std(), 1e-9)
        lat = t.lick_latency.to_numpy(float)[m]
        lat = (lat - lat.mean()) / max(lat.std(), 1e-9)
        isW, isF = (cls[m] == "W").astype(float), (cls[m] == "FA").astype(float)
        Z = np.c_[np.ones(m.sum()), p_, isW, isF, p_ * isW, p_ * isF, lat]
        b = np.linalg.lstsq(Z, z[m], rcond=None)[0]
        out.update(b_pos=b[1], b_W=b[2], b_FA=b[3], b_posW=b[4], b_posFA=b[5], b_lat=b[6])
    # performance bins (whisker licks)
    pw = t.p_wh_i.to_numpy(float)
    edges = np.array([0, 0.25, 0.5, 0.75, 1.01])
    mw = cls == "W"
    out["perf_W"] = [float(z[mw & (pw >= edges[i]) & (pw < edges[i + 1])].mean())
                     if (mw & (pw >= edges[i]) & (pw < edges[i + 1])).sum() >= 3 else np.nan for i in range(4)]
    # time courses
    tc = d["t_centres"] * 1e3
    out["tc"] = tc.tolist()
    for nm, sel in (("W_first", mw & (pos < 1 / 3)), ("W_last", mw & (pos > 2 / 3)), ("A", cls == "A"),
                    ("FA", cls == "FA"), ("still", still)):
        out[f"tc_{nm}"] = P[sel].mean(0).tolist() if sel.sum() >= 3 else [np.nan] * len(tc)

    def onset(y):
        y = np.asarray(y)
        if not np.isfinite(y).all() or y.max() <= 0:
            return np.nan
        return float(tc[np.argmax(y >= 0.5 * y.max())])
    out["onset_W_first"], out["onset_W_last"] = onset(out["tc_W_first"]), onset(out["tc_W_last"])
    out["onset_A"] = onset(out["tc_A"])
    return out


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(stats.wilcoxon(v).pvalue) if len(v) >= 5 and np.any(v != 0) else np.nan


def two(a, b):
    a, b = (np.asarray(x, float) for x in (a, b))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def sem(x, axis=0):
    x = np.asarray(x, float)
    return np.nanstd(x, axis) / np.sqrt(np.maximum(np.sum(np.isfinite(x), axis) - 1, 1))


def dots(ax, S, col, title, zero=0.0, ylab=""):
    for i, coh in enumerate(("R+", "R-")):
        v = S.loc[S.cohort == coh, col].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.scatter(i + (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35, v, s=3, color=COH[coh], lw=0)
        if len(v):
            ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="k", lw=1)
            ax.text(i, 0.99, f"n={len(v)}\np={wil(v - zero):.3f}", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=4.5)
    mw, we = two(S.loc[S.cohort == "R+", col], S.loc[S.cohort == "R-", col])
    ax.axhline(zero, color="0.6", lw=0.4)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_xlim(-0.6, 1.6)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    ax.set_ylabel(ylab, fontsize=5)
    ax.set_title(f"{title}\nR+ vs R-: MW {mw:.3f}, Welch {we:.3f}", fontsize=5.5)
    return mw, we


def page(S, meth, k, nexp, out):
    s0, se = S[S.stage == "learning"], S[S.stage == "expert"]
    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(3, 4, hspace=0.95, wspace=0.65)
    xb = np.arange(NB)
    for j, coh in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        h = s0[s0.cohort == coh]
        for kk, col, ls in (("W", WH, "-"), ("A", AU, "-"), ("FA", FA_C, "--"), ("move", MV, ":")):
            Y = np.array(h[f"f_{kk}"].tolist(), float)
            if len(Y):
                ax.errorbar(xb + {"W": 0, "A": 0.05, "FA": 0.1, "move": 0.15}[kk], np.nanmean(Y, 0), sem(Y), color=col,
                            ls=ls, marker="o", ms=2.3, lw=0.8, elinewidth=0.4,
                            label={"W": "whisker lick", "A": "auditory lick", "FA": "false alarm",
                                   "move": "jaw move, no lick"}[kk])
        ax.set_xticks(xb, [str(i + 1) for i in xb])
        ax.set(xlabel="Fifth of the active block", ylabel="Weight (SD; 0 = active still)")
        ax.set_title(f"A  {coh}: session course (day 0, {len(h)} mice)", fontsize=6, color=COH[coh])
        if j == 0:
            ax.legend(frameon=False, fontsize=4.5)
    dots(fig.add_subplot(gs[1, 0]), s0, "d", "B  d = slope W lick - A lick", ylab="d (per block)")
    dots(fig.add_subplot(gs[1, 1]), s0, "b_posW", "C  position x whisker\n(OLS, latency controlled)", ylab="coef.")
    ax = fig.add_subplot(gs[1, 2])
    for i, coh in enumerate(("R+", "R-")):
        h = s0[s0.cohort == coh]
        for kk, x0 in (("ratio_first", 0), ("ratio_last", 1)):
            v = h[kk].to_numpy(float)
            ax.errorbar(i * 2.6 + x0, np.nanmedian(v), sem(v), color=COH[coh], marker="o" if x0 else "s", ms=3,
                        mfc="white" if x0 == 0 else COH[coh], elinewidth=0.5)
        pr = h[["ratio_first", "ratio_last"]].dropna()
        p_ = stats.wilcoxon(pr.ratio_first, pr.ratio_last).pvalue if len(pr) >= 5 else np.nan
        ax.text(i * 2.6 + 0.5, 0.99, f"p={p_:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.5)
        v = se[se.cohort == coh]["mean_W"] / se[se.cohort == coh]["mean_A"]
        if len(v):
            ax.errorbar(i * 2.6 + 1.8, np.nanmedian(v), sem(v), color=COH[coh], marker="D", ms=3, elinewidth=0.5)
    ax.axhline(1, color="0.6", lw=0.4)
    ax.set_xticks([0, 1, 1.8, 2.6, 3.6, 4.4], ["1st", "last", "exp.", "1st", "last", "exp."], fontsize=4.5)
    ax.set(ylabel="W lick / A lick weight (median)")
    ax.set_title("D  whisker / auditory lick weight\n(R+ left, R- right; first vs last third)", fontsize=5.5)
    ax = fig.add_subplot(gs[1, 3])
    for coh in ("R+", "R-"):
        Y = np.array(s0[s0.cohort == coh].perf_W.tolist(), float)
        ax.errorbar(np.arange(4) * 0.25 + 0.125, np.nanmean(Y, 0), sem(Y), color=COH[coh], marker="o", ms=2.3,
                    elinewidth=0.4)
    ax.set(xlabel="P(lick | whisker)", ylabel="W-lick weight (SD)")
    ax.set_title("E  whisker licks by\nperformance level", fontsize=5.5)
    for j, coh in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[2, j])
        h = s0[s0.cohort == coh]
        tc = np.array(h.tc.iloc[0]) if len(h) else np.arange(60)
        for nm, col, ls in (("W_first", WH, ":"), ("W_last", WH, "-"), ("A", AU, "-"), ("FA", FA_C, "--"),
                            ("still", MV, "-")):
            Y = np.array(h[f"tc_{nm}"].tolist(), float)
            if len(Y):
                mu = np.nanmean(Y, 0)
                ax.plot(tc, mu, color=col, ls=ls, lw=0.8, label=nm.replace("_", " "))
                ax.fill_between(tc, mu - sem(Y), mu + sem(Y), color=col, alpha=0.15, lw=0)
        ax.axvline(0, color="0.6", lw=0.4)
        ax.set(xlabel="Time from jaw onset (ms)", ylabel="Projection (norm.)")
        ax.set_title(f"F  {coh}: model-free time course", fontsize=5.5, color=COH[coh])
        if j == 0:
            ax.legend(frameon=False, fontsize=4.2)
    s0 = s0.assign(d_onset=s0.onset_W_last - s0.onset_W_first)
    dots(fig.add_subplot(gs[2, 2]), s0, "d_onset", "G  onset W lick\nlast - first third (ms)", ylab="ms")
    dots(fig.add_subplot(gs[2, 3]), se, "d", f"H  expert sessions: d\n({nexp} matched)", ylab="d (per block)")
    fig.suptitle(f"Go component and learning: jaw_all, {FITN[meth]} cl. {k} (whisker day 0 unless stated; weights in "
                 "SD of active licks, 0 = active still trials). Exploratory, uncorrected.", fontsize=6.5, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"go_learning{meth}_cl{k}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    tdir = L.OUT / "tables"
    out = L.OUT / "figures" / "aligned_gpu" / "go2"
    out.mkdir(parents=True, exist_ok=True)
    ses = pd.read_csv(tdir / "sessions.csv").set_index("session_id")
    mem = pd.read_parquet(tdir / "cluster_members_go.parquet")
    comp = pd.read_parquet(tdir / "components_go.parquet")
    comp = comp[comp.error.isna()] if "error" in comp else comp
    f = tdir / "go_learning_sessions.parquet"
    if a.reuse and f.exists():
        S = pd.read_parquet(f)
    else:
        jobs, nexp = [], {}
        for meth, k in TARGETS:
            m = mem[(mem.method == meth) & (mem.alignment == AL) & (mem.cluster == k) & (mem.stage == "learning")]
            ex, thr = match_expert(comp, m)
            nexp[(meth, k)] = len(ex)
            print(meth, k, "expert matched", len(ex), "thr", round(thr, 3))
            for src in (m, ex):
                for sid, g in src.groupby("session_id"):
                    r = ses.loc[sid]
                    if r.mouse_id in NO_JAW:
                        continue
                    meta = dict(session_id=sid, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, method=meth, cluster=k)
                    jobs.append(delayed(session)(sid, meta, [(int(x.rank), int(x.comp)) for x in g.itertuples()]))
        res = Parallel(n_jobs=a.n_jobs, verbose=2)(jobs)
        S = pd.DataFrame([x for x in res if x is not None])
        S.to_parquet(f, index=False)
    rows = []
    for (meth, k), g in S.groupby(["method", "cluster"]):
        page(g, meth, k, int((g.stage == "expert").sum()), out)
        for stg, h in g.groupby("stage"):
            for v in ("d", "d_FA", "slope_W", "slope_A", "slope_FA", "b_posW", "b_posFA", "b_pos", "b_lat"):
                r = dict(method=meth, cluster=k, stage=stg, var=v)
                for coh in ("R+", "R-"):
                    x = h.loc[h.cohort == coh, v]
                    r[f"{coh}_median"], r[f"{coh}_p"], r[f"{coh}_n"] = float(x.median()), wil(x), int(x.notna().sum())
                r["mw_p"], r["welch_p"] = two(h.loc[h.cohort == "R+", v], h.loc[h.cohort == "R-", v])
                rows.append(r)
            for coh in ("R+", "R-"):
                pr = h[h.cohort == coh][["ratio_first", "ratio_last"]].dropna()
                rows.append(dict(method=meth, cluster=k, stage=stg, var=f"ratio_last_minus_first_{coh}",
                                 **{f"{coh}_median": float((pr.ratio_last - pr.ratio_first).median()) if len(pr) else np.nan,
                                    f"{coh}_p": wil(pr.ratio_last - pr.ratio_first), f"{coh}_n": len(pr)}))
    st = pd.DataFrame(rows)
    st.to_csv(tdir / "go_learning_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
