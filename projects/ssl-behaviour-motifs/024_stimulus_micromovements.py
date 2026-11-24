"""024 -- stimulus-locked micro-movements at 200 Hz (haas, full run sessions).

Usage: python 024_stimulus_micromovements.py compute   (per session -> cache/full/024/<sid>.npz)
       python 024_stimulus_micromovements.py stats     (tables/024_*.csv, figures/024_*.png|pdf|svg)

Data: NWB_ks4 DLC series (ephys_utilities dlc_helpers.load_dlc; physical units; no likelihood mask; NWB reconverted
2026-11-20 WITHOUT dlc2kinematics smoothing), each series on its own camera's timestamps, linearly resampled on a 5 ms
grid from -0.5 to +1.0 s around the stimulus (trial start_time = stimulus onset for whisker, auditory and no-stim
trials); samples > 7.5 ms from a real frame are NaN.
Signals: side jaw_distance (mm), jaw_angle (deg), nose tip x / y (mm), nose speed (mm/s, keypoint_speed), tongue out
(tongue likelihood > 0.6); top whisker_angle (deg), whisker speed (|whisker_velocity|, deg/s), top nose tip x / y,
top nose speed, top particle x / y (whisker particle). Positions / angles: minus the trial's mean over -200..0 ms.
Pre-lick truncation: per trial, samples from the first of (jaw onset [dlc_jaw_onset_times], corrected first lick, first
piezo lick, first tongue-out frame) after the stimulus are NaN in the 'pre-lick' version. Trials with a piezo lick or
tongue-out in -1..0 s are dropped.
Kept trials (bm_lib.load_trials conventions) of the 76 full-run sessions.
Analyses (unit = session = mouse; cluster-mass permutation over 5 ms bins, |t| > 2.0, 2000 permutations; window
-0.1..0.6 s):
 1. evoked responses per condition (WM, WH pre-lick, AM, AH pre-lick, CR, FA pre-lick), per cohort;
 2. detection without report: WM - CR (paired, sign-flip) per cohort; R+ vs R- on WM - CR (Welch, label permutation);
 3. pre-lick hit vs miss: WH (pre-lick) - WM per cohort; single-trial AUC of the 0-100 / 0-200 ms mean (hits with
    jaw onset after the window end only) per mouse -> sign-flip vs 0.5;
 4. learning: WM - CR before vs after the learning trial (learners with an LT; whisker trials) per cohort.
Caveat: the whisker stimulus deflects the whisker directly; whisker signals in the first ~25 ms can be passive.
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
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402  (NAS checkout first, via bm_lib)

_sp = importlib.util.spec_from_file_location("m18", HERE / "018_report_examples.py")
M18 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(M18)

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
OUT = L.CACHE / "full" / "024"
DT = 0.005
TGRID = np.round(np.arange(-0.5, 1.0 + 1e-9, DT), 4)
BASE = (TGRID >= -0.2) & (TGRID < 0)
SIGNALS = [("side", "jaw_distance", "jaw opening (mm)", True), ("side", "jaw_angle", "jaw angle (deg)", True),
           ("side", "nose_tip_x", "nose x side (mm)", True), ("side", "nose_tip_y", "nose y side (mm)", True),
           ("side", "nose_speed", "nose speed side (mm/s)", False), ("side", "tongue", "tongue out (P)", False),
           ("top", "whisker_angle", "whisker angle (deg)", True), ("top", "whisker_speed", "whisker speed (deg/s)", False),
           ("top", "top_nose_tip_x", "nose x top (mm)", True), ("top", "top_nose_tip_y", "nose y top (mm)", True),
           ("top", "top_nose_speed", "nose speed top (mm/s)", False), ("top", "top_particle_x", "whisker particle x (mm)", True),
           ("top", "top_particle_y", "whisker particle y (mm)", True)]
SIGN = [s for _, s, _, _ in SIGNALS]
LAB = {s: lab for _, s, lab, _ in SIGNALS}
CONDS = {"WM": ("whisker", 0), "WH": ("whisker", 1), "AM": ("auditory", 0), "AH": ("auditory", 1), "CR": ("no_stim", 0),
         "FA": ("no_stim", 1)}
WIN = (-0.1, 0.6)
NPERM = 2000
LT_TABLE = L.ROOT / "combined_results_ks4" / "ssl-learning-trial-identification" / "artifacts" / "017_learning_trials_final.csv"
rng = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------ compute
def resample(x, ts, events):
    """trials x TGRID by linear interpolation on the camera timestamps; NaN > 7.5 ms from a frame"""
    out = np.full((len(events), len(TGRID)), np.nan, np.float32)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return out
    tq = events[:, None] + TGRID[None, :]
    y = np.interp(tq.ravel(), ts[ok], x[ok], left=np.nan, right=np.nan).reshape(tq.shape)
    i = np.clip(np.searchsorted(ts, tq.ravel()), 1, len(ts) - 1)
    near = np.minimum(np.abs(tq.ravel() - ts[i - 1]), np.abs(ts[i] - tq.ravel())).reshape(tq.shape)
    y[near > 0.0075] = np.nan
    return y.astype(np.float32)


def one(sid):
    nwb = L.NWB_DIR / f"{sid}.nwb"
    V = dh.load_dlc(str(nwb))
    tr = L.load_trials(sid)
    k = tr[tr.keep].reset_index(drop=True)
    ev = k.start_time.to_numpy()
    X = np.full((len(k), len(SIGNALS), len(TGRID)), np.nan, np.float32)
    tongue_t = np.array([])
    for si, (view, s, _, _) in enumerate(SIGNALS):
        if view not in V:
            continue
        v = V[view]
        ts = v.timestamps
        if s == "nose_speed":
            x = dh.keypoint_speed(v.data["nose_tip_x"].to_numpy(), v.data["nose_tip_y"].to_numpy(), ts)
        elif s == "top_nose_speed":
            x = dh.keypoint_speed(v.data["top_nose_tip_x"].to_numpy(), v.data["top_nose_tip_y"].to_numpy(), ts)
        elif s == "whisker_speed":
            x = np.abs(v.data["whisker_velocity"].to_numpy())
        elif s == "tongue":
            x = dh.tongue_presence(v)
            tongue_t = ts[np.nan_to_num(x) > 0.5]
        elif s in v.data:
            x = v.data[s].to_numpy()
        else:
            continue
        X[:, si] = resample(x.astype(float), ts, ev)
    # baseline subtraction for positions / angles
    for si, (_, s, _, rel) in enumerate(SIGNALS):
        if rel:
            X[:, si] -= np.nanmean(X[:, si][:, BASE], 1, keepdims=True)
    # first movement / lick event after the stimulus
    licks = L.piezo_licks(sid)
    try:
        jo = pd.read_parquet(M18.INP / f"{sid}_jaw_onsets.parquet")
        jo_abs = dict(zip(np.round(jo.start_time, 3), jo.jaw_onset_abs))
    except Exception:  # noqa: BLE001
        jo_abs = {}
    first = np.full(len(k), np.inf)
    pre_lick = np.zeros(len(k), bool)
    allev = np.sort(np.r_[licks, tongue_t])
    for i, t0 in enumerate(ev):
        c = []
        j = np.searchsorted(allev, t0)
        if j < len(allev):
            c.append(allev[j] - t0)
        if np.isfinite(k.first_lick_time.iloc[i]):
            c.append(k.first_lick_time.iloc[i] - t0)
        ja = jo_abs.get(round(t0, 3), np.nan)
        if np.isfinite(ja) and ja > t0:
            c.append(ja - t0)
        first[i] = min(c) if c else np.inf
        pre_lick[i] = np.any((allev >= t0 - 1.0) & (allev < t0))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{sid}.npz", X=X, stim=k.stim.to_numpy().astype(str), lick=k.lick_flag.to_numpy().astype(int),
                        first_event=first, lick_before=pre_lick, whisker_trial_index=k.whisker_trial_index.to_numpy())
    return sid


def compute():
    sids = list(M18.sessions().session_id)
    with ProcessPoolExecutor(int(os.environ.get("BM_WORKERS", 12))) as ex:
        for s in ex.map(one, sids):
            print("done", s, flush=True)


# ------------------------------------------------------------------------------------------------ stats
def bouts(b):
    on = np.flatnonzero(np.diff(np.r_[0, b.astype(int)]) == 1)
    off = np.flatnonzero(np.diff(np.r_[b.astype(int), 0]) == -1) + 1
    return list(zip(on, off))


def welch_t(A, B):
    with np.errstate(all="ignore"):
        return np.nan_to_num((np.nanmean(A, 0) - np.nanmean(B, 0)) /
                             np.sqrt(np.nanvar(A, 0, ddof=1) / np.sum(np.isfinite(A), 0) + np.nanvar(B, 0, ddof=1) / np.sum(np.isfinite(B), 0)))


def one_t(Dm):
    with np.errstate(all="ignore"):
        return np.nan_to_num(np.nanmean(Dm, 0) / (np.nanstd(Dm, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Dm), 0))))


def clusters(t, thr=2.0):
    out = []
    for sgn in (1, -1):
        for a, b in bouts(sgn * t > thr):
            out.append((a, b, float(np.sum(t[a:b]))))
    return out


def cluster_paired(Dm, thr=2.0):
    """sign-flip cluster-mass test of per-mouse differences (mice x bins)"""
    Dm = Dm[np.isfinite(Dm).any(1)]
    t0 = one_t(Dm)
    obs = clusters(t0, thr)
    mx = []
    for _ in range(NPERM):
        s = rng.choice([-1, 1], size=(len(Dm), 1))
        c = clusters(one_t(Dm * s), thr)
        mx.append(max([abs(x[2]) for x in c] or [0]))
    mx = np.array(mx)
    return t0, [(a, b, m, float((np.sum(mx >= abs(m)) + 1) / (NPERM + 1))) for a, b, m in obs]


def cluster_two(A, B, thr=2.0):
    X = np.vstack([A, B])
    lab = np.r_[np.ones(len(A), bool), np.zeros(len(B), bool)]
    t0 = welch_t(A, B)
    obs = clusters(t0, thr)
    mx = []
    for _ in range(NPERM):
        p = rng.permutation(lab)
        c = clusters(welch_t(X[p], X[~p]), thr)
        mx.append(max([abs(x[2]) for x in c] or [0]))
    mx = np.array(mx)
    return t0, [(a, b, m, float((np.sum(mx >= abs(m)) + 1) / (NPERM + 1))) for a, b, m in obs]


def signflip(x, n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    s = rng.choice([-1, 1], size=(n, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= abs(x.mean())) + 1) / (n + 1))


def run_stats():
    from sklearn.metrics import roc_auc_score
    S = M18.sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    lt = pd.read_csv(LT_TABLE)
    ltd = dict(zip(lt.session_id, lt.lt_final))
    W = (TGRID >= WIN[0]) & (TGRID <= WIN[1])
    tw = TGRID[W]
    M = {}       # (cond, cohort) -> list of mice x sig x bins (pre-lick truncated)
    DIFF = {}    # (contrast, cohort) -> list of mice x sig x bins
    AUC = []
    LRN = {}
    for f in sorted(OUT.glob("*.npz")):
        sid = f.stem
        if sid not in coh:
            continue
        z = np.load(f)
        X, stim, lick, first, lb = z["X"].astype(float), z["stim"], z["lick"], z["first_event"], z["lick_before"]
        # pre-lick truncation
        Xt = X.copy()
        cut = TGRID[None, None, :] >= first[:, None, None]
        Xt[np.broadcast_to(cut, Xt.shape)] = np.nan
        good = ~lb
        cm = {}
        for cn, (st, lk) in CONDS.items():
            msk = good & (stim == st) & (lick == lk)
            if msk.sum() >= 3:
                with np.errstate(all="ignore"):
                    cm[cn] = np.nanmean(Xt[msk], 0)
                    cm[cn][np.sum(np.isfinite(Xt[msk]), 0) < 3] = np.nan
                M.setdefault((cn, coh[sid]), []).append(cm[cn])
        if "WM" in cm and "CR" in cm:
            DIFF.setdefault(("WM-CR", coh[sid]), []).append(cm["WM"] - cm["CR"])
        if "WH" in cm and "WM" in cm:
            DIFF.setdefault(("WH-WM", coh[sid]), []).append(cm["WH"] - cm["WM"])
        if "AM" in cm and "CR" in cm:
            DIFF.setdefault(("AM-CR", coh[sid]), []).append(cm["AM"] - cm["CR"])
        # single-trial AUC hit vs miss (whisker), window means; hits only if the first event is after the window
        for wname, (a, b) in (("0-100 ms", (0.0, 0.1)), ("0-200 ms", (0.0, 0.2))):
            ww = (TGRID >= a) & (TGRID < b)
            sel = good & (stim == "whisker") & ((lick == 0) | (first >= b))
            y = lick[sel]
            if (y == 1).sum() >= 5 and (y == 0).sum() >= 5:
                for si, s in enumerate(SIGN):
                    with np.errstate(all="ignore"):
                        v = np.nanmean(X[sel][:, si][:, ww], 1)
                    ok = np.isfinite(v)
                    if (y[ok] == 1).sum() >= 5 and (y[ok] == 0).sum() >= 5 and np.std(v[ok]) > 0:
                        AUC.append(dict(session_id=sid, cohort=coh[sid], window=wname, signal=s,
                                        auc=roc_auc_score(y[ok], v[ok]), n_hit=int((y[ok] == 1).sum()), n_miss=int((y[ok] == 0).sum())))
        # learning: WM - CR before / after the LT (whisker trials by index; CR by position between them)
        if np.isfinite(ltd.get(sid, np.nan)):
            wti = z["whisker_trial_index"]
            wpos = np.maximum.accumulate(np.where(wti >= 0, wti, -1))
            for ph, sel in (("before LT", wpos < ltd[sid]), ("after LT", wpos >= ltd[sid])):
                mw = good & sel & (stim == "whisker") & (lick == 0)
                mc = good & sel & (stim == "no_stim") & (lick == 0)
                if mw.sum() >= 3 and mc.sum() >= 3:
                    with np.errstate(all="ignore"):
                        LRN.setdefault((ph, coh[sid]), []).append(np.nanmean(Xt[mw], 0) - np.nanmean(Xt[mc], 0))
    # cluster tests
    rows = []
    for (cname, cg), L_ in DIFF.items():
        A = np.array(L_)
        for si, s in enumerate(SIGN):
            t0, cl = cluster_paired(A[:, si, W])
            for a, b, m, p in cl:
                rows.append(dict(test=f"{cname} (paired)", cohort=cg, signal=s, t_start=tw[a], t_stop=tw[b - 1], mass=m, p=p, n=len(A)))
    for cname in ("WM-CR", "WH-WM", "AM-CR"):
        if (cname, "R+") in DIFF and (cname, "R-") in DIFF:
            A, B = np.array(DIFF[(cname, "R+")]), np.array(DIFF[(cname, "R-")])
            for si, s in enumerate(SIGN):
                t0, cl = cluster_two(A[:, si, W], B[:, si, W])
                for a, b, m, p in cl:
                    rows.append(dict(test=f"{cname}: R+ vs R-", cohort="R+ vs R-", signal=s, t_start=tw[a], t_stop=tw[b - 1], mass=m,
                                     p=p, n=len(A) + len(B)))
    for cg in ("R+", "R-"):
        if ("before LT", cg) in LRN and ("after LT", cg) in LRN:
            pre = {id(x): x for x in LRN[("before LT", cg)]}
            A, B = np.array(LRN[("after LT", cg)]), np.array(LRN[("before LT", cg)])
            n = min(len(A), len(B))
            for si, s in enumerate(SIGN):
                t0, cl = cluster_two(A[:, si, W], B[:, si, W])
                for a, b, m, p in cl:
                    rows.append(dict(test="WM-CR after vs before LT", cohort=cg, signal=s, t_start=tw[a], t_stop=tw[b - 1], mass=m,
                                     p=p, n=n))
    Cl = pd.DataFrame(rows)
    Cl.to_csv(T / "024_clusters.csv", index=False)
    A_ = pd.DataFrame(AUC)
    A_.to_csv(T / "024_auc_per_session.csv", index=False)
    ag = []
    for (w, s, cg), g in A_.groupby(["window", "signal", "cohort"]):
        ag.append(dict(window=w, signal=s, cohort=cg, n=len(g), auc_mean=g.auc.mean(), p_signflip=signflip(g.auc - 0.5)))
    for (w, s), g in A_.groupby(["window", "signal"]):
        a, b = g[g.cohort == "R+"].auc, g[g.cohort == "R-"].auc
        ag.append(dict(window=w, signal=s, cohort="R+ vs R-", n=len(g), auc_mean=a.mean() - b.mean(),
                       p_signflip=np.nan, p_mw=stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan,
                       p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 2 and len(b) > 2 else np.nan))
    AG = pd.DataFrame(ag)
    AG.to_csv(T / "024_auc_group.csv", index=False)
    figures(M, DIFF, LRN, Cl, AG, W, tw)
    json.dump(dict(script="024_stimulus_micromovements.py", dt=DT, window=WIN, nperm=NPERM, n_sessions=len(list(OUT.glob("*.npz"))),
                   versions=L.versions()), open(L.RES / "provenance" / "024.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(Cl[Cl.p < 0.05].sort_values(["test", "cohort", "signal", "t_start"]).round(4).to_string())
    print(AG[(AG.p_signflip < 0.05) | (AG.get("p_mw", pd.Series(dtype=float)) < 0.05)].round(4).to_string())


def figures(M, DIFF, LRN, Cl, AG, W, tw):
    show = ["whisker_angle", "whisker_speed", "top_particle_y", "top_nose_speed", "nose_speed", "nose_tip_y", "jaw_distance",
            "jaw_angle"]
    si_of = {s: i for i, s in enumerate(SIGN)}
    cond_col = {"WM": L.TASK_COL["whisker"], "WH": "#8a5a00", "AM": L.TASK_COL["auditory"], "AH": "#000080", "CR": "#888888",
                "FA": "#000000"}
    # Fig 1: evoked responses per condition (pre-lick), per cohort
    fig, axs = plt.subplots(len(show), 2, figsize=(7.4, 1.05 * len(show) + 0.8), sharex=True, squeeze=False)
    for i, s in enumerate(show):
        for j, cg in enumerate(("R+", "R-")):
            ax = axs[i, j]
            for cn in ("CR", "WM", "AM", "WH", "AH"):
                Xs = np.array(M.get((cn, cg), []))
                if not len(Xs):
                    continue
                X = Xs[:, si_of[s], W]
                with np.errstate(all="ignore"):
                    m, e = np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(np.sum(np.isfinite(X), 0).clip(1))
                m[np.sum(np.isfinite(X), 0) < 5] = np.nan
                ax.fill_between(tw, m - e, m + e, color=cond_col[cn], alpha=0.15, lw=0)
                ax.plot(tw, m, color=cond_col[cn], lw=0.7, ls="--" if cn in ("WH", "AH") else "-", label=f"{cn} ({len(X)})")
            ax.axvline(0, color="k", lw=0.4)
            ax.axvspan(0, 0.025, color="#ffe9a8", lw=0, alpha=0.6)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if j == 0:
                ax.set_ylabel(LAB[s], fontsize=5)
            if i == 0:
                ax.set_title(f"{cg}: evoked, pre-lick (hits truncated at jaw onset / first lick)", fontsize=5.5, color=L.COH_COL[cg])
                ax.legend(frameon=False, fontsize=4, ncol=5, loc="upper left")
            if i == len(show) - 1:
                ax.set_xlabel("s from stimulus (yellow: 0-25 ms, possible passive deflection)", fontsize=5)
    fig.tight_layout(h_pad=0.3)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"024_evoked.{ext}", dpi=300)
    plt.close(fig)
    # Fig 2: contrasts with clusters
    contrasts = [("WM-CR", "whisker miss − correct rejection (detection without report)"),
                 ("WH-WM", "whisker hit (pre-lick) − whisker miss"), ("AM-CR", "auditory miss − correct rejection")]
    fig, axs = plt.subplots(len(show), 3, figsize=(7.4, 1.05 * len(show) + 0.8), sharex=True, squeeze=False)
    for i, s in enumerate(show):
        for j, (cname, ctitle) in enumerate(contrasts):
            ax = axs[i, j]
            for cg in ("R+", "R-"):
                Xs = np.array(DIFF.get((cname, cg), []))
                if not len(Xs):
                    continue
                X = Xs[:, si_of[s], W]
                with np.errstate(all="ignore"):
                    m, e = np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(np.sum(np.isfinite(X), 0).clip(1))
                m[np.sum(np.isfinite(X), 0) < 5] = np.nan
                ax.fill_between(tw, m - e, m + e, color=L.COH_COL[cg], alpha=0.2, lw=0)
                ax.plot(tw, m, color=L.COH_COL[cg], lw=0.7)
                q = Cl[(Cl.test == f"{cname} (paired)") & (Cl.cohort == cg) & (Cl.signal == s) & (Cl.p < 0.05)] if len(Cl) else []
                y0 = ax.get_ylim()
                for r in (q.itertuples() if len(q) else []):
                    ax.plot([r.t_start, r.t_stop], [1.0 - 0.06 * (cg == "R-")] * 2, color=L.COH_COL[cg], lw=2,
                            transform=ax.get_xaxis_transform(), solid_capstyle="butt")
            q = Cl[(Cl.test == f"{cname}: R+ vs R-") & (Cl.signal == s) & (Cl.p < 0.05)] if len(Cl) else []
            for r in (q.itertuples() if len(q) else []):
                ax.axvspan(r.t_start, r.t_stop, color="#ffd400", alpha=0.3, lw=0)
            ax.axhline(0, color="k", lw=0.3)
            ax.axvline(0, color="k", lw=0.4)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if j == 0:
                ax.set_ylabel(LAB[s], fontsize=5)
            if i == 0:
                ax.set_title(ctitle, fontsize=5.3)
            if i == len(show) - 1:
                ax.set_xlabel("s from stimulus", fontsize=5)
    fig.suptitle("Contrasts per mouse, mean ± s.e.m.; bars on top: paired cluster p < 0.05 (R+ green, R− magenta); "
                 "yellow: R+ vs R− cluster p < 0.05", fontsize=5.8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=0.3)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"024_contrasts.{ext}", dpi=300)
    plt.close(fig)
    # Fig 3: AUC + learning
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.5, left=0.08, right=0.98, top=0.92, bottom=0.1)
    for j, w in enumerate(("0-100 ms", "0-200 ms")):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        g = AG[(AG.window == w) & AG.cohort.isin(["R+", "R-"])]
        for h, cg in enumerate(("R+", "R-")):
            q = g[g.cohort == cg].set_index("signal").reindex(SIGN)
            x = np.arange(len(SIGN)) + (h - 0.5) * 0.38
            ax.bar(x, q.auc_mean - 0.5, 0.36, bottom=0.5, color=L.COH_COL[cg], lw=0, label=cg)
            for xx, p, a in zip(x, q.p_signflip, q.auc_mean):
                if p < 0.05:
                    ax.text(xx, a + (0.005 if a >= 0.5 else -0.012), "*", ha="center", fontsize=6)
        ax.axhline(0.5, color="k", lw=0.4)
        ax.set_xticks(range(len(SIGN)))
        ax.set_xticklabels([LAB[s] for s in SIGN], rotation=60, ha="right", fontsize=4.3)
        ax.set_ylabel("AUC hit vs miss (single trials)", fontsize=5)
        ax.set_title(f"whisker trials, mean of {w} after the stimulus (hits only if jaw onset / lick later); * sign-flip p < 0.05",
                     fontsize=5.3)
        ax.legend(frameon=False, fontsize=4.5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    for j, s in enumerate(["whisker_angle", "top_particle_y", "top_nose_speed", "nose_speed"]):
        ax = fig.add_subplot(gs[1, j])
        for cg in ("R+", "R-"):
            for ph, ls in (("before LT", "--"), ("after LT", "-")):
                Xs = np.array(LRN.get((ph, cg), []))
                if not len(Xs):
                    continue
                X = Xs[:, si_of[s], W]
                with np.errstate(all="ignore"):
                    m = np.nanmean(X, 0)
                m[np.sum(np.isfinite(X), 0) < 4] = np.nan
                ax.plot(tw, m, ls, color=L.COH_COL[cg], lw=0.7, label=f"{cg} {ph} ({len(X)})")
        ax.axhline(0, color="k", lw=0.3)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_title(f"WM − CR: {LAB[s]}", fontsize=5.3)
        ax.set_xlabel("s from stimulus", fontsize=5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if j == 0:
            ax.legend(frameon=False, fontsize=3.8)
    fig.suptitle("Single-trial prediction of whisker hits from early micro-movements, and the detection response (WM − CR) "
                 "before vs after the learning trial", fontsize=5.8, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"024_auc_learning.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    {"compute": compute, "stats": run_stats}[sys.argv[1]]()
