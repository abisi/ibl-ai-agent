"""018 -- examples for the article report (haas, full run, BM_SET=full): what the states look like, example trials and
sessions, trial-epoch time courses and worked examples of the learning correlations.

Usage: BM_SET=full python 018_report_examples.py compute   (per-session pass, cache/full/018/<sid>.pkl)
       BM_SET=full python 018_report_examples.py figures   (figures/018_*.png|pdf|svg, tables/018_*.csv)

Inputs (same as 015): cache/full/inputs (014), kuma/full state sequences (tables/full_selected_models.json),
cache/full/009_meta.npz (meta-states of every method x view on the side 200 Hz grid), tables/full_trial_measures.parquet.
Sessions: full_sessions.csv at 200 Hz (76).

Definitions
- Trial epochs as in 015: pre = [stim - 2, stim); resp = [stim, response_window_stop]; iti = (response_window_stop,
  min(next stim - 2, response_window_stop + 5)]; other = every frame outside the epochs of kept trials.
- Peri-stimulus time courses: kept trials, -2 to +6.5 s from the stimulus in 25 ms bins; per bin the fraction of frames in
  each lightning-action class / meta-state and the mean of each feature. Per session the mean over the trials of a
  condition (whisker / auditory / no-stim x lick or not; whisker hits of the first / last third of whisker trials);
  groups = mean +- s.e.m. over sessions with >= 3 trials of the condition.
- State atlas per state set (lightning-action classes, kpms side meta-states, kpms side / top syllables, AR-HMM side /
  top states): per session up to 200 randomly drawn bouts (>= 25 ms) -> onset-triggered feature average (-0.5 to +1 s,
  10 ms); bout durations (log histogram); frames per epoch; up to 150 random frames of keypoint deviation from the
  session's resting pose (side and top, mm), drawn on the mean resting pose over sessions.
- Whisker angle is shown as the deviation from the session median (deg).
- Example sessions: one per cohort x learner group, chosen by rule: the session whose number of whisker trials is >= the
  group median and whose whisker P(lick) is closest to the group median (learners: also with a learning trial).
"""
import json
import os
import pickle
import textwrap
import zlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import bm_states as BS  # noqa: E402

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.8})
INP = L.CACHE / "full" / "inputs"
OUT = L.CACHE / "full" / "018"
T = L.TABLES
LA = ["background", "still", "whisking", "jaw", "lick"]
LA_COL = {"still": "#4d4d4d", "whisking": "#1f77b4", "jaw": "#ff7f0e", "lick": "#d62728", "background": "#e6e6e6"}
META_COL = ["#4d4d4d", "#17becf", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
FEATS = [("whisker_angle_dev", "whisker angle\n(deg from median)"), ("whisking_amp", "whisking amp.\n(deg)"),
         ("jaw_open", "jaw opening\n(mm)"), ("tongue_out", "tongue out\n(P)"), ("nose_speed", "nose speed side\n(mm/s)"),
         ("top_nose_speed", "nose speed top\n(mm/s)")]
SHORT = ["whisker angle (deg)", "whisking amp. (deg)", "jaw opening (mm)", "tongue out (P)"]
PERI = (-2.0, 16.0)  # to the ITI (response end + 5 s -> next quiet window); bins after the next quiet window are NaN
BIN = 0.025
NB = int(round((PERI[1] - PERI[0]) / BIN))
TRIG = np.arange(-100, 201, 2)  # frames at 200 Hz: -0.5 to +1.0 s, 10 ms
MIN_BOUT = 5
DUR_EDGES = np.logspace(np.log10(0.025), np.log10(20), 31)
EPN = ["other", "pre", "resp", "post", "iti"]
rng0 = 0


# ------------------------------------------------------------------------------------------------ per-session pass
def interp(x, ts_from, ts_to, tol=0.0075):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    y = np.interp(ts_to, ts_from[ok], x[ok], left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ts_from, ts_to), 1, len(ts_from) - 1)
    near = np.minimum(np.abs(ts_to - ts_from[i - 1]), np.abs(ts_from[i] - ts_to))
    y[near > tol] = np.nan
    return y


def nearest_idx(t_src, t_dst, tol=0.0075):
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t_dst) <= np.abs(t_src[j] - t_dst), j - 1, j)
    return np.where(np.abs(t_src[j] - t_dst) <= tol, j, -1)


def load_session(sid):
    sg = BS.session_grid(sid)
    t = sg["t"]
    F = {}
    for view in L.VIEWS:
        f = np.load(INP / f"{sid}_{view}_features.npz")
        for k, n in enumerate(f["names"]):
            F[str(n)] = interp(f["X"][:, k], f["t"], t)
    F["whisker_angle_dev"] = F["whisker_angle"] - np.nanmedian(F["whisker_angle"])
    feats = np.stack([F[k] for k, _ in FEATS], 1).astype(np.float32)
    meta = np.load(L.CACHE / "full" / "009_meta.npz")
    Z = {}
    f = BS.KR / "lightning_action" / "side" / f"{sid}_probs.npz"
    if f.exists():
        tv = BS.view_t(sid, "side")
        p = np.load(f)["pred"]
        n = min(len(p), len(tv))
        Z["la"] = BS.to_grid(p[:n], tv[:n], t)
    for view in L.VIEWS:
        if f"{sid}|kpms|{view}" in meta:
            Z[f"meta_{view}"] = meta[f"{sid}|kpms|{view}"].astype(int)
        for m in ("kpms", "arhmm"):
            z = BS.load_states(sid, m, view)
            if z is not None:
                Z[f"{m}_{view}"] = z
    pose = {}
    for view in L.VIEWS:
        k = np.load(INP / f"{sid}_{view}_keypoints.npz", allow_pickle=True)
        j = nearest_idx(k["t"], t)
        pose[view] = (k["xy"], k["rest"], j, [str(b) for b in k["bodyparts"]])
    tr = pd.read_parquet(INP / f"{sid}_trials.parquet")
    k = tr[tr.keep].reset_index(drop=True)
    return dict(t=t, feats=feats, Z=Z, pose=pose, trials=k, licks=sg["licks"])


def epoch_codes(t, k, sid):
    """per frame: 0 other, 1 pre, 2 resp, 3 post (response end -> +5 s), 4 iti (+5 s -> next quiet window);
    windows from bm_lib.trial_windows"""
    ep = np.zeros(len(t), np.int8)
    W = L.trial_windows(sid, k)
    for code, e in ((1, "pre"), (2, "resp"), (3, "post"), (4, "iti")):
        for a, b in zip(W[f"{e}_start"], W[f"{e}_stop"]):
            if np.isfinite(a) and np.isfinite(b):
                ep[np.searchsorted(t, a):np.searchsorted(t, b)] = code
    return ep


def bouts(z):
    ch = np.flatnonzero(np.diff(z) != 0) + 1
    on = np.r_[0, ch]
    off = np.r_[ch, len(z)]
    return on, off - on, z[on]


def nstates(name, z):
    if name == "la":
        return 5
    return int(max(z.max() + 1, {"kpms": 25, "arhmm": 15}.get(name.split("_")[0], 0)))


def one(sid):
    S = load_session(sid)
    t, Fm, Z, k = S["t"], S["feats"], S["Z"], S["trials"]
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    ep = epoch_codes(t, k, sid)
    W = L.trial_windows(sid, k)
    out = dict(session_id=sid)
    # ---- atlas
    atlas = {}
    for name, z in Z.items():
        ns = nstates(name, z)
        on, ln, st = bouts(z)
        A = dict(n_frames=np.bincount(z[z >= 0], minlength=ns)[:ns],
                 epoch_frames=np.bincount((z * 5 + ep)[z >= 0], minlength=5 * ns)[:5 * ns].reshape(ns, 5),
                 epoch_total=np.bincount(ep, minlength=5), trig=np.full((ns, len(TRIG), Fm.shape[1]), np.nan, np.float32),
                 n_bouts=np.zeros(ns, int), dur_hist=np.zeros((ns, len(DUR_EDGES) - 1), int), pose={})
        for s in range(ns):
            sel = (st == s) & (ln >= MIN_BOUT)
            A["n_bouts"][s] = sel.sum()
            A["dur_hist"][s] = np.histogram(ln[st == s] / L.FS_DS, DUR_EDGES)[0]
            o = on[sel]
            o = o[(o + TRIG[0] >= 0) & (o + TRIG[-1] < len(z))]
            if len(o) >= 5:
                o = rng.choice(o, min(200, len(o)), replace=False)
                A["trig"][s] = np.nanmean(Fm[o[:, None] + TRIG[None, :]], 0)
            fr = np.flatnonzero(z == s)
            if len(fr):
                fr = rng.choice(fr, min(150, len(fr)), replace=False)
                for view, (xy, rest, j, _) in S["pose"].items():
                    jj = j[fr]
                    jj = jj[jj >= 0]
                    A["pose"][view] = A["pose"].get(view, {})
                    A["pose"][view][s] = (xy[jj] - rest[None]).astype(np.float32)
        atlas[name] = A
    out["atlas"] = atlas
    out["rest"] = {v: S["pose"][v][1] for v in L.VIEWS}
    out["bodyparts"] = {v: S["pose"][v][3] for v in L.VIEWS}
    # ---- peri-stimulus
    chans = []
    for name, cls in (("la", [1, 2, 3, 4]), ("meta_side", range(6)), ("meta_top", range(6))):
        if name in Z:
            chans += [(name, c) for c in cls]
    nch = len(chans) + Fm.shape[1]
    P = np.full((len(k), nch, NB), np.nan, np.float32)
    modes = {n: np.full((len(k), NB), -1, np.int8) for n in ("la", "meta_side", "meta_top") if n in Z}
    for i, r in k.iterrows():
        stop = r.start_time + PERI[1]
        nq = W.next_quiet_start.iloc[i]
        if np.isfinite(nq):
            stop = min(stop, nq)  # never into the next trial's quiet window
        i0, i1 = np.searchsorted(t, r.start_time + PERI[0]), np.searchsorted(t, stop)
        b = np.floor((t[i0:i1] - r.start_time - PERI[0]) / BIN).astype(int)
        b = np.clip(b, 0, NB - 1)
        for c, (name, cl) in enumerate(chans):
            zz = Z[name][i0:i1]
            v = zz >= 0
            num = np.bincount(b[v], weights=(zz[v] == cl), minlength=NB)
            den = np.bincount(b[v], minlength=NB)
            P[i, c] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        for f in range(Fm.shape[1]):
            x = Fm[i0:i1, f]
            v = np.isfinite(x)
            num = np.bincount(b[v], weights=x[v], minlength=NB)
            den = np.bincount(b[v], minlength=NB)
            P[i, len(chans) + f] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        for n in modes:
            zz = Z[n][i0:i1]
            v = zz >= 0
            if v.sum():
                M = np.zeros((NB, int(zz.max()) + 1))
                np.add.at(M, (b[v], zz[v]), 1)
                md = M.argmax(1)
                md[M.sum(1) == 0] = -1
                modes[n][i] = md
    out["peri_channels"] = [f"{n}:{c}" for n, c in chans] + [f for f, _ in FEATS]
    out["peri"] = P.astype(np.float16)
    out["modes"] = modes
    out["trials"] = k[["start_time", "stim", "lick_flag", "whisker_trial_index", "response_window_stop_time"]].assign(
        trial=np.arange(len(k)), **{f"{c}_rel": (W[c] - k.start_time.to_numpy()).to_numpy()
                                    for c in ("resp_stop", "post_stop", "iti_start", "iti_stop", "next_quiet_start")})
    out["rw_stop"] = float(np.median(k.response_window_stop_time - k.start_time))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{sid}.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return sid


def sessions():
    S = pd.read_csv(T / "full_sessions.csv")
    return S[S.rate_side > 150].reset_index(drop=True)


def compute():
    sids = list(sessions().session_id)
    with ProcessPoolExecutor(int(os.environ.get("BM_WORKERS", 24))) as ex:
        for s in ex.map(one, sids):
            print("done", s, flush=True)


# ------------------------------------------------------------------------------------------------ figures
def wrap(s, w=165):
    return "\n".join(textwrap.wrap(s, w))


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"018_{name}.{ext}", dpi=300)
    plt.close(fig)


def letter(ax, s, x=-0.25, y=1.12):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def load_all():
    D = {}
    for f in sorted(OUT.glob("*.pkl")):
        with open(f, "rb") as fh:
            D[f.stem] = pickle.load(fh)
    return D


def meta_names():
    M = pd.read_csv(T / "full_009_metastates.csv")
    return list(M.name)


def state_sets():
    """(set, state, label, colour) of every atlas page"""
    mn = meta_names()
    Pp = pd.read_csv(T / "full_009_fingerprints_pooled.csv")
    main = [("la", c, f"action: {LA[c]}", LA_COL[LA[c]]) for c in (1, 2, 3, 4)] + \
           [("meta_side", m, f"meta-state {n}", META_COL[m]) for m, n in enumerate(mn)]
    out = {"main": main}
    for m, v, n in (("kpms", "side", 12), ("kpms", "top", 12), ("arhmm", "side", 12), ("arhmm", "top", 12)):
        g = Pp[(Pp.method == m) & (Pp.view == v) & Pp.shared].sort_values("usage", ascending=False).head(n)
        out[f"{m}_{v}"] = [(f"{m}_{v}", int(r.state), f"{'S' if m == 'kpms' else 'A'}{int(r.state)} -> {r.meta_name} "
                            f"({100 * r.usage:.0f}%, {int(r.n_mice)} mice)", META_COL[int(r.meta)]) for r in g.itertuples()]
    return out


def atlas_figure(D, items, name, title, coh):
    nr = len(items)
    cols = ["pose side", "pose top"] + [f for f, _ in FEATS[:5]] + ["duration", "epoch"]
    fig = plt.figure(figsize=(7.4, 0.95 * nr + 0.9))
    gs = fig.add_gridspec(nr, len(cols), left=0.07, right=0.99, top=1 - 0.55 / (0.95 * nr + 0.9),
                          bottom=0.35 / (0.95 * nr + 0.9), wspace=0.55, hspace=0.45,
                          width_ratios=[1.1, 1.1, 1, 1, 1, 1, 1, 0.9, 1.1])
    rest = {v: np.mean([d["rest"][v] for d in D.values()], 0) for v in L.VIEWS}
    bps = next(iter(D.values()))["bodyparts"]
    tt = TRIG / L.FS_DS
    ylim = {}
    for r, (sname, s, lab, col) in enumerate(items):
        A = {sid: d["atlas"].get(sname) for sid, d in D.items() if sname in d["atlas"]}
        # pose
        for c, v in enumerate(L.VIEWS):
            ax = fig.add_subplot(gs[r, c])
            pts = [a["pose"].get(v, {}).get(s) for a in A.values()]
            pts = [p for p in pts if p is not None and len(p)]
            for kk in range(rest[v].shape[0]):
                if pts:
                    P = np.concatenate([p[:, kk] for p in pts])
                    P = P[np.random.default_rng(0).permutation(len(P))[:2500]]
                    ax.scatter(rest[v][kk, 0] + P[:, 0], rest[v][kk, 1] + P[:, 1], s=0.6, color=col, alpha=0.25, lw=0,
                               rasterized=True)
            ax.plot(*rest[v].T, "x", color="k", ms=2.5, mew=0.6)
            if r == 0:
                for kk, b in enumerate(bps[v]):
                    dy = -0.5 - 0.7 * (kk % 2)
                    ax.text(rest[v][kk, 0], rest[v][kk, 1] + dy, b.replace("whisker_", "wh_"), fontsize=3.8, ha="center",
                            va="bottom")
                ax.set_title(f"{v} keypoints\n(frames, mm)", fontsize=5.5)
            ax.set_aspect("equal")
            cx, cy = rest[v].mean(0)
            span = 1.15 * max(np.ptp(rest[v][:, 0]), np.ptp(rest[v][:, 1])) / 2 + 1.5
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy + span, cy - span)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.text(-0.08, 0.5, "\n".join(textwrap.wrap(lab, 26)), transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=4.6, color=col, fontweight="bold")
                if r == nr - 1:
                    ax.plot([cx - span + 0.3, cx - span + 1.3], [cy + span - 0.3] * 2, "k-", lw=1)
                    ax.text(cx - span + 0.8, cy + span - 0.5, "1 mm", fontsize=4, ha="center", va="bottom")
        # triggered features
        for f in range(5):
            ax = fig.add_subplot(gs[r, 2 + f])
            M = np.array([a["trig"][s, :, f] for a in A.values() if s < len(a["n_bouts"]) and a["n_bouts"][s] >= 10])
            if len(M):
                m, e = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(np.isfinite(M), 0).clip(1))
                ax.fill_between(tt, m - e, m + e, color=col, alpha=0.3, lw=0)
                ax.plot(tt, m, color=col)
                ylim.setdefault(f, []).append((np.nanmin(m - e), np.nanmax(m + e)))
            ax.axvline(0, color="k", lw=0.4, ls=":")
            ax.set_xlim(tt[0], tt[-1])
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if r == 0:
                ax.set_title(FEATS[f][1], fontsize=5.5)
            if r == nr - 1:
                ax.set_xlabel("s from bout onset", fontsize=5)
            else:
                ax.set_xticklabels([])
            ax._feat = f
        # duration
        ax = fig.add_subplot(gs[r, 7])
        H = np.sum([a["dur_hist"][s] for a in A.values() if s < len(a["dur_hist"])], 0)
        if np.sum(H):
            ax.stairs(H / H.sum(), DUR_EDGES, color=col, fill=True, alpha=0.6)
            ctr = np.sqrt(DUR_EDGES[:-1] * DUR_EDGES[1:])
            med = ctr[np.searchsorted(np.cumsum(H) / H.sum(), 0.5)]
            ax.axvline(med, color="k", lw=0.5, ls="--")
            ax.text(0.98, 0.95, f"{1000 * med:.0f} ms", transform=ax.transAxes, ha="right", va="top", fontsize=4.5)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if r == 0:
            ax.set_title("bout duration", fontsize=5.5)
        if r == nr - 1:
            ax.set_xlabel("s", fontsize=5)
        else:
            ax.set_xticklabels([])
        # epoch usage by cohort
        ax = fig.add_subplot(gs[r, 8])
        for h, cg in enumerate(("R+", "R-")):
            U = np.array([a["epoch_frames"][s] / np.maximum(a["epoch_total"], 1) for sid, a in A.items()
                          if coh.get(sid) == cg and s < len(a["epoch_frames"])])
            if len(U):
                x = np.arange(5) + (h - 0.5) * 0.38
                ax.bar(x, U.mean(0), 0.36, yerr=U.std(0) / np.sqrt(len(U)), color=L.COH_COL[cg], lw=0,
                       error_kw=dict(lw=0.4), label=cg)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["oth", "pre", "resp", "post", "ITI"] if r == nr - 1 else [], fontsize=4.0, rotation=90)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if r == 0:
            ax.set_title("P(state | epoch)", fontsize=5.5)
            ax.legend(frameon=False, fontsize=4, loc="upper left", handlelength=0.8)
    for ax in fig.axes:
        f = getattr(ax, "_feat", None)
        if f is not None and f in ylim:
            lo, hi = np.nanmin([a for a, _ in ylim[f]]), np.nanmax([b for _, b in ylim[f]])
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    fig.suptitle(wrap(title), fontsize=6, x=0.01, ha="left", y=0.999, va="top")
    save(fig, name)


def outcome_mask(tr):
    lk = tr.lick_flag.to_numpy() == 1
    st = tr.stim.to_numpy()
    w = np.flatnonzero(st == "whisker")
    third = np.zeros(len(tr), int)
    if len(w) >= 6:
        third[w[: len(w) // 3]] = 1
        third[w[-(len(w) // 3):]] = 3
    return {"WH": (st == "whisker") & lk, "WM": (st == "whisker") & ~lk, "AH": (st == "auditory") & lk,
            "AM": (st == "auditory") & ~lk, "FA": (st == "no_stim") & lk, "CR": (st == "no_stim") & ~lk,
            "WH_early": (st == "whisker") & lk & (third == 1), "WH_late": (st == "whisker") & lk & (third == 3)}


def trial_epoch_means(d):
    """per trial x channel x epoch mean over that trial's own windows (pre, resp, post, iti) -> (n, nch, 4)"""
    tr = d["trials"]
    P = d["peri"].astype(np.float32)
    tb = PERI[0] + BIN * (np.arange(NB) + 0.5)
    out = np.full((len(tr), P.shape[1], 4), np.nan, np.float32)
    bounds = np.stack([np.full(len(tr), -2.0), np.zeros(len(tr)), tr.resp_stop_rel, tr.resp_stop_rel, tr.resp_stop_rel,
                       tr.post_stop_rel, tr.iti_start_rel, tr.iti_stop_rel], 1)
    for i in range(len(tr)):
        for e in range(4):
            a, b = bounds[i, 2 * e], bounds[i, 2 * e + 1]
            if np.isfinite(a) and np.isfinite(b) and b > a:
                k = (tb >= a) & (tb < b)
                if k.sum():
                    with np.errstate(all="ignore"):
                        out[i, :, e] = np.nanmean(P[i][:, k], 1)
    return out


EPOCHS4 = ["pre", "resp", "post", "iti"]


def peri_means(D):
    """per session x condition: mean time course (bins with >= 3 contributing trials) and per-trial epoch means
    -> rows (session, condition, channel, array) and epoch rows"""
    rows, ep_rows = [], []
    for sid, d in D.items():
        msk = outcome_mask(d["trials"])
        P = d["peri"].astype(np.float32)
        E = trial_epoch_means(d)
        for cond, m in msk.items():
            if m.sum() >= 3:
                with np.errstate(all="ignore"):
                    mu = np.nanmean(P[m], 0)
                    em = np.nanmean(E[m], 0)
                mu[np.sum(np.isfinite(P[m]), 0) < 3] = np.nan
                ne = np.sum(np.isfinite(E[m][:, 0, :]), 0)
                for c, ch in enumerate(d["peri_channels"]):
                    rows.append((sid, cond, ch, mu[c]))
                    for e, epn in enumerate(EPOCHS4):
                        ep_rows.append(dict(session_id=sid, condition=cond, channel=ch, epoch=epn, value=float(em[c, e]),
                                            n_trials=int(ne[e])))
    return rows, pd.DataFrame(ep_rows)


def time_course_figure(D, coh, chans, labels, cols_by_ch, name, title, ylabel_unit=None):
    rows, EPR = peri_means(D)
    EPR["cohort"] = EPR.session_id.map(coh)
    R = {}
    for sid, cond, ch, arr in rows:
        R.setdefault((cond, ch, coh[sid]), []).append(arr)
    tb = PERI[0] + BIN * (np.arange(NB) + 0.5)
    rw = np.median([d["rw_stop"] for d in D.values()])
    panels = [("whisker: hit (solid) / miss (dashed)", [("WH", "-"), ("WM", "--")]),
              ("auditory: hit (solid) / miss (dashed)", [("AH", "-"), ("AM", "--")]),
              ("no stimulus: FA (solid) / CR (dashed)", [("FA", "-"), ("CR", "--")]),
              ("whisker hits: last third (solid) / first third (dashed)", [("WH_late", "-"), ("WH_early", "--")])]
    fig, axs = plt.subplots(len(chans), 4, figsize=(7.4, 1.0 * len(chans) + 0.7), sharex=True, squeeze=False)
    summ = []
    for i, ch in enumerate(chans):
        for j, (ptitle, conds) in enumerate(panels):
            ax = axs[i, j]
            ax.axvspan(PERI[0], 0, color="#f0f0f0", lw=0)
            ax.axvspan(0, rw, color="#fff3d6", lw=0)
            ax.axvspan(rw, rw + 5, color="#fde7d4", lw=0)
            ax.axvspan(rw + 5, PERI[1], color="#e8f1fa", lw=0)
            for cond, ls in conds:
                for cg in ("R+", "R-"):
                    X = np.array(R.get((cond, ch, cg), []))
                    if not len(X):
                        continue
                    m = np.nanmean(X, 0)
                    e = np.nanstd(X, 0) / np.sqrt(np.sum(np.isfinite(X), 0).clip(1))
                    ax.fill_between(tb, m - e, m + e, color=L.COH_COL[cg], alpha=0.18, lw=0)
                    ax.plot(tb, m, ls, color=L.COH_COL[cg], lw=0.7, label=f"{cg} {cond} (n={len(X)})")
                    q = EPR[(EPR.channel == ch) & (EPR.condition == cond) & (EPR.cohort == cg) & (EPR.n_trials >= 3)]
                    for epn, g in q.groupby("epoch"):
                        v = g.value.dropna()
                        summ.append(dict(channel=ch, condition=cond, cohort=cg, epoch=epn, n_sessions=len(v),
                                         mean=float(v.mean()), sem=float(v.sem())))
            ax.axvline(0, color="k", lw=0.4)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if i == 0:
                ax.set_title(ptitle, fontsize=5.5)
                if j == 0:
                    for x0, x1, lab in ((-2, 0, "pre"), (0, rw, "resp"), (rw, rw + 5, "post"), (rw + 5, PERI[1], "ITI")):
                        ax.text((x0 + x1) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.5)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=5, color=cols_by_ch.get(ch, "k"))
            if i == len(chans) - 1:
                ax.set_xlabel("s from stimulus", fontsize=5)
        lo = min(a.get_ylim()[0] for a in axs[i])
        hi = max(a.get_ylim()[1] for a in axs[i])
        for a in axs[i]:
            a.set_ylim(lo, hi)
    hs = [plt.Line2D([], [], color=L.COH_COL[c], lw=1) for c in ("R+", "R-")]
    fig.legend(hs, ["R+", "R−"], loc="upper right", ncol=1, frameon=False, fontsize=5.5)
    fig.suptitle(wrap(title), fontsize=6, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.94, 0.975), h_pad=0.4, w_pad=0.6)
    save(fig, name)
    return pd.DataFrame(summ)


def example_sessions():
    S = sessions()
    lt = pd.read_csv(L.ROOT / "combined_results_ks4" / "ssl-learning-trial-identification" / "artifacts" /
                     "017_learning_trials_final.csv")
    ltd = dict(zip(lt.session_id, lt.lt_final))
    S["lt_trial"] = S.session_id.map(ltd)
    ex = []
    for cg in ("R+", "R-"):
        for lr in ("learner", "non-learner"):
            g = S[(S.reward_group == cg) & (S.learner == lr)]
            if lr == "learner":
                g = g[g.lt_trial.notna()] if g.lt_trial.notna().any() else g
            if not len(g):
                continue
            g = g[g.n_whisker >= g.n_whisker.median()]
            r = g.iloc[np.argmin(np.abs(g.p_lick_whisker - g.p_lick_whisker.median()))]
            ex.append(dict(session_id=r.session_id, cohort=cg, learner=lr, lt_trial=r.lt_trial, n_whisker=int(r.n_whisker),
                           p_lick_whisker=float(r.p_lick_whisker)))
    E = pd.DataFrame(ex)
    E.to_csv(T / "018_example_sessions.csv", index=False)
    return E


def smooth_nan(x, sig=5):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.full(len(x), np.nan)
    num = gaussian_filter1d(np.where(ok, x, 0), sig, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), sig, mode="nearest")
    return np.where(den > 0.2, num / den, np.nan)


def raster_cmap(name, D):
    if name == "la":
        return ListedColormap([LA_COL[c] for c in LA]), 0, 4, LA
    mn = meta_names()
    return ListedColormap(META_COL[:len(mn)]), 0, len(mn) - 1, [n.split(" ", 1)[1] for n in mn]


def example_sessions_figure(D, E):
    n = len(E)
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(4, n, height_ratios=[0.8, 2.2, 2.2, 0.15], hspace=0.28, wspace=0.3, left=0.07, right=0.99,
                          top=0.92, bottom=0.05)
    tb = PERI[0] + BIN * (np.arange(NB) + 0.5)
    for j, r in E.iterrows():
        d = D[r.session_id]
        tr = d["trials"]
        ax = fig.add_subplot(gs[0, j])
        for stim in ("whisker", "auditory", "no_stim"):
            m = (tr.stim == stim).to_numpy()
            ax.plot(np.flatnonzero(m), smooth_nan(tr.lick_flag.to_numpy(float)[m], 3), color=L.TASK_COL[stim], lw=0.8)
        w = np.flatnonzero((tr.stim == "whisker").to_numpy())
        if np.isfinite(r.lt_trial) and int(r.lt_trial) < len(w):
            ax.axvline(w[int(r.lt_trial)], color="r", ls="--", lw=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("kept trial", fontsize=5)
        if j == 0:
            ax.set_ylabel("P(lick)")
        ax.set_title(f"{r.session_id}\n{r.cohort} {r.learner}", fontsize=5.5, color=L.COH_COL[r.cohort])
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        letter(ax, "ABCD"[j], -0.25, 1.32)
        for i, name in enumerate(("la", "meta_side")):
            ax = fig.add_subplot(gs[1 + i, j])
            if name not in d["modes"]:
                continue
            M = d["modes"][name][w].astype(float)
            M[M < 0] = np.nan
            cm, lo, hi, _ = raster_cmap(name, D)
            ax.imshow(M, aspect="auto", cmap=cm, vmin=lo - 0.5, vmax=hi + 0.5, interpolation="nearest",
                      extent=[PERI[0], PERI[1], len(w), 0], rasterized=True)
            hit = tr.lick_flag.to_numpy()[w] == 1
            ax.scatter(np.full(len(w), PERI[1] + 0.15), np.arange(len(w)) + 0.5, s=0.6, marker="s",
                       c=np.where(hit, "#000000", "#cccccc"), lw=0, clip_on=False)
            if np.isfinite(r.lt_trial) and int(r.lt_trial) < len(w):
                ax.axhline(int(r.lt_trial), color="r", ls="--", lw=0.6)
            for x in (0, d["rw_stop"], d["rw_stop"] + 5):
                ax.axvline(x, color="w", lw=0.5)
            nq = tr.next_quiet_start_rel.to_numpy()[w]
            ax.scatter(np.clip(nq, PERI[0], PERI[1]), np.arange(len(w)) + 0.5, s=0.4, color="k", marker="|", lw=0.4)
            ax.set_xlim(PERI[0], PERI[1] + 0.3)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if j == 0:
                ax.set_ylabel(("lightning-action actions" if name == "la" else "kpms side meta-states") +
                              "\nwhisker trial", fontsize=5)
            if i == 1:
                ax.set_xlabel("s from whisker stimulus", fontsize=5)
            if j == 0 and i == 0:
                for x0, x1, lab in ((-2, 0, "pre"), (0, d["rw_stop"], "resp"), (d["rw_stop"], d["rw_stop"] + 5, "post"),
                                    (d["rw_stop"] + 5, PERI[1], "ITI")):
                    ax.text((x0 + x1) / 2, 1.01, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.5)
    for i, name in enumerate(("la", "meta_side")):
        cm, lo, hi, nm = raster_cmap(name, D)
        ax = fig.add_subplot(gs[3, 2 * i:2 * i + 2])
        ax.axis("off")
        hs = [plt.Rectangle((0, 0), 1, 1, color=cm(k - lo)) for k in range(lo, hi + 1)]
        ax.legend(hs, nm, ncol=len(hs), frameon=False, fontsize=4.5, loc="center", handlelength=0.8,
                  title="actions" if name == "la" else "meta-states", title_fontsize=5)
    fig.suptitle("Example sessions: P(lick) per trial type (whisker orange, auditory blue, no-stim black; red dashed = "
                 "learning trial); whisker trials aligned to the stimulus (right strip: hit black / miss grey)",
                 fontsize=5.5, x=0.01, ha="left", y=0.995)
    save(fig, "fig_example_sessions")


def example_trials_figure(D, E):
    """one hit and one miss whisker trial of each example session: features + ethograms, -2 to +6.5 s"""
    picks = []
    for _, r in E.iterrows():
        tr = D[r.session_id]["trials"]
        w = tr[tr.stim == "whisker"]
        late = w.iloc[len(w) // 2:]
        h = late[late.lick_flag == 1]
        mi = w[w.lick_flag == 0]
        if len(h):
            picks.append((r, int(h.trial.iloc[len(h) // 2]), "whisker hit (2nd half)"))
        if len(mi):
            picks.append((r, int(mi.trial.iloc[len(mi) // 2]), "whisker miss"))
    picks = picks[:8]
    fig = plt.figure(figsize=(7.4, 9.6))
    outer = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.16, left=0.07, right=0.99, top=0.95, bottom=0.04)
    cache = {}
    for p, (r, ti, lab) in enumerate(picks):
        sid = r.session_id
        if sid not in cache:
            cache[sid] = load_session(sid)
        S = cache[sid]
        tr = S["trials"].iloc[ti]
        t0 = tr.start_time
        i0, i1 = np.searchsorted(S["t"], t0 + PERI[0]), np.searchsorted(S["t"], t0 + PERI[1])
        tt = S["t"][i0:i1] - t0
        inner = outer[p // 2, p % 2].subgridspec(5, 1, height_ratios=[1, 1, 1, 1, 1.5], hspace=0.12)
        rw = tr.response_window_stop_time - t0
        for f, fi in enumerate((0, 1, 2, 3)):
            ax = fig.add_subplot(inner[f])
            ax.axvspan(0, rw, color="#fff3d6", lw=0)
            ax.plot(tt, S["feats"][i0:i1, fi], color="k", lw=0.4)
            ax.set_xlim(*PERI)
            ax.set_xticklabels([])
            ax.tick_params(labelsize=4, length=1.2, pad=1)
            ax.text(0.005, 0.97, SHORT[fi], transform=ax.transAxes, fontsize=4, ha="left", va="top")
            lk = S["licks"][(S["licks"] >= t0 + PERI[0]) & (S["licks"] < t0 + PERI[1])] - t0
            if f == 0:
                ax.plot(lk, np.full(len(lk), np.nanmax(S["feats"][i0:i1, fi])), "|", color="#d62728", ms=3, mew=0.6)
                ax.set_title(f"{sid} ({r.cohort} {r.learner}): {lab}, trial {ti}", fontsize=5, color=L.COH_COL[r.cohort])
        ax = fig.add_subplot(inner[4])
        strips = []
        for name in ("la", "meta_side", "meta_top", "kpms_side", "kpms_top"):
            z = S["Z"].get(name)
            strips.append(z[i0:i1].astype(float) if z is not None else np.full(i1 - i0, np.nan))
        y = 0
        for name, z in zip(("actions", "meta side", "meta top", "kpms side", "kpms top"), strips):
            z = z.copy()
            z[z < 0] = np.nan
            if name == "actions":
                cm, lo, hi = ListedColormap([LA_COL[c] for c in LA]), 0, 4
            elif name.startswith("meta"):
                cm, lo, hi = ListedColormap(META_COL[:4]), 0, 3
            else:
                cm, lo, hi = plt.get_cmap("tab20"), 0, 24
                z = z % 20
            ax.imshow(z[None], aspect="auto", cmap=cm, vmin=lo - 0.5, vmax=hi + 0.5, interpolation="nearest",
                      extent=[tt[0], tt[-1], y + 1, y], rasterized=True)
            ax.text(PERI[0] - 0.1, y + 0.5, name, ha="right", va="center", fontsize=4)
            y += 1
        ax.set_ylim(y, 0)
        ax.set_xlim(*PERI)
        ax.set_yticks([])
        for x in (0, rw, rw + 5):
            ax.axvline(x, color="w", lw=0.6)
        W1 = L.trial_windows(sid, S["trials"]).iloc[ti]
        if np.isfinite(W1.next_quiet_start):
            ax.axvline(W1.next_quiet_start - t0, color="k", lw=0.8, ls="--")
        ax.tick_params(labelsize=4, length=1.2, pad=1)
        ax.set_xlabel("s from whisker stimulus (shade = response window; white = response end, +5 s; dashed = next quiet window)",
                      fontsize=4.5)
    hs = [plt.Rectangle((0, 0), 1, 1, color=LA_COL[c]) for c in LA[1:]] + \
         [plt.Rectangle((0, 0), 1, 1, color=META_COL[k]) for k in range(4)]
    fig.legend(hs, [f"action {c}" for c in LA[1:]] + [n for n in meta_names()], ncol=8, frameon=False, fontsize=4.5,
               loc="lower center", handlelength=0.8)
    fig.suptitle("Example whisker trials (one hit, one miss per example session): features, piezo licks (red ticks) and "
                 "the state sequence of every method", fontsize=5.5, x=0.01, ha="left")
    save(fig, "fig_example_trials")


def correlation_figure(E, coh):
    Dm = pd.read_parquet(T / "full_trial_measures.parquet")
    A = pd.read_csv(T / "016_A_curve_correlation_per_session.csv")
    Dp = pd.read_csv(T / "016_D_prestim_auc_per_session.csv")
    meas = [("iti", "act_whisking_nl", "ITI whisking (no-lick frames)"), ("iti", "act_still_nl", "ITI still (no-lick)"),
            ("pre", "act_still_nl", "pre-stim still (no-lick)")]
    n = len(E)
    fig, axs = plt.subplots(n + 1, 4, figsize=(7.4, 1.25 * (n + 1) + 0.6), squeeze=False)
    for i, r in E.iterrows():
        g = Dm[(Dm.session_id == r.session_id) & (Dm.stim == "whisker")]
        for j, (ep, m, lab) in enumerate(meas):
            ax = axs[i, j]
            ge = g[g.epoch == ep].sort_values("trial")
            x = np.arange(len(ge))
            curve = smooth_nan(ge.lick_flag.to_numpy(float))
            ms = smooth_nan(ge[m].to_numpy(float))
            ax.plot(x, ge[m], ".", color="#999999", ms=1)
            ax.plot(x, ms, color="k", lw=0.8)
            ax2 = ax.twinx()
            ax2.plot(x, curve, color=L.TASK_COL["whisker"], lw=0.8)
            ax2.set_ylim(-0.02, 1.02)
            ax2.tick_params(labelsize=4.5, length=1.5, pad=1, colors=L.TASK_COL["whisker"])
            ax2.spines["right"].set_visible(True)
            q = A[(A.session_id == r.session_id) & (A.epoch == ep) & (A.measure == m)]
            if len(q):
                ax.text(0.02, 0.97, f"r {q.r.iloc[0]:+.2f}, shift null {q.null_mean.iloc[0]:+.2f}\nexcess {q.excess.iloc[0]:+.2f}",
                        transform=ax.transAxes, fontsize=4.3, va="top")
            if np.isfinite(r.lt_trial):
                ax.axvline(r.lt_trial, color="r", ls="--", lw=0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if i == 0:
                ax.set_title(lab, fontsize=5.5)
            if j == 0:
                ax.set_ylabel(f"{r.session_id}\n{r.cohort} {r.learner}\nfraction of frames", fontsize=4.5, color=L.COH_COL[r.cohort])
            if j == 2:
                ax2.set_ylabel("whisker P(lick)", fontsize=4.5, color=L.TASK_COL["whisker"])
            ax.set_xlabel("whisker trial" if i == n - 1 else "", fontsize=5)
        # AUC example
        ax = axs[i, 3]
        ge = g[g.epoch == "pre"].sort_values("trial")
        half = len(ge) // 2
        for h, (nm, sl) in enumerate((("early", slice(0, half)), ("late", slice(half, None)))):
            gg = ge.iloc[sl]
            for k, lk in enumerate((0, 1)):
                v = gg[gg.lick_flag == lk].act_still_nl.dropna()
                xpos = h * 2.6 + k
                ax.scatter(xpos + np.random.default_rng(0).uniform(-0.25, 0.25, len(v)), v, s=1.2,
                           color="#cccccc" if lk == 0 else "#333333", lw=0)
                ax.plot([xpos - 0.3, xpos + 0.3], [v.median()] * 2, color=L.TASK_COL["whisker"], lw=1)
            q = Dp[(Dp.session_id == r.session_id) & (Dp.measure == "act_still_nl") & (Dp.half == nm)]
            if len(q):
                ax.text(h * 2.6 + 0.5, 1.04, f"AUC {q.auc.iloc[0]:.2f}", ha="center", fontsize=4.3)
        ax.set_xticks([0, 1, 2.6, 3.6])
        ax.set_xticklabels(["miss", "hit", "miss", "hit"], fontsize=4.5)
        ax.text(0.5, -0.17, "early half", ha="center", va="top", fontsize=4.3, transform=ax.get_xaxis_transform())
        ax.text(3.1, -0.17, "late half", ha="center", va="top", fontsize=4.3, transform=ax.get_xaxis_transform())
        ax.set_ylim(-0.02, 1.1)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if i == 0:
            ax.set_title("pre-stim still vs whisker hit", fontsize=5.5)
    # population row
    ex = set(E.session_id)
    for j, (ep, m, lab) in enumerate(meas):
        ax = axs[n, j]
        q = A[(A.epoch == ep) & (A.measure == m)].assign(cohort=lambda d: d.session_id.map(coh))
        for h, cg in enumerate(("R+", "R-")):
            v = q[q.cohort == cg]
            x = h + np.random.default_rng(1).uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v.excess, s=3, color=L.COH_COL[cg], lw=0, alpha=0.7)
            isx = v.session_id.isin(ex).to_numpy()
            ax.scatter(x[isx], v.excess[isx], s=14, facecolor="none", edgecolor="k", lw=0.6)
            ax.plot([h - 0.25, h + 0.25], [v.excess.mean()] * 2, color="k", lw=1)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R+", "R−"])
        ax.set_ylabel("excess r (per mouse)", fontsize=5)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        ax.set_title(f"all mice: {lab}", fontsize=5)
    ax = axs[n, 3]
    q = Dp[Dp.measure == "act_still_nl"].pivot_table(index="session_id", columns="half", values="auc").dropna()
    for sid, rr in q.iterrows():
        c = L.COH_COL.get(coh.get(sid), "k")
        ax.plot([0, 1], [rr.early, rr.late], color=c, lw=0.3, alpha=0.6)
        if sid in ex:
            ax.plot([0, 1], [rr.early, rr.late], "o-", color="k", lw=0.8, ms=2.5, mfc="none")
    ax.axhline(0.5, color="k", lw=0.4, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["early half", "late half"])
    ax.set_ylabel("AUC pre-stim still", fontsize=5)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax.set_title("all mice (black = examples)", fontsize=5)
    fig.suptitle("Worked examples of the learning analyses: per-trial measure (grey), smoothed (black), whisker P(lick) "
                 "(orange, right axis); last row: every mouse, examples circled", fontsize=5.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.9, w_pad=1.2)
    save(fig, "fig_example_correlations")


def figures():
    D = load_all()
    S = sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    D = {k: v for k, v in D.items() if k in coh}
    print(len(D), "sessions")
    sets = state_sets()
    atlas_figure(D, sets["main"], "fig_atlas_main", "Actions and meta-states (76 sessions): keypoint "
                 "positions in the state (coloured, deviation from each session's rest drawn on the mean resting pose; x = "
                 "rest), onset-triggered features (mean +- s.e.m. over mice), bout durations (dashed = median) and the "
                 "fraction of each trial epoch spent in the state (R+ green, R− magenta).", coh)
    for k, title in (("kpms_side", "keypoint-MoSeq side syllables"), ("kpms_top", "keypoint-MoSeq top syllables"),
                     ("arhmm_side", "AR-HMM side states"), ("arhmm_top", "AR-HMM top states")):
        if sets.get(k):
            atlas_figure(D, sets[k], f"fig_atlas_{k}", f"{title}: 12 most used shared states (label -> meta-state, usage, "
                         f"mice); panels as in the main atlas", coh)
    mn = meta_names()
    ch_a = [f"la:{c}" for c in (1, 2, 3, 4)] + [f"meta_side:{m}" for m in range(len(mn))]
    lab_a = [f"P({LA[c]})" for c in (1, 2, 3, 4)] + [f"P({n})" for n in mn]
    cols = {**{f"la:{c}": LA_COL[LA[c]] for c in (1, 2, 3, 4)}, **{f"meta_side:{m}": META_COL[m] for m in range(len(mn))}}
    s1 = time_course_figure(D, coh, ch_a, lab_a, cols, "fig_timecourse_states",
                            "Trial-epoch time courses of actions (lightning-action) and kpms side meta-states: mean +- s.e.m. "
                            "over mice; grey = pre-stim, yellow = response window, blue = ITI (5 s)")
    s2 = time_course_figure(D, coh, [f for f, _ in FEATS], [lab for _, lab in FEATS], {}, "fig_timecourse_features",
                            "Trial-epoch time courses of the features (physical units), same layout")
    pd.concat([s1, s2]).to_csv(T / "018_timecourse_epoch_means.csv", index=False)
    E = example_sessions()
    example_sessions_figure(D, E)
    example_trials_figure(D, E)
    correlation_figure(E, coh)
    json.dump(dict(script="018_report_examples.py", n_sessions=len(D), peri=PERI, bin_s=BIN, trig_s=[TRIG[0] / L.FS_DS, TRIG[-1] / L.FS_DS],
                   min_bout_frames=MIN_BOUT, examples=list(E.session_id), versions=L.versions()),
              open(L.RES / "provenance" / "018.json", "w"), indent=1)


if __name__ == "__main__":
    {"compute": compute, "figures": figures}[sys.argv[1]]()
