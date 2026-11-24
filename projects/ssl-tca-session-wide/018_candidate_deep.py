"""Deeper look at the trend candidate (user 2026-11-23): examples, trial factors, area attribution, expert sessions,
other methods. Exploratory, uncorrected; per-mouse values; R+ vs R- with Mann-Whitney and Welch.

Candidate: stim_sensory, main fit (split non-negative) cluster 10 (Ss-whisker, peak ~22.5 ms); reference cl. 40.

Per session (all whisker day-0 sessions, one per mouse, and the 36 expert sessions), on the stim_sensory tensor prepared as
in 006g (kept units, rate - same-trial baseline, / unit SD over trials x time; artefact-filled, 10-ms bins):
- member score: trial weights of the session's member components, each divided by its mean over active whisker trials,
  averaged (sessions without a member: NaN). Expert sessions: components matched to the cluster (below).
- area attribution of the member score: projection of the data on each member component restricted to the units of one
  area group, p_g(trial) = sum_{u in g} net_u * <tf, X[trial, u, :]>, divided by the mean total projection over active
  whisker trials -> the area contributions add up to the total, so their slopes add up to the total slope.
- other methods (no TCA), all on bins >= +7.5 ms (the corrected bins excluded):
  template: projection on the mean whisker response of the passive-pre whisker trials (units x time; independent of the
            active trials); PCA: unsupervised PCA (trials x units.time, all stimulus trials), the PC among the first 5 with
            the largest whisker vs auditory AUC, scores uncentred (0 = baseline) and signed whisker > auditory;
  stimulus axis (dPCA-like): whisker - auditory mean response (units x time) of the other half of the trials (2-fold,
            odd / even), i.e. the stimulus marginal of demixed PCA without regularisation.
  Each score / its mean over active whisker trials; same slopes and contrast d as 017.
- single units (model-free): evoked response = mean z over bins 12.5-42.5 ms; slopes over position for whisker hits,
  whisker misses, auditory hits; d; responsive = whisker evoked > 0 with Wilcoxon p < 0.01 over all whisker trials; evoked
  Hz (rate - baseline) per class for cross-session comparison (expert vs day 0).
- trial factors (active whisker trials, member score): OLS with standardised predictors: position, hit, previous trial
  rewarded, pre-stimulus population rate (mean over units of the per-unit z-scored baseline, -1..-0.01 s), log ITI.
Expert matching: each expert component's mean 013 similarity to the day-0 members; accepted if >= the 10th percentile of
the members' leave-one-out mean similarity to each other; best component per session; false-positive rate = share of
day-0 sessions without a member whose best component passes.

Run (haas): python 018_candidate_deep.py [--method _splitnn --cluster 10] [--n-jobs 40] [--reuse]
     -> tables/deep_<fit>_cl<k>_{trials,units,expert}.parquet, figures/aligned_gpu/candidates/deep_<fit>_cl<k>_*.png|pdf
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
P4 = importlib.import_module("004_pilot_figures")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
WH, AU = "#f7b519", "#2c2cdb"
AL = "stim_sensory"
MIN_N, NB = 8, 5
EV = (10.0, 45.0)          # evoked bins (centres 12.5-42.5 ms)
T0 = 5.0                   # bins used by the other methods: centres >= 7.5 ms (corrected bins -7.5 / 2.5 excluded)
SCORES = {"member": "TCA member", "template": "Template (passive pre)", "pca": "PCA (best whisker PC)",
          "stimaxis": "Stimulus axis (dPCA-like)"}
CLS = {"wh_hit": ("whisker_trial", 1), "wh_miss": ("whisker_trial", 0), "au_hit": ("auditory_trial", 1)}
REG = ["pos", "hit", "prev_rew", "base_pop", "log_iti"]
REG2 = REG + ["p_wh_i", "p_au_i"]   # model 2: + whisker and auditory learning curves (engagement / performance state)
REG_LAB = {"pos": "Position", "hit": "Hit", "prev_rew": "Prev. rewarded", "base_pop": "Pre-stim rate", "log_iti": "log ITI", "p_wh_i": "P(lick | whisker)",
           "p_au_i": "P(lick | auditory)"}


def prepare(sid):
    d = np.load(A5.OUT_DIR / sid / f"{AL}.npz")
    R, base = d["rate"], d["baseline"]
    keep = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= 0.5
    D = R[:, keep] - base[:, keep, None]
    D = np.nan_to_num(D / (np.nanstd(D, (0, 2), keepdims=True) + 1e-9))
    return d, R, base, keep, D


def slope(pos, y):
    ok = np.isfinite(y)
    return float(np.polyfit(pos[ok], y[ok], 1)[0]) if ok.sum() >= MIN_N else np.nan


def auc(a, b):
    if len(a) < 3 or len(b) < 3:
        return 0.5
    return float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def session(sid, meta, comps):
    """comps: list of (rank, comp) of member / matched components (may be empty)"""
    t_all = pd.read_parquet(A5.OUT_DIR / sid / "trials.parquet").sort_values("trial_index").reset_index(drop=True)
    rew = ((t_all.lick_flag.fillna(0) == 1) & (t_all.reward_available == 1) & (t_all.epoch == "active")).to_numpy()
    t_all["prev_rew"] = np.r_[0, rew[:-1]].astype(float)
    t_all["log_iti"] = np.log(np.r_[np.nan, np.diff(t_all.start_time.to_numpy(float))])
    d, R, base, keep, D = prepare(sid)
    t = t_all.set_index("trial_index").loc[d["trial_index"]].reset_index()
    units = pd.read_parquet(A5.OUT_DIR / sid / "units.parquet").iloc[np.flatnonzero(keep)].reset_index(drop=True)
    _, _, grp = P4.area_colours(units.area_acronym_custom.str.strip().tolist())
    grp = np.asarray(grp)
    tc = d["t_centres"] * 1e3
    act = (t.epoch == "active").to_numpy()
    tt, lk = t.trial_type.to_numpy(), t.lick_flag.fillna(0).to_numpy().astype(int)
    aw = act & (tt == "whisker_trial")
    wh, au = tt == "whisker_trial", tt == "auditory_trial"
    pos = np.full(len(t), np.nan)
    pos[act] = np.arange(act.sum()) / max(act.sum() - 1, 1)
    bz = (base[:, keep] - base[:, keep].mean(0)) / (base[:, keep].std(0) + 1e-9)
    out = t[["trial_index", "epoch", "trial_type", "lick_flag", "p_wh_i", "p_au_i", "prev_rew", "log_iti",
           "lick_latency"]].copy()
    out["pos"], out["base_pop"] = pos, bz.mean(1)
    for k, v in meta.items():
        out[k] = v
    # TCA member score + area projections
    groups = sorted(set(grp))
    if comps:
        f = np.load(C12.D / sid / f"{AL}_zscore{meta['method']}.npz")
        J = len(f["units_kept"])
        assert np.array_equal(f["units_kept"], np.flatnonzero(keep)) and np.array_equal(f["trial_index"], d["trial_index"])
        ys, P = [], {g: np.zeros(len(t)) for g in groups}
        for r, c in comps:
            Fa, Fu, Ft = f[f"r{r}_f0"][:, c - 1], f[f"r{r}_f1"][:, c - 1], f[f"r{r}_f2"][:, c - 1]
            net = Fu[:J] - Fu[J:] if Fu.shape[0] == 2 * J else Fu
            ys.append(Fa / max(np.std(Fa[aw]), 1e-12))
            proj = (D @ Ft) * net[None]                     # trials x units
            sdp = np.std(proj.sum(1)[aw])
            for g in groups:
                P[g] += proj[:, grp == g].sum(1) / max(sdp, 1e-12) / len(comps)
        out["member"] = np.mean(ys, 0)
        for g in groups:
            out[f"area::{g}"] = P[g]
        out["_tot"] = sum(P.values())
    else:
        out["member"] = np.nan
        out["_tot"] = np.nan
    # other methods
    tb = tc >= T0
    X = D[:, :, tb].reshape(len(t), -1)
    pre = (t.epoch == "passive_pre").to_numpy() & wh
    if pre.sum() < 6:
        pre = wh & ~act
    if pre.sum() >= 6:
        ip = np.flatnonzero(pre)
        tm_ = X[ip].mean(0)
        sc = X @ tm_ / max(tm_ @ tm_, 1e-12)
        for h in (0, 1):   # template trials scored with the template of the other half (no self-projection bias)
            th = X[ip[1 - h::2]].mean(0)
            sc[ip[h::2]] = X[ip[h::2]] @ th / max(th @ th, 1e-12)
        out["template"] = sc
    else:
        out["template"] = np.nan
    st_ = wh | au
    Xc = X[st_] - X[st_].mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    best, bs = None, -1
    for v in Vt[:5]:
        s = X @ v
        a = auc(s[wh], s[au])
        if abs(a - 0.5) > bs:
            bs, best = abs(a - 0.5), v * (1 if a >= 0.5 else -1)
    out["pca"] = X @ best
    sa = np.full(len(t), np.nan)
    half = np.arange(len(t)) % 2
    for h in (0, 1):
        tr = half != h
        ax_ = X[tr & wh].mean(0) - X[tr & au].mean(0)
        sa[half == h] = X[half == h] @ ax_ / max(np.linalg.norm(ax_), 1e-12)
    out["stimaxis"] = sa
    # all scores (also the member score and area projections) in SD units of the active whisker trials, relative to
    # their mean there: dividing by the mean is unstable when it is near 0 (PCA / stimulus axis, weak members)
    for s in ["member", "template", "pca", "stimaxis"] + [c for c in out.columns if c.startswith("area::")]:
        y = out[s].to_numpy(float)
        if not np.isfinite(y).any():
            continue
        tot = out["_tot"].to_numpy(float) if s.startswith("area::") else y
        mu, sd = np.nanmean(tot[aw]), np.nanstd(tot[aw])
        out[s] = (y - (mu if not s.startswith("area::") else 0)) / sd if sd > 0 else np.nan
    out = out.drop(columns=["_tot"], errors="ignore")
    # single units
    eb = (tc > EV[0]) & (tc < EV[1])
    ez = D[:, :, eb].mean(2)                               # trials x units, z
    ehz = (R[:, keep][:, :, eb] - base[:, keep][:, :, None]).mean(2)
    urows = []
    p_act = pos
    for j in range(ez.shape[1]):
        x = ez[wh, j]
        resp = bool(x.mean() > 0 and np.any(x != 0) and stats.wilcoxon(x).pvalue < 0.01)
        r = dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=meta["cohort"], stage=meta["stage"],
                 electrode_group=units.electrode_group.iloc[j], cluster_id=int(units.cluster_id.iloc[j]),
                 area=units.area_acronym_custom.iloc[j].strip(), area_group=grp[j], responsive=resp,
                 ev_z_wh=float(x.mean()))
        for nm, (ty, l_) in CLS.items():
            m = act & (tt == ty) & (lk == l_)
            r[f"slope_{nm}"] = slope(p_act[m], ez[m, j])
        r["d"] = r["slope_wh_hit"] - r["slope_au_hit"]
        for nm, m in (("pre", (t.epoch == "passive_pre").to_numpy() & wh),
                      ("post", (t.epoch == "passive_post").to_numpy() & wh),
                      ("hit_first", aw & (lk == 1) & (pos < 0.33)), ("hit_last", aw & (lk == 1) & (pos > 0.67)),
                      ("hit", aw & (lk == 1)), ("miss", aw & (lk == 0))):
            r[f"hz_{nm}"] = float(ehz[m, j].mean()) if m.sum() >= 3 else np.nan
        urows.append(r)
    U_ = pd.DataFrame(urows)
    frows = []
    resp = U_.responsive.to_numpy()
    for g in groups:
        sel = resp & (grp == g)
        if sel.sum() < 3:
            continue
        y = ez[:, sel].mean(1)
        base_row = dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=meta["cohort"], stage=meta["stage"],
                        area_group=g, n_units=int(sel.sum()))
        for nm, (ty, l_) in CLS.items():
            m = act & (tt == ty) & (lk == l_)
            b = np.minimum((pos[m] * NB).astype(int), NB - 1)
            for i in range(NB):
                if (b == i).sum() >= 2:
                    frows.append(dict(base_row, cls=nm, bin=i, ez=float(y[m][b == i].mean())))
        for nm, ep in (("pre", "passive_pre"), ("post", "passive_post")):
            m = (t.epoch == ep).to_numpy() & wh
            if m.sum() >= 3:
                frows.append(dict(base_row, cls=nm, bin=-1 if nm == "pre" else NB, ez=float(y[m].mean())))
    return out, U_, pd.DataFrame(frows)


def match_expert(comp, members):
    """expert components vs the day-0 members (013 similarity)"""
    feat = lambda r: dict(tf=np.array(json.loads(r.time_factor)), ap=json.loads(r.area_profile),
                          cp=np.array([getattr(r, k) for k in C13.CLS], float))
    M = [feat(r) for r in members.itertuples()]
    msess = members.session_id.to_numpy()
    loo = [np.mean([C13.similarity(M[i], M[j]) for j in range(len(M)) if msess[j] != msess[i]]) for i in range(len(M))]
    thr = float(np.percentile(loo, 10))
    g = comp[(comp.method == members.method.iloc[0]) & (comp.alignment == AL)].copy()
    g["sim"] = [np.mean([C13.similarity(feat(r), m) for m in M]) for r in g.itertuples()]
    best = g.loc[g.groupby("session_id").sim.idxmax()].copy()
    best["passes"] = best.sim >= thr
    best["is_member"] = best.session_id.isin(set(msess))
    return best, thr, np.array(loo)


def ols(df, reg=REG, suffix=""):
    df = df.dropna(subset=["member"] + reg)
    if len(df) < 20:
        return {}
    Z = df[reg].to_numpy(float)
    sd = Z.std(0)
    if np.any(sd == 0):
        return {}
    Z = np.c_[np.ones(len(Z)), (Z - Z.mean(0)) / sd]
    b = np.linalg.lstsq(Z, df.member.to_numpy(float), rcond=None)[0]
    return {k + suffix: v for k, v in zip(reg, b[1:])}


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


def per_session_stats(T, score):
    """slopes, d, fifths per session for one score column (active trials)"""
    rows = []
    for sid, g in T.groupby("session_id"):
        a = g[g.epoch == "active"]
        r = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0],
                 is_member=bool(np.isfinite(g.member).any()))
        y = a[score].to_numpy(float)
        if not np.isfinite(y).any():
            continue
        for nm, (ty, l_) in CLS.items():
            m = ((a.trial_type == ty) & (a.lick_flag.fillna(0) == l_)).to_numpy()
            r[f"slope_{nm}"] = slope(a.pos.to_numpy()[m], y[m])
            b = np.minimum((a.pos.to_numpy()[m] * NB).astype(int), NB - 1)
            r[f"f_{nm}"] = [np.nanmean(y[m][b == i]) if (b == i).sum() >= 2 else np.nan for i in range(NB)]
        r["d"] = r["slope_wh_hit"] - r["slope_au_hit"]
        for ep in ("passive_pre", "passive_post"):
            m = ((g.epoch == ep) & (g.trial_type == "whisker_trial")).to_numpy()
            r[ep] = float(np.nanmean(g[score].to_numpy(float)[m])) if m.sum() >= 3 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def dots(ax, S, col, title, ylab, zero=0.0):
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
    ax.set_ylabel(ylab)
    ax.set_title(f"{title}\nR+ vs R-: MW p={mw:.3f}, Welch p={we:.3f}", fontsize=5.5, pad=4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))


def fifths(ax, S, title, classes=("wh_hit", "wh_miss", "au_hit"), passive=True):
    xb = (np.arange(NB) + 0.5) / NB
    sty = {"wh_hit": ("-", "o", True), "wh_miss": ("--", "s", False), "au_hit": (":", "^", True)}
    for coh in ("R+", "R-"):
        s = S[S.cohort == coh]
        for nm in classes:
            Y = np.array([x for x in s[f"f_{nm}"]], float)
            if not len(Y):
                continue
            ls, mk, fill = sty[nm]
            ax.errorbar(xb + (0.012 if coh == "R-" else 0), np.nanmean(Y, 0), sem(Y), color=COH[coh], ls=ls, marker=mk,
                        ms=2.3, lw=0.7, elinewidth=0.4, mfc=COH[coh] if fill else "white",
                        label=f"{coh} {nm.replace('_', ' ')}")
        if passive:
            for x, ep in ((-0.12, "passive_pre"), (1.12, "passive_post")):
                v = s[ep].to_numpy(float)
                ax.errorbar(x + (0.02 if coh == "R-" else 0), np.nanmean(v), sem(v), color=COH[coh], marker="D", ms=2.5,
                            elinewidth=0.4, mfc="white")
    ax.set_xticks([-0.12, *xb, 1.12], ["pre", "1", "2", "3", "4", "5", "post"])
    ax.axvspan(-0.2, -0.04, color="0.93", lw=0, zorder=-1)
    ax.axvspan(1.04, 1.2, color="0.93", lw=0, zorder=-1)
    ax.set(xlabel="Passive pre | fifth of the active block | passive post", ylabel="Score (SD; 0 = active whisker mean)",
           title=title)


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / f"{stem}.png")


def page_examples(T, S, out, stem):
    s0 = S[(S.stage == "learning") & S.is_member].dropna(subset=["d"])
    for coh in ("R+", "R-"):   # every member mouse, one page per cohort, sorted by d
        pick = list(s0[s0.cohort == coh].sort_values("d").itertuples())
        example_page(T, pick, coh, out, stem)


def example_page(T, pick, coh, out, stem):
    nc = 5
    nr = int(np.ceil(len(pick) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.6, 1.25 * nr + 0.6), gridspec_kw=dict(hspace=0.9, wspace=0.6),
                            squeeze=False)
    for ax in axs.ravel()[len(pick):]:
        ax.axis("off")
    for ax, r in zip(axs.ravel(), pick):
        g = T[T.session_id == r.session_id].reset_index(drop=True)
        x = np.arange(len(g))
        pas = g.epoch.str.startswith("passive").to_numpy()
        e = np.flatnonzero(np.diff(np.r_[0, pas.astype(int), 0]))
        for a0, a1 in zip(e[::2], e[1::2]):
            ax.axvspan(a0 - 0.5, a1 - 0.5, color="0.93", lw=0, zorder=-1)
        tt, lk = g.trial_type.to_numpy(), g.lick_flag.fillna(0).to_numpy().astype(int)
        y = g.member.to_numpy(float)
        for ty, col, s in (("whisker_trial", WH, 5), ("auditory_trial", AU, 2)):
            for l_ in (1, 0):
                m = (tt == ty) & (lk == l_)
                ax.scatter(x[m], y[m], s=s, facecolors=col if l_ else "none", edgecolors=col, lw=0.4,
                           alpha=0.4 if ty == "auditory_trial" else 0.85)
        w = np.flatnonzero(tt == "whisker_trial")
        rm = pd.Series(y[w]).rolling(15, center=True, min_periods=5).median()
        ax.plot(x[w], rm, color="k", lw=0.8)
        ax2 = ax.twinx()
        ax2.plot(x, g.p_wh_i.to_numpy(float), color=COH[r.cohort], lw=0.7, alpha=0.8)
        ax2.set_ylim(0, 1.05)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(labelsize=4.5)
        ax.set_ylim(np.nanpercentile(y, 0.5) - 0.2, np.nanpercentile(y, 99.5) + 0.2)
        ax.set_title(f"{r.cohort} {r.mouse_id}  d={r.d:+.2f}", fontsize=5.5, color=COH[r.cohort])
        ax.tick_params(labelsize=4.5)
    for ax in axs[-1]:
        ax.set_xlabel("Trial in session", fontsize=5)
    for ax in axs[:, 0]:
        ax.set_ylabel("Trial weight (SD)", fontsize=5)
    fig.suptitle(f"All {coh} mice with a member ({len(pick)}), sorted by d (whisker-hit minus auditory-hit slope, "
                 "lowest first). Orange = whisker (filled hit, open miss), blue = auditory;\nblack = running median of "
                 "whisker trials (15); coloured line = P(lick | whisker) (right axis); grey = passive blocks. Weight in "
                 "SD of the active whisker trials, 0 = their mean.", fontsize=6, y=1.0)
    save(fig, out, f"{stem}_examples_{'Rp' if coh == 'R+' else 'Rm'}")


def page_factors(T, S, R_, out, stem):
    fig = plt.figure(figsize=(7.6, 9.6))
    gs = fig.add_gridspec(4, 3, hspace=0.95, wspace=0.6)
    s0 = S[(S.stage == "learning") & S.is_member]
    ax = fig.add_subplot(gs[0, :2])
    fifths(ax, s0, "A  Session course per trial class (mean ± SEM over mice)")
    ax.legend(frameon=False, fontsize=4.5, ncol=3, loc="upper left")
    # B performance-binned
    ax = fig.add_subplot(gs[0, 2])
    tm = T[(T.stage == "learning") & (T.epoch == "active") & (T.trial_type == "whisker_trial") & T.member.notna()].copy()
    edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.01])
    tm["pb"] = np.digitize(tm.p_wh_i, edges) - 1
    for coh in ("R+", "R-"):
        for l_, ls, fill in ((1, "-", True), (0, "--", False)):
            g = tm[(tm.cohort == coh) & (tm.lick_flag == l_)]
            M = g.groupby(["mouse_id", "pb"]).member.mean().unstack().reindex(columns=range(5))
            ax.errorbar(np.arange(5) * 0.2 + 0.1, M.mean(), sem(M.to_numpy()), color=COH[coh], ls=ls, marker="o", ms=2.3,
                        lw=0.7, elinewidth=0.4, mfc=COH[coh] if fill else "white")
    ax.set(xlabel="P(lick | whisker), learning curve", ylabel="Trial weight (SD)",
           title="B  By performance level\n(whisker hits filled, misses open)")
    # C regression coefficients
    for j, v in enumerate(REG):
        ax = fig.add_subplot(gs[1 + j // 3, j % 3]) if j < 5 else None
        dots(ax, R_, v, f"{'CDEFG'[j]}  {REG_LAB[v]} coefficient", "Std. coefficient")
    # H passive
    ax = fig.add_subplot(gs[2, 2])
    s0 = s0.assign(pp=s0.passive_post - s0.passive_pre)
    dots(ax, s0, "pp", "H  Passive post - pre (whisker)", "Difference (SD)")
    for j, v in enumerate(["pos", "p_wh_i", "p_au_i"]):
        ax = fig.add_subplot(gs[3, j])
        dots(ax, R_, f"{v}_m2", f"{'IJK'[j]}  {REG_LAB[v]} coefficient, model 2", "Std. coefficient")
    fig.suptitle("Trial factors of the member score (whisker day 0, mice with a member). C-G: per-mouse OLS on active "
                 "whisker trials with standardised predictors (position, hit, previous trial rewarded, pre-stimulus "
                 "population rate, log ITI);\nthe position coefficient is the trend that remains after the other "
                 "factors. I-K: model 2 = the same + P(lick | whisker) and P(lick | auditory) of the learning curves (performance /"
                 " engagement state).\nExploratory, uncorrected; dots = mice, bar = median, p = Wilcoxon vs 0.", fontsize=6,
                 y=0.995)
    save(fig, out, f"{stem}_factors")


def page_area(T, U, G, out, stem):
    from ephys_utilities.allen_utils import allen_utils as AU_
    order, colours = AU_.get_area_group_custom_order(), AU_.get_custom_area_groups_colors()
    fig = plt.figure(figsize=(7.6, 10.6))
    gs = fig.add_gridspec(4, 2, hspace=1.05, wspace=0.35)
    tm = T[(T.stage == "learning") & (T.epoch == "active") & T.member.notna()]
    acols = [c for c in tm.columns if c.startswith("area::")]
    rows = []
    for sid, g in tm.groupby("session_id"):
        aw_ = (g.trial_type == "whisker_trial").to_numpy()
        tot = g[acols].fillna(0).sum(1).to_numpy()[aw_].mean()
        for c in acols:
            y = g[c].to_numpy(float)
            if not np.isfinite(y).any():
                continue
            r = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], group=c[6:],
                     level=float(np.nanmean(y[aw_]) / tot) if tot > 0 else np.nan)
            for nm, (ty, l_) in CLS.items():
                m = ((g.trial_type == ty) & (g.lick_flag.fillna(0) == l_)).to_numpy()
                r[nm] = slope(g.pos.to_numpy()[m], y[m])
            r["d"] = r["wh_hit"] - r["au_hit"]
            rows.append(r)
    A_ = pd.DataFrame(rows)
    gl = [g for g in order if A_[A_.group == g].mouse_id.nunique() >= 5]
    x = np.arange(len(gl))
    for i, (col, lab) in enumerate((
            ("level", "A  Share of the member projection per area group (active whisker trials; median, IQR)"),
            ("d", "B  Contribution to the projection trend d (SD units; contributions add up; mean ± SEM)"))):
        ax = fig.add_subplot(gs[i, :])
        for k, coh in enumerate(("R+", "R-")):
            a = A_[A_.cohort == coh].groupby(["mouse_id", "group"])[col].mean().unstack().reindex(columns=gl)
            a0 = a.fillna(0)   # an area not recorded in a mouse contributes 0 to its total
            if col == "level":
                ctr = a0.median()
                err = np.vstack([ctr - a0.quantile(0.25), a0.quantile(0.75) - ctr])
            else:
                ctr, err = a0.mean(), sem(a0.to_numpy())
            ax.bar(x + (k - 0.5) * 0.38, ctr, 0.36, yerr=err, color=[colours[g] for g in gl], edgecolor=COH[coh],
                   lw=0.8, alpha=0.9 if coh == "R+" else 0.55, error_kw=dict(lw=0.4))
        if col == "d":
            for j, g in enumerate(gl):
                a = A_[A_.group == g].groupby(["mouse_id", "cohort"])[col].mean().reset_index()
                mw, we = two(a.loc[a.cohort == "R+", col], a.loc[a.cohort == "R-", col])
                if np.isfinite(mw) and mw < 0.05:
                    ax.text(j, 0.98, f"MW {mw:.3f}\nW {we:.3f}", transform=ax.get_xaxis_transform(), ha="center",
                            va="top", fontsize=4)
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_xticks(x, [g.replace("Somatosensory-", "Ss-") for g in gl], rotation=45, ha="right", fontsize=5)
        ax.set_title(lab + " (left bar R+, right R-)", fontsize=6)
    # C / D model-free single units
    u = U[(U.stage == "learning") & U.responsive].dropna(subset=["d"])
    for j, (key, lab) in enumerate((("area_group", "C  Single units (no TCA), area groups"),
                                    ("area", "D  Same, single areas"))):
        s = u.groupby(["mouse_id", "cohort", key]).d.mean().reset_index()
        cnt = s.groupby([key, "cohort"]).mouse_id.nunique().unstack().reindex(columns=["R+", "R-"]).fillna(0)
        keep = cnt[cnt.min(1) >= 5].index.tolist()
        if key == "area_group":
            keep = [g for g in order if g in keep]
        else:
            keep = sorted(keep, key=lambda a: -cnt.loc[a].sum())[:14]
        ax = fig.add_subplot(gs[2, j])
        for i, g in enumerate(keep):
            for k, coh in enumerate(("R+", "R-")):
                v = s[(s[key] == g) & (s.cohort == coh)].d.to_numpy()
                ax.errorbar(i + (k - 0.5) * 0.3, np.mean(v), sem(v), color=COH[coh], marker="o", ms=2.5,
                            elinewidth=0.6)
            mw, we = two(s[(s[key] == g) & (s.cohort == "R+")].d, s[(s[key] == g) & (s.cohort == "R-")].d)
            ax.text(i, 0.99, f"{mw:.2f}\n{we:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=4, color="k" if mw < 0.05 else "0.6")
        ax.axhline(0, color="0.5", lw=0.4)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.25 * (hi - lo))
        ax.set_xticks(range(len(keep)), [g.replace("Somatosensory-", "Ss-") for g in keep], rotation=45, ha="right",
                      fontsize=5)
        ax.set(ylabel="d of evoked z (per block)")
        ax.set_title(f"{lab}: whisker-responsive units, mean d per mouse\n(>= 5 mice per cohort; top: MW / Welch p)",
                     fontsize=5.5)
    for j, g in enumerate(["Somatosensory-whisker", "Thalamus", "Midbrain", "Striatum"]):
        ax = fig.add_subplot(gs[3, :].subgridspec(1, 4, wspace=0.45)[0, j])
        h = G[(G.stage == "learning") & (G.area_group == g)]
        for coh in ("R+", "R-"):
            for nm, ls, mk, fill in (("wh_hit", "-", "o", True), ("wh_miss", "--", "s", False),
                                     ("au_hit", ":", "^", True)):
                M = h[(h.cohort == coh) & (h.cls == nm)].groupby(["mouse_id", "bin"]).ez.mean().unstack()
                M = M.reindex(columns=range(NB))
                ax.errorbar(np.arange(NB) + (0.1 if coh == "R-" else 0), M.mean(), sem(M.to_numpy()), color=COH[coh],
                            ls=ls, marker=mk, ms=2, lw=0.6, elinewidth=0.4, mfc=COH[coh] if fill else "white")
            for nm, xx in (("pre", -1), ("post", NB)):
                v = h[(h.cohort == coh) & (h.cls == nm)].groupby("mouse_id").ez.mean()
                ax.errorbar(xx + (0.1 if coh == "R-" else 0), v.mean(), sem(v.to_numpy()), color=COH[coh], marker="D",
                            ms=2.3, mfc="white", elinewidth=0.4)
        nmz = h.groupby("cohort").mouse_id.nunique()
        ax.set_xticks(range(-1, NB + 1), ["pre", "1", "2", "3", "4", "5", "post"], fontsize=4.5)
        ax.set_title(f"{'EFGH'[j]}  {g.replace('Somatosensory-', 'Ss-')}\n(R+ {nmz.get('R+', 0)}, R- "
                     f"{nmz.get('R-', 0)} mice)", fontsize=5.5)
        if j == 0:
            ax.set_ylabel("Evoked z, responsive units")
    fig.suptitle("Where is the trend? A-B: member projection split by area group (whisker day 0, mice with a member). "
                 "C-H: no TCA, all day-0 mice, whisker-responsive units\n(Wilcoxon p < 0.01 over whisker trials); evoked "
                 "z = mean of the 12.5-42.5 ms bins; E-H: session course (whisker hit filled circles, miss open squares, "
                 "auditory hit triangles;\npassive pre / post diamonds; mean ± SEM over mice). Exploratory, uncorrected.",
                 fontsize=6, y=0.995)
    save(fig, out, f"{stem}_area")
    return A_


def page_expert(T, S, U, best, thr, loo, out, stem):
    fig = plt.figure(figsize=(7.6, 7.0))
    gs = fig.add_gridspec(3, 3, hspace=1.0, wspace=0.6)
    ax = fig.add_subplot(gs[0, 0])
    b0 = best[best.stage == "learning"]
    ax.hist(loo, bins=20, color="0.6", alpha=0.7, density=True, label="day-0 members (LOO)")
    ax.hist(b0[~b0.is_member].sim, bins=20, color="k", histtype="step", density=True, label="day 0, no member (best)")
    ax.hist(best[best.stage == "expert"].sim, bins=20, color="tab:red", histtype="step", density=True,
            label="expert (best)")
    ax.axvline(thr, color="k", ls=":", lw=0.6)
    ax.legend(frameon=False, fontsize=4.5)
    fp = float(b0[~b0.is_member].passes.mean())
    ax.set(xlabel="Mean similarity to the day-0 members", ylabel="Density",
           title=f"A  Matching (threshold = 10th pct;\nfalse positive {fp:.0%} of day-0 non-member sessions)")
    ax = fig.add_subplot(gs[0, 1])
    e = best[best.stage == "expert"]
    for i, coh in enumerate(("R+", "R-")):
        d0 = S[(S.stage == "learning") & (S.cohort == coh)]
        ax.bar(i - 0.18, d0.is_member.mean(), 0.34, color=COH[coh], alpha=0.5)
        ee = e[e.cohort == coh]
        ax.bar(i + 0.18, ee.passes.mean(), 0.34, color=COH[coh])
        ax.text(i - 0.18, d0.is_member.mean() + 0.02, f"{d0.is_member.sum()}/{len(d0)}", ha="center", fontsize=4.5)
        ax.text(i + 0.18, ee.passes.mean() + 0.02, f"{ee.passes.sum()}/{len(ee)}", ha="center", fontsize=4.5)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set(ylim=(0, 1.1), ylabel="Share of sessions", title="B  Sessions with the component\n(light day 0, dark expert)")
    ax = fig.add_subplot(gs[0, 2])
    se = S[S.stage == "expert"]
    fifths(ax, se, "C  Expert sessions with a match", classes=("wh_hit", "wh_miss", "au_hit"))
    ax.set_xlabel("pre | fifth | post", fontsize=5)
    for j, (stg, lab) in enumerate((("learning", "day 0"), ("expert", "expert"))):
        ax = fig.add_subplot(gs[1, j])
        dots(ax, S[S.stage == stg], "d", f"{'DE'[j]}  d, {lab} (member / matched)", "Whisker-hit - auditory-hit slope")
    # F evoked Hz Ss-whisker responsive units: day0 first / last third of hits, expert
    ax = fig.add_subplot(gs[1, 2])
    u = U[U.responsive & (U.area_group.str.contains("whisker"))]
    for i, coh in enumerate(("R+", "R-")):
        uu = u[u.cohort == coh]
        vals = [uu[uu.stage == "learning"].groupby("mouse_id").hz_hit_first.mean(),
                uu[uu.stage == "learning"].groupby("mouse_id").hz_hit_last.mean(),
                uu[uu.stage == "expert"].groupby("mouse_id").hz_hit.mean()]
        for k, v in enumerate(vals):
            xx = i * 4 + k
            ax.scatter(xx + (np.random.default_rng(k).random(len(v)) - 0.5) * 0.3, v, s=2.5, color=COH[coh], lw=0)
            ax.plot([xx - 0.25, xx + 0.25], [np.nanmedian(v)] * 2, color="k", lw=0.9)
        cc = pd.concat(vals[:2], axis=1).dropna()
        p_ = stats.wilcoxon(cc.iloc[:, 0], cc.iloc[:, 1]).pvalue if len(cc) >= 5 else np.nan
        ax.text(i * 4 + 0.5, 1.0, f"first vs last p={p_:.3f}", transform=ax.get_xaxis_transform(), ha="center",
                fontsize=4.5)
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["d0 1st", "d0 last", "exp."] * 2, fontsize=5, rotation=30)
    ax.set(ylabel="Evoked rate (Hz), whisker hits", title="F  Ss-whisker responsive units\n(mouse means; R+ left, R- right)")
    # G paired day0 vs expert member ratio hit/miss
    for j, col in enumerate(("slope_wh_hit", "slope_wh_miss")):
        ax = fig.add_subplot(gs[2, j])
        dots(ax, se, col, f"{'GH'[j]}  Expert: {col.replace('slope_', '').replace('_', ' ')} slope", "Slope")
    ax = fig.add_subplot(gs[2, 2])
    hm = []
    for stg in ("learning", "expert"):
        g = T[(T.stage == stg) & (T.epoch == "active") & (T.trial_type == "whisker_trial") & T.member.notna()]
        r = g.groupby(["mouse_id", "cohort", "lick_flag"]).member.mean().unstack()
        r = (r[1] - r[0]).rename("ratio").reset_index()
        r["stage"] = stg
        hm.append(r)
    hm = pd.concat(hm)
    for i, coh in enumerate(("R+", "R-")):
        for k, stg in enumerate(("learning", "expert")):
            v = hm[(hm.cohort == coh) & (hm.stage == stg)].ratio.dropna()
            xx = i * 3 + k
            ax.scatter(xx + (np.random.default_rng(k).random(len(v)) - 0.5) * 0.3, v, s=2.5, color=COH[coh], lw=0)
            ax.plot([xx - 0.25, xx + 0.25], [np.median(v)] * 2, color="k", lw=0.9)
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_xticks([0, 1, 3, 4], ["d0", "exp.", "d0", "exp."])
    ax.set(ylabel="Hit - miss weight (SD)", title="I  Hit - miss weight\n(R+ left, R- right)")
    fig.suptitle("Expert sessions (whisker day 1-4; 36 sessions). A-B: matching of the expert components to the day-0 "
                 "cluster; C-E, G-I: matched components; F: model-free evoked rates (no TCA). Exploratory, uncorrected.",
                 fontsize=6, y=0.995)
    save(fig, out, f"{stem}_expert")


def page_methods(T, out, stem):
    fig, axs = plt.subplots(len(SCORES), 3, figsize=(7.6, 2.0 * len(SCORES)),
                            gridspec_kw=dict(hspace=1.0, wspace=0.55, width_ratios=[1.6, 0.8, 0.8]))
    rows = []
    for i, (sc, lab) in enumerate(SCORES.items()):
        S = per_session_stats(T[T.stage == "learning"], sc)
        S = S[S.is_member] if sc == "member" else S
        fifths(axs[i, 0], S, f"{lab}: session course" + ("" if sc == "member" else " (all day-0 mice)"))
        if i == 0:
            axs[i, 0].legend(frameon=False, fontsize=4.5, ncol=3, loc="upper left")
        dots(axs[i, 1], S, "d", f"{lab}: d", "Whisker-hit - auditory-hit slope")
        if sc != "member":
            sm = S[S.is_member]
            dots(axs[i, 2], sm, "d", f"{lab}: d,\nmember mice only", "d")
        else:
            cc = []
            for sid, g in T[(T.stage == "learning") & T.member.notna() & (T.trial_type == "whisker_trial")
                            & (T.epoch == "active")].groupby("session_id"):
                cc.append(dict(cohort=g.cohort.iloc[0], **{s: stats.spearmanr(g.member, g[s], nan_policy="omit")[0]
                                                            for s in list(SCORES)[1:]}))
            cc = pd.DataFrame(cc)
            ax = axs[i, 2]
            for k, s in enumerate(list(SCORES)[1:]):
                v = cc[s].dropna()
                ax.scatter(k + (np.random.default_rng(k).random(len(v)) - 0.5) * 0.3, v, s=2.5, color="0.4", lw=0)
                ax.plot([k - 0.25, k + 0.25], [v.median()] * 2, color="k")
            ax.set_xticks(range(3), ["templ.", "PCA", "stim ax."])
            ax.set(ylim=(-0.3, 1.05), ylabel="Spearman r with member score",
                   title="Agreement per session\n(active whisker trials)")
        for coh in ("R+", "R-"):
            rows.append(dict(score=sc, cohort=coh, n=int((S.cohort == coh).sum()),
                             d_median=float(S.loc[S.cohort == coh, "d"].median()),
                             d_p=wil(S.loc[S.cohort == coh, "d"])))
        rows[-1]["mw_p"], rows[-1]["welch_p"] = two(S.loc[S.cohort == "R+", "d"], S.loc[S.cohort == "R-", "d"])
    fig.suptitle("Is the trend found without TCA? Template = projection on the passive-pre whisker response; PCA = PC "
                 "(of 5) best separating whisker from auditory, uncentred scores;\nstimulus axis = cross-validated whisker "
                 "- auditory mean response (stimulus marginal of dPCA). Bins >= +7.5 ms. Exploratory, uncorrected.",
                 fontsize=6, y=0.995)
    save(fig, out, f"{stem}_methods")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="_splitnn")
    ap.add_argument("--cluster", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    stem = f"deep_{a.method.strip('_')}_cl{a.cluster}"
    tdir = L.OUT / "tables"
    out = L.OUT / "figures" / "aligned_gpu" / "candidates"
    sess = pd.read_csv(tdir / "sessions.csv")
    mem = pd.read_parquet(tdir / "cluster_members.parquet")
    mem = mem[(mem.method == a.method) & (mem.alignment == AL) & (mem.cluster == a.cluster)]
    comp = pd.read_parquet(tdir / "components.parquet")
    comp = comp[comp.error.isna()] if "error" in comp else comp
    best, thr, loo = match_expert(comp, mem)
    best.drop(columns=[c for c in best.columns if c.startswith(("learn_", "perf_"))]).to_parquet(
        tdir / f"{stem}_expert.parquet", index=False)
    print(f"threshold {thr:.3f}; expert matches", best[best.stage == "expert"].groupby("cohort").passes.agg(["sum", "size"]))
    if a.reuse and (tdir / f"{stem}_trials.parquet").exists():
        T = pd.read_parquet(tdir / f"{stem}_trials.parquet")
        U = pd.read_parquet(tdir / f"{stem}_units.parquet")
        G = pd.read_parquet(tdir / f"{stem}_areafifths.parquet")
    else:
        jobs = []
        for r in sess.itertuples():
            if r.stage == "learning":
                c = mem[mem.session_id == r.session_id]
                comps = [(int(x.rank), int(x.comp)) for x in c.itertuples()]
            else:
                b = best[(best.session_id == r.session_id) & best.passes]
                comps = [(int(x.rank), int(x.comp)) for x in b.itertuples()]
            meta = dict(session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, day=r.day,
                        method=a.method)
            jobs.append(delayed(session)(r.session_id, meta, comps))
        res = Parallel(n_jobs=a.n_jobs, verbose=2)(jobs)
        T = pd.concat([x[0] for x in res], ignore_index=True)
        U = pd.concat([x[1] for x in res], ignore_index=True)
        G = pd.concat([x[2] for x in res], ignore_index=True)
        T.to_parquet(tdir / f"{stem}_trials.parquet", index=False)
        U.to_parquet(tdir / f"{stem}_units.parquet", index=False)
        G.to_parquet(tdir / f"{stem}_areafifths.parquet", index=False)
    S = per_session_stats(T, "member")
    S = S[S.is_member | (S.stage == "expert")]
    regs = []
    for sid, g in T[(T.epoch == "active") & (T.trial_type == "whisker_trial") & T.member.notna()].groupby("session_id"):
        g = g.assign(hit=g.lick_flag.fillna(0))
        regs.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0],
                         **ols(g), **ols(g, REG2, "_m2")))
    R_ = pd.DataFrame(regs)
    page_examples(T, S, out, stem)
    page_factors(T, S, R_[R_.stage == "learning"], out, stem)
    A_ = page_area(T, U, G, out, stem)
    page_expert(T, S, U, best, thr, loo, out, stem)
    M = page_methods(T, out, stem)
    R_.to_csv(tdir / f"{stem}_regression.csv", index=False)
    A_.to_csv(tdir / f"{stem}_area.csv", index=False)
    M.to_csv(tdir / f"{stem}_methods.csv", index=False)
    print(M.to_string())
    print(R_.groupby(["stage", "cohort"])[REG + [c + "_m2" for c in REG2]].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
