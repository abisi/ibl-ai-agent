"""Facemap lick decoding, step 3 (haas): pre-stimulus Facemap features -> upcoming whisker lick, all whisker days,
and correlation with the exact learning curves. Same models and measures as ssl-facemap-pilot/003_analysis.py.

Inputs: clips/<session>_clips.npz (step 1: per clips-video frame the clip, NWB timestamp, lead flag), facemap/
<session>_facemap_small.npz (step 2, rclone'd back from Kuma), NWB_ks4 trials and DLC keypoints, exact learning curves
combined_results_ks4/<mouse>/whisker_<day>/learning_curve/<mouse>_whisker_<day>_whisker_trial_learning_curve_exact.h5.

Per session (whisker trials, context != passive, perf != 6, early_lick == 0 = the exact-curve trials):
  features   Facemap SVDs z-scored over the session's clip frames (lead frames dropped); each trial's features are
             taken from its own clip (matched on start_time): fm_mot / fm_mov (first N_COMP components) and fm_energy
             averaged over -0.2, -1, -2 s, fm_mot_bins_1s (20 components, 250 ms bins over -1 s); kpca_mean_1s = DLC
             keypoint PCA (dlc_kpca.py, from the ssl-face-lick-decoding pilot) for comparison
  decoding   upcoming lick, logistic regression (class-balanced, L2 with C chosen by inner 3-fold CV), blocked 5-fold
             CV (train on 4 contiguous blocks, predict the held-out block); models: time (trial index, index^2),
             history (previous lick, reward, whisker lick, log time since last lick), base = time + history, each
             feature set alone and base + set. Metrics on the held-out predictions: r = Pearson correlation between
             predicted P(lick) and the actual lick (0/1), balanced accuracy (threshold 0.5, chance 0.5), AUC.
             Linear-shift null for NULL_KEYS: licks shifted against the features by s trials (10 < |s| <= n/3,
             N_SHIFT_DEC shifts), same CV pipeline; z and one-sided P of r and balanced accuracy per session
  learning   Pearson r (lag 0) between per-trial measures (pre-stimulus motion energy -1 s; cross-validated lick
             probability of the fm_mot -1 s model) and the exact curve p_mean / p_mean - p_chance; linear-shift null
             (500 shifts, |shift| > 10 trials), z and P per session
Outputs: combined_results_ks4/ssl-facemap-lick-decoding/analysis/ (tables, figure).
Run on haas: nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 003_analysis.py [--jobs 24]
"""

import zlib
import argparse
import importlib.util
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dlc_kpca", HERE / "dlc_kpca.py")
dlc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dlc)  # load, trial_table, keypoint_features, blocked_auc

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
BASE = ROOT / "combined_results_ks4" / "ssl-facemap-lick-decoding"
OUT = BASE / "analysis"
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
N_COMP = 50
WINDOWS = {"0.2s": (-0.2, 0.0), "1s": (-1.0, 0.0), "2s": (-2.0, 0.0)}
BIN_S = 0.25
N_SHIFT, SHIFT_EXCLUDE = 500, 10
COLORS = {"R+": "#00B400", "R-": "#C800C8"}
NULL_KEYS = ["time", "history", "base", "kpca_mean_1s", "fm_mot_1s", "base+kpca_mean_1s", "base+fm_mot_1s"]
N_SHIFT_DEC = 100
METRICS = ("r", "bacc", "auc")
KEYS = ["time", "history", "base", "kpca_mean_1s", "fm_energy_1s", "fm_mot_1s", "fm_mov_1s", "fm_mot_bins_1s",
        "base+kpca_mean_1s", "base+fm_mot_1s", "base+fm_both_1s"]


def window_means(X, frames, ftimes, clip_of_trial, starts, a, b, bins=None):
    """Mean of X over [start + a, start + b) within each trial's own clip; frames[c] = row indices of clip c in X,
    ftimes[c] = their NWB timestamps. NaN row when the trial has no clip or the window is too short."""
    width = X.shape[1] * (bins or 1)
    rows = []
    for c, s in zip(clip_of_trial, starts):
        if c < 0:
            rows.append(np.full(width, np.nan))
            continue
        j0, j1 = np.searchsorted(ftimes[c], [s + a, s + b])
        sel = frames[c][j0:j1]
        if len(sel) < 2:
            rows.append(np.full(width, np.nan))
        elif bins:
            e = np.linspace(0, len(sel), bins + 1).astype(int)
            rows.append(np.concatenate([X[sel[e0:e1]].mean(0) for e0, e1 in zip(e[:-1], e[1:])]))
        else:
            rows.append(X[sel].mean(0))
    return np.array(rows)


def cv_proba(X, y):
    """Blocked-CV predicted lick probability per trial (same model as dlc.blocked_auc)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    X = np.where(np.isfinite(X), X, 0.0)
    n = len(y)
    pred = np.full(n, np.nan)
    edges = np.linspace(0, n, dlc.N_FOLDS + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.zeros(n, bool)
        test[a:b] = True
        if len(np.unique(y[~test])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3,
                                                                   class_weight="balanced", scoring="roc_auc",
                                                                   max_iter=2000))
        clf.fit(X[~test], y[~test])
        pred[test] = clf.predict_proba(X[test])[:, 1]
    return pred


def dec_metrics(X, y):
    """Held-out metrics of the blocked-CV model: r (predicted P(lick) vs lick 0/1), balanced accuracy, AUC."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    nan = dict(r=np.nan, bacc=np.nan, auc=np.nan)
    if len(y) < 30 or y.sum() < 5 or (1 - y).sum() < 5:
        return nan
    p = cv_proba(X, y)
    ok = np.isfinite(p)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.std(p[ok]) == 0:
        return nan
    return dict(r=np.corrcoef(p[ok], y[ok])[0, 1], bacc=balanced_accuracy_score(y[ok], p[ok] >= 0.5),
                auc=roc_auc_score(y[ok], p[ok]))


def shift_null(X, y, obs, rng):
    """Linear-shift null of the decoding metrics: licks shifted against the features, same CV pipeline."""
    n = len(y)
    shifts = [s for s in range(-(n // 3), n // 3 + 1) if abs(s) > SHIFT_EXCLUDE]
    if not shifts:
        return {}
    null = pd.DataFrame([dec_metrics(X[s:], y[:-s]) if s > 0 else dec_metrics(X[:s], y[-s:])
                         for s in rng.choice(shifts, min(N_SHIFT_DEC, len(shifts)), replace=False)])
    out = {}
    for met in ("r", "bacc"):
        v = null[met].dropna().to_numpy()
        out[f"{met}_null_mean"] = v.mean() if len(v) else np.nan
        out[f"{met}_null_z"] = (obs[met] - v.mean()) / v.std() if len(v) > 1 and v.std() > 0 else np.nan
        out[f"{met}_null_p"] = (1 + np.sum(v >= obs[met])) / (1 + len(v)) if len(v) and np.isfinite(obs[met]) else np.nan
    return out


def shift_corr(x, y, rng):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 * SHIFT_EXCLUDE or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    shifts = [s for s in range(-n + 1, n) if abs(s) > SHIFT_EXCLUDE and n - abs(s) >= 10]
    null = []
    for s in rng.choice(shifts, min(N_SHIFT, len(shifts)), replace=False):
        null.append(np.corrcoef(x[s:], y[:-s])[0, 1] if s > 0 else np.corrcoef(x[:s], y[-s:])[0, 1])
    null = np.array(null)
    return r, (r - np.nanmean(null)) / np.nanstd(null), float(np.nanmean(np.abs(null) >= abs(r)))


def session(sid, cohort):
    rng = np.random.default_rng(zlib.crc32(sid.encode()))  # stable across runs (hash() is salted per process)
    fm = np.load(BASE / "facemap" / f"{sid}_facemap_small.npz")
    ix = np.load(BASE / "clips" / f"{sid}_clips.npz", allow_pickle=True)
    keep = ix["lead"] == 0
    assert fm["motSVD"].shape[0] == len(keep), (sid, fm["motSVD"].shape, len(keep))
    clip, ft = ix["clip"][keep], ix["t"][keep]
    z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-9)  # noqa: E731
    mot, mov = z(fm["motSVD"][keep, :N_COMP]), z(fm["movSVD"][keep, :N_COMP])
    energy = fm["motion"][keep][:, None].astype(float)
    frames = {c: np.flatnonzero(clip == c) for c in np.unique(clip)}
    ftimes = {c: ft[f] for c, f in frames.items()}
    clip_start = np.asarray(ix["clips"]["start_time"], float)

    nwb = ROOT / "NWB_ks4" / f"{sid}.nwb"
    desc, trials, series, bp4, mismatch = dlc.load(nwb)
    t = dlc.trial_table(trials)
    y = t.lick_flag.to_numpy()
    starts = t.start_time.to_numpy()
    d = np.abs(starts[:, None] - clip_start[None, :])
    clip_of_trial = np.where(d.min(1) < 1e-3, d.argmin(1), -1)

    sets = {"time": np.column_stack([t.trial_idx, t.trial_idx ** 2]),
            "history": t[["prev_lick", "prev_reward", "prev_whisker_lick", "log_since_lick"]].to_numpy(float)}
    sets["base"] = np.hstack([sets["time"], sets["history"]])
    for w, (a, b) in WINDOWS.items():
        sets[f"fm_mot_{w}"] = window_means(mot, frames, ftimes, clip_of_trial, starts, a, b)
        sets[f"fm_mov_{w}"] = window_means(mov, frames, ftimes, clip_of_trial, starts, a, b)
        sets[f"fm_energy_{w}"] = window_means(energy, frames, ftimes, clip_of_trial, starts, a, b)
    sets["fm_mot_bins_1s"] = window_means(mot[:, :20], frames, ftimes, clip_of_trial, starts, -1.0, 0.0,
                                          bins=int(1 / BIN_S))
    sets["fm_both_1s"] = np.hstack([sets["fm_mot_1s"], sets["fm_mov_1s"]])
    if series and max(mismatch.values(), default=0) <= dlc.MAX_LEN_MISMATCH:
        kf, _ = dlc.keypoint_features(series, starts)
        sets["kpca_mean_1s"] = kf["kpca_mean_1s"]
    for k in [k for k in list(sets) if k.startswith(("fm_", "kpca_"))]:
        sets[f"base+{k}"] = np.hstack([sets["base"], sets[k]])
    dec = {k: dec_metrics(X, y) for k, X in sets.items()}
    nulls = {k: shift_null(sets[k], y, dec[k], rng) for k in NULL_KEYS if k in sets}

    m = sid.split("_")[0]
    behavior, day = desc.replace("+", "").rsplit("_", 1)
    day = int(day)
    measures = {"motion_energy_1s": sets["fm_energy_1s"][:, 0], "lick_proba_fm_mot_1s": cv_proba(sets["fm_mot_1s"], y)}
    corr, p_mean, p_diff = [], np.full(len(y), np.nan), np.full(len(y), np.nan)
    cpath = ROOT / "combined_results_ks4" / m / f"whisker_{day}" / "learning_curve" / \
        f"{m}_whisker_{day}_whisker_trial_learning_curve_exact.h5"
    if cpath.exists():
        cur = pd.read_hdf(cpath)
        pm = np.asarray(cur.p_mean.values[0], float)
        if len(pm) == len(y):
            p_mean, p_diff = pm, pm - np.asarray(cur.p_chance.values[0], float)
    for name, x in measures.items():
        for cname, c in (("p_mean", p_mean), ("p_mean_minus_chance", p_diff)):
            r, zval, p = shift_corr(x, c, rng)
            corr.append(dict(session_id=sid, mouse_id=m, cohort=cohort, day=day, measure=name, curve=cname, r=r,
                             z_shift=zval, p_shift=p))
    trial_df = pd.DataFrame(dict(session_id=sid, mouse_id=m, cohort=cohort, day=day, trial=np.arange(len(y)), lick=y,
                                 p_mean=p_mean, p_diff=p_diff, has_clip=clip_of_trial >= 0, **measures))
    row = dict(session_id=sid, mouse_id=m, cohort=cohort, day=day, n_trials=len(y), n_with_clip=int((clip_of_trial >= 0).sum()),
               lick_rate=y.mean(), curve_ok=bool(np.isfinite(p_mean).any()),
               **{f"{met}_{k}": v[met] for k, v in dec.items() for met in METRICS},
               **{f"{name}_{k}": val for k, d in nulls.items() for name, val in d.items()})
    return row, corr, trial_df


def safe(args):
    sid, cohort = args
    try:
        return session(sid, cohort)
    except Exception as e:
        return dict(session_id=sid, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-600:]), [], None


def tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def figure(rows, corr):
    rows = rows.assign(stage=np.where(rows.day == 0, "day 0", np.where(rows.day > 0, "expert (day > 0)", "pre (day < 0)")))
    stages = [s for s in ("day 0", "expert (day > 0)") if s in set(rows.stage)]
    fig, axs = plt.subplots(2, 2 + len(stages) - 1, figsize=(5 * (1 + len(stages)), 9), squeeze=False)
    for j, st in enumerate(stages):
        ax = axs[0, j]
        sub = rows[rows.stage == st]
        for k, coh in enumerate(("R+", "R-")):
            s = sub[sub.cohort == coh]
            vals = [s[f"r_{key}"].dropna() for key in KEYS]
            pos = np.arange(len(KEYS)) + (k - 0.5) * 0.35
            ax.boxplot(vals, positions=pos, widths=0.3, vert=False, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=COLORS[coh], alpha=0.35), medianprops=dict(color=COLORS[coh]))
            ax.plot([], [], color=COLORS[coh], label=f"{coh} (n = {len(s)} sessions)")
        for i, key in enumerate(KEYS):
            if f"r_null_mean_{key}" in sub:
                ax.plot([sub[f"r_null_mean_{key}"].median()] * 2, [i - 0.4, i + 0.4], color="0.4", lw=1.5)
        ax.plot([], [], color="0.4", label="linear-shift null (median)")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(range(len(KEYS)))
        ax.set_yticklabels(KEYS, fontsize=7)
        ax.set_xlabel("r(predicted P(lick), lick), held-out trials")
        ax.set_title(st)
        ax.legend(fontsize=7, frameon=False)
    ax = axs[0, -1]
    for coh in ("R+", "R-"):
        s = rows[(rows.cohort == coh) & rows.stage.isin(stages)]
        ax.scatter(s.r_base, s["r_base+fm_mot_1s"], s=10, color=COLORS[coh], alpha=0.6, label=coh)
    ax.plot([-0.4, 0.8], [-0.4, 0.8], color="k", lw=0.5)
    ax.set_xlabel("r base (time + history)")
    ax.set_ylabel("r base + Facemap motion SVD (-1 s)")
    ax.legend(fontsize=7, frameon=False)
    for j, (meas, title) in enumerate((("motion_energy_1s", "Pre-stim motion energy vs exact curve"),
                                       ("lick_proba_fm_mot_1s", "Facemap lick probability vs exact curve"))):
        ax = axs[1, j]
        c = corr[(corr.measure == meas) & (corr.curve == "p_mean")]
        for k, coh in enumerate(("R+", "R-")):
            for i, st in enumerate((0, 1)):
                v = c[(c.cohort == coh) & ((c.day == 0) if st == 0 else (c.day > 0))].z_shift.dropna()
                x = i * 2 + k * 0.8
                ax.scatter(x + np.random.default_rng(0).uniform(-0.2, 0.2, len(v)), v, s=8, color=COLORS[coh], alpha=0.5)
                ax.plot([x - 0.3, x + 0.3], [v.median()] * 2, color=COLORS[coh], lw=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks([0.4, 2.4])
        ax.set_xticklabels(["day 0", "expert"])
        ax.set_ylabel("r vs linear-shift null (z)")
        ax.set_title(title, fontsize=9)
    axs[1, -1].axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"facemap_lick_decoding_summary.{ext}", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_excel(SHEET).drop_duplicates("mouse_id")
    cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
    sids = sorted(p.name.replace("_facemap_small.npz", "") for p in (BASE / "facemap").glob("*_facemap_small.npz"))
    sids = [s for s in sids if (BASE / "clips" / f"{s}_clips.npz").exists()]
    rows, corr, trials = [], [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r, c, td in ex.map(safe, [(s, cohort.get(s.split("_")[0], "?")) for s in sids]):
            rows.append(r), corr.extend(c)
            if td is not None:
                trials.append(td)
            print(r["session_id"], r.get("error") or round(r.get("r_base+fm_mot_1s", np.nan), 3), flush=True)
    rows, corr = pd.DataFrame(rows), pd.DataFrame(corr)
    rows.to_csv(OUT / "decoding_per_session.csv", index=False)
    corr.to_csv(OUT / "learning_curve_correlations.csv", index=False)
    pd.concat(trials).to_csv(OUT / "per_trial_measures.csv", index=False)
    ok = rows[rows.get("error").isna()] if "error" in rows else rows
    summ = []
    for st, sub in (("day0", ok[ok.day == 0]), ("expert", ok[ok.day > 0])):
        for key in KEYS:
            for met in ("r", "bacc"):
                col = f"{met}_{key}"
                if col not in sub:
                    continue
                p_mw, p_w = tests(sub[sub.cohort == "R+"][col], sub[sub.cohort == "R-"][col])
                d = dict(stage=st, model=key, metric=met, n=int(sub[col].notna().sum()), median=sub[col].median(),
                         median_Rplus=sub[sub.cohort == "R+"][col].median(),
                         median_Rminus=sub[sub.cohort == "R-"][col].median(), p_mannwhitney=p_mw, p_welch=p_w)
                if f"{met}_null_mean_{key}" in sub:
                    d.update(median_null=sub[f"{met}_null_mean_{key}"].median(),
                             median_z_vs_null=sub[f"{met}_null_z_{key}"].median(),
                             frac_sessions_p05=(sub[f"{met}_null_p_{key}"] < 0.05).mean())
                summ.append(d)
    pd.DataFrame(summ).to_csv(OUT / "decoding_summary.csv", index=False)
    figure(ok, corr)
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print("ALL DONE")


if __name__ == "__main__":
    main()
