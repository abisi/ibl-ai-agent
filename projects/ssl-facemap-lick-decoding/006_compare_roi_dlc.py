"""Facemap lick decoding, ROI step 3 (haas): is Facemap on face ROIs better than DLC? All sessions with clips + ROI
Facemap outputs, day 0 and expert (day > 0) separately.

Trials: exact-curve whisker trials (context != passive, perf != 6, early_lick == 0). Pre-stimulus windows -1 s and
-0.2 s. Feature sets:
  DLC         dlc_bp4  posture (mean) + movement (mean |diff|) of jaw distance, top-nose distance, whisker angle,
                       pupil area (the beh_correlates set; pupil uses -2..0 s for the 0.2 s window, as there)
              dlc_kpca keypoint PCA (10 posture + 10 movement PCs; dlc_kpca.py), -1 s only
  Facemap     roi_energy  motion energy of the eye, jaw, whisker-pad and snout ROIs (004_roi_auto.py)
              roi_svd     first N_COMP motion-SVD components of each ROI (z-scored), -1 s only
              full_svd    full-frame motion SVD (50 components, earlier clips run), -1 s only, reference
  pairs       single signals: eye ROI energy vs pupil-area movement, jaw ROI vs jaw, whisker ROI vs whisker angle,
              snout ROI vs top-nose distance (DLC movement = mean |diff|)
Metrics (each reported as the observed value minus the MEDIAN of its linear-shift null; no time / history model):
  lick     upcoming whisker lick. Sets: blocked 5-fold CV logistic regression (003_analysis.dec_metrics), held-out
           r(predicted P(lick), lick) and balanced accuracy; null = licks shifted against the features
           (N_SHIFT_SET shifts, 10 < |s| <= n/3). Pairs: Pearson r(signal, lick), N_SHIFT_PAIR shifts
  curve    exact learning curve p_mean and p_mean - p_chance. Sets: in-sample R2 of a ridge regression of the curve on
           the features (2026-11-05, user: replaces the blocked-CV held-out r, which was biased negative because each
           block of a smooth curve is extrapolated); the inflation of in-sample R2 by the number of features is
           removed by the null = the same fit with the curve shifted against the features (shifted fits use fewer
           trials, so slightly more inflation: conservative). Pairs: Pearson r(signal, curve) with its shift null
Session is the unit; ROI vs DLC compared with paired Wilcoxon tests on the per-session differences.
Outputs: combined_results_ks4/ssl-facemap-lick-decoding/roi/compare/ (per-session table, summary, figure).
Run on haas: OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 006_compare_roi_dlc.py --jobs 48
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
spec = importlib.util.spec_from_file_location("full", HERE / "003_analysis.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)  # window_means, dec_metrics; full.dlc = dlc_kpca.py
dlc = full.dlc

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
BASE = ROOT / "combined_results_ks4" / "ssl-facemap-lick-decoding"
ROI = BASE / "roi"
OUT = ROI / "compare"
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
ROIS = ["eye", "jaw", "whisker", "snout"]
PAIRS = {"eye": "pupil_area", "jaw": "jaw_distance", "whisker": "whisker_angle", "snout": "top_nose_distance"}
N_COMP = 20
N_SHIFT_SET, N_SHIFT_PAIR, SHIFT_EXCLUDE = 100, 200, 10
SETS = ["dlc_bp4_1s", "dlc_bp4_0.2s", "dlc_kpca_1s", "roi_energy_1s", "roi_energy_0.2s", "roi_svd_1s", "full_svd_1s"]
COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def shifts(n, max_frac):
    lim = int(n * max_frac)
    return [s for s in range(-lim, lim + 1) if abs(s) > SHIFT_EXCLUDE]


def shifted(X, y, s):
    return (X[s:], y[:-s]) if s > 0 else (X[:s], y[-s:])


def ridge_curve_r2(X, c):
    """In-sample R2 of the ridge fit of the curve on X (alpha by RidgeCV's leave-one-out)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    X = np.where(np.isfinite(X), X, 0.0)
    ok = np.isfinite(c)
    if ok.sum() < 20 or np.std(c[ok]) == 0:
        return np.nan
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    m.fit(X[ok], c[ok])
    return m.score(X[ok], c[ok])


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 10 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0 else np.nan


def session(args):
    sid, cohort = args
    rng = np.random.default_rng(zlib.crc32(sid.encode()))  # stable across runs (hash() is salted per process)
    ix = np.load(BASE / "clips" / f"{sid}_clips.npz", allow_pickle=True)
    keep = ix["lead"] == 0
    clip, ft = ix["clip"][keep], ix["t"][keep]
    frames = {c: np.flatnonzero(clip == c) for c in np.unique(clip)}
    ftimes = {c: ft[f] for c, f in frames.items()}
    clip_start = np.asarray(ix["clips"]["start_time"], float)
    z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-9)  # noqa: E731

    nwb = ROOT / "NWB_ks4" / f"{sid}.nwb"
    desc, trials, series, bp4, mismatch = dlc.load(nwb)
    t = dlc.trial_table(trials)
    y = t.lick_flag.to_numpy().astype(float)
    starts = t.start_time.to_numpy()
    d = np.abs(starts[:, None] - clip_start[None, :])
    cot = np.where(d.min(1) < 1e-3, d.argmin(1), -1)
    wm = lambda X, a, b: full.window_means(X, frames, ftimes, cot, starts, a, b)  # noqa: E731

    sets, pairs = {}, {}
    energy = {}
    for r in ROIS:
        f = np.load(ROI / "facemap" / f"{sid}_{r}_facemap_small.npz")
        assert f["motSVD"].shape[0] == len(keep), (sid, r)
        energy[r] = {w: wm(f["motion"][keep][:, None].astype(float), a, 0.0)[:, 0] for w, a in (("1s", -1.0), ("0.2s", -0.2))}
        sets.setdefault("roi_svd_1s", []).append(wm(z(f["motSVD"][keep, :N_COMP]), -1.0, 0.0))
    sets["roi_svd_1s"] = np.hstack(sets["roi_svd_1s"])
    for w in ("1s", "0.2s"):
        sets[f"roi_energy_{w}"] = np.column_stack([energy[r][w] for r in ROIS])
    ff = np.load(BASE / "facemap" / f"{sid}_facemap_small.npz")
    sets["full_svd_1s"] = wm(z(ff["motSVD"][keep, :50]), -1.0, 0.0)
    ok_dlc = max(mismatch.values(), default=0) <= dlc.MAX_LEN_MISMATCH
    if ok_dlc and bp4:
        bf = dlc.bp4_features({k: v for k, v in bp4.items() if k in dlc.BP4}, starts)
        sets["dlc_bp4_1s"], sets["dlc_bp4_0.2s"] = bf["bp4_1s"], bf["bp4_0.2s"]
        present = [k for k in dlc.BP4 if k in bp4]
        for r, k in PAIRS.items():
            if k in present:
                j = present.index(k)
                for w in ("1s", "0.2s"):
                    pairs[(r, w)] = (energy[r][w], bf[f"bp4_{w}"][:, 2 * j + 1])
    if ok_dlc and series:
        kf, _ = dlc.keypoint_features(series, starts)
        sets["dlc_kpca_1s"] = kf["kpca_mean_1s"]

    m = sid.split("_")[0]
    behavior, day = desc.replace("+", "").rsplit("_", 1)
    day = int(day)
    cur = pd.read_hdf(ROOT / "combined_results_ks4" / m / f"whisker_{day}" / "learning_curve" /
                      f"{m}_whisker_{day}_whisker_trial_learning_curve_exact.h5")
    p_mean = np.asarray(cur.p_mean.values[0], float)
    curves = {"pmean": p_mean, "pdelta": p_mean - np.asarray(cur.p_chance.values[0], float)} if len(p_mean) == len(y) else {}

    row = dict(session_id=sid, mouse_id=m, cohort=cohort, day=day, n_trials=len(y), lick_rate=y.mean())
    n = len(y)
    lick_ok = 5 <= y.sum() <= n - 5
    sh = shifts(n, 1 / 3)
    for k, X in sets.items():
        if lick_ok:
            obs = full.dec_metrics(X, y)
            null = pd.DataFrame([full.dec_metrics(*shifted(X, y, s)) for s in rng.choice(sh, min(N_SHIFT_SET, len(sh)), replace=False)])
            for met in ("r", "bacc"):
                row[f"lick_{met}_{k}"] = obs[met]
                row[f"lick_{met}_null_{k}"] = null[met].median()
        for cn, c in curves.items():
            obs = ridge_curve_r2(X, c)
            null = [ridge_curve_r2(*shifted(X, c, s)) for s in rng.choice(sh, min(N_SHIFT_SET, len(sh)), replace=False)]
            row[f"curve_{cn}_{k}"] = obs
            row[f"curve_{cn}_null_{k}"] = np.nanmedian(null)
    shp = [s for s in range(-n + 10, n - 9) if abs(s) > SHIFT_EXCLUDE]
    for (r, w), (fm_x, dlc_x) in pairs.items():
        for src, x in (("roi", fm_x), ("dlc", dlc_x)):
            for tn, target in [("lick", y)] + [(f"curve_{cn}", c) for cn, c in curves.items()]:
                key = f"pair_{tn}_{src}_{r}_{w}"
                row[key] = pearson(x, target)
                row[f"{key}_null"] = np.nanmedian([pearson(*shifted(x, target, s))
                                                   for s in rng.choice(shp, min(N_SHIFT_PAIR, len(shp)), replace=False)])
    return row


def safe(args):
    try:
        return session(args)
    except Exception as e:
        return dict(session_id=args[0], error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-600:])


def deltas(D):
    """Observed minus null median for every metric column."""
    out = D[["session_id", "mouse_id", "cohort", "day", "n_trials"]].copy()
    for met in ("lick_r", "lick_bacc", "curve_pmean", "curve_pdelta"):
        for k in SETS:
            if f"{met}_{k}" in D and f"{met}_null_{k}" in D:
                out[f"d_{met}_{k}"] = D[f"{met}_{k}"] - D[f"{met}_null_{k}"]
    for c in D.columns:
        if c.startswith("pair_") and not c.endswith("_null") and f"{c}_null" in D:
            out[f"d_{c}"] = D[c] - D[f"{c}_null"]
    return out


def summary(Dd):
    rows = []
    stage = np.where(Dd.day == 0, "day0", "expert")
    for st in ("day0", "expert"):
        for coh in ("all", "R+", "R-"):
            g = Dd[(stage == st) & ((Dd.cohort == coh) if coh != "all" else True)]
            for c in [c for c in Dd.columns if c.startswith("d_")]:
                v = g[c].dropna()
                if len(v) < 5:
                    continue
                rows.append(dict(stage=st, cohort=coh, metric=c[2:], n=len(v), median_delta=v.median(),
                                 frac_positive=(v > 0).mean(), p_wilcoxon_vs0=stats.wilcoxon(v).pvalue))
    S = pd.DataFrame(rows)
    comps = [("roi_energy_1s", "dlc_bp4_1s"), ("roi_energy_0.2s", "dlc_bp4_0.2s"), ("roi_svd_1s", "dlc_kpca_1s"),
             ("roi_svd_1s", "full_svd_1s"), ("roi_svd_1s", "roi_energy_1s")]
    prow = []
    for st in ("day0", "expert"):
        for coh in ("all", "R+", "R-"):
            g = Dd[(stage == st) & ((Dd.cohort == coh) if coh != "all" else True)]
            for met in ("lick_r", "lick_bacc", "curve_pmean", "curve_pdelta"):
                for a, b in comps:
                    ca, cb = f"d_{met}_{a}", f"d_{met}_{b}"
                    if ca in g and cb in g:
                        h = g[[ca, cb]].dropna()
                        if len(h) >= 5:
                            diff = h[ca] - h[cb]
                            prow.append(dict(stage=st, cohort=coh, metric=met, roi=a, dlc_or_ref=b, n=len(h),
                                             median_roi=h[ca].median(), median_ref=h[cb].median(),
                                             median_diff=diff.median(), frac_roi_better=(diff > 0).mean(),
                                             p_wilcoxon_paired=stats.wilcoxon(diff).pvalue))
            for tn in ("lick", "curve_pmean", "curve_pdelta"):
                for r in ROIS:
                    for w in ("1s", "0.2s"):
                        ca, cb = f"d_pair_{tn}_roi_{r}_{w}", f"d_pair_{tn}_dlc_{r}_{w}"
                        if ca in g and cb in g:
                            h = g[[ca, cb]].dropna()
                            if len(h) >= 5:
                                diff = h[ca].abs() - h[cb].abs()
                                prow.append(dict(stage=st, cohort=coh, metric=f"pair_{tn}_|dr|", roi=f"roi_{r}_{w}",
                                                 dlc_or_ref=f"dlc_{PAIRS[r]}_{w}", n=len(h),
                                                 median_roi=h[ca].median(), median_ref=h[cb].median(),
                                                 median_diff=diff.median(), frac_roi_better=(diff > 0).mean(),
                                                 p_wilcoxon_paired=stats.wilcoxon(diff).pvalue))
    return S, pd.DataFrame(prow)


def figure(Dd):
    mets = [("lick_r", "lick: held-out r - null median"), ("lick_bacc", "lick: balanced acc. - null median"),
            ("curve_pmean", "curve p_mean: in-sample R2 - null median"),
            ("curve_pdelta", "curve p_mean - p_chance: in-sample R2 - null median")]
    stage = np.where(Dd.day == 0, "day0", "expert")
    fig, axs = plt.subplots(len(mets), 2, figsize=(14, 3.4 * len(mets)), squeeze=False)
    rng = np.random.default_rng(0)
    for i, (met, lab) in enumerate(mets):
        for j, st in enumerate(("day0", "expert")):
            ax = axs[i, j]
            g = Dd[stage == st]
            for k, s in enumerate(SETS):
                col = f"d_{met}_{s}"
                if col not in g:
                    continue
                for q, coh in enumerate(("R+", "R-")):
                    v = g[g.cohort == coh][col].dropna()
                    x = k + (q - 0.5) * 0.35
                    ax.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v, s=5, alpha=0.35, color=COLORS[coh], linewidths=0)
                    ax.plot([x - 0.14, x + 0.14], [v.median()] * 2, color=COLORS[coh], lw=2.2)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.axvspan(-0.5, 2.5, color="0.93", zorder=-1)
            ax.set_xticks(range(len(SETS)))
            ax.set_xticklabels(SETS, rotation=25, ha="right", fontsize=7)
            ax.set_ylabel(lab, fontsize=7)
            ax.set_title(f"{st} (n = {g.session_id.nunique()} sessions; grey = DLC)", fontsize=8, loc="left")
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    for coh, c in COLORS.items():
        axs[0, 0].plot([], [], color=c, lw=2, label=f"{coh} median")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"roi_vs_dlc.{ext}", dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--sessions", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_excel(SHEET).drop_duplicates("mouse_id")
    cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
    have = {p.name.replace("_whisker_facemap_small.npz", "") for p in (ROI / "facemap").glob("*_whisker_facemap_small.npz")}
    sids = a.sessions or sorted(s for s in have if all((ROI / "facemap" / f"{s}_{r}_facemap_small.npz").exists()
                                                    for r in ROIS) and (BASE / "facemap" / f"{s}_facemap_small.npz").exists())
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(safe, [(s, cohort.get(s[:5], "?")) for s in sids]))
    D = pd.DataFrame(rows)
    tag = "_test" if a.sessions else ""
    D.to_csv(OUT / f"per_session{tag}.csv", index=False)
    if "error" in D:
        print("errors:", D.error.dropna().str[:80].value_counts().to_dict())
        D = D[D.error.isna()]
    D = D[D.cohort.isin(COLORS) & (D.day >= 0)]
    Dd = deltas(D)
    Dd.to_csv(OUT / f"per_session_deltas{tag}.csv", index=False)
    S, P = summary(Dd)
    S.to_csv(OUT / f"summary_vs_null{tag}.csv", index=False)
    P.to_csv(OUT / f"roi_vs_dlc_paired{tag}.csv", index=False)
    if not a.sessions:
        figure(Dd)
    with pd.option_context("display.width", 250):
        print(S[S.cohort == "all"].round(3).to_string(index=False))
        print(P[P.cohort == "all"].round(3).to_string(index=False))
    print("ALL DONE")


if __name__ == "__main__":
    main()
