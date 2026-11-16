"""Facemap lick decoding, ROI step 4 (haas): feature-by-feature comparison of DLC and ROI-Facemap signals, and
real-score vs null-median figures (sets from 006, single features, and the user's beh_correlates results).

Single features per session (whisker trials of the exact curves; pre-stimulus windows -0.2 s and -1 s):
  DLC   posture (mean) and movement (mean |diff|) of jaw distance, top-nose distance, whisker angle, pupil area
        (the beh_correlates definitions; pupil uses -2..0 s for the 0.2 s window, as there)
  ROI   motion energy of the eye, jaw, whisker-pad and snout ROIs (004_roi_auto.py)
For each feature and target (upcoming whisker lick 0/1, exact-curve p_mean, p_mean - p_chance): Pearson r across
trials and the median of its linear-shift null (N_SHIFT shifts, |shift| > 10 trials, >= 10 trials overlap).
Figures (real scores and null medians side by side, per session, R+ / R-, day 0 and expert):
  compare/sets_real_vs_null.png        feature sets of 006 (held-out decoding r / balanced accuracy, curve in-sample R2)
  compare/features_real_vs_null.png    single features
  compare/beh_correlates_real_vs_null.png   the beh_correlates outputs (combined_results_ks4/
      beh_curves_correlation_linear_shift: correlation_results_lagged_v2.csv lag 0, null = shift-null MEAN as saved
      there; lick_decoding_results_svc.csv, null = label-permutation mean as saved there), day 0, per mouse
Tables: compare/features_per_session.csv, compare/features_ranking.csv (median real - null median per feature).
Run on haas: OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 007_feature_by_feature.py --jobs 48
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

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cmp", HERE / "006_compare_roi_dlc.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)  # ROIS, pearson, shifted, full (003), dlc
full, dlc = cmp.full, cmp.dlc

ROOT, BASE, ROI = cmp.ROOT, cmp.BASE, cmp.ROI
OUT = ROI / "compare"
BEH = ROOT / "combined_results_ks4" / "beh_curves_correlation_linear_shift"
N_SHIFT = 200
COLORS = cmp.COLORS
TARGETS = {"lick": "upcoming whisker lick", "pmean": "p_mean", "pdelta": "p_mean - p_chance"}
DLC_LABEL = {"jaw_distance": "jaw", "top_nose_distance": "nose", "whisker_angle": "whisker", "pupil_area": "pupil"}


def session(args):
    sid, cohort = args
    try:
        rng = np.random.default_rng(zlib.crc32(sid.encode()))  # stable across runs (hash() is salted per process)
        ix = np.load(BASE / "clips" / f"{sid}_clips.npz", allow_pickle=True)
        keep = ix["lead"] == 0
        clip, ft = ix["clip"][keep], ix["t"][keep]
        frames = {c: np.flatnonzero(clip == c) for c in np.unique(clip)}
        ftimes = {c: ft[f] for c, f in frames.items()}
        clip_start = np.asarray(ix["clips"]["start_time"], float)
        desc, trials, series, bp4, mismatch = dlc.load(ROOT / "NWB_ks4" / f"{sid}.nwb")
        t = dlc.trial_table(trials)
        y = t.lick_flag.to_numpy().astype(float)
        starts = t.start_time.to_numpy()
        d = np.abs(starts[:, None] - clip_start[None, :])
        cot = np.where(d.min(1) < 1e-3, d.argmin(1), -1)
        feats = {}
        for r in cmp.ROIS:
            f = np.load(ROI / "facemap" / f"{sid}_{r}_facemap_small.npz")
            e = f["motion"][keep][:, None].astype(float)
            for w, a in (("1s", -1.0), ("0.2s", -0.2)):
                feats[f"roi_{r}_energy_{w}"] = full.window_means(e, frames, ftimes, cot, starts, a, 0.0)[:, 0]
        if bp4 and max(mismatch.values(), default=0) <= dlc.MAX_LEN_MISMATCH:
            bf = dlc.bp4_features({k: v for k, v in bp4.items() if k in dlc.BP4}, starts)
            present = [k for k in dlc.BP4 if k in bp4]
            for w in ("1s", "0.2s"):
                for j, k in enumerate(present):
                    feats[f"dlc_{DLC_LABEL[k]}_posture_{w}"] = bf[f"bp4_{w}"][:, 2 * j]
                    feats[f"dlc_{DLC_LABEL[k]}_movement_{w}"] = bf[f"bp4_{w}"][:, 2 * j + 1]
        m = sid.split("_")[0]
        day = int(desc.replace("+", "").rsplit("_", 1)[1])
        cur = pd.read_hdf(ROOT / "combined_results_ks4" / m / f"whisker_{day}" / "learning_curve" /
                          f"{m}_whisker_{day}_whisker_trial_learning_curve_exact.h5")
        p_mean = np.asarray(cur.p_mean.values[0], float)
        targets = {"lick": y}
        if len(p_mean) == len(y):
            targets.update(pmean=p_mean, pdelta=p_mean - np.asarray(cur.p_chance.values[0], float))
        n = len(y)
        shp = [s for s in range(-n + 10, n - 9) if abs(s) > cmp.SHIFT_EXCLUDE]
        chosen = rng.choice(shp, min(N_SHIFT, len(shp)), replace=False)
        rows = []
        for fn, x in feats.items():
            for tn, yv in targets.items():
                null = [cmp.pearson(*cmp.shifted(x, yv, s)) for s in chosen]
                rows.append(dict(session_id=sid, mouse_id=m, cohort=cohort, day=day, feature=fn, target=tn,
                                 r=cmp.pearson(x, yv), r_null_median=np.nanmedian(null)))
        return rows
    except Exception as e:
        return [dict(session_id=sid, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-500:])]


def strip_pair(ax, x, real, null, color, rng):
    """Real scores (filled) and null medians (open) side by side at x - 0.1 / x + 0.1, with median bars."""
    for dx, v, filled in ((-0.1, real, True), (0.1, null, False)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        ax.scatter(x + dx + rng.uniform(-0.05, 0.05, len(v)), v, s=5, linewidths=0.5, alpha=0.5,
                   facecolors=color if filled else "none", edgecolors=color)
        if len(v):
            ax.plot([x + dx - 0.07, x + dx + 0.07], [np.median(v)] * 2, color="k" if filled else "0.5", lw=2)


def panel(ax, df, items, real_col, null_col, item_col, title, ylabel):
    rng = np.random.default_rng(0)
    for i, it in enumerate(items):
        g = df[df[item_col] == it]
        for q, coh in enumerate(("R+", "R-")):
            h = g[g.cohort == coh]
            strip_pair(ax, i + (q - 0.5) * 0.45, h[real_col], h[null_col], COLORS[coh], rng)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(items, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=8, loc="left")
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)


LEGEND = ("filled = real score per session, open = median of its linear-shift null; black / grey bar = medians; "
          "green R+, magenta R-")


def fig_sets():
    D = pd.read_csv(OUT / "per_session.csv")
    D = D[D.get("error").isna() if "error" in D else slice(None)]
    D = D[D.cohort.isin(COLORS) & (D.day >= 0)]
    D["stage"] = np.where(D.day == 0, "day0", "expert")
    mets = [("lick_r", "lick: held-out r"), ("lick_bacc", "lick: balanced accuracy"),
            ("curve_pmean", "p_mean: in-sample R2"), ("curve_pdelta", "p_mean - p_chance: in-sample R2")]
    long = []
    for met, _ in mets:
        for s in cmp.SETS:
            if f"{met}_{s}" in D:
                long.append(pd.DataFrame(dict(stage=D.stage, cohort=D.cohort, set=s, metric=met,
                                              real=D[f"{met}_{s}"], null=D[f"{met}_null_{s}"])))
    L = pd.concat(long)
    fig, axs = plt.subplots(len(mets), 2, figsize=(15, 3.4 * len(mets)))
    for i, (met, lab) in enumerate(mets):
        for j, st in enumerate(("day0", "expert")):
            g = L[(L.metric == met) & (L.stage == st)]
            panel(axs[i, j], g, cmp.SETS, "real", "null", "set", f"{st}: {lab}", lab)
            axs[i, j].axvspan(-0.5, 2.5, color="0.94", zorder=-1)
    fig.suptitle("Feature sets (grey band = DLC). " + LEGEND, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"sets_real_vs_null.{ext}", dpi=140)
    plt.close(fig)


def fig_features(F):
    F = F[F.cohort.isin(COLORS) & (F.day >= 0)].copy()
    F["stage"] = np.where(F.day == 0, "day0", "expert")
    order = []
    for w in ("0.2s", "1s"):
        order += [f"dlc_{b}_{k}_{w}" for b in ("jaw", "nose", "whisker", "pupil") for k in ("posture", "movement")]
        order += [f"roi_{r}_energy_{w}" for r in cmp.ROIS]
    order = [o for o in order if o in set(F.feature)]
    fig, axs = plt.subplots(3, 2, figsize=(22, 11))
    for i, (tn, lab) in enumerate(TARGETS.items()):
        for j, st in enumerate(("day0", "expert")):
            g = F[(F.target == tn) & (F.stage == st)]
            panel(axs[i, j], g, order, "r", "r_null_median", "feature", f"{st}: r(feature, {lab})", "Pearson r")
            for k, o in enumerate(order):
                if o.startswith("roi_"):
                    axs[i, j].axvspan(k - 0.5, k + 0.5, color="#fff2cc", zorder=-1)
    fig.suptitle("Single features (yellow = ROI Facemap motion energy, white = DLC). " + LEGEND, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"features_real_vs_null.{ext}", dpi=130)
    plt.close(fig)


def fig_beh_correlates():
    C = pd.read_csv(BEH / "correlation_results_lagged_v2.csv")
    C = C[(C.lag == 0)].copy()
    C["cohort"] = C.reward_group.map({1: "R+", 0: "R-", "R+": "R+", "R-": "R-"})
    C["item"] = C.bodypart.map(DLC_LABEL) + "_" + C.metric_type + "_" + C.window
    items = [f"{b}_{k}_{w}" for w in ("pre", "post") for b in ("jaw", "nose", "whisker", "pupil")
             for k in ("posture", "movement")]
    Dd = pd.read_csv(BEH / "lick_decoding_results_svc.csv")
    Dd["cohort"] = Dd.reward_group.map({1: "R+", 0: "R-", "R+": "R+", "R-": "R-"})
    Dd["item"] = Dd.metric_type + "_" + Dd.window
    fig, axs = plt.subplots(2, 2, figsize=(20, 8.5), gridspec_kw=dict(width_ratios=[4, 1]))
    for ax, tc, lab in ((axs[0, 0], "p_mean", "p_mean"), (axs[1, 0], "p_diff", "p_mean - p_chance")):
        panel(ax, C, items, f"corr_{tc}", f"shuffle_mean_corr_{tc}", "item",
              f"beh_correlates, day 0, per mouse: r(feature, {lab}), lag 0 (open = shift-null mean as saved)",
              "Pearson r")
        ax.axvspan(7.5, 15.5, color="0.94", zorder=-1)
    panel(axs[0, 1], Dd, ["posture_pre", "movement_pre", "posture_post", "movement_post"], "accuracy_real",
          "shuffle_mean", "item", "beh_correlates lick decoding (SVC)\n(open = label-permutation mean)",
          "balanced accuracy")
    axs[0, 1].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axs[1, 1].axis("off")
    axs[1, 1].text(0, 0.9, "grey band = post-stimulus windows\n(contain the lick: sanity check only)\n\n"
                   "reward group as saved by beh_correlates\n(wh_reward from the NWB)", fontsize=8, va="top")
    fig.suptitle(LEGEND, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"beh_correlates_real_vs_null.{ext}", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if not a.figures_only:
        sheet = pd.read_excel(cmp.SHEET).drop_duplicates("mouse_id")
        cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
        sids = sorted(p.name.replace("_whisker_facemap_small.npz", "")
                      for p in (ROI / "facemap").glob("*_whisker_facemap_small.npz"))
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = [r for rs in ex.map(session, [(s, cohort.get(s[:5], "?")) for s in sids]) for r in rs]
        F = pd.DataFrame(rows)
        if "error" in F:
            print("errors:", F[F.error.notna()][["session_id", "error"]].to_string(index=False))
            F = F[F.error.isna()].drop(columns=["error", "trace"], errors="ignore")
        F.to_csv(OUT / "features_per_session.csv", index=False)
    F = pd.read_csv(OUT / "features_per_session.csv")
    F["delta"] = F.r - F.r_null_median
    F["stage"] = np.where(F.day == 0, "day0", "expert")
    R = F[F.cohort.isin(COLORS)].groupby(["stage", "target", "feature", "cohort"]).agg(
        n=("r", "size"), median_r=("r", "median"), median_null=("r_null_median", "median"),
        median_delta=("delta", "median")).reset_index()
    R["abs_delta"] = R.median_delta.abs()
    R.sort_values(["stage", "target", "cohort", "abs_delta"], ascending=[True, True, True, False]).to_csv(
        OUT / "features_ranking.csv", index=False)
    fig_sets()
    fig_features(F)
    fig_beh_correlates()
    with pd.option_context("display.width", 200):
        for (st, tn), g in R.groupby(["stage", "target"]):
            top = g.pivot_table(index="feature", columns="cohort", values="median_delta")
            top = top.reindex(top.abs().max(1).sort_values(ascending=False).index).head(8)
            print(f"\n== {st}, target {tn}: top features by |median(r - null median)|")
            print(top.round(3).to_string())
    print("ALL DONE")


if __name__ == "__main__":
    main()
