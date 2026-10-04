"""133 -- Whisker vs auditory at STIMULUS onset across passive_pre / active / passive_post, v2 (user 2026-10-02: "Use rates
Hz-based, make sure data is zscore. Use as baseline -55 ms to -20 ms for baseline correction. Baseline is specific to
pre, active, post. Can you also do a single decoder but evaluated on each epoch? ... compute angles between vectors and look
at how the angles change? Port these updates to the area_group level analysis. Improve single area group visualization:
focus on the change across the three epochs, between cohorts and within an area").
Sessions / trials: as 132 (learning-stage R+ / R- sessions with passive trials before AND after the main active block;
active = prep_modality_trials from the first whisker trial on, perf != 6, A1-trimmed; passive: context == 'passive', no perf
filter -- skills/ssl-trial-exclusion).
Preprocessing (per session x area; QC units good + mua, >= MIN_UNITS):
  response  rate (Hz) in 5-35 ms after stimulus onset (dead zone -10..+5 ms);
  baseline  rate (Hz) in -55..-20 ms; EPOCH-SPECIFIC: each unit's mean baseline rate within its epoch (pre / active / post)
            is subtracted from that epoch's responses;
  z-score   per unit over all trials of the session (pooled epochs, one transform for every epoch); units with zero variance
            dropped.
Matching: each epoch subsampled to n_match trials per class (min over epochs and classes), K_SUB subsamples.
Metrics (each with a trial-shuffle null; corrected = value - null mean where applicable):
  within    5-fold CV balanced accuracy per epoch (L2 logistic regression, class_weight balanced, C fixed per session x area);
  cross     train on one epoch's subsample, test on another's (3 x 3);
  single    ONE decoder trained on all three epochs (pooled subsamples, 5-fold CV stratified by epoch x class), held-out
            balanced accuracy evaluated separately on each epoch;
  distance  cross-validated (crossnobis-like) squared distance between the whisker and auditory mean patterns per epoch:
            d = <D_1, D_2> / n_units, D_h = mean(whisker) - mean(auditory) in random half h of the trials (N_SPLIT splits);
            unbiased, ~0 without signal, does not saturate like accuracy;
  angles    coding vector per epoch D = mean(whisker) - mean(auditory); between-epoch similarity = cos(D_a,h1, D_b,h2) averaged
            over halves, normalised by the split-half within-epoch reliabilities: cos_norm = cos_ab / sqrt(cos_aa * cos_bb)
            (1 = same direction, 0 = unrelated; reliabilities floored at 0.05); raw angle in degrees also saved.
Outputs: 133_modality_stim_epochs.parquet (session x area: per-epoch / per-pair metrics), 133_stats.csv; figures:
  133_whole_brain.{pdf,png,svg}  per metric, pre -> active -> post per cohort (mean +- SEM, faint per mouse) + angles + cross matrix;
  133_area_groups_<metric>.{pdf,png,svg}  small multiples, one panel per area group, same layout; stats in each panel:
     within cohort epoch effect (Friedman AND repeated-measures ANOVA), R+ vs R- on the changes active-pre, post-active,
     post-pre (Mann-Whitney AND Welch). Uncorrected.
Units (env SSL_UNITS; user 2026-10-02 "rerun on good only, i.e. tracking the same units over the entire session"): all =
good + mua (default); good = quality_label 'good' only AND firing >= 0.5 Hz in the time span of every epoch (outputs *_good).
Run (haas, repo root): python .../133_modality_stim_epochs_v2.py   |   python ... plot
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, str(OUT))
COL = {"R+": "#00B400", "R-": "#C800C8"}
EP = ["passive_pre", "active", "passive_post"]
EPLAB = {"passive_pre": "pre", "active": "active", "passive_post": "post"}
PAIRS = [("passive_pre", "active"), ("active", "passive_post"), ("passive_pre", "passive_post")]
WIN, BASE, DZ = (0.005, 0.035), (-0.055, -0.020), (-0.010, 0.005)
MIN_UNITS, MIN_PER_CLASS = 5, 8
K_SUB, N_SHUF, N_FOLD, N_SPLIT = 5, 20, 5, 20
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
UNITS = os.environ.get("SSL_UNITS", "all")                 # all = good + mua; good = quality_label 'good' AND present in every epoch
MIN_EPOCH_RATE = 0.5                                       # Hz over each epoch's span (good mode): unit tracked across the session
TAG = "" if UNITS == "all" else "_good"
OUT_PATH = OUT / f"133_modality_stim_epochs{TAG}.parquet"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


def clf(C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000)


def bacc(y, p):
    return 0.5 * (np.mean(p[y]) + np.mean(~p[~y])) if y.any() and (~y).any() else np.nan


def cv_pred(X, y, C, rng, strat=None):
    from sklearn.model_selection import StratifiedKFold
    pred = np.zeros(len(y), bool)
    s = y if strat is None else strat
    for tr, te in StratifiedKFold(N_FOLD, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, s):
        pred[te] = clf(C).fit(X[tr], y[tr]).predict(X[te])
    return pred


def subsample(y, n, rng):
    return np.sort(np.r_[rng.choice(np.where(y)[0], n, replace=False), rng.choice(np.where(~y)[0], n, replace=False)])


def halves(y, rng):
    a, b = [], []
    for cls in (True, False):
        i = rng.permutation(np.where(y == cls)[0])
        a.append(i[: len(i) // 2])
        b.append(i[len(i) // 2:])
    return np.concatenate(a), np.concatenate(b)


def coding_vec(X, y):
    return X[y].mean(0) - X[~y].mean(0)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else np.nan


def metrics_once(X, ys, idx, C, rng, shuffle=False):
    """All metrics on one subsample; with shuffle=True the labels are permuted within each epoch (null)."""
    Xs = {e: X[e][idx[e]] for e in EP}
    Ys = {e: ys[e][idx[e]] for e in EP}
    if shuffle:
        Ys = {e: rng.permutation(Ys[e]) for e in EP}
    out = {}
    for e in EP:
        out[f"within_{e}"] = bacc(Ys[e], cv_pred(Xs[e], Ys[e], C, rng))
    for a in EP:
        m = clf(C).fit(Xs[a], Ys[a])
        for b in EP:
            if a != b:
                yb = ys[b][idx[b]]                                   # test on TRUE labels (null: trained on shuffled)
                out[f"cross_{a}__{b}"] = bacc(yb, m.predict(Xs[b]))
    Xp = np.vstack([Xs[e] for e in EP])
    yp = np.concatenate([Ys[e] for e in EP])
    ep = np.concatenate([[k] * len(Ys[e]) for k, e in enumerate(EP)])
    pred = cv_pred(Xp, yp, C, rng, strat=ep * 2 + yp)
    for k, e in enumerate(EP):
        out[f"single_{e}"] = bacc(yp[ep == k], pred[ep == k])
    if not shuffle:
        nu = Xs[EP[0]].shape[1]
        dist = {e: [] for e in EP}
        cab = {p: [] for p in PAIRS}
        caa = {e: [] for e in EP}
        for _ in range(N_SPLIT):
            H = {e: halves(Ys[e], rng) for e in EP}
            D = {e: (coding_vec(Xs[e][H[e][0]], Ys[e][H[e][0]]), coding_vec(Xs[e][H[e][1]], Ys[e][H[e][1]])) for e in EP}
            for e in EP:
                dist[e].append(D[e][0] @ D[e][1] / nu)
                caa[e].append(cos(D[e][0], D[e][1]))
            for a, b in PAIRS:
                cab[(a, b)].append(0.5 * (cos(D[a][0], D[b][1]) + cos(D[a][1], D[b][0])))
        for e in EP:
            out[f"distance_{e}"] = float(np.mean(dist[e]))
            out[f"reliab_{e}"] = float(np.nanmean(caa[e]))
        for a, b in PAIRS:
            c = float(np.nanmean(cab[(a, b)]))
            ra, rb = max(out[f"reliab_{a}"], 0.05), max(out[f"reliab_{b}"], 0.05)
            out[f"cos_{a}__{b}"] = c
            out[f"cosnorm_{a}__{b}"] = float(np.clip(c / np.sqrt(ra * rb), -1.5, 1.5))
            full_a, full_b = coding_vec(Xs[a], Ys[a]), coding_vec(Xs[b], Ys[b])
            out[f"angle_{a}__{b}"] = float(np.degrees(np.arccos(np.clip(cos(full_a, full_b), -1, 1))))
    return out


def process(args):
    sid, subject, rg = args
    _init()
    sys.path.insert(0, SCRIPTS)
    sys.path.insert(0, str(OUT))
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(session_id=sid, mouse_id=subject, reward_group=rg, skipped_reason="no passive pre+post or no active")]
    spikes = T.load_session_unit_spikes(root, sid)
    ys = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in EP}
    n_match = min(min(int(ys[e].sum()), int((~ys[e]).sum())) for e in EP)
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg, n_match=n_match)
    if n_match < MIN_PER_CLASS:
        return [dict(base, skipped_reason=f"n_match {n_match} < {MIN_PER_CLASS}")]
    sl = labels[labels.session_id == sid]
    areas = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(sl["area_group"].dropna().unique())]
    rows = []
    for area_col, area in areas:
        units = T.area_units(sid, area_col, area, labels)
        if UNITS == "good":
            # good units only (quality_label == 'good'), and tracked over the whole session: firing >= MIN_EPOCH_RATE Hz in the
            # time span of EVERY epoch (first trial - 1 s to last trial + 1 s), so the same units contribute to pre, active, post
            lab_s = labels[(labels.session_id == sid) & labels.cluster_id.isin(units)]
            units = lab_s[lab_s.quality_label == "good"].cluster_id.to_numpy()
            keep_u = []
            for cid in units:
                sp = spikes.get(cid, np.array([]))
                ok = True
                for e in EP:
                    a0, a1 = trs[e].start_time.min() - 1.0, trs[e].start_time.max() + 1.0
                    n_sp = np.searchsorted(sp, a1) - np.searchsorted(sp, a0)
                    if n_sp / max(a1 - a0, 1e-6) < MIN_EPOCH_RATE:
                        ok = False
                        break
                if ok:
                    keep_u.append(cid)
            units = np.asarray(keep_u)
        if len(units) < MIN_UNITS:
            continue
        X = {}
        for e in EP:
            t0 = trs[e].start_time.to_numpy()
            resp, bas = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
            # sliding_bin_population_matrices returns rates per window (spikes / s); epoch-specific baseline subtraction
            X[e] = resp - np.nanmean(bas, 0, keepdims=True)
        if any(np.isnan(X[e]).any() for e in EP):
            continue
        Z = np.vstack([X[e] for e in EP])
        mu, sd = Z.mean(0), Z.std(0)
        keep = sd > 0
        if keep.sum() < MIN_UNITS:
            continue
        X = {e: (X[e][:, keep] - mu[keep]) / sd[keep] for e in EP}
        rng = np.random.default_rng(zlib.crc32(f"{sid}|{area}|v2".encode()))
        C = T.select_fixed_c_pooled(np.vstack([X[e] for e in EP]), np.concatenate([ys[e] for e in EP]), rng, n_folds=5)
        real, null = [], []
        for _ in range(K_SUB):
            idx = {e: subsample(ys[e], n_match, rng) for e in EP}
            real.append(metrics_once(X, ys, idx, C, rng))
            for _s in range(N_SHUF):
                null.append(metrics_once(X, ys, idx, C, rng, shuffle=True))
        R, N = pd.DataFrame(real).mean(), pd.DataFrame(null).mean()
        row = dict(base, units_mode=UNITS, area_col=area_col, area=area, n_units=int(keep.sum()), C=C, skipped_reason=None)
        for k, v in R.items():
            row[k] = float(v)
            if k in N.index:
                row[f"{k}_null"] = float(N[k])
                row[f"{k}_corr"] = float(v - N[k])
        rows.append(row)
    return rows or [dict(base, skipped_reason="no area with enough units")]


def run():
    os.chdir(OUT.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples() if r.session_id not in done]
    print(f"[133] {len(args)} sessions", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, a): a for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                rows = [dict(session_id=a[0], mouse_id=a[1], reward_group=a[2], skipped_reason=f"error: {e!r}"[:300])]
            new = pd.DataFrame(rows)
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            print(f"[133] [{i}/{len(args)}] {a[0]} -- {time.time() - t0:.0f}s", flush=True)
    plot()


# --------------------------------------------------------------------------------------------------------------- plots
METRICS = [("within", "within-epoch decoder\n(acc − shuffle)", "_corr"), ("single", "single decoder, per epoch\n(acc − shuffle)", "_corr"),
           ("distance", "cv distance\n(whisker − auditory)", "")]


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def epoch_stats(g, cols):
    """Within-cohort epoch effect (Friedman AND RM-ANOVA) and cohort comparison of changes (MW AND Welch)."""
    from scipy import stats
    out = {}
    for rg in ("R+", "R-"):
        x = g[g.reward_group == rg][cols].dropna()
        if len(x) >= 4:
            out[f"friedman_{rg}"] = stats.friedmanchisquare(*[x[c] for c in cols]).pvalue
            try:
                from statsmodels.stats.anova import AnovaRM
                long = x.reset_index().melt(id_vars="index", value_vars=cols, var_name="epoch", value_name="v")
                out[f"rmanova_{rg}"] = float(AnovaRM(long, "v", "index", within=["epoch"]).fit().anova_table["Pr > F"].iloc[0])
            except Exception:  # noqa: BLE001
                out[f"rmanova_{rg}"] = np.nan
    for name, (i, j) in (("active-pre", (1, 0)), ("post-active", (2, 1)), ("post-pre", (2, 0))):
        a = (g[g.reward_group == "R+"][cols[i]] - g[g.reward_group == "R+"][cols[j]]).dropna()
        b = (g[g.reward_group == "R-"][cols[i]] - g[g.reward_group == "R-"][cols[j]]).dropna()
        if min(len(a), len(b)) >= 3:
            out[f"mw_{name}"] = stats.mannwhitneyu(a, b).pvalue
            out[f"welch_{name}"] = stats.ttest_ind(a, b, equal_var=False).pvalue
    return out


def epoch_panel(ax, g, cols, ylabel, st, rng, small=False):
    xs = np.arange(len(cols))
    for k, rg in enumerate(("R+", "R-")):
        x = g[g.reward_group == rg][cols].dropna()
        off = (k - 0.5) * 0.12
        for _, r in x.iterrows():
            ax.plot(xs + off, r.to_numpy(), color=COL[rg], lw=0.3, alpha=0.25)
        if len(x):
            m, se = x.mean().to_numpy(), (x.std(ddof=1) / np.sqrt(len(x))).to_numpy()
            ax.errorbar(xs + off, m, yerr=se, color=COL[rg], lw=1.6, marker="o", ms=3, capsize=0,
                        label=f"{'R+' if rg == 'R+' else 'R−'} n={len(x)}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["pre", "active", "post"] if len(cols) == 3 else ["pre·act", "act·post", "pre·post"],
                       fontsize=5.5 if small else 6.5)
    ax.set_ylabel(ylabel, fontsize=5.5 if small else 6.5)
    t = (f"R+ F {pf(st.get('friedman_R+', np.nan))} / RM {pf(st.get('rmanova_R+', np.nan))}; "
         f"R− F {pf(st.get('friedman_R-', np.nan))} / RM {pf(st.get('rmanova_R-', np.nan))}\n"
         + " · ".join(f"Δ{n}: MW {pf(st.get('mw_' + n, np.nan))} W {pf(st.get('welch_' + n, np.nan))}"
                      for n in ("active-pre", "post-active", "post-pre")))
    ax.set_title(t, fontsize=4.2 if small else 5.5)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_parquet(OUT_PATH)
    d = d[d.skipped_reason.isna()]
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 6.5})
    rng = np.random.default_rng(0)
    srows = []
    blocks = [(m, lab, [f"{m}_{e}{suf}" for e in EP]) for m, lab, suf in METRICS]
    blocks += [("cosnorm", "coding-axis similarity\n(reliability-normalised cos)", [f"cosnorm_{a}__{b}" for a, b in PAIRS]),
               ("angle", "angle between coding vectors (°)", [f"angle_{a}__{b}" for a, b in PAIRS])]
    for area, g in d.groupby("area"):
        for m, lab, cols in blocks:
            if all(c in g for c in cols):
                st = epoch_stats(g, cols)
                srows.append(dict(area=area, metric=m, n_Rplus=int((g.reward_group == "R+").sum()),
                                  n_Rminus=int((g.reward_group == "R-").sum()),
                                  **{f"mean_{rg}_{EPLAB.get(c.split('_', 1)[1].replace('_corr', ''), c)}":
                                     g[g.reward_group == rg][c].mean() for rg in ("R+", "R-") for c in cols}, **st))
    S = pd.DataFrame(srows)
    S.to_csv(OUT / f"133_stats{TAG}.csv", index=False)
    # whole brain
    g = d[d.area == "All units"]
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 5.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.08, wspace=0.5, hspace=0.75)
    for ax, (m, lab, cols) in zip(axes.flat[:5], blocks):
        epoch_panel(ax, g, cols, lab, S[(S.area == "All units") & (S.metric == m)].iloc[0].to_dict(), rng)
    axes.flat[0].legend(fontsize=5.5, frameon=False)
    for ax, rg in zip(axes.flat[5:7], ("R+", "R-")):
        x = g[g.reward_group == rg]
        M = np.full((3, 3), np.nan)
        for i, a in enumerate(EP):
            for j, b in enumerate(EP):
                M[i, j] = x[f"within_{a}_corr"].mean() if a == b else x[f"cross_{a}__{b}_corr"].mean()
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanmax(M))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["pre", "act", "post"])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["pre", "act", "post"])
        ax.set_xlabel("test")
        ax.set_ylabel("train")
        ax.set_title(f"{'R+' if rg == 'R+' else 'R−'}: train/test (acc − shuffle)", color=COL[rg], fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes.flat[7].set_axis_off()
    n = g.drop_duplicates("session_id").groupby("reward_group").size().to_dict()
    fig.suptitle("Whisker vs auditory, 5-35 ms, whole brain: rates (Hz), epoch-specific baseline (−55 to −20 ms), z-scored; "
                 f"R+ n={n.get('R+', 0)}, R− n={n.get('R-', 0)}. Within cohort: Friedman (F) / RM-ANOVA (RM); "
                 "R+ vs R− on changes: MW / Welch (W). Uncorrected. Units: " + ("good + mua" if UNITS == "all" else
                 f"good only, >= {MIN_EPOCH_RATE} Hz in every epoch"), fontsize=6.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"133_whole_brain{TAG}.{ext}", dpi=250)
    plt.close(fig)
    # area groups: one figure per metric, one panel per area group
    areas = sorted(a for a in d.area.unique() if a != "All units")
    ncol = 5
    nrow = int(np.ceil(len(areas) / ncol))
    for m, lab, cols in blocks:
        fig, axes = plt.subplots(nrow, ncol, figsize=(11.7, 2.3 * nrow + 0.6), squeeze=False)
        fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.6 / (2.3 * nrow + 0.6), bottom=0.04, wspace=0.4, hspace=0.95)
        for ax, ar in zip(axes.flat, areas):
            ga = d[d.area == ar]
            st = S[(S.area == ar) & (S.metric == m)]
            epoch_panel(ax, ga, cols, lab if ar == areas[0] else "", st.iloc[0].to_dict() if len(st) else {}, rng, small=True)
            ax.text(0.02, 0.98, ar, transform=ax.transAxes, fontsize=6, fontweight="bold", va="top")
            ax.tick_params(labelsize=5)
        for ax in axes.flat[len(areas):]:
            ax.set_axis_off()
        fig.suptitle(f"{lab.replace(chr(10), ' ')} across passive-pre → active → passive-post, per area group (mean ± SEM per "
                     "cohort; faint = mice). Panel titles: within-cohort epoch effect (Friedman / RM-ANOVA) and R+ vs R− on "
                     "each change (MW / Welch); uncorrected.", fontsize=7)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / "figures" / f"133_area_groups_{m}{TAG}.{ext}", dpi=220)
        plt.close(fig)
    pd.set_option("display.width", 250)
    print(S[S.area == "All units"].round(3).T.to_string())


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else run()
