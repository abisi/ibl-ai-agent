"""132 -- Whisker vs auditory decoding at STIMULUS onset (all stimulus trials, any outcome) in passive_pre, active and
passive_post, within and across conditions, whole brain and per area group (user 2026-10-02: "extend the whisker vs
auditory stim-aligned decoding on all stim trials to passive_pre, active and passive_post. Use the window 5 to 35 ms post
stimulus. Here the null is from trial shuffling. Also do cross-condition evaluation and compare, across passive_pre-active-
passive_post, corrected performance between cohorts. Do whole-brain and per-area_group").
Sessions: learning-stage (whisker day 0) R+ / R- sessions with passive trials BEFORE and AFTER the main active block (trials
context; pre-only sessions count as no passive -- user definition).
Trials (skills/ssl-trial-exclusion):
  active        prep_modality_trials (active context, perf != 6, warm-up cut) restricted to trials from the first whisker
                trial on (the kept warm-up trial removed), whisker + auditory, any outcome; A1 end-of-session trim;
  passive_pre   context == 'passive' trials before the first active trial (whisker + auditory; passive: no perf filter);
  passive_post  context == 'passive' trials after the last active trial.
Features: spike count per unit in 5-35 ms after start_time (stimulus onset; dead zone -10..+5 ms, window starts after it),
QC units (good + mua) of the area (whole brain, or one area_group); >= MIN_UNITS units.
Decoder: StandardScaler -> L2 logistic regression (liblinear, class_weight balanced), C fixed per (session, area) by
select_fixed_c_pooled on all three conditions pooled; balanced accuracy.
Matching: every condition subsampled to n_match trials PER CLASS (min over conditions and classes, >= MIN_PER_CLASS), K_SUB
random subsamples; within-condition = stratified 5-fold CV on the subsample; cross-condition A -> B = fit on the whole A
subsample, test on the B subsample (disjoint trials).
Null (trial shuffling): training labels permuted (N_SHUF per subsample) -- within: permuted labels throughout the CV;
cross: fit on A with permuted labels, test on B's true labels. corrected = accuracy - mean null.
Stats (mouse = unit, one session per mouse; uncorrected): per area x train x test cell, corrected vs 0 per cohort (Wilcoxon
AND one-sample t); R+ vs R- (Mann-Whitney AND Welch).
Outputs: 132_modality_stim_passive_active.parquet (session x area x train x test), 132_stats.csv,
figures/132_whole_brain.{pdf,png,svg} (3x3 train/test matrices per cohort + R+ - R- with p; within-condition dots),
figures/132_area_groups.{pdf,png,svg} (within-condition and active -> passive cells per area group, cohorts side by side).
Baseline (env SSL_BASELINE; user 2026-10-02): none = raw counts; trial = response minus the same trial's count in -40..-10 ms
(equal length, before the dead zone); condition = response minus each unit's mean -40..-10 ms count within its own
condition (removes epoch-specific state offsets that would otherwise shift cross-condition inputs).
Run (haas, repo root): python .../132_modality_stim_passive_active.py   |   python ... plot
"""

from __future__ import annotations

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
COL = {"R+": "#00B400", "R-": "#C800C8"}
CONDS = ["passive_pre", "active", "passive_post"]
WIN, DZ = (0.005, 0.035), (-0.010, 0.005)
MIN_UNITS, MIN_PER_CLASS = 5, 8
K_SUB, N_SHUF, N_FOLD = 5, 20, 5
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
BASELINE = os.environ.get("SSL_BASELINE", "none")          # none | trial | condition (user 2026-10-02: epoch-specific baselines)
BASE_WIN = (-0.040, -0.010)                                # equal length (30 ms), ends where the artefact dead zone starts
TAG = "" if BASELINE == "none" else f"_base-{BASELINE}"
OUT_PATH = OUT / f"132_modality_stim_passive_active{TAG}.parquet"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


def fit_predict(Xa, ya, Xb, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", solver="liblinear", class_weight="balanced",
                                                           max_iter=2000))
    return m.fit(Xa, ya).predict(Xb)


def bacc(y, p):
    return 0.5 * (np.mean(p[y]) + np.mean(~p[~y])) if y.any() and (~y).any() else np.nan


def within(X, y, C, rng):
    from sklearn.model_selection import StratifiedKFold
    pred = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(N_FOLD, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, y):
        pred[te] = fit_predict(X[tr], y[tr], X[te], C)
    return bacc(y, pred)


def subsample(y, n, rng):
    pos, neg = np.where(y)[0], np.where(~y)[0]
    return np.sort(np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])


def session_trials(sid, st, tt, T):
    from ssl_bwm_trial_prep import prep_session  # noqa: F401  (prep_modality_trials uses it)
    t = tt[tt.session_id == sid].sort_values("start_time").reset_index(drop=True)
    act_idx = np.where(t.context == "active")[0]
    if not len(act_idx):
        return None
    t0, t1 = t.start_time.iloc[act_idx[0]], t.start_time.iloc[act_idx[-1]]
    pas = t[(t.context == "passive") & t.trial_type.isin(["whisker_trial", "auditory_trial"])]
    pre, post = pas[pas.start_time < t0], pas[pas.start_time > t1]
    if not len(pre) or not len(post):
        return None
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    act = T.prep_modality_trials(resolve_dataset_dir("ssl_ephys"), sid, st, tt)
    if act is None:
        return None
    fw = act.loc[act.trial_type == "whisker_trial", "start_time"].min()
    act = act[act.start_time >= fw]                         # remove the kept warm-up trial too
    dis = T.detect_terminal_disengagement(sid, tt)
    if dis["disengaged"]:
        act = act[act.start_time < dis["t_cut"]]
    return {"passive_pre": pre, "active": act.reset_index(drop=True), "passive_post": post}


def process(args):
    sid, subject, rg = args
    _init()
    sys.path.insert(0, SCRIPTS)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(session_id=sid, mouse_id=subject, reward_group=rg, skipped_reason="no passive pre+post or no active")]
    spikes = T.load_session_unit_spikes(root, sid)
    ys = {c: (trs[c].trial_type == "whisker_trial").to_numpy() for c in CONDS}
    counts = {c: (int(ys[c].sum()), int((~ys[c]).sum())) for c in CONDS}
    n_match = min(min(v) for v in counts.values())
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg, n_match=n_match,
                **{f"n_{c}_{k}": counts[c][i] for c in CONDS for i, k in enumerate(("whisker", "auditory"))})
    if n_match < MIN_PER_CLASS:
        return [dict(base, skipped_reason=f"n_match {n_match} < {MIN_PER_CLASS}")]
    sess_labels = labels[labels.session_id == sid] if "session_id" in labels else labels
    areas = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(sess_labels["area_group"].dropna().unique())]
    rows = []
    for area_col, area in areas:
        units = T.area_units(sid, area_col, area, labels)
        if len(units) < MIN_UNITS:
            continue
        X = {}
        for c in CONDS:
            st_ = trs[c].start_time.to_numpy()
            resp, base_ = T.sliding_bin_population_matrices(spikes, units, st_, np.ones(len(st_), bool), [WIN, BASE_WIN],
                                                            dead_zone=DZ)
            if BASELINE == "trial":            # per-trial: response minus the same trial's pre-stimulus count
                X[c] = resp - base_
            elif BASELINE == "condition":      # per-epoch: response minus the unit's mean pre-stimulus count in this condition
                X[c] = resp - np.nanmean(base_, 0, keepdims=True)
            else:
                X[c] = resp
        if any(np.isnan(X[c]).any() for c in CONDS):
            continue
        rng = np.random.default_rng(zlib.crc32(f"{sid}|{area}".encode()))
        Xall = np.vstack([X[c] for c in CONDS])
        yall = np.concatenate([ys[c] for c in CONDS])
        C = T.select_fixed_c_pooled(Xall, yall, rng, n_folds=5)
        acc = {(a, b): [] for a in CONDS for b in CONDS}
        nul = {(a, b): [] for a in CONDS for b in CONDS}
        for _ in range(K_SUB):
            idx = {c: subsample(ys[c], n_match, rng) for c in CONDS}
            for a in CONDS:
                Xa, ya = X[a][idx[a]], ys[a][idx[a]]
                acc[(a, a)].append(within(Xa, ya, C, rng))
                for _s in range(N_SHUF):
                    nul[(a, a)].append(within(Xa, rng.permutation(ya), C, rng))
                for b in CONDS:
                    if b == a:
                        continue
                    Xb, yb = X[b][idx[b]], ys[b][idx[b]]
                    acc[(a, b)].append(bacc(yb, fit_predict(Xa, ya, Xb, C)))
                    for _s in range(N_SHUF):
                        nul[(a, b)].append(bacc(yb, fit_predict(Xa, rng.permutation(ya), Xb, C)))
        for (a, b), v in acc.items():
            rows.append(dict(base, baseline=BASELINE, area_col=area_col, area=area, n_units=len(units), C=C, train=a, test=b,
                             acc=float(np.nanmean(v)), null_mean=float(np.nanmean(nul[(a, b)])),
                             null_sd=float(np.nanstd(nul[(a, b)])), corrected=float(np.nanmean(v) - np.nanmean(nul[(a, b)])),
                             skipped_reason=None))
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
    print(f"[132] {len(args)} sessions, window {WIN}, K_SUB {K_SUB}, N_SHUF {N_SHUF}, {N_WORKERS} workers", flush=True)
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
            print(f"[132] [{i}/{len(args)}] {a[0]}: {int(new.skipped_reason.isna().sum())} rows -- {time.time() - t0:.0f}s",
                  flush=True)
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    d = pd.read_parquet(OUT_PATH)
    print("skipped:", d[d.skipped_reason.notna()].drop_duplicates("session_id").skipped_reason.str[:40].value_counts().to_dict())
    d = d[d.skipped_reason.isna()]
    rows = []
    for (area, a, b), g in d.groupby(["area", "train", "test"]):
        r = dict(area=area, train=a, test=b)
        v = {}
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg].corrected.to_numpy()
            v[rg] = x
            r.update(**{f"n_{rg}": len(x), f"mean_{rg}": x.mean() if len(x) else np.nan,
                        f"sem_{rg}": x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                        f"p_wilcoxon_{rg}": stats.wilcoxon(x).pvalue if len(x) >= 5 else np.nan,
                        f"p_t_{rg}": stats.ttest_1samp(x, 0).pvalue if len(x) >= 5 else np.nan})
        if min(len(v["R+"]), len(v["R-"])) >= 3:
            r.update(p_mw=stats.mannwhitneyu(v["R+"], v["R-"]).pvalue,
                     p_welch=stats.ttest_ind(v["R+"], v["R-"], equal_var=False).pvalue)
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"132_stats{TAG}.csv", index=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 6.5})
    lab = {"passive_pre": "pre", "active": "active", "passive_post": "post"}

    def pf(p):
        return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
    # whole brain
    s = S[S.area == "All units"].set_index(["train", "test"])
    fig, axes = plt.subplots(1, 4, figsize=(8.27, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.18, wspace=0.55)
    vmax = np.nanmax(np.abs(s[["mean_R+", "mean_R-"]].to_numpy()))
    for ax, key, title, cmap, vm in ((axes[0], "mean_R+", "R+", "viridis", (0, vmax)), (axes[1], "mean_R-", "R−", "viridis", (0, vmax)),
                                     (axes[2], "diff", "R+ − R−", "RdBu_r", (-vmax / 2, vmax / 2))):
        M = np.full((3, 3), np.nan)
        for i, a in enumerate(CONDS):
            for j, b in enumerate(CONDS):
                if (a, b) in s.index:
                    M[i, j] = s.loc[(a, b), "mean_R+"] - s.loc[(a, b), "mean_R-"] if key == "diff" else s.loc[(a, b), key]
        im = ax.imshow(M, cmap=cmap, vmin=vm[0], vmax=vm[1])
        for i, a in enumerate(CONDS):
            for j, b in enumerate(CONDS):
                if (a, b) not in s.index:
                    continue
                txt = f"{M[i, j]:.3f}"
                if key == "diff":
                    txt += f"\nMW {pf(s.loc[(a, b), 'p_mw'])}\nW {pf(s.loc[(a, b), 'p_welch'])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5, color="w" if key != "diff" else "k")
        ax.set_xticks(range(3))
        ax.set_xticklabels([lab[c] for c in CONDS])
        ax.set_yticks(range(3))
        ax.set_yticklabels([lab[c] for c in CONDS])
        ax.set_xlabel("test")
        ax.set_ylabel("train")
        ax.set_title(title, color=COL["R+"] if title == "R+" else COL["R-"] if title == "R−" else "k")
        fig.colorbar(im, ax=ax, shrink=0.7)
    ax = axes[3]
    g = d[(d.area == "All units") & (d.train == d.test)]
    rng = np.random.default_rng(0)
    for i, c in enumerate(CONDS):
        for k, rg in enumerate(("R+", "R-")):
            v = g[(g.train == c) & (g.reward_group == rg)].corrected.to_numpy()
            x = i + (k - 0.5) * 0.35
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=5, color=COL[rg], lw=0, alpha=0.7)
            if len(v) > 1:
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="_", color="k", ms=8, elinewidth=1)
        p = s.loc[(c, c)] if (c, c) in s.index else None
        if p is not None:
            ax.text(i, 1.0, f"MW {pf(p.p_mw)}\nW {pf(p.p_welch)}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=5)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab[c] for c in CONDS])
    ax.set_ylabel("balanced acc. − shuffle null")
    ax.set_title("within condition", pad=14)
    n = d[d.area == "All units"].drop_duplicates("session_id").groupby("reward_group").size().to_dict()
    fig.suptitle(f"Whisker vs auditory, 5-35 ms after stimulus onset, whole brain (R+ n={n.get('R+', 0)}, R− n={n.get('R-', 0)}): "
                 "train/test across passive-pre, active, passive-post; trial-shuffle-corrected; R+ vs R−: MW / Welch; "
                 f"baseline: {BASELINE}",
                 fontsize=7)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"132_whole_brain{TAG}.{ext}", dpi=250)
    plt.close(fig)
    # area groups
    areas = [a for a in S.area.unique() if a != "All units"]
    cells = [(c, c) for c in CONDS] + [("active", "passive_pre"), ("active", "passive_post"), ("passive_pre", "passive_post")]
    fig, axes = plt.subplots(len(cells), 1, figsize=(8.27, 1.6 * len(cells) + 0.6), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.07, hspace=0.9)
    for ax, (a, b) in zip(axes[:, 0], cells):
        for i, ar in enumerate(areas):
            for k, rg in enumerate(("R+", "R-")):
                r = S[(S.area == ar) & (S.train == a) & (S.test == b)]
                if not len(r):
                    continue
                r = r.iloc[0]
                m, se, pw, pt = r[f"mean_{rg}"], r[f"sem_{rg}"], r[f"p_wilcoxon_{rg}"], r[f"p_t_{rg}"]
                sig = (np.isfinite(pw) and pw < 0.05) or (np.isfinite(pt) and pt < 0.05)
                ax.errorbar(i + (k - 0.5) * 0.3, m, yerr=se, fmt="o", color=COL[rg], mfc=COL[rg] if sig else "white", ms=3,
                            elinewidth=0.8, capsize=0)
            r = S[(S.area == ar) & (S.train == a) & (S.test == b)]
            if len(r) and np.isfinite(r.iloc[0].get("p_mw", np.nan)) and (r.iloc[0].p_mw < 0.05 or r.iloc[0].p_welch < 0.05):
                ax.text(i, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas, rotation=40, ha="right", fontsize=5.5)
        ax.set_ylabel("acc. − null")
        ax.set_title(f"train {lab[a]} → test {lab[b]}", fontsize=6.5, loc="left")
    fig.suptitle("Whisker vs auditory 5-35 ms, per area group: mean ± SEM per cohort (filled = differs from 0, Wilcoxon or t); "
                 "* = R+ vs R− (MW or Welch) p < 0.05; uncorrected", fontsize=7)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"132_area_groups{TAG}.{ext}", dpi=250)
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(S[S.area == "All units"].round(3).to_string(index=False))


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else run()
