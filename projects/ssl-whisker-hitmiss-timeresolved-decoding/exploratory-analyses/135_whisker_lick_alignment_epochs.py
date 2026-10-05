"""135 -- Whisker axis . lick axis alignment, 5-35 ms after stimulus (no movement), across passive_pre, active first half, active
second half and passive_post (user 2026-10-02: "For measure 3 we're only interested in 5-35 ms ... compare within passive pre
then in the active epoch ... split the active epoch in two halves ... single time window for passive pre, first active half,
second active half and passive post").
Passive mice almost never lick (0.3% of passive whisker / auditory trials; 2 / 114 sessions with >= 5 licked and unlicked), so
the LICK axis is defined once, in the active epoch: mean(licked) - mean(unlicked) whisker trials (lick after 35 ms; hits vs
misses), and the WHISKER axis (mean(whisker) - mean(auditory)) is computed in each of the four epochs and compared with it.
Independence: on each of N_SPLIT random splits, the active trials are halved at random (stratified by trial type x lick);
the lick axis is estimated from one half and every ACTIVE-epoch whisker axis from the other half's trials of that active
half -- the two are never estimated from the same trials. Passive whisker axes use their own epoch's trials.
cos_norm = cos(whisker axis, lick axis) / sqrt(rel_whisker * rel_lick), reliabilities = split-half cos within the same trial
sets (floored at 0.05), averaged over splits; raw cos also saved.
Safeguards (as 133/134): good units with >= 0.5 Hz in every epoch; rates (Hz) 5-35 ms; epoch-specific baseline (-55..-20 ms,
the active baseline shared by both active halves); z-scored per unit over all trials (pooled epochs); active trials with a lick
before 35 ms excluded; trials from skills/ssl-trial-exclusion.
Stats (mouse = unit; uncorrected): per cohort, epoch effect (Friedman AND RM-ANOVA); vs 0 per epoch (Wilcoxon AND t);
R+ vs R- per epoch and on the changes from passive_pre (Mann-Whitney AND Welch).
Outputs: 135_alignment_epochs.parquet, 135_stats.csv, figures/135_alignment_epochs.{pdf,png,svg}
Run (haas, repo root): python .../135_whisker_lick_alignment_epochs.py   |   python ... plot
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
EP4 = ["passive_pre", "active_1", "active_2", "passive_post"]
LAB = {"passive_pre": "passive\npre", "active_1": "active\n1st half", "active_2": "active\n2nd half", "passive_post": "passive\npost"}
WIN, BASE, DZ = (0.005, 0.035), (-0.055, -0.020), (-0.010, 0.005)
MIN_UNITS, N_SPLIT = 5, 50
MIN_CLASS = int(os.environ.get("SSL_MIN_CLASS", "5"))     # per class PER HALF -> >= 2 * MIN_CLASS licked and unlicked trials
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
TAG = "" if MIN_CLASS == 5 else f"_min{MIN_CLASS}"
# 2026-10-05: SSL_135_UNITSET=good|stable uses the 137 unit sets (skills/ssl-valid-data "Unit sets": good = good AND stable)
# instead of the label-table quality_label (no drift check); outputs carry the suffix _<unitset>.
# SSL_135_UNITSET=tracked: the shared Part III tracked stable units (tracked_units.py / 137b), no further rate filter.
UNITSET = os.environ.get("SSL_135_UNITSET", "")
TAG = TAG + (f"_{UNITSET}" if UNITSET else "")
OUT_PATH = OUT / f"135_alignment_epochs{TAG}.parquet"
SET_IDS = {}


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else np.nan


def halve(idx, rng):
    i = rng.permutation(idx)
    return i[: len(i) // 2], i[len(i) // 2:]


def wvec(Z, w, a):
    return Z[w].mean(0) - Z[a].mean(0)


def process(args):
    sid, subject, rg = args
    _init()
    sys.path.insert(0, SCRIPTS)
    sys.path.insert(0, str(OUT))
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    M134 = importlib.import_module("134_whisker_specific_change")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(session_id=sid, mouse_id=subject, reward_group=rg, skipped_reason="no passive pre+post or no active")]
    act = trs["active"].reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    ok_act = ~licked | (rt > WIN[1])                                    # no lick before the window ends
    act = act[ok_act].reset_index(drop=True)
    licked = licked[ok_act]
    trs = dict(trs, active=act)
    half = np.zeros(len(act), int)
    half[len(act) // 2:] = 1                                           # chronological halves of the active epoch
    spikes = T.load_session_unit_spikes(root, sid)
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    sl = labels[labels.session_id == sid]
    areas = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(sl["area_group"].dropna().unique())]
    rows = []
    for area_col, area in areas:
        if UNITSET == "tracked":                  # shared Part III tracked stable units (tracked_units.py / 137b), taken as is
            units = np.intersect1d(T.area_units(sid, area_col, area, labels), SET_IDS.get(sid, np.array([], dtype=np.int64)))
        elif UNITSET:
            au = np.intersect1d(T.area_units(sid, area_col, area, labels), SET_IDS.get(sid, np.array([], dtype=np.int64)))
            lab = labels.assign(quality_label=np.where(labels.cluster_id.isin(au) & (labels.session_id == sid), "good", "mua"))
            units = M134.tracked_good(sid, au, lab, spikes, {e: trs[e] for e in ("passive_pre", "active", "passive_post")})
        else:
            units = M134.tracked_good(sid, T.area_units(sid, area_col, area, labels), labels, spikes,
                                      {e: trs[e] for e in ("passive_pre", "active", "passive_post")})
        if len(units) < MIN_UNITS:
            continue
        X = {}
        for e in ("passive_pre", "active", "passive_post"):
            t0 = trs[e].start_time.to_numpy()
            r, b = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
            X[e] = r - np.nanmean(b, 0, keepdims=True)
        if any(np.isnan(v).any() for v in X.values()):
            continue
        P = np.vstack(list(X.values()))
        mu, sd = P.mean(0), P.std(0)
        keep = sd > 0
        if keep.sum() < MIN_UNITS:
            continue
        Z = {e: (v[:, keep] - mu[keep]) / sd[keep] for e, v in X.items()}
        E = {e: v[:, keep] / sd[keep] for e, v in X.items()}      # EVOKED (baseline-subtracted, scaled, not mean-centred)
        isw = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in Z}
        wl, wn = np.where(isw["active"] & licked)[0], np.where(isw["active"] & ~licked)[0]
        if min(len(wl), len(wn)) < 2 * MIN_CLASS:
            rows.append(dict(base, area=area, skipped_reason=f"too few licked/unlicked active whisker trials ({len(wl)}/{len(wn)})"))
            continue
        need = {"passive_pre": isw["passive_pre"], "passive_post": isw["passive_post"]}
        if any(min(m.sum(), (~m).sum()) < 2 * MIN_CLASS for m in need.values()):
            continue
        rng = np.random.default_rng(zlib.crc32(f"{sid}|{area}|135".encode()))
        acc = {e: dict(cab=[], rw=[], rl=[], cW=[], rW=[], cA=[], rA=[]) for e in EP4}
        for _ in range(N_SPLIT):
            # split ALL active trials (stratified by type x lick) into L (lick-axis) and V (whisker-axis) sets
            Lset, Vset = [], []
            for m in (isw["active"] & licked, isw["active"] & ~licked, ~isw["active"]):
                a_, b_ = halve(np.where(m)[0], rng)
                Lset.append(a_)
                Vset.append(b_)
            L = np.concatenate(Lset)
            V = np.concatenate(Vset)
            # lick axis from L, split again for its reliability
            lw, ln = np.intersect1d(L, wl), np.intersect1d(L, wn)
            lw1, lw2 = halve(lw, rng)
            ln1, ln2 = halve(ln, rng)
            lick1 = Z["active"][lw1].mean(0) - Z["active"][ln1].mean(0)
            lick2 = Z["active"][lw2].mean(0) - Z["active"][ln2].mean(0)
            rel_l = cos(lick1, lick2)
            lick = Z["active"][lw].mean(0) - Z["active"][ln].mean(0)
            for e in EP4:
                if e.startswith("active"):
                    h = int(e[-1]) - 1
                    pool = V[half[V] == h]
                    Ze, Ee, w_ = Z["active"], E["active"], isw["active"]
                else:
                    pool = np.arange(len(isw[e]))
                    Ze, Ee, w_ = Z[e], E[e], isw[e]
                wi, ai = pool[w_[pool]], pool[~w_[pool]]
                if min(len(wi), len(ai)) < 4:
                    continue
                w1, w2 = halve(wi, rng)
                a1, a2 = halve(ai, rng)
                acc[e]["rw"].append(cos(wvec(Ze, w1, a1), wvec(Ze, w2, a2)))
                acc[e]["cab"].append(cos(wvec(Ze, wi, ai), lick))
                acc[e]["rl"].append(rel_l)
                # decomposition: whisker-evoked and auditory-evoked patterns alone vs the lick axis (is it the auditory side?)
                for key, idx_, h1, h2 in (("W", wi, w1, w2), ("A", ai, a1, a2)):
                    acc[e][f"c{key}"].append(cos(Ee[idx_].mean(0), lick))
                    acc[e][f"r{key}"].append(cos(Ee[h1].mean(0), Ee[h2].mean(0)))
        row = dict(base, area=area, n_units=int(keep.sum()), n_wh_licked=len(wl), n_wh_unlicked=len(wn), skipped_reason=None)
        for e in EP4:
            if not acc[e]["cab"]:
                continue
            c = np.nanmean(acc[e]["cab"])
            rw, rl = max(np.nanmean(acc[e]["rw"]), 0.05), max(np.nanmean(acc[e]["rl"]), 0.05)
            row[f"cos_{e}"] = float(c)
            row[f"cosnorm_{e}"] = float(np.clip(c / np.sqrt(rw * rl), -1.5, 1.5))
            row[f"rel_whisker_{e}"] = float(np.nanmean(acc[e]["rw"]))
            row[f"rel_lick_{e}"] = float(np.nanmean(acc[e]["rl"]))
            for key in ("W", "A"):
                cc = np.nanmean(acc[e][f"c{key}"])
                rr = max(np.nanmean(acc[e][f"r{key}"]), 0.05)
                row[f"evoked{key}_cosnorm_{e}"] = float(np.clip(cc / np.sqrt(rr * rl), -1.5, 1.5))
        rows.append(row)
    return rows or [dict(base, skipped_reason="no area")]


def run():
    os.chdir(OUT.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    if UNITSET == "tracked":
        sys.path.insert(0, SCRIPTS)
        SET_IDS.update(importlib.import_module("tracked_units").load("stable"))
    elif UNITSET:                                 # filled before the worker pool forks, so workers inherit it
        sys.path.insert(0, SCRIPTS)
        from axel_bisi_paths import axel_bisi_root
        U = pd.read_parquet(axel_bisi_root() / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet",
                            columns=["session_id", "cluster_id", UNITSET])
        SET_IDS.update({s: g.cluster_id.to_numpy() for s, g in U[U[UNITSET]].groupby("session_id")})
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples() if r.session_id not in done]
    print(f"[135] {len(args)} sessions", flush=True)
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
            print(f"[135] [{i}/{len(args)}] {a[0]} -- {time.time() - t0:.0f}s", flush=True)
    plot()


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.stats.anova import AnovaRM
    d = pd.read_parquet(OUT_PATH)
    d = d[d.skipped_reason.isna()]
    cols = [f"cosnorm_{e}" for e in EP4]
    rows = []
    for area, g in d.groupby("area"):
        r = dict(area=area)
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg][cols].dropna()
            r[f"n_{rg}"] = len(x)
            for c, e in zip(cols, EP4):
                r[f"mean_{rg}_{e}"] = x[c].mean() if len(x) else np.nan
                r[f"sem_{rg}_{e}"] = x[c].std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                if len(x) >= 5:
                    r[f"pW0_{rg}_{e}"] = stats.wilcoxon(x[c]).pvalue
                    r[f"pt0_{rg}_{e}"] = stats.ttest_1samp(x[c], 0).pvalue
            if len(x) >= 4:
                r[f"friedman_{rg}"] = stats.friedmanchisquare(*[x[c] for c in cols]).pvalue
                long = x.reset_index().melt(id_vars="index", value_vars=cols, var_name="ep", value_name="v")
                try:
                    r[f"rmanova_{rg}"] = float(AnovaRM(long, "v", "index", within=["ep"]).fit().anova_table["Pr > F"].iloc[0])
                except Exception:  # noqa: BLE001
                    r[f"rmanova_{rg}"] = np.nan
        a, b = g[g.reward_group == "R+"], g[g.reward_group == "R-"]
        for c, e in zip(cols, EP4):
            xa, xb = a[c].dropna(), b[c].dropna()
            if min(len(xa), len(xb)) >= 3:
                r[f"pMW_{e}"] = stats.mannwhitneyu(xa, xb).pvalue
                r[f"pWelch_{e}"] = stats.ttest_ind(xa, xb, equal_var=False).pvalue
            if e != "passive_pre":
                da = (a[c] - a["cosnorm_passive_pre"]).dropna()
                db = (b[c] - b["cosnorm_passive_pre"]).dropna()
                if min(len(da), len(db)) >= 3:
                    r[f"pMW_change_{e}"] = stats.mannwhitneyu(da, db).pvalue
                    r[f"pWelch_change_{e}"] = stats.ttest_ind(da, db, equal_var=False).pvalue
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"135_stats{TAG}.csv", index=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 6.5})
    key = ["All units", "Somatosensory-whisker", "Motor areas", "Frontal areas", "Striatum", "Thalamus", "Midbrain", "Hippocampus"]
    key = [k for k in key if k in set(S.area)]
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 5.2), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12, wspace=0.45, hspace=0.95)
    for ax, ar in zip(axes.flat, key):
        g = d[d.area == ar]
        s = S[S.area == ar].iloc[0]
        xs = np.arange(4)
        for k, rg in enumerate(("R+", "R-")):
            x = g[g.reward_group == rg][cols].dropna()
            off = (k - 0.5) * 0.12
            for _, rr in x.iterrows():
                ax.plot(xs + off, rr.to_numpy(), color=COL[rg], lw=0.3, alpha=0.2)
            if len(x):
                ax.errorbar(xs + off, x.mean(), yerr=x.std(ddof=1) / np.sqrt(len(x)), color=COL[rg], lw=1.6, marker="o",
                            ms=3, capsize=0, label=f"{'R+' if rg == 'R+' else 'R−'} n={len(x)}")
        for i, e in enumerate(EP4):
            pm, pw = s.get(f"pMW_{e}", np.nan), s.get(f"pWelch_{e}", np.nan)
            mark = "**" if (pm < 0.05 and pw < 0.05) else ("*" if (pm < 0.05 or pw < 0.05) else "")
            ax.text(i, 1.0, mark, transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([LAB[e] for e in EP4], fontsize=5.5)
        ax.set_ylabel("whisker axis · lick axis (norm. cos)", fontsize=6)
        ax.set_title(f"{ar}\nR+ F {pf(s.get('friedman_R+', np.nan))}/RM {pf(s.get('rmanova_R+', np.nan))}; "
                     f"R− F {pf(s.get('friedman_R-', np.nan))}/RM {pf(s.get('rmanova_R-', np.nan))}", fontsize=5.5)
        if ar == "All units":
            ax.legend(fontsize=5.5, frameon=False)
    for ax in axes.flat[len(key):]:
        ax.set_axis_off()
    fig.suptitle("Whisker axis (whisker − auditory, per epoch) vs lick axis (active hits − misses, independent trials), 5-35 ms "
                 "after stimulus; tracked good units; epoch-specific baseline. ** = R+ vs R− MW AND Welch p < .05, * = one test. "
                 "Within cohort: Friedman (F) / RM-ANOVA (RM). Uncorrected.", fontsize=6.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"135_alignment_epochs{TAG}.{ext}", dpi=250)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(S[S.area == "All units"].T.to_string())


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else run()
