"""015 -- Test 6: the hit - miss axis before vs after each mouse's behavioural learning trial (user 2026-10-16).
Does the R- decoupling of the passive whisker response appear on the axis of the post-learning-trial active trials, and does the axis itself
change at the learning trial more than at placebo split points? Expert sessions with pseudo-learning trials as control.
Data and preprocessing exactly as ssl-hitmiss-axis-passive 135 (SSL_135_HARM, tracked stable units, whole brain): rate 5-35 ms after
stimulus onset minus the epoch-mean baseline (-55 to -20 ms), units scaled by their active-trial mean / SD; active whisker trials with a lick
before 35 ms excluded; passive pre / post whisker (W) and auditory (A) trials.
Split at time t_s: active whisker trials before t_s (pre-split) and after (post-split); per side the unit-norm hit - miss axis
(>= MIN_CLASS hits and misses per side). Per axis u (pre-split, post-split), passive projections per neuron pi(e, s; u) = mean_z(e, s) . u /
sqrt(N) and the passive pre -> post change d_s(u) = pi(post, s; u) - pi(pre, s; u) for s = W, A, W - A. Quantities:
  dWA_pre, dWA_post   whisker - auditory change on the pre- and post-split axes;  axis_effect = dWA_post - dWA_pre
  cos_axes            cosine between the pre- and post-split axes (rotation of the axis at the split; biased low by noise)
Real split: day-0 learning trial (017 lt_final, sigma1; time from 008); placebo splits: every active whisker trial >= MIN_DIST whisker trials
from it with enough hits / misses on both sides; excess = real - mean placebo, percentile. Expert sessions: pseudo-learning trial at the
cohort's median relative position of the day-0 learning trials.
Output: tables/015_sessions.csv, 015_stats.csv; figures/015_hitmiss_lt.{png,pdf,svg}
Run (haas, repo env): cd ~/code/ibl-ai-agent && ./.venv/bin/python projects/ssl-within-day-two-axes/exploratory-analyses/015_hitmiss_lt.py
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HM = REPO / "projects" / "ssl-hitmiss-axis-passive" / "exploratory-analyses"
SCRIPTS = str(REPO / "scripts")
sys.path[:0] = [SCRIPTS, str(HM)]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-within-day-two-axes"
TAB, FIG = ROOT / "tables", ROOT / "figures"
WIN, BASE, DZ = (0.005, 0.035), (-0.055, -0.020), (-0.010, 0.005)
MIN_UNITS, MIN_CLASS, MIN_DIST = 5, 4, 10
COH = {"R+": "#00B400", "R-": "#C800C8"}
SET_IDS = {}
Q = ["dWA_pre", "dWA_post", "axis_effect", "dW_pre", "dW_post", "cos_axes"]


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def metrics(Z, isw, wt, licked, split_t, n_units):
    pre, post = wt < split_t, wt >= split_t
    out = {}
    U = {}
    for k, m in (("pre", pre), ("post", post)):
        h, mi = np.where(isw & m & licked)[0], np.where(isw & m & ~licked)[0]
        if min(len(h), len(mi)) < MIN_CLASS:
            return None
        U[k] = unit(Z["active"][h].mean(0) - Z["active"][mi].mean(0))
    pw = {e: Z[e]["W"].mean(0) for e in ("passive_pre", "passive_post")}; pa = {e: Z[e]["A"].mean(0) for e in ("passive_pre", "passive_post")}
    for k, u in U.items():
        dW = (pw["passive_post"] - pw["passive_pre"]) @ u / np.sqrt(n_units); dA = (pa["passive_post"] - pa["passive_pre"]) @ u / np.sqrt(n_units)
        out[f"dW_{k}"], out[f"dA_{k}"], out[f"dWA_{k}"] = dW, dA, dW - dA
    out["axis_effect"] = out["dWA_post"] - out["dWA_pre"]; out["cos_axes"] = float(U["pre"] @ U["post"])
    return out


def process(args):
    sid, subject, rg, stage, lt_time, relpos = args
    _init()
    sys.path[:0] = [SCRIPTS, str(HM)]
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = M132.session_trials(sid, st, tt, T)
    base = dict(session_id=sid, mouse_id=subject, cohort=rg, stage=stage)
    if trs is None:
        return dict(base, skipped="no passive pre + post")
    act = trs["active"].reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    ok = ~licked | (rt > WIN[1]); act = act[ok].reset_index(drop=True); licked = licked[ok]
    units = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), SET_IDS.get(sid, np.array([], dtype=np.int64)))
    if len(units) < MIN_UNITS:
        return dict(base, skipped="units")
    spikes = T.load_session_unit_spikes(root, sid)
    X = {}
    for e, d in (("passive_pre", trs["passive_pre"]), ("active", act), ("passive_post", trs["passive_post"])):
        t0 = d.start_time.to_numpy()
        r, b = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
        X[e] = r - np.nanmean(b, 0, keepdims=True)
    mu, sd = X["active"].mean(0), X["active"].std(0); keep = sd > 1e-6
    if keep.sum() < MIN_UNITS:
        return dict(base, skipped="units")
    Z = {"active": (X["active"][:, keep] - mu[keep]) / sd[keep]}
    for e in ("passive_pre", "passive_post"):
        z = (X[e][:, keep] - mu[keep]) / sd[keep]; ty = trs[e].trial_type.to_numpy()
        Z[e] = {"W": z[ty == "whisker_trial"], "A": z[ty == "auditory_trial"]}
        if min(len(Z[e]["W"]), len(Z[e]["A"])) < 3:
            return dict(base, skipped="passive trials")
    isw = (act.trial_type == "whisker_trial").to_numpy(); t_all = act.start_time.to_numpy()
    wt_times = t_all[isw]
    if stage == "learning":
        k_real = int(np.searchsorted(wt_times, lt_time))
    else:
        k_real = int(round(relpos * len(wt_times)))
    if not 0 < k_real < len(wt_times):
        return dict(base, skipped="split outside")
    n = int(keep.sum())
    real = metrics(Z, isw, t_all, licked, wt_times[k_real], n)
    if real is None:
        return dict(base, skipped="too few hits / misses on one side")
    pl = []
    for k in range(1, len(wt_times)):
        if abs(k - k_real) < MIN_DIST:
            continue
        m = metrics(Z, isw, t_all, licked, wt_times[k], n)
        if m is not None:
            pl.append(m)
    if len(pl) < 5:
        return dict(base, skipped="too few placebo splits")
    P = pd.DataFrame(pl)
    out = dict(base, n_units=n, k=k_real, n_wt=len(wt_times), n_placebo=len(P))
    for q in Q:
        out[q] = real[q]; out[f"{q}_placebo"] = P[q].mean(); out[f"{q}_excess"] = real[q] - P[q].mean()
        out[f"{q}_pct"] = float(np.mean(P[q] < real[q]))
    return out


def run(n_workers):
    os.chdir(REPO)
    tu = importlib.import_module("tracked_units")
    for path in (tu.table_path(), tu.table_path().with_name(tu.table_path().stem + "_expert.parquet")):   # day-0 and expert lists (137b)
        d = pd.read_parquet(path); d = d[d["stable"]]
        SET_IDS.update({s: np.sort(g.cluster_id.to_numpy()) for s, g in d.groupby("session_id")})
    S8 = pd.read_csv(TAB / "008_sessions.csv")
    d0 = S8[(S8.stage == "learning") & S8.lt_time.notna()]
    relpos = (d0.lt_wtrial / d0.n_wtrials).groupby(d0.cohort).median()
    args = [(r.session_id, r.mouse_id, r.cohort, r.stage, r.lt_time, relpos[r.cohort]) for r in S8.itertuples()
            if (r.stage == "learning" and np.isfinite(r.lt_time)) or r.stage == "expert"]
    print(f"[015] {len(args)} sessions", flush=True)
    rows = []
    with ProcessPoolExecutor(n_workers, initializer=_init) as ex:
        futs = {ex.submit(process, a): a for a in args}
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as e:  # noqa: BLE001
                a = futs[f]; rows.append(dict(session_id=a[0], mouse_id=a[1], cohort=a[2], stage=a[3], skipped=f"error {e!r}"[:200]))
    S = pd.DataFrame(rows); S.to_csv(TAB / "015_sessions.csv", index=False)
    return S


def stats_table(S):
    S = S[S.get("skipped").isna()] if "skipped" in S else S
    St = []
    for q in Q:
        for c in ("R+", "R-"):
            for s_ in ("learning", "expert"):
                v = S[(S.cohort == c) & (S.stage == s_)][f"{q}_excess"].dropna()
                p = S[(S.cohort == c) & (S.stage == s_)][f"{q}_pct"].dropna()
                if len(v) >= 3:
                    St.append(dict(measure=q, cohort=c, stage=s_, n=len(v), excess=v.mean(), sem=v.sem(), p_excess=stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan,
                                   pct=p.mean(), p_pct=stats.wilcoxon(p - 0.5).pvalue if len(p) >= 5 else np.nan, real=S[(S.cohort == c) & (S.stage == s_)][q].mean()))
            a, b = (S[(S.cohort == c) & (S.stage == s_)][f"{q}_excess"].dropna() for s_ in ("learning", "expert"))
            if len(a) >= 3 and len(b) >= 3:
                St.append(dict(measure=q, cohort=c, stage="day 0 vs expert", p_excess=stats.mannwhitneyu(a, b).pvalue))
        a, b = (S[(S.cohort == c) & (S.stage == "learning")][f"{q}_excess"].dropna() for c in ("R+", "R-"))
        if len(a) >= 3 and len(b) >= 3:
            St.append(dict(measure=q, cohort="R+ vs R-", stage="learning", p_excess=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return pd.DataFrame(St)


def fp(p):
    return "n/a" if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.2f}".replace("0.", "."))


def figure(S, St):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    S = S[S.skipped.isna()] if "skipped" in S else S
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.5), gridspec_kw=dict(wspace=0.65))
    # a: dWA on pre- vs post-split axes (day 0 learners), per session lines
    ax = axs[0]
    for j, c in enumerate(("R+", "R-")):
        d = S[(S.cohort == c) & (S.stage == "learning")]
        for _, r in d.iterrows():
            ax.plot([j * 2.4, j * 2.4 + 1], [r.dWA_pre, r.dWA_post], color=COH[c], lw=0.4, alpha=0.35)
        ax.errorbar([j * 2.4, j * 2.4 + 1], [d.dWA_pre.mean(), d.dWA_post.mean()], [d.dWA_pre.sem(), d.dWA_post.sem()], color=COH[c], marker="o", ms=3, capsize=0)
    ax.set_xticks([0, 1, 2.4, 3.4], ["pre-LT\naxis", "post-LT\naxis"] * 2); ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.set_ylabel("passive W − A change (post − pre)"); ax.set_title("day 0: W − A change on the\naxis before / after the LT", fontsize=5.6)
    ax.text(-0.45, 1.12, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    for ax, (q, lab), l in zip(axs[1:], (("axis_effect", "axis effect (post − pre axis)"), ("dWA_post", "W − A change, post-split axis"),
                                          ("cos_axes", "cos(pre-, post-split axis)")), "bcd"):
        for j, c in enumerate(("R+", "R-")):
            for i, s_ in enumerate(("learning", "expert")):
                v = S[(S.cohort == c) & (S.stage == s_)][f"{q}_excess"].dropna(); x = j * 2.4 + i
                ax.scatter(x + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=3, color=COH[c], alpha=0.35, lw=0)
                ax.errorbar(x, v.mean(), v.sem(), fmt="o", ms=3.4, color=COH[c], mfc="white" if s_ == "learning" else COH[c], capsize=0)
                r = St[(St.measure == q) & (St.cohort == c) & (St.stage == s_)]
                ax.text(x, 1.0, fp(r.p_excess.iloc[0]) if len(r) else "n/a", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.2, color=COH[c])
        ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_xticks([0, 1, 2.4, 3.4], ["D0", "E", "D0", "E"])
        ax.set_ylabel("excess over placebo splits"); ax.set_title(lab, fontsize=5.6, pad=12); ax.text(-0.45, 1.12, l, transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Test 6: hit − miss axis split at the behavioural learning trial (day 0) or a pseudo-learning trial (expert), vs placebo splits "
                 "(green R+, magenta R−)", fontsize=6.4, x=0.01, y=1.08, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"015_hitmiss_lt.{ext}", dpi=300, bbox_inches="tight")


def main():
    S = pd.read_csv(TAB / "015_sessions.csv") if "--replot" in sys.argv else run(int(os.environ.get("N_WORKERS", "24")))
    St = stats_table(S); St.to_csv(TAB / "015_stats.csv", index=False)
    figure(S, St)
    pd.set_option("display.width", 220)
    print(S.groupby(["cohort", "stage"]).skipped.value_counts(dropna=False).to_string() if "skipped" in S else "")
    print(St.round(4).to_string())


if __name__ == "__main__":
    main()
