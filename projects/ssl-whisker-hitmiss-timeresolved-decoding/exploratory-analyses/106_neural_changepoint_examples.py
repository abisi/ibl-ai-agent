"""106 -- Agnostic, cross-validated "neural change point" of hit/miss decodability on example sessions
(user request 2026-09-25: "How to be more agnostic about finding where decodability changes most? ... in
a cross-validated manner ... try this on a few example sessions ... avoid any data filtering (except class
balance in folds), and do every 1% percentile").

Per session (whole brain; all learning-stage whisker trials, disengaged trials KEPT, no gates):
  - split positions at every 1% of the session (percentile p = 1..99 of the decoded trials in time order;
    trials before the split = pre).
  - at each split: size-matched balanced accuracy pre and post (both epochs subsampled to the per-class
    minimum over epochs, N_SUB subsamples x N_REP repeats of stratified CV with n_folds = min(5, minority
    count)); change = post - pre. Splits where an epoch has < 2 hits or < 2 misses cannot be decoded and are
    left empty (the only constraint: class balance in folds).
  - three profiles: ALL trials, ODD trials, EVEN trials (interleaved, so both halves span the whole session
    and share its slow drift). C chosen once per profile (all its trials), not per split.
  - neural change point k* = split with the largest change in the cohort's expected direction (R+ max,
    R- min) and with the largest |change|.
  - cross-validation: k* chosen on ODD is evaluated on EVEN (change at that split, and its percentile among
    EVEN's own profile), and vice versa; |k*_odd - k*_even| = location reproducibility.
  - behavioural learning trial (ssl-learning-trial-identification 020, cohort-specific, sigma = 1) converted
    to the same percentile scale; non-learners have none.
Windows: sensory (5-50 ms) and sensory minus baseline.
Outputs: 106_neural_cp_examples.parquet (profiles), 106_neural_cp_examples_summary.csv,
         figures/whole_brain/learning/106_neural_cp_examples_<window>.png
"""

from __future__ import annotations

import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
EXAMPLES = ["AB119", "AB125", "MH029", "AB085", "MH018", "AB120", "MH030", "AB159"]
WINDOWS = {"sensory": (0.005, 0.050), "baseline": (-0.200, -0.010)}
WIN_USE = ["sensory", "sensory_minus_base"]
PCTS = np.arange(1, 100)
N_SUB, N_REP, MAX_FOLDS = 20, 2, 5
DZ = (-0.010, 0.005)
# Edge exclusion (user 2026-09-25: "avoid the session edges by leaving 5 trials off both sides"): splits leaving fewer
# than EDGE_TRIALS whisker trials before or after them are ignored when locating k* (applied at plot/summary time).
EDGE_TRIALS = int(os.environ.get("SSL_EDGE_TRIALS", "0"))
SUF = f"_edge{EDGE_TRIALS}" if EDGE_TRIALS else ""


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def split_task(args):
    key, X, y, is_pre, C, seed = args
    sys.path.insert(0, SCRIPTS_DIR)
    from ssl_timeresolved_decoding import decode_bin_pooled
    rng = np.random.default_rng(seed)
    idx = {"pre": np.where(is_pre)[0], "post": np.where(~is_pre)[0]}
    cnt = {e: (int(y[i].sum()), int((~y[i]).sum())) for e, i in idx.items()}
    th, tm = min(cnt["pre"][0], cnt["post"][0]), min(cnt["pre"][1], cnt["post"][1])
    if min(th, tm) < 2:
        return key, dict(acc_pre=np.nan, acc_post=np.nan, delta=np.nan, matched_hit=th, matched_miss=tm)
    nf = min(MAX_FOLDS, th, tm)
    acc = {}
    for e, ie in idx.items():
        hi, mi = ie[y[ie]], ie[~y[ie]]
        acc[e] = float(np.nanmean([decode_bin_pooled(X[s], y[s], C, rng, n_repeats=N_REP, n_folds=nf) for s in
                                   (np.concatenate([rng.choice(hi, th, replace=False), rng.choice(mi, tm, replace=False)])
                                    for _ in range(N_SUB))]))
    return key, dict(acc_pre=acc["pre"], acc_post=acc["post"], delta=acc["post"] - acc["pre"], matched_hit=th, matched_miss=tm)


def build():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        load_session_unit_spikes, prep_hitmiss_trials, select_fixed_c_pooled, sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st = pd.read_parquet(root / "metadata" / "sessions.parquet")
    tt = pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    lt_tab = pd.read_csv(LTP / "artifacts" / "020_lt_eval.csv").set_index("session_id")
    sids = [next(s for s in lt_tab.index if s.startswith(p)) for p in EXAMPLES]
    tasks, meta = [], {}
    for sid in sids:
        trials = prep_hitmiss_trials(root, sid, st, tt)
        y = trials["lick_flag"].to_numpy().astype(bool)
        t = trials["start_time"].to_numpy()
        n = len(y)
        cw = _active_trials_from_whisker_onset_for_curve(sid, tt)
        cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
        lt = lt_tab.loc[sid, "lt_cohort"]
        lt_pct = 100 * np.mean(t < cw_t[int(lt)]) if not pd.isna(lt) else np.nan
        units = area_units(sid, "whole_brain", "All units", labels)
        spikes = load_session_unit_spikes(root, sid)
        mats = sliding_bin_population_matrices(spikes, units, t, np.ones(n, bool), [WINDOWS["sensory"], WINDOWS["baseline"]],
                                               dead_zone=DZ)
        feats = {"sensory": mats[0], "sensory_minus_base": mats[0] - mats[1]}
        meta[sid] = dict(reward_group=lt_tab.loc[sid, "reward_group"], group=lt_tab.loc[sid, "group"], lt=lt, lt_pct=lt_pct,
                         n_trials=n, n_hits=int(y.sum()), n_units=len(units), y=y.astype(int))
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        for w in WIN_USE:
            for prof, sel in (("all", np.ones(n, bool)), ("odd", np.arange(n) % 2 == 1), ("even", np.arange(n) % 2 == 0)):
                Xs, ys = feats[w][sel], y[sel]
                pos = np.arange(n)[sel]
                C = select_fixed_c_pooled(Xs, ys, rng, n_folds=min(5, int(ys.sum()), int((~ys).sum())))
                for p in PCTS:
                    cut = np.percentile(np.arange(n), p)
                    tasks.append(((sid, w, prof, int(p)), Xs, ys, pos < cut, C, int(rng.integers(1 << 31))))
    print(f"{len(tasks)} split tasks", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SSL_DECODE_N_WORKERS", "96")), initializer=_init) as ex:
        res = list(ex.map(split_task, tasks, chunksize=4))
    rows = [dict(session_id=k[0], window=k[1], profile=k[2], pct=k[3], **v, **{kk: vv for kk, vv in meta[k[0]].items() if kk != "y"})
            for k, v in res]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "106_neural_cp_examples.parquet", index=False)
    pickle.dump(meta, open(OUT / "106_neural_cp_examples_meta.pkl", "wb"))


def summarize_and_plot():
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _s = importlib.util.spec_from_file_location("q034", OUT / "034_area_window_quant_grid.py")
    q034 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(q034)
    df = pd.read_parquet(OUT / "106_neural_cp_examples.parquet")
    meta = pickle.load(open(OUT / "106_neural_cp_examples_meta.pkl", "rb"))
    curves = pickle.load(open(LTP / "artifacts" / "018_curves_sigma1.pkl", "rb"))
    col = {"R+": "#00B400", "R-": "#C800C8"}
    rows = []
    for w in WIN_USE:
        sids = list(dict.fromkeys(df.session_id))
        fig, axes = plt.subplots(2, len(sids), figsize=(4.2 * len(sids), 7.2), constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.6]))
        for c, sid in enumerate(sids):
            m = meta[sid]
            rg = m["reward_group"]
            sign = 1 if rg == "R+" else -1
            g = df[(df.session_id == sid) & (df.window == w)]
            prof = {p: g[g.profile == p].set_index("pct").delta.reindex(PCTS) for p in ("all", "odd", "even")}
            if EDGE_TRIALS:
                cut = PCTS / 100 * (m["n_trials"] - 1)
                edge = (cut < EDGE_TRIALS) | (cut > m["n_trials"] - 1 - EDGE_TRIALS)
                prof = {p_: v_.where(~edge) for p_, v_ in prof.items()}
            ks = {}
            for p in ("all", "odd", "even"):
                s = prof[p]
                ks[p] = int((sign * s).idxmax()) if s.notna().any() else np.nan
                ks[p + "_abs"] = int(s.abs().idxmax()) if s.notna().any() else np.nan
            cv = {}
            for a_, b_ in (("odd", "even"), ("even", "odd")):
                k = ks[a_]
                vb = prof[b_]
                val = vb.get(k, np.nan)
                cv[f"{a_}->{b_}"] = (val, float(np.nanmean(sign * vb < sign * val)) if not np.isnan(val) else np.nan)
            rows.append(dict(session_id=sid, window=w, reward_group=rg, group=m["group"], lt_pct=m["lt_pct"],
                             k_all=ks["all"], k_odd=ks["odd"], k_even=ks["even"], k_abs_all=ks["all_abs"],
                             k_odd_even_diff=abs(ks["odd"] - ks["even"]) if not np.isnan(ks["odd"] + ks["even"]) else np.nan,
                             delta_all_at_k=prof["all"].get(ks["all"], np.nan),
                             cv_odd_to_even=cv["odd->even"][0], cv_odd_to_even_pct=cv["odd->even"][1],
                             cv_even_to_odd=cv["even->odd"][0], cv_even_to_odd_pct=cv["even->odd"][1],
                             delta_all_at_lt=prof["all"].get(int(round(m["lt_pct"])), np.nan) if not np.isnan(m["lt_pct"]) else np.nan,
                             n_trials=m["n_trials"], n_hits=m["n_hits"], n_units=m["n_units"]))
            # top: behaviour on the percentile axis
            ax = axes[0, c]
            cu = curves[sid]["sigma1"]
            xb = np.linspace(0, 100, len(cu["p_mean"]))
            ax.fill_between(xb, cu["p_low80"], cu["p_high80"], color=col[rg], alpha=0.2, lw=0)
            ax.plot(xb, cu["p_mean"], color=col[rg], lw=1.2, label="whisker (sigma=1)")
            ax.plot(xb, cu["fa_time"], color="#555555", lw=1, ls="--", label="FA")
            if not np.isnan(m["lt_pct"]):
                ax.axvline(m["lt_pct"], color="#1f77b4", lw=2, label=f"behavioural LT ({m['lt_pct']:.0f}%)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{sid[:5]} {rg} {m['group']}\n{m['n_trials']} trials, {m['n_hits']} hits, {m['n_units']} units", fontsize=8.5)
            ax.legend(fontsize=6.5, frameon=False, loc="upper right")
            # bottom: profiles
            ax = axes[1, c]
            for p, cc, lw in (("all", "k", 2), ("odd", "#1f77b4", 1.1), ("even", "#ff7f0e", 1.1)):
                ax.plot(PCTS, prof[p].values, color=cc, lw=lw, label=f"{p} trials (k*={ks[p]})")
                if not np.isnan(ks[p]):
                    ax.plot(ks[p], prof[p][ks[p]], marker="v" if sign < 0 else "^", color=cc, ms=8)
            if not np.isnan(m["lt_pct"]):
                ax.axvline(m["lt_pct"], color="#1f77b4", lw=2, alpha=0.6)
            ax.axhline(0, color="#888888", ls=":")
            r = rows[-1]
            ax.set_title(f"change post - pre ({'max' if sign > 0 else 'min'} expected)\nCV odd->even: {r['cv_odd_to_even']:+.2f} "
                         f"(pct {r['cv_odd_to_even_pct']:.2f}); even->odd: {r['cv_even_to_odd']:+.2f} (pct {r['cv_even_to_odd_pct']:.2f})"
                         f"\n|k*odd - k*even| = {r['k_odd_even_diff']:.0f}%".replace("nan", "-"), fontsize=7.5)
            ax.set_xlabel("split position (% of session trials)")
            ax.legend(fontsize=6.5, frameon=False)
            for a_ in axes[:, c]:
                a_.set_xlim(0, 100)
        axes[1, 0].set_ylabel("matched balanced accuracy change (post - pre)")
        fig.suptitle(f"Agnostic neural change point, {w}: pre/post change at every 1% split (black = all trials, blue = odd, "
                     f"orange = even); triangles = k*; blue vertical = behavioural learning trial. No filtering besides class "
                     f"balance{f'; splits within {EDGE_TRIALS} trials of the session edges excluded' if EDGE_TRIALS else ''}.", fontsize=10.5)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"106_neural_cp_examples_{w}{SUF}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / f"106_neural_cp_examples_summary{SUF}.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(2).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        summarize_and_plot()
    else:
        build()
        summarize_and_plot()
