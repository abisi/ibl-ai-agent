"""134 -- Measures that should separate the cohorts (user 2026-10-02: "Keeping those safeguards, implement 1 and 2 and 3"):
  1  whisker-SPECIFIC response change, auditory as the within-mouse control (both cohorts get identical auditory trials);
  2  time-resolved: sliding windows 5-200 ms after stimulus onset (later recurrent / frontal / striatal stages);
  3  alignment of the whisker coding direction with the lick direction in the active epoch.
Safeguards (kept from 133):
  units      quality_label 'good' only AND firing >= 0.5 Hz in the time span of every epoch (same units in pre, active, post);
  baseline   rates (Hz), each unit's mean -55..-20 ms rate within its epoch subtracted (epoch-specific); z-scored per unit and
             window over all trials of the session (pooled epochs);
  control    auditory responses, treated identically in both cohorts, subtracted from whisker changes;
  licks      active trials whose corrected first lick (start_time + lick_time - response_window_start_time) falls before the END
             of a window are excluded from that window (no lick-motor contamination); passive trials all kept.
Sessions / trials: as 132/133 (learning stage, passive before AND after the active block; active = prep_modality_trials from
the first whisker trial on, perf != 6, A1-trimmed; skills/ssl-trial-exclusion).
Windows: 30 ms, starts 5, 20, ..., 170 ms (dead zone -10..+5 ms precedes all).
Measures per session x area x window:
  gain change  for each unit, its preferred sign s_u per modality from a random HALF of the passive-pre trials of that
               modality; change in the OTHER half's mean response: g_u = s_u * (R_epoch - R_pre,heldout) (> 0 = response grew in
               its preferred direction); averaged over units and N_SPLIT splits; per modality (W, A) and per comparison:
               state = active vs pre, plast = post vs pre; SPECIFIC = g_W - g_A (whisker change beyond the auditory change);
  lick align   active epoch: whisker axis = mean(whisker) - mean(auditory); lick axis = mean(licked) - mean(unlicked) within
               each modality (licks AFTER the window; >= MIN_CLASS trials per class), averaged over the modalities available;
               similarity = cos of split halves normalised by split-half reliabilities (as 133); also the raw cos.
Stats per window (mouse = unit; uncorrected): R+ vs R- (Mann-Whitney AND Welch); per cohort vs 0 (Wilcoxon AND t).
Outputs: 134_whisker_specific_change.parquet, 134_stats.csv; figures 134_whole_brain.{pdf,png,svg} (time courses per cohort),
134_area_groups_<measure>.{pdf,png,svg} (small multiples).
Run (haas, repo root): python .../134_whisker_specific_change.py   |   python ... plot
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
WINDOWS = [(s / 1000, (s + 30) / 1000) for s in range(5, 171, 15)]
BASE, DZ = (-0.055, -0.020), (-0.010, 0.005)
MIN_UNITS, MIN_EPOCH_RATE, MIN_CLASS, N_SPLIT = 5, 0.5, 5, 10
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
# 2026-10-05 SSL_UNITS=tracked: the shared Part III tracked stable units (tracked_units.py / 137b), taken as is (outputs *_tracked);
# default: the original quality_label good + >= 0.5 Hz per epoch selection (no drift check)
UNITS = os.environ.get("SSL_UNITS", "good")
TAG = "_tracked" if UNITS == "tracked" else ""
TRACKED = importlib.import_module("tracked_units").load("stable") if UNITS == "tracked" else {}
OUT_PATH = OUT / f"134_whisker_specific_change{TAG}.parquet"
MEASURES = [("spec_state", "whisker − auditory gain change\nactive vs pre (z)"),
            ("spec_plast", "whisker − auditory gain change\npost vs pre (z)"),
            ("gW_state", "whisker gain change\nactive vs pre (z)"), ("gA_state", "auditory gain change\nactive vs pre (z)"),
            ("lick_cosnorm", "whisker axis · lick axis\n(active, norm. cos)")]


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


def tracked_good(sid, units, labels, spikes, trs):
    lab = labels[(labels.session_id == sid) & labels.cluster_id.isin(units)]
    out = []
    for cid in lab[lab.quality_label == "good"].cluster_id.to_numpy():
        sp = spikes.get(cid, np.array([]))
        ok = True
        for e in EP:
            a0, a1 = trs[e].start_time.min() - 1.0, trs[e].start_time.max() + 1.0
            if (np.searchsorted(sp, a1) - np.searchsorted(sp, a0)) / max(a1 - a0, 1e-6) < MIN_EPOCH_RATE:
                ok = False
                break
        if ok:
            out.append(cid)
    return np.asarray(out)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else np.nan


def gain_changes(Z, isw, rng):
    """Z: dict epoch -> (trials x units) z-scored responses, isw: dict epoch -> whisker mask (valid trials only)."""
    res = {k: [] for k in ("gW_state", "gA_state", "gW_plast", "gA_plast")}
    for _ in range(N_SPLIT):
        for mod, key in ((True, "W"), (False, "A")):
            pre_idx = rng.permutation(np.where(isw["passive_pre"] == mod)[0])
            if len(pre_idx) < 4:
                continue
            h1, h2 = pre_idx[: len(pre_idx) // 2], pre_idx[len(pre_idx) // 2:]
            s = np.sign(Z["passive_pre"][h1].mean(0))
            ref = Z["passive_pre"][h2].mean(0)
            for ep, lab in (("active", "state"), ("passive_post", "plast")):
                m = isw[ep] == mod
                if m.sum() < 3:
                    continue
                res[f"g{key}_{lab}"].append(float(np.mean(s * (Z[ep][m].mean(0) - ref))))
    out = {k: float(np.mean(v)) if v else np.nan for k, v in res.items()}
    out["spec_state"] = out["gW_state"] - out["gA_state"]
    out["spec_plast"] = out["gW_plast"] - out["gA_plast"]
    return out


def lick_alignment(Za, isw, lick, rng):
    """Active epoch: cos between the whisker axis (W - A) and the lick axis (licked - unlicked within modality)."""
    mods = [m for m in (True, False) if min(((isw == m) & lick).sum(), ((isw == m) & ~lick).sum()) >= MIN_CLASS]
    if not mods or min(isw.sum(), (~isw).sum()) < 2 * MIN_CLASS:
        return dict(lick_cos=np.nan, lick_cosnorm=np.nan, lick_mods="none")
    cab, cww, cll = [], [], []
    for _ in range(N_SPLIT * 2):
        halves = {}
        for grp in [(m, l) for m in (True, False) for l in (True, False)]:
            i = rng.permutation(np.where((isw == grp[0]) & (lick == grp[1]))[0])
            halves[grp] = (i[: len(i) // 2], i[len(i) // 2:])
        W, L = [], []
        for h in (0, 1):
            wi = np.concatenate([halves[(True, l)][h] for l in (True, False)])
            ai = np.concatenate([halves[(False, l)][h] for l in (True, False)])
            W.append(Za[wi].mean(0) - Za[ai].mean(0))
            L.append(np.mean([Za[halves[(m, True)][h]].mean(0) - Za[halves[(m, False)][h]].mean(0) for m in mods], 0))
        cab.append(0.5 * (cos(W[0], L[1]) + cos(W[1], L[0])))
        cww.append(cos(W[0], W[1]))
        cll.append(cos(L[0], L[1]))
    c, rw, rl = np.nanmean(cab), max(np.nanmean(cww), 0.05), max(np.nanmean(cll), 0.05)
    return dict(lick_cos=float(c), lick_cosnorm=float(np.clip(c / np.sqrt(rw * rl), -1.5, 1.5)),
                lick_rel_w=float(np.nanmean(cww)), lick_rel_l=float(np.nanmean(cll)),
                lick_mods="+".join("W" if m else "A" for m in mods))


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
    act = trs["active"]
    rt = (act.lick_time - act.response_window_start_time).to_numpy()          # corrected first lick, s from stimulus onset
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    spikes = T.load_session_unit_spikes(root, sid)
    isw_all = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in EP}
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    sl = labels[labels.session_id == sid]
    areas = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(sl["area_group"].dropna().unique())]
    rows = []
    for area_col, area in areas:
        if UNITS == "tracked":
            units = np.intersect1d(T.area_units(sid, area_col, area, labels), TRACKED.get(sid, np.array([], dtype=np.int64)))
        else:
            units = tracked_good(sid, T.area_units(sid, area_col, area, labels), labels, spikes, trs)
        if len(units) < MIN_UNITS:
            continue
        mats = {}
        for e in EP:
            t0 = trs[e].start_time.to_numpy()
            ms = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), WINDOWS + [BASE], dead_zone=DZ)
            b = np.nanmean(ms[-1], 0, keepdims=True)                         # epoch-specific mean baseline per unit
            mats[e] = [m - b for m in ms[:-1]]
        rng = np.random.default_rng(zlib.crc32(f"{sid}|{area}|134".encode()))
        for wi, (w0, w1) in enumerate(WINDOWS):
            valid = {e: np.ones(len(isw_all[e]), bool) for e in EP}
            valid["active"] = ~licked | (rt > w1)                             # no lick before the window ends
            X = {e: mats[e][wi][valid[e]] for e in EP}
            isw = {e: isw_all[e][valid[e]] for e in EP}
            if any(min(isw[e].sum(), (~isw[e]).sum()) < 4 for e in EP) or any(np.isnan(X[e]).any() for e in EP):
                continue
            P = np.vstack([X[e] for e in EP])
            mu, sd = P.mean(0), P.std(0)
            keep = sd > 0
            if keep.sum() < MIN_UNITS:
                continue
            Z = {e: (X[e][:, keep] - mu[keep]) / sd[keep] for e in EP}
            r = dict(base, area_col=area_col, area=area, n_units=int(keep.sum()), win_start=w0 * 1000, win_end=w1 * 1000,
                     n_active_valid=int(valid["active"].sum()), **gain_changes(Z, isw, rng),
                     **lick_alignment(Z["active"], isw["active"], licked[valid["active"]], rng), skipped_reason=None)
            rows.append(r)
    return rows or [dict(base, skipped_reason="no area with enough tracked good units")]


def run():
    os.chdir(OUT.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples() if r.session_id not in done]
    print(f"[134] {len(args)} sessions, {len(WINDOWS)} windows", flush=True)
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
            print(f"[134] [{i}/{len(args)}] {a[0]} -- {time.time() - t0:.0f}s", flush=True)
    plot()


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def stats_table(d):
    from scipy import stats
    rows = []
    for (area, ws), g in d.groupby(["area", "win_start"]):
        for m, _ in MEASURES:
            a, b = g[g.reward_group == "R+"][m].dropna(), g[g.reward_group == "R-"][m].dropna()
            r = dict(area=area, win_start=ws, measure=m, n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=a.mean(), mean_Rminus=b.mean(),
                     sem_Rplus=a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan,
                     sem_Rminus=b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else np.nan)
            for lab, x in (("Rplus", a), ("Rminus", b)):
                if len(x) >= 5:
                    r[f"p_wilcoxon_{lab}"] = stats.wilcoxon(x).pvalue
                    r[f"p_t_{lab}"] = stats.ttest_1samp(x, 0).pvalue
            if min(len(a), len(b)) >= 3:
                r["p_mw"] = stats.mannwhitneyu(a, b).pvalue
                r["p_welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
            rows.append(r)
    return pd.DataFrame(rows)


def time_panel(ax, s, ylabel, small=False):
    for rg, lab in (("R+", "Rplus"), ("R-", "Rminus")):
        x = s.win_start + 15
        ax.fill_between(x, s[f"mean_{lab}"] - s[f"sem_{lab}"], s[f"mean_{lab}"] + s[f"sem_{lab}"], color=COL[rg], alpha=0.2, lw=0)
        ax.plot(x, s[f"mean_{lab}"], color=COL[rg], lw=1.4, marker="o", ms=2)
        pw = s.get(f"p_wilcoxon_{lab}", pd.Series(np.nan, index=s.index))
        pt = s.get(f"p_t_{lab}", pd.Series(np.nan, index=s.index))
        sig0 = (pw < 0.05) & (pt < 0.05)
        ax.scatter(x[sig0], s[f"mean_{lab}"][sig0], s=14 if not small else 8, facecolor="none", edgecolor=COL[rg], lw=0.8, zorder=4)
    both = (s.p_mw < 0.05) & (s.p_welch < 0.05)
    one = ((s.p_mw < 0.05) | (s.p_welch < 0.05)) & ~both
    for msk, c in ((both, "k"), (one, "0.65")):
        for xx in (s.win_start + 15)[msk]:
            ax.plot([xx - 7.5, xx + 7.5], [1.03, 1.03], color=c, lw=2.5, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("window centre (ms after stimulus)", fontsize=5.5 if small else 6.5)
    ax.set_ylabel(ylabel, fontsize=5.5 if small else 6.5)
    ax.tick_params(labelsize=5 if small else 6)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_parquet(OUT_PATH)
    d = d[d.skipped_reason.isna()]
    S = stats_table(d)
    S.to_csv(OUT / f"134_stats{TAG}.csv", index=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 6.5})
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(11.7, 2.6))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.18, wspace=0.45)
    for ax, (m, lab) in zip(axes, MEASURES):
        time_panel(ax, S[(S.area == "All units") & (S.measure == m)].sort_values("win_start"), lab)
    n = d[d.area == "All units"].drop_duplicates("session_id").groupby("reward_group").size().to_dict()
    fig.suptitle(f"Whole brain, good units tracked across pre/active/post (R+ n={n.get('R+', 0)}, R− n={n.get('R-', 0)}): mean ± "
                 "SEM per cohort; open circle = differs from 0 (Wilcoxon AND t); bar = R+ vs R− (black: MW AND Welch p < .05; "
                 "grey: one of them). Active trials with a lick before the window end excluded. Uncorrected.", fontsize=7)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"134_whole_brain{TAG}.{ext}", dpi=250)
    plt.close(fig)
    areas = sorted(a for a in S.area.unique() if a != "All units")
    ncol = 5
    nrow = int(np.ceil(len(areas) / ncol))
    for m, lab in MEASURES:
        fig, axes = plt.subplots(nrow, ncol, figsize=(11.7, 2.1 * nrow + 0.5), squeeze=False)
        fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.5 / (2.1 * nrow + 0.5), bottom=0.05, wspace=0.4, hspace=0.75)
        for ax, ar in zip(axes.flat, areas):
            s = S[(S.area == ar) & (S.measure == m)].sort_values("win_start")
            time_panel(ax, s, lab if ar == areas[0] else "", small=True)
            nn = f"R+ {int(s.n_Rplus.max()) if len(s) else 0}, R− {int(s.n_Rminus.max()) if len(s) else 0}"
            ax.set_title(f"{ar} ({nn})", fontsize=6, fontweight="bold")
        for ax in axes.flat[len(areas):]:
            ax.set_axis_off()
        fig.suptitle(f"{lab.replace(chr(10), ' ')} per area group (tracked good units; same conventions as the whole-brain "
                     "figure)", fontsize=7)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / "figures" / f"134_area_groups_{m}{TAG}.{ext}", dpi=220)
        plt.close(fig)
    pd.set_option("display.width", 250)
    w = S[(S.area == "All units")]
    print(w[["measure", "win_start", "n_Rplus", "n_Rminus", "mean_Rplus", "mean_Rminus", "p_mw", "p_welch"]].round(3).to_string(index=False))


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else run()
