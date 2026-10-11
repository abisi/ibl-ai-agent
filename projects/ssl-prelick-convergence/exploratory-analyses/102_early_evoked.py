"""102 -- Early stimulus-evoked responses in the active task, per area, across learning (user 2026-10-11: "focus on early evoked
responses, hit - miss differences and auditory hit / whisker hit differences; no passive"). Exploratory.
Trials: 051 active selection; whisker hits (WH), whisker misses (WM), auditory hits (AH); hit trials with RT < RT_MIN (100 ms)
excluded, so no lick falls inside the early window. Units: tested units of 081 (stable run), by area group (>= 5 per session).
Rates in 5-ms bins from -100 to +200 ms after stimulus onset (10-ms boxcar); on hit trials, bins after the trial's first lick are masked
(NaN) and each unit's PSTH is the mean over the unmasked trials, so later bins are not lick-driven. z per unit: (rate - mean
pre-stimulus baseline) / across-trial SD of that baseline (baseline [-1, -0.015] s before trial start, SD floor 0.5 Hz).
Early window: EARLY = 5-50 ms. Quantities per session x area: early WH, WM, AH (z); WH - WM (hit - miss); WH - AH (whisker vs
auditory hit); whole brain = unit-weighted mean over areas.
Figure:
  a  heatmaps area x time (z, lick-masked), WH / WM / AH (columns) for R+ day 0, R+ expert, R- day 0, R- expert (rows)
  b  forest plot: expert - day 0 of the early-window values, R+ and R- (mean +- combined s.e.m.), columns early WH, WH - WM, WH - AH;
     filled: within-cohort MWU p < 0.05; * cohort x stage (mouse permutation) BH q < 0.05 within column, (*) p < 0.05
Learners rule (--pop all to keep all). Output: across_days/<ref>/early_evoked/{psth.parquet, early_evoked_<pop>.{png,pdf}, stats_<pop>.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../102_early_evoked.py [--n-proc 25] [--replot] [--pop learners]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m81 = importlib.import_module("081_single_unit_classes")
OUT = m51.OUTROOT / "early_evoked"
BIN, LO, HI, BASE, SD_MIN, MIN_UNITS, RT_MIN = 0.005, -0.1, 0.2, (-1.0, -0.015), 0.5, 5, 0.100
EARLY = (5, 50)


def session(args):
    sid, units = args
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        U, _ = m51.ru.process_nwb_tables(nwb)
        t, _ = m51.active_trials(nwb.trials.to_dataframe(), sid)
    rt = (t.lick_time - t.response_window_start_time).to_numpy(); lick = (t.lick_flag == 1).to_numpy() & np.isfinite(rt)
    isw, isa = (t.trial_type == "whisker_trial").to_numpy(), (t.trial_type == "auditory_trial").to_numpy()
    st = t.start_time.to_numpy()
    cls = {"WH": isw & lick & (rt >= RT_MIN), "WM": isw & ~lick, "AH": isa & lick & (rt >= RT_MIN)}
    if min(c.sum() for c in cls.values()) < 3:
        return []
    edges = np.arange(LO, HI + BIN / 2, BIN); nb = len(edges) - 1
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str)).merge(units, on=["electrode_group", "cluster_id"], how="inner")
    rows = []
    for ag, g in U.groupby("area_group"):
        if len(g) < MIN_UNITS:
            continue
        acc = {c: [] for c in cls}
        for r in g.itertuples():
            s = np.sort(np.asarray(r.spike_times))
            for c, m in cls.items():
                bt = (np.searchsorted(s, st[m] + BASE[1]) - np.searchsorted(s, st[m] + BASE[0])) / (BASE[1] - BASE[0])
                mu, sd = bt.mean(), max(bt.std(), SD_MIN)
                H = np.array([np.histogram(s[np.searchsorted(s, e + LO):np.searchsorted(s, e + HI)] - e, edges)[0] for e in st[m]], float) / BIN
                if c in ("WH", "AH"):                                                # mask bins after the first lick
                    H[edges[1:][None, :] > rt[m][:, None]] = np.nan
                ps = np.nanmean(H, 0)
                ps = np.convolve(np.nan_to_num(ps), np.ones(2) / 2, "same")          # 10-ms boxcar
                acc[c].append((ps - mu) / sd)
        for c, v in acc.items():
            rows.append(dict(session_id=sid, area_group=ag, cls=c, n_units=len(g), n_trials=int(cls[c].sum()), psth_z=np.mean(v, 0).tolist()))
    return rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    jobs = [(sid, g[["electrode_group", "cluster_id", "area_group"]]) for sid, g in U0.groupby("session_id")]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    D = pd.DataFrame(res).merge(U0.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]], on="session_id", how="left")
    D.to_parquet(OUT / "psth.parquet", index=False); print("computed", len(D), D.session_id.nunique(), flush=True)


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2")
    m62, m61 = m88.m62, m88.m61
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup(); rng = np.random.default_rng(0); gcol = au.get_custom_area_groups_colors(); COH = m88.COH
    D = pd.read_parquet(OUT / "psth.parquet"); D = m61.learner_filter(D) if pop == "learners" else D
    D["psth_z"] = D.psth_z.apply(np.asarray)
    tc = (np.arange(LO, HI + BIN / 2, BIN)[:-1] + BIN / 2) * 1e3
    win = (tc >= EARLY[0]) & (tc < EARLY[1])
    W = D.pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "area_group", "n_units"], columns="cls", values="psth_z", aggfunc="first").reset_index().dropna(subset=["WH", "WM", "AH"])
    for c in ("WH", "WM", "AH"):
        W[f"e_{c}"] = [v[win].mean() for v in W[c]]
    W["e_HM"] = W.e_WH - W.e_WM; W["e_WA"] = W.e_WH - W.e_AH
    WB = W.groupby(["session_id", "mouse_id", "cohort", "stage"]).apply(
        lambda g: pd.Series({k: np.average(g[k], weights=g.n_units) for k in ("e_WH", "e_WM", "e_AH", "e_HM", "e_WA")})).reset_index().assign(area_group="Whole brain")
    V = pd.concat([WB, W.drop(columns=["WH", "WM", "AH"])], ignore_index=True)
    GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
    n = W.groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    areas = [g for g in au.get_area_group_custom_order() if g in n.index and all(n.loc[g].get(k, 0) >= 3 for k in GROUPS)]
    rows_f = ["Whole brain"] + areas
    COLS = (("e_WH", "early WH (z)"), ("e_HM", "WH − WM (z)"), ("e_WA", "WH − AH (z)"))
    ST = []
    for col, _ in COLS:
        for ag in rows_f:
            d = V[V.area_group == ag]; r = dict(area=ag, measure=col)
            for co in ("R+", "R-"):
                L_, E_ = (d[(d.cohort == co) & (d.stage == s_)][col] for s_ in ("learning", "expert"))
                r.update({f"D0_{co}": L_.mean(), f"E_{co}": E_.mean(), f"chg_{co}": E_.mean() - L_.mean(),
                          f"sem_{co}": np.sqrt(L_.sem() ** 2 + E_.sem() ** 2), f"p_{co}": stats.mannwhitneyu(L_, E_).pvalue, f"n_{co}": f"{len(L_)}/{len(E_)}"})
            r["p_D0_cohort"] = stats.mannwhitneyu(*(d[(d.cohort == co) & (d.stage == "learning")][col] for co in ("R+", "R-"))).pvalue
            r["p_int"] = m62.group_stats(d, col, f"{ag} {col}", rng, "102")["interaction"][1]
            ST.append(r)
    ST = pd.DataFrame(ST)
    for col, _ in COLS:
        m = ST.measure == col; ST.loc[m, "q_int"] = bh(ST.loc[m, "p_int"])
    ST.to_csv(OUT / f"stats_{pop}.csv", index=False)
    # ---- figure
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28, left=0.07, right=0.99, top=0.9, bottom=0.08)
    gh = gs[0, 0].subgridspec(4, 3, hspace=0.25, wspace=0.12)
    mask = tc <= 150
    v = np.nanpercentile(np.abs(np.concatenate([np.concatenate(W[c].to_numpy()) for c in ("WH", "WM", "AH")])), 98)
    for i, g in enumerate(GROUPS):
        for j, c in enumerate(("WH", "WM", "AH")):
            ax = fig.add_subplot(gh[i, j]); d = W[(W.cohort == g[0]) & (W.stage == g[1])]
            Z = np.vstack([np.stack(d[d.area_group == ag][c].to_numpy()).mean(0)[mask] for ag in areas])
            im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=(tc[mask][0], tc[mask][-1], len(areas) - 0.5, -0.5), interpolation="nearest")
            ax.axvline(0, color="k", lw=0.4, ls=":"); ax.axvspan(*EARLY, ymin=0.97, ymax=1.0, color="k", lw=0)
            ax.set_yticks(range(len(areas))); ax.set_yticklabels([m88.m83.m62_short(a) for a in areas] if j == 0 else [], fontsize=3.9)
            ax.tick_params(labelsize=4, length=1.5)
            if i == 0:
                ax.set_title({"WH": "whisker hit", "WM": "whisker miss", "AH": "auditory hit"}[c], fontsize=5.2)
            if j == 0:
                ax.set_ylabel({("R+", "learning"): "R+ day 0", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− day 0", ("R-", "expert"): "R− expert"}[g],
                              fontsize=5, color=COH[g[0]])
            if i < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("from stimulus (ms)", fontsize=4.6)
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01, location="left", anchor=(0, 0.5)) if False else None
    fig.text(0.07, 0.94, f"a  z-scored responses, lick-masked (bar: early window {EARLY[0]}–{EARLY[1]} ms; colour ±{v:.1f} z)", fontsize=5.6)
    gf = gs[0, 1].subgridspec(1, 3, wspace=0.12); y = np.arange(len(rows_f))[::-1]
    for j, (col, lab) in enumerate(COLS):
        ax = fig.add_subplot(gf[j]); s = ST[ST.measure == col].set_index("area")
        for i, ag in enumerate(rows_f):
            r = s.loc[ag]
            for k, co in enumerate(("R+", "R-")):
                ax.errorbar(r[f"chg_{co}"], y[i] + 0.15 - 0.3 * k, xerr=r[f"sem_{co}"], fmt="o", ms=2.6, color=COH[co], lw=0.8, capsize=0,
                            mfc=COH[co] if r[f"p_{co}"] < 0.05 else "white")
            mk = "*" if r.q_int < 0.05 else ("(*)" if r.p_int < 0.05 else "")
            if mk:
                ax.text(0.99, y[i], mk, transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", ha="right", weight="bold")
        ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(-0.6, len(rows_f) - 0.4); ax.axhline(len(rows_f) - 1.5, color="0.85", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([m88.m83.m62_short(g) for g in rows_f] if j == 0 else [], fontsize=4.6)
        if j == 0:
            for tl, g in zip(ax.get_yticklabels(), rows_f):
                tl.set_color(gcol.get(g, "k")); tl.set_weight("bold" if g == "Whole brain" else "normal")
        ax.set_title(f"{lab}\n{EARLY[0]}–{EARLY[1]} ms", fontsize=5); ax.set_xlabel("expert − day 0", fontsize=4.6); ax.tick_params(labelsize=4.3)
    fig.text(0.58, 0.94, "b  Change across learning; filled: within-cohort p < 0.05; * c×s q < 0.05, (*) p < 0.05", fontsize=5.4)
    fig.suptitle(f"Early stimulus-evoked responses, active task ({pop}; hits with RT ≥ {int(RT_MIN * 1e3)} ms)", x=0.02, y=0.995, ha="left", fontsize=6.5, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"early_evoked_{pop}.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    print(ST[["area", "measure", "D0_R+", "E_R+", "p_R+", "D0_R-", "E_R-", "p_R-", "n_R+", "n_R-", "p_D0_cohort", "p_int", "q_int"]].round(3).to_string(index=False))


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
