"""120 -- Projections on the reward-lick CD around jaw-opening onset (user 2026-10-13: "aligning to jaw onset but projection around it").
CD as within-day 026 / 087 but from the jaw-aligned events of 118 (PRELICK_JAW_WIN=pre): AH vs SL rates in [-100, 0) ms before jaw onset,
units = 026 unit set (stable, mean raw pre-jaw rate >= MIN_FR), whole brain. Two-fold cross-validation over AH + SL (stratified, 10 random
splits): units z-scored on the training events, u = unit(mean AH - mean SL); held-out AH / SL and all WH binned rates around jaw onset
(118 jbins: 50-ms windows, centres -300 .. +200 ms, minus the event's baseline) z-scored with the training mean / SD and projected on u;
per neuron (/ sqrt(N)); session mean per class and bin, averaged over splits. d_WS(t) = WH - SL.
Population: learners rule; session = unit. Output: across_days/sl_stable_jaw/cd_timecourse/{timecourse.parquet, cd_jaw_timecourse.{png,pdf,svg}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PRELICK_ALIGN=jaw python .../120_cd_jaw_timecourse.py [--n-proc 30]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m51 = importlib.import_module("051_roc_prelick")
m002 = importlib.import_module("002_trial_slopes")
OUT = m51.OUTROOT / "cd_timecourse"
N_SPLIT = 10


def session(args):
    sid, f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    if "jbins" not in z:
        return []
    X, raw, lab, JB, c = z["rates"].astype(float), z["raw"].astype(float), z["cls"], z["jbins"].astype(float), z["jbins_c"]
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))).merge(W_s, how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    if ok.sum() < m002.MIN_UNITS or (lab == "WH").sum() < m002.MIN_WH or min((lab == k).sum() for k in ("AH", "FA")) < m002.MIN_EV:
        return []
    X, JB = X[ok], JB[ok]; N = ok.sum()
    ah, sl, wh = (np.where(lab == k)[0] for k in ("AH", "FA", "WH"))
    rng = np.random.default_rng(seed); acc = {k: [] for k in ("AH", "SL", "WH")}
    for _ in range(N_SPLIT):
        pa, ps = rng.permutation(ah), rng.permutation(sl)
        tr = np.r_[pa[: len(pa) // 2], ps[: len(ps) // 2]]; te = {"AH": pa[len(pa) // 2:], "SL": ps[len(ps) // 2:], "WH": wh}
        mu, sd = X[:, tr].mean(1), X[:, tr].std(1); sd[sd == 0] = 1
        Z = (X - mu[:, None]) / sd[:, None]
        u = Z[:, pa[: len(pa) // 2]].mean(1) - Z[:, ps[: len(ps) // 2]].mean(1); u /= max(np.linalg.norm(u), 1e-12)
        for k, idx in te.items():
            Zb = (JB[:, idx, :] - mu[:, None, None]) / sd[:, None, None]
            acc[k].append(np.einsum("u,uet->t", u, Zb) / len(idx) / np.sqrt(N))
    return [dict(meta, cls=k, t=float(cc * 1e3), value=float(v)) for k in acc for cc, v in zip(c, np.mean(acc[k], 0))]


def figure(T):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH, CL, GROUPS = m88.COH, m88.CL, m88.GROUPS
    plt = m62.setup(); T = m61.learner_filter(T)
    with plt.rc_context({"font.size": 5.2, "axes.labelsize": 5.0, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6, "svg.fonttype": "none"}):
        fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.7), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 1, 1, 1.1]))
        lim = [np.inf, -np.inf]
        for ax, (coh, st) in zip(axs[:4], GROUPS):
            d = T[(T.cohort == coh) & (T.stage == st)]
            for k, key in (("SL", "FA"), ("AH", "AH"), ("WH", "WH")):
                piv = d[d.cls == k].pivot_table(index="session_id", columns="t", values="value"); m, se = piv.mean(0), piv.sem(0)
                ax.fill_between(piv.columns, m - se, m + se, color=CL[key], alpha=0.25, lw=0); ax.plot(piv.columns, m, color=CL[key], lw=0.9, label=k)
                lim = [min(lim[0], (m - se).min()), max(lim[1], (m + se).max())]
            ax.axvspan(-100, 0, color="#FDD49E", alpha=0.5, lw=0, zorder=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
            ax.set_title(f"{coh.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'} ({d.session_id.nunique()})", color=COH[coh], fontsize=5.2)
            ax.set_xlabel("time from jaw onset (ms)"); ax.set_box_aspect(1)
        for ax in axs[:4]:
            ax.set_ylim(*lim)
        axs[0].set_ylabel("projection on CD (per neuron)"); axs[0].legend(frameon=False, fontsize=4.2, loc="upper left")
        ax = axs[4]
        for coh, st in GROUPS:
            d = T[(T.cohort == coh) & (T.stage == st)].pivot_table(index=["session_id", "t"], columns="cls", values="value").reset_index()
            d["dWS"] = d.WH - d.SL; piv = d.pivot_table(index="session_id", columns="t", values="dWS"); m, se = piv.mean(0), piv.sem(0)
            ax.fill_between(piv.columns, m - se, m + se, color=COH[coh], alpha=0.15, lw=0)
            ax.plot(piv.columns, m, color=COH[coh], lw=0.9, ls="-" if st == "expert" else (0, (2.5, 1.5)), label=f"{coh.replace('-', '−')} {'D0' if st == 'learning' else 'E'}")
        ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2))); ax.axvspan(-100, 0, color="#FDD49E", alpha=0.5, lw=0, zorder=0)
        ax.set_xlabel("time from jaw onset (ms)"); ax.set_ylabel("d_WS = WH − SL"); ax.legend(frameon=False, fontsize=4.0); ax.set_box_aspect(1)
        fig.suptitle("Reward-lick CD (defined 100 ms before jaw-opening onset), projections around jaw onset (held out; learners)",
                     x=0.01, y=1.0, ha="left", fontsize=6.2, weight="bold")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"cd_jaw_timecourse.{ext}", dpi=300, bbox_inches="tight")


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet"); W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0]),
                         Ws[["electrode_group", "cluster_id", "quality_label"]], i))
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        T = pd.DataFrame([x for r in pool.imap_unordered(session, jobs) for x in r])
    T.to_parquet(OUT / "timecourse.parquet", index=False)
    figure(T); print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    main(ap.parse_args())
