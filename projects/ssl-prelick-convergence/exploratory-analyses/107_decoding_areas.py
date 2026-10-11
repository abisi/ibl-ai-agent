"""107 -- Decoding per area group and stage-overlaid time courses (user 2026-10-11: "overlay the learning stages, a decoder per area;
per session or pseudopopulation?"). Exploratory.
Decoders are per session (one decoder per session x area, units of that area only, >= MIN_UNITS); no pseudopopulation. Same decoders,
classifier and label-shuffle chance as 106 (logistic regression, L2, C = 0.1, balanced class weights, units z-scored on the training
events, 5-fold CV balanced accuracy; transfer = fraction of test events decoded as the positive class; chance = mean over N_SHUF = 20
label shuffles), in one summary window each (to keep the cost per area manageable):
  AH vs SL -> WH        lick-aligned, mean of the 50-ms bins centred in [-100, 0) ms
  AH vs CR -> W         stimulus window [50, 150) ms after stimulus onset, truncated at the first lick (105 stimwin)
  WH vs SL -> AH        lick-aligned, as the first
Whole brain = all units of the session (same windows; for comparison with the areas).
Figures:
  timecourses_<pop>: whole-brain decoding over time from 106 (accuracy - chance, transfer - chance), one panel per cohort, day 0 dashed /
  expert solid, mean +- s.e.m. over sessions
  areas_<pop>: forest plot per area (rows) x decoder and measure (columns): expert - day 0 of accuracy - chance and transfer - chance,
  R+ and R-; filled: within-cohort MWU p < 0.05; * cohort x stage (mouse permutation) BH q < 0.05 within column, (*) p < 0.05
Output: across_days/<ref>/convergence_divergence/{areas.parquet, timecourses_<pop>.{png,pdf}, areas_<pop>.{png,pdf}, areas_stats_<pop>.csv}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../107_decoding_areas.py [--n-proc 25] [--replot] [--pop learners]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m106 = importlib.import_module("106_convergence_divergence")
m105, m51 = m106.m105, m106.m51
OUT = m106.OUT
MIN_UNITS = 10
DEC = [("AH vs SL -> WH", "lick"), ("AH vs CR -> W", "stim"), ("WH vs SL -> AH", "lick")]


def session(args):
    f, = args
    sid = f.stem; z = np.load(f, allow_pickle=True)
    seed = zlib.crc32(f"{sid}|107".encode()) % 100000
    ag = z["area_group"].astype(str)
    lab, X = z["stim_lab"], z["stimwin"].astype(float)
    okS = np.isfinite(X).all(0)
    llab, L, lc = z["lick_lab"], z["lickbins"].astype(float), z["lick_c"]
    Lw = L[:, :, (lc >= -0.1) & (lc < 0.0)].mean(2)
    a_, s_, w_ = (np.where(llab == k)[0] for k in ("AH", "SL", "WH"))
    iw, ia, ic = (np.where(okS & (lab == k))[0] for k in ("W", "AH", "CR"))
    groups = [("Whole brain", np.ones(len(ag), bool))] + [(g, ag == g) for g in sorted(set(ag)) if g not in ("nan", "None")]
    rows = []
    for g, m in groups:
        if m.sum() < MIN_UNITS:
            continue
        for name, align in DEC:
            if name == "AH vs SL -> WH":
                o = m106.transfer(Lw[m], a_, s_, w_, seed)
            elif name == "AH vs CR -> W":
                o = m106.transfer(X[m], ia, ic, iw, seed)
            else:
                o = m106.transfer(Lw[m], w_, s_, a_, seed)
            rows.append(dict(session_id=sid, area=g, n_units=int(m.sum()), decoder=name, **o))
    return rows


def compute(n_proc):
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, [(f,) for f in files]) for x in r]
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    D = pd.DataFrame(res).merge(meta, on="session_id")
    D.to_parquet(OUT / "areas.parquet", index=False); print("computed", len(D), D.session_id.nunique(), flush=True)


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def fig_timecourses(pop, m88, m62, flt):
    COH = m88.COH; plt = m62.setup()
    T = flt(pd.read_parquet(OUT / "timecourses.parquet")); T["acc_ex"] = T.acc - T.acc_null; T["frac_ex"] = T.frac - T.frac_null
    TESTS = [("lick", "WH as AH (train AH vs SL)", "AH vs SL decoder → whisker hits"), ("stim", "W as go (train AH vs CR)", "AH vs CR decoder → whisker trials"),
             ("lick", "AH as WH (train WH vs SL)", "WH vs SL decoder → auditory hits")]
    fig, axs = plt.subplots(3, 4, figsize=(m62.W_IN, m62.W_IN * 0.72))
    for i, (align, test, ttl) in enumerate(TESTS):
        d0 = T[(T["align"] == align) & (T.test == test)]
        for j, (col, coh) in enumerate((("acc_ex", "R+"), ("acc_ex", "R-"), ("frac_ex", "R+"), ("frac_ex", "R-"))):
            ax = axs[i, j]
            for st, ls in (("learning", (0, (2.5, 1.5))), ("expert", "-")):
                d = d0[(d0.cohort == coh) & (d0.stage == st)]
                piv = d.pivot_table(index="session_id", columns="t", values=col); mu, se = piv.mean(0), piv.sem(0)
                ax.fill_between(piv.columns, mu - se, mu + se, color=COH[coh], alpha=0.15, lw=0)
                ax.plot(piv.columns, mu, color=COH[coh], lw=1.0, ls=ls, label=f"{'day 0' if st == 'learning' else 'expert'} ({piv.shape[0]})")
            ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.4, ls=":")
            ax.set_box_aspect(1); ax.tick_params(labelsize=4.6); ax.legend(frameon=False, fontsize=4, loc="upper left")
            if j in (0, 2):
                ax.set_ylabel(f"{ttl}\n{'balanced acc − chance' if col == 'acc_ex' else 'transfer − chance'}" if j == 0 else "transfer − chance", fontsize=4.6)
            if i == 0:
                ax.set_title(f"{coh.replace('-', '−')}: {'accuracy' if col == 'acc_ex' else 'transfer'}", color=COH[coh], fontsize=5.4)
            if i == 2 or (i == 1):
                ax.set_xlabel("from lick (ms)" if align == "lick" else "from stimulus (ms)", fontsize=4.6)
    for r in range(3):                                                  # shared y within accuracy / transfer of a row
        for c0 in (0, 2):
            yl = (min(axs[r, c0].get_ylim()[0], axs[r, c0 + 1].get_ylim()[0]), max(axs[r, c0].get_ylim()[1], axs[r, c0 + 1].get_ylim()[1]))
            axs[r, c0].set_ylim(*yl); axs[r, c0 + 1].set_ylim(*yl)
    fig.suptitle(f"Whole-brain decoding over time, day 0 (dashed) vs expert (solid), relative to label-shuffle chance ({pop})", fontsize=6.2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"timecourses_{pop}.{ext}", dpi=300)


def fig_areas(pop, m88, m62, flt):
    import ephys_utilities.allen_utils.allen_utils as au
    COH = m88.COH; plt = m62.setup(); rng = np.random.default_rng(0); gcol = au.get_custom_area_groups_colors()
    D = flt(pd.read_parquet(OUT / "areas.parquet")); D["acc_ex"] = D.acc - D.acc_null; D["frac_ex"] = D.frac - D.frac_null
    GR = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
    n = D.groupby(["area", "decoder", "cohort", "stage"]).session_id.nunique()
    areas = ["Whole brain"] + [g for g in au.get_area_group_custom_order() if all(n.get((g, DEC[0][0], c, s), 0) >= 3 for c, s in GR)]
    COLS = [(dec, col) for dec, _ in DEC for col in ("acc_ex", "frac_ex")]
    ST = []
    for dec, col in COLS:
        for g in areas:
            d = D[(D.area == g) & (D.decoder == dec)].dropna(subset=[col]); r = dict(area=g, decoder=dec, measure=col)
            if not all(((d.cohort == c) & (d.stage == s)).sum() >= 3 for c, s in GR):
                continue
            for co in ("R+", "R-"):
                L_, E_ = (d[(d.cohort == co) & (d.stage == s_)][col] for s_ in ("learning", "expert"))
                r.update({f"D0_{co}": L_.mean(), f"E_{co}": E_.mean(), f"chg_{co}": E_.mean() - L_.mean(), f"sem_{co}": np.sqrt(L_.sem() ** 2 + E_.sem() ** 2),
                          f"p_{co}": stats.mannwhitneyu(L_, E_).pvalue, f"n_{co}": f"{len(L_)}/{len(E_)}"})
            r["p_int"] = m62.group_stats(d, col, f"{g} {dec} {col}", rng, "107")["interaction"][1]
            ST.append(r)
    ST = pd.DataFrame(ST)
    for dec, col in COLS:
        m = (ST.decoder == dec) & (ST.measure == col); ST.loc[m, "q_int"] = bh(ST.loc[m, "p_int"])
    ST.to_csv(OUT / f"areas_stats_{pop}.csv", index=False)
    fig, axs = plt.subplots(1, len(COLS), figsize=(m62.W_IN * 1.1, 0.22 * len(areas) + 1.2), sharey=True)
    y = np.arange(len(areas))[::-1]
    for j, (dec, col) in enumerate(COLS):
        ax = axs[j]; s = ST[(ST.decoder == dec) & (ST.measure == col)].set_index("area")
        for i, g in enumerate(areas):
            if g not in s.index:
                continue
            r = s.loc[g]
            for k, co in enumerate(("R+", "R-")):
                ax.errorbar(r[f"chg_{co}"], y[i] + 0.15 - 0.3 * k, xerr=r[f"sem_{co}"], fmt="o", ms=2.6, color=COH[co], lw=0.8, capsize=0,
                            mfc=COH[co] if r[f"p_{co}"] < 0.05 else "white")
            mk = "*" if r.q_int < 0.05 else ("(*)" if r.p_int < 0.05 else "")
            if mk:
                ax.text(0.99, y[i], mk, transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", ha="right", weight="bold")
        ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(areas) - 1.5, color="0.85", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([m88.m83.m62_short(g) for g in areas], fontsize=4.6)
        ax.set_title(f"{dec}\n{'accuracy' if col == 'acc_ex' else 'transfer'} − chance", fontsize=4.8); ax.tick_params(labelsize=4.3)
        ax.set_xlabel("expert − day 0", fontsize=4.6)
    for tl, g in zip(axs[0].get_yticklabels(), areas):
        tl.set_color(gcol.get(g, "k")); tl.set_weight("bold" if g == "Whole brain" else "normal")
    fig.suptitle(f"Per-area decoding (per session, units of the area, >= {MIN_UNITS}); change across learning; filled: within-cohort p < 0.05; "
                 f"* c×s q < 0.05, (*) p < 0.05 ({pop})", fontsize=5.6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"areas_{pop}.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    print(ST[["area", "decoder", "measure", "D0_R+", "E_R+", "p_R+", "D0_R-", "E_R-", "p_R-", "n_R+", "n_R-", "p_int", "q_int"]].round(3).to_string(index=False))


def main(a):
    if not a.replot:
        compute(a.n_proc)
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61
    flt = m61.learner_filter if a.pop == "learners" else (lambda d: d)
    fig_timecourses(a.pop, m88, m62, flt); fig_areas(a.pop, m88, m62, flt); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
