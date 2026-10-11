"""110 -- Cross-modal cue CCGP (109) per area group, ranked from most to least generalising (user 2026-10-11). Exploratory.
Per session x area group (units of the area, >= MIN_UNITS; 105 lick-aligned data, mean of the bins centred in [-100, 0) ms): the same
decoders, SL halves (same split as 109, seed crc32("<sid>|109")) and label-shuffle chance as 109:
  CCGP_cue = mean of [train WH vs SL-A -> test AH vs SL-B] and [train AH vs SL-A -> test WH vs SL-B] (balanced accuracy) - shuffle chance
  PS_cue   = cosine(WH - SL-A, AH - SL-B)
Whole brain (all units of the session) included for reference. Areas with >= 3 sessions in every cohort x stage group.
Ranking: areas sorted by the mean CCGP_cue of R+ expert sessions (also printed for the other groups).
Statistics per area: MWU day 0 vs expert within cohort, R+ vs R- at each stage, cohort x stage (mouse permutation); BH over areas.
Output: across_days/<ref>/ccgp/{cue_areas.parquet, ccgp_cue_areas_<pop>.{png,pdf}, cue_areas_stats_<pop>.csv}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../110_ccgp_cue_areas.py [--n-proc 25] [--replot] [--pop learners]
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
m109 = importlib.import_module("109_ccgp_cue")
m108, m105, m51 = m109.m108, m109.m105, m109.m51
OUT = m108.OUT
MIN_UNITS, MIN_EV, N_SHUF = 10, 6, 20
GR = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]


def session(args):
    f, = args
    sid = f.stem; z = np.load(f, allow_pickle=True)
    lab, L, c = z["lick_lab"], z["lickbins"].astype(float), z["lick_c"]
    ag = z["area_group"].astype(str)
    wh, ah, sl = (np.where(lab == k)[0] for k in ("WH", "AH", "SL"))
    if min(len(wh), len(ah), len(sl) // 2) < MIN_EV:
        return []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|109".encode())); _ = int(rng.integers(1e6))
    p = rng.permutation(sl); sa, sb = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
    Xw = L[:, :, (c >= -0.1) & (c < 0.0)].mean(2)
    r2 = np.random.default_rng(zlib.crc32(f"{sid}|110".encode()))
    rows = []
    for g, m in [("Whole brain", np.ones(len(ag), bool))] + [(g, ag == g) for g in sorted(set(ag)) if g not in ("nan", "None")]:
        if m.sum() < MIN_UNITS:
            continue
        X = Xw[m]
        cc = m109.ccgp(X, wh, ah, sa, sb); null = float(np.mean([m109.ccgp(X, wh, ah, sa, sb, r2) for _ in range(N_SHUF)]))
        mu, sd = m108.zfit(X[:, np.r_[wh, ah, sl]].T); Z = (X.T - mu) / sd
        v1, v2 = Z[wh].mean(0) - Z[sa].mean(0), Z[ah].mean(0) - Z[sb].mean(0)
        ps = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
        rows.append(dict(session_id=sid, area=g, n_units=int(m.sum()), CCGP_cue=cc, CCGP_cue_null=null, CCGP_ex=cc - null, PS_cue=ps))
    return rows


def compute(n_proc):
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, [(f,) for f in files]) for x in r]
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    D = pd.DataFrame(res).merge(meta, on="session_id"); D.to_parquet(OUT / "cue_areas.parquet", index=False)
    print("computed", len(D), "session x area rows", flush=True)


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH = m88.COH
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup(); rng = np.random.default_rng(0); gcol = au.get_custom_area_groups_colors()
    D = pd.read_parquet(OUT / "cue_areas.parquet"); D = m61.learner_filter(D) if pop == "learners" else D
    n = D.groupby(["area", "cohort", "stage"]).session_id.nunique()
    areas = [a for a in D.area.unique() if all(n.get((a, c, s), 0) >= 3 for c, s in GR)]
    M = D.groupby(["area", "cohort", "stage"]).agg(CCGP=("CCGP_ex", "mean"), sem=("CCGP_ex", "sem"), PS=("PS_cue", "mean"), PS_sem=("PS_cue", "sem"),
                                                     n=("session_id", "nunique")).reset_index()
    order = M[(M.cohort == "R+") & (M.stage == "expert") & M.area.isin(areas) & (M.area != "Whole brain")].sort_values("CCGP", ascending=False).area.tolist()
    order = ["Whole brain"] + order
    ST = []
    for a in order:
        d = D[D.area == a]; r = dict(area=a)
        for c, s in GR:
            v = d[(d.cohort == c) & (d.stage == s)].CCGP_ex
            r[f"{c} {s}"] = v.mean(); r[f"n {c} {s}"] = len(v); r[f"p_vs0 {c} {s}"] = stats.wilcoxon(v).pvalue if len(v) >= 6 else np.nan
        for c in ("R+", "R-"):
            L_, E_ = (d[(d.cohort == c) & (d.stage == s)].CCGP_ex for s in ("learning", "expert")); r[f"p_D0vsE {c}"] = stats.mannwhitneyu(L_, E_).pvalue
        for s in ("learning", "expert"):
            P_, M_ = (d[(d.cohort == c) & (d.stage == s)].CCGP_ex for c in ("R+", "R-")); r[f"p_RpvsRm {s}"] = stats.mannwhitneyu(P_, M_).pvalue
        r["p_int"] = m62.group_stats(d, "CCGP_ex", f"{a} cue CCGP", rng, "110")["interaction"][1]
        ST.append(r)
    ST = pd.DataFrame(ST)
    for k in ("p_D0vsE R+", "p_D0vsE R-", "p_RpvsRm learning", "p_RpvsRm expert", "p_int"):
        m = ST.area != "Whole brain"; ST.loc[m, k.replace("p_", "q_")] = bh(ST.loc[m, k])
    ST.to_csv(OUT / f"cue_areas_stats_{pop}.csv", index=False)
    # figure: dot plots sorted
    fig, axs = plt.subplots(1, 3, figsize=(m62.W_IN, 0.24 * len(order) + 1.3), gridspec_kw=dict(width_ratios=[1.2, 1, 0.9]), sharey=True)
    y = np.arange(len(order))[::-1]
    off = {("R+", "learning"): 0.27, ("R+", "expert"): 0.09, ("R-", "learning"): -0.09, ("R-", "expert"): -0.27}
    for j, (col, se, xl) in enumerate((("CCGP", "sem", "cue CCGP − chance"), ("PS", "PS_sem", "parallelism (cosine)"))):
        ax = axs[j]
        for i, a in enumerate(order):
            for (c, s) in GR:
                q = M[(M.area == a) & (M.cohort == c) & (M.stage == s)]
                if len(q):
                    ax.errorbar(q[col].iloc[0], y[i] + off[(c, s)], xerr=q[se].iloc[0], fmt="o", ms=2.8, color=COH[c], mfc=COH[c] if s == "expert" else "white",
                                lw=0.8, capsize=0)
        ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
        ax.set_xlabel(xl, fontsize=5); ax.tick_params(labelsize=4.6)
    axs[0].set_yticks(y); axs[0].set_yticklabels([m88.m83.m62_short(a) for a in order], fontsize=4.8)
    for tl, a in zip(axs[0].get_yticklabels(), order):
        tl.set_color(gcol.get(a, "k")); tl.set_weight("bold" if a == "Whole brain" else "normal")
    axs[0].set_title("Cue CCGP per area, sorted by R+ expert\n(open: day 0, filled: expert)", fontsize=5.4)
    axs[1].set_title("Parallelism\n(WH − SL vs AH − SL)", fontsize=5.4)
    ax = axs[2]; ax.axis("off")
    for i, a in enumerate(order):
        r = ST[ST.area == a].iloc[0]
        f = lambda k: ("q" if a != "Whole brain" else "p")
        txt = (f"R+ D0→E {m62.fmt_p(r['p_D0vsE R+'])}  |  R+ vs R− E {m62.fmt_p(r['p_RpvsRm expert'])}")
        star = "*" if a != "Whole brain" and (r.get("q_D0vsE R+", 1) < 0.05) else ""
        ax.text(0.0, y[i], txt + (" *" if star else ""), fontsize=4.0, va="center", transform=ax.get_yaxis_transform(), color="0.25")
    ax.set_ylim(axs[0].get_ylim()); ax.set_title("p (Mann-Whitney); * R+ D0→E q < 0.05", fontsize=5)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", color=COH[c], mfc=COH[c] if s == "expert" else "white", ms=3) for c, s in GR],
               ["R+ day 0", "R+ expert", "R− day 0", "R− expert"], frameon=False, fontsize=4.8, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.0))
    fig.suptitle(f"Cross-modal cued-lick CCGP per area (per session, ≥ {MIN_UNITS} units; {pop})", fontsize=6.2, y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ccgp_cue_areas_{pop}.{ext}", dpi=300)
    pd.set_option("display.width", 260)
    cols = ["area"] + [f"{c} {s}" for c, s in GR] + ["p_D0vsE R+", "q_D0vsE R+", "p_RpvsRm learning", "p_RpvsRm expert", "q_RpvsRm expert", "p_int", "q_int"]
    print(ST[[k for k in cols if k in ST]].round(3).to_string(index=False))


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
