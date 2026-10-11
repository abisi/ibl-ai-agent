"""114 -- Cross-modal cue CCGP per session and area with the neuron count equalised by subsampling (user 2026-10-11: "keep single-session
decoding" instead of the pseudo-population of 111/112). Exploratory.
Same data, window, decoders, SL halves and linear-shift null as 111 (105 lick-aligned rates, stable tested units, [-100, 0) ms before the
lick, WH / AH / SL ordered in time; CCGP = mean of [train WH vs SL-A -> test AH vs SL-B] and the reverse, balanced accuracy, logistic
regression C = 0.05, units z-scored on the training events; PS = cos(WH - SL-A, AH - SL-B)).
Equalisation: per session x area group (and whole brain) with >= N_SUB units, R_SUB random draws of N_SUB units (without replacement);
per draw CCGP / PS and their linear-shift null (N_SHIFT shifts, the same shift set for all draws and areas of the session); the session
value is the mean over draws of (value - null). Correlations between simultaneously recorded neurons are kept; noise from the unit draw is
averaged out; differences in unit counts between areas / cohorts / stages no longer enter the accuracy.
Statistics (session = unit): per area, day 0 vs expert within cohort (Mann-Whitney + Welch), R+ vs R- per stage (Mann-Whitney + Welch),
cohort x stage interaction (mouse-level permutation, 062 group_stats); BH over areas per test. Area inclusion within cohort: >= 3 sessions
at both stages; R+ vs R- and the interaction only where both cohorts pass.
Output: across_days/<ref>/ccgp_sessions_eq/{sessions.parquet, stats_<pop>.csv, ccgp_sessions_eq_<pop>.{png,pdf}}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../114_ccgp_cue_sessions_equalised.py [--n-proc 25] [--replot]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m111 = importlib.import_module("111_ccgp_cue_pub")
m105, m51 = m111.m105, m111.m51
OUT = m51.OUTROOT / "ccgp_sessions_eq"
N_SUB, R_SUB, N_SHIFT, MIN_SESS = 20, 50, 10, 3
GROUPS = m111.GROUPS


def session(args):
    f, = args
    sid = f.stem; S = m111.load_session(f); idx = S["idx"]
    if min(len(idx["WH"]), len(idx["AH"]), len(idx["SL"]) // 2) < m111.MIN_EV:
        return []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|111".encode()))          # same SL halves as 111
    p = rng.permutation(idx["SL"]); sa, sb = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
    base = dict(WH=idx["WH"], AH=idx["AH"], SA=sa, SB=sb)
    r2 = np.random.default_rng(zlib.crc32(f"{sid}|114".encode()))
    shifts = [m111.shifted(base, S["n"], r2) for _ in range(N_SHIFT)]

    def run(X, I):
        if min(len(I[k]) for k in I) < m111.MIN_EV:
            return np.nan, np.nan
        return m111.ccgp_ps(X[:, I["WH"]].T, X[:, I["AH"]].T, X[:, I["SA"]].T, X[:, I["SB"]].T)
    rows = []
    groups = [("Whole brain", np.ones(len(S["area"]), bool))] + [(g, S["area"] == g) for g in sorted(set(S["area"])) if g not in ("nan", "None")]
    for g, m in groups:
        u = np.where(m)[0]
        if len(u) < N_SUB:
            continue
        cc, ps, cn, pn = [], [], [], []
        for _ in range(R_SUB):
            X = S["X"][r2.choice(u, N_SUB, replace=False)]
            a, b = run(X, base); nul = np.array([run(X, I) for I in shifts])
            cc.append(a); ps.append(b); cn.append(np.nanmean(nul[:, 0])); pn.append(np.nanmean(nul[:, 1]))
        cc, ps, cn, pn = map(np.asarray, (cc, ps, cn, pn))
        rows.append(dict(session_id=sid, area=g, n_units=len(u), n_WH=len(idx["WH"]), n_AH=len(idx["AH"]), n_SL=len(idx["SL"]),
                         CCGP=np.nanmean(cc), CCGP_null=np.nanmean(cn), ex=np.nanmean(cc - cn), ex_sd_draws=np.nanstd(cc - cn),
                         PS=np.nanmean(ps), PS_null=np.nanmean(pn), psx=np.nanmean(ps - pn), N_SUB=N_SUB, R_SUB=R_SUB, N_SHIFT=N_SHIFT))
    return rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, [(f,) for f in files]) for x in r]
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    D = pd.DataFrame(res).merge(meta, on="session_id"); D.to_parquet(OUT / "sessions.parquet", index=False)
    print("computed", len(D), "session x area rows", flush=True)


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def two(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def stats_table(D, col, m62, rng):
    n = D.groupby(["area", "cohort", "stage"]).session_id.nunique()
    ok = {(a, c): n.get((a, c, "learning"), 0) >= MIN_SESS and n.get((a, c, "expert"), 0) >= MIN_SESS for a in D.area.unique() for c in ("R+", "R-")}
    rows = []
    for a, d in D.groupby("area"):
        if not (ok[(a, "R+")] or ok[(a, "R-")]):
            continue
        r = dict(area=a, measure=col)
        g = lambda c, s: d[(d.cohort == c) & (d.stage == s)][col].dropna().to_numpy()
        for c, s in GROUPS:
            v = g(c, s); r[f"{c} {s}"] = v.mean() if len(v) else np.nan; r[f"sem {c} {s}"] = stats.sem(v) if len(v) > 1 else np.nan; r[f"n {c} {s}"] = len(v)
        for c in ("R+", "R-"):
            r[f"p_MWU D0vsE {c}"], r[f"p_Welch D0vsE {c}"] = two(g(c, "learning"), g(c, "expert")) if ok[(a, c)] else (np.nan, np.nan)
        both = ok[(a, "R+")] and ok[(a, "R-")]
        for s in ("learning", "expert"):
            r[f"p_MWU RpvsRm {s}"], r[f"p_Welch RpvsRm {s}"] = two(g("R+", s), g("R-", s)) if both else (np.nan, np.nan)
        r["p_int"] = m62.group_stats(d.dropna(subset=[col]), col, f"114 {a} {col}", rng, "114")["interaction"][1] if both else np.nan
        rows.append(r)
    ST = pd.DataFrame(rows)
    for k in [c for c in ST.columns if c.startswith("p_")]:
        m = ST.area != "Whole brain"; ST.loc[m, k.replace("p_", "q_", 1)] = bh(ST.loc[m, k])
    return ST


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH = m88.COH
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors(); plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(OUT / "sessions.parquet"); D = m61.learner_filter(D) if pop == "learners" else D
    ST = pd.concat([stats_table(D, c, m62, rng) for c in ("ex", "psx")]); ST.to_csv(OUT / f"stats_{pop}.csv", index=False)
    S = ST[ST.measure == "ex"].copy()
    order = ["Whole brain"] + S[S.area != "Whole brain"].sort_values("R+ expert", ascending=False).area.tolist()
    star = lambda q: "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
    with plt.rc_context({"font.size": 5.4, "axes.labelsize": 5.4, "xtick.labelsize": 4.9, "ytick.labelsize": 4.9}):
        fig = plt.figure(figsize=(7.2, 0.17 * len(order) + 1.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.5, 1.1, 1.1], wspace=0.45, left=0.05, right=0.99, top=0.86, bottom=0.12)
        # a whole brain
        ax = fig.add_subplot(gs[0, 0]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, D[D.area == "Whole brain"], "ex", "CCGP − shift null", rng, "114 wb", f"Whole brain\n({N_SUB} units × {R_SUB} draws)", ref=0)
        # b per area, sorted
        y = np.arange(len(order))[::-1]; off = {GROUPS[0]: 0.27, GROUPS[1]: 0.09, GROUPS[2]: -0.09, GROUPS[3]: -0.27}
        ax = fig.add_subplot(gs[0, 1])
        for i, a in enumerate(order):
            r = S[S.area == a].iloc[0]
            for k in GROUPS:
                key = f"{k[0]} {k[1]}"
                if r[f"n {key}"] >= MIN_SESS:
                    ax.errorbar(r[key], y[i] + off[k], xerr=r[f"sem {key}"], fmt="o", ms=2.6, color=COH[k[0]], mfc=COH[k[0]] if k[1] == "expert" else "white",
                                lw=0.7, capsize=0, mew=0.6)
        ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([m88.m83.m62_short(a) for a in order], fontsize=4.8)
        for tl, a in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(a, "k")); tl.set_weight("bold" if a == "Whole brain" else "normal")
        ax.set_xlabel("CCGP − shift null (session mean ± SEM)"); ax.set_title("Per area, ranked by R+ expert", fontsize=5.4)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], marker="o", ls="", color=COH[c], mfc=COH[c] if s == "expert" else "white", ms=3) for c, s in GROUPS],
                  ["R+ day 0", "R+ expert", "R− day 0", "R− expert"], frameon=False, fontsize=4.2, loc="lower right", handletextpad=0.2)
        # c learning change per cohort; d R+ - R- at expert
        for j, (ttl, pairs) in enumerate((("Expert − day 0", [("R+", "D0vsE R+"), ("R-", "D0vsE R-")]), ("R+ − R−", [("learning", "RpvsRm learning"), ("expert", "RpvsRm expert")]))):
            ax = fig.add_subplot(gs[0, 2 + j], sharey=None)
            for i, a in enumerate(order):
                r = S[S.area == a].iloc[0]
                for kk, (lab, test) in enumerate(pairs):
                    if j == 0:
                        dv = r[f"{lab} expert"] - r[f"{lab} learning"]; se = np.hypot(r[f"sem {lab} expert"], r[f"sem {lab} learning"]); col = COH[lab]; mfc = col
                    else:
                        dv = r[f"R+ {lab}"] - r[f"R- {lab}"]; se = np.hypot(r[f"sem R+ {lab}"], r[f"sem R- {lab}"]); col = "0.2"; mfc = col if lab == "expert" else "white"
                    if not np.isfinite(dv) or not np.isfinite(r.get(f"p_MWU {test}", np.nan)):
                        continue
                    yy = y[i] + (0.15 if kk == 0 else -0.15)
                    ax.errorbar(dv, yy, xerr=se, fmt="o", ms=2.6, color=col, mfc=mfc, lw=0.7, capsize=0, mew=0.6)
                    q = r[f"p_MWU {test}"] if a == "Whole brain" else r.get(f"q_MWU {test}", np.nan)
                    if np.isfinite(q) and q < 0.05:
                        ax.text(dv + se, yy, " " + star(q), fontsize=5, va="center", color=col)
            ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
            ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(-0.7, len(order) - 0.3)
            ax.set_title(ttl + ("\n(green R+, magenta R−)" if j == 0 else "\n(open day 0, filled expert)"), fontsize=5.2)
            ax.set_xlabel("Δ CCGP − shift null")
        fig.axes[1].set_ylim(-0.7, len(order) - 0.3)
        fig.suptitle(f"Cross-modal cue CCGP per session with equalised neuron count ({N_SUB} units; {pop}); "
                     "stars: MWU, BH over areas (whole brain uncorrected)", x=0.01, ha="left", fontsize=6, weight="bold")
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"ccgp_sessions_eq_{pop}.{ext}", dpi=300)
    pd.set_option("display.width", 300)
    cols = ["area"] + [f"{c} {s}" for c, s in GROUPS] + [f"n {c} {s}" for c, s in GROUPS] + \
        ["p_MWU D0vsE R+", "q_MWU D0vsE R+", "p_Welch D0vsE R+", "p_MWU D0vsE R-", "q_MWU D0vsE R-", "p_MWU RpvsRm expert", "q_MWU RpvsRm expert",
         "p_Welch RpvsRm expert", "p_int", "q_int"]
    S = S.set_index("area").loc[order].reset_index()
    print(S[[c for c in cols if c in S]].round(3).to_string(index=False))
    P = ST[ST.measure == "psx"]; print("\nPS whole brain:\n", P[P.area == "Whole brain"][[c for c in cols if c in P]].round(3).to_string(index=False))


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
