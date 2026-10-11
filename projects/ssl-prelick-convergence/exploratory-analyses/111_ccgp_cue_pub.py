"""111 -- Cross-modal CCGP of the cued-lick code: per session and pseudo-population, linear-shift null, publication figure
(user 2026-10-11: "pseudopopulation to account for sampling; linear-shift null for autocorrelated time series; report acc minus null;
publication quality; relate to the coding direction"). 109 / 110 kept.
Data: 105 lick-aligned rates (stable tested units; baseline-corrected spikes/s), window = mean of the 50-ms bins centred in [-100, 0) ms
before the lick; events WH (whisker hit), AH (auditory hit), SL (spontaneous lick), ordered in time per session. Learners rule.
CCGP (Bernardi et al. 2020): variable cued vs spontaneous lick, condition cue modality. SL split at random into halves A / B (training
always SL-A, testing SL-B); CCGP = mean of [train WH vs SL-A -> test AH vs SL-B] and [train AH vs SL-A -> test WH vs SL-B]; balanced
accuracy; logistic regression (L2), units z-scored on the training events.
Parallelism PS = cos(mean WH - mean SL-A, mean AH - mean SL-B) (z-scored units); WH - SL vs AH - SL is the angle of WH - SL to the
reward-lick CD (AH - SL; 087 theta), measured on independent SL halves.
Linear-shift null (as 064 / within-day 026): the event labels keep their time order and each label takes the activity of the event k
positions later (|k| in [MIN_SHIFT, n // 3], random sign, no wrap-around; events shifted outside the session dropped), decoders retrained;
null = mean over shifts. Reported: value - null.
Part A, per session (session x area group, >= MIN_UNITS units; whole brain = all units): CCGP and PS, N_SHIFT shifts; time course for the
whole brain (each 50-ms bin, -400 .. +100 ms; N_SHIFT_TC shifts per bin).
Part B, pseudo-population with hierarchical bootstrap (064): per cohort x stage group and region, per iteration: mice with replacement ->
one session per sampled mouse -> M neurons from the pooled units (without replacement; with if the pool is smaller) -> per session random
SL halves and N_PS pseudo-trials per class (same event indices for all neurons of a session); real and linear-shift null (one random k per
session, same neurons); N_SPLIT trial draws averaged. B iterations. Regions with >= MIN_MICE mice (with >= 1 unit) in every group.
Statistics: part A as 088 (session = unit; MWU, cohort x stage mouse permutation); part B: percentile 95 % CI of each group, of the
expert - day 0 change per cohort, of R+ - R- per stage and of the interaction; bootstrap p = 2 x fraction on the other side of 0.
NOTE: B = 200 and N_SPLIT = 5 are pilot values (increase for the final figure).
Output: across_days/<ref>/ccgp_pub/{sessions.parquet, timecourse.parquet, pseudopop.parquet, pseudopop_stats.csv, stats_<pop>.csv,
ccgp_cue_pub_<pop>.{png,pdf,svg}}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../111_ccgp_cue_pub.py [--n-proc 25] [--replot] [--pop learners]
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

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m105 = importlib.import_module("105_trial_rates")
m51 = m105.m51
OUT = m51.OUTROOT / "ccgp_pub"
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
MIN_UNITS, MIN_EV, MIN_SHIFT, N_SHIFT, N_SHIFT_TC = 10, 6, 5, 20, 10
M, N_PS, N_SPLIT, B, MIN_MICE, C_REG = 40, 50, 5, 200, 3, 0.05


# ------------------------------------------------------------------ core
def zfit(A):
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
    return mu, sd


def bacc_tt(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    mu, sd = zfit(Xtr)
    clf = LogisticRegression(C=C_REG, max_iter=1000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    p = clf.predict((Xte - mu) / sd)
    return 0.5 * ((p[yte == 1] == 1).mean() + (p[yte == 0] == 0).mean())


def ccgp_ps(WH, AH, SA, SB):
    """inputs: events x units; returns CCGP and PS"""
    y = lambda a, b: np.r_[np.ones(len(a)), np.zeros(len(b))]
    c1 = bacc_tt(np.vstack([WH, SA]), y(WH, SA), np.vstack([AH, SB]), y(AH, SB))
    c2 = bacc_tt(np.vstack([AH, SA]), y(AH, SA), np.vstack([WH, SB]), y(WH, SB))
    mu, sd = zfit(np.vstack([WH, AH, SA, SB])); z = lambda A: (A - mu) / sd
    v1, v2 = z(WH).mean(0) - z(SA).mean(0), z(AH).mean(0) - z(SB).mean(0)
    return 0.5 * (c1 + c2), float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))


def shifted(idx, n, rng):
    kmax = max(MIN_SHIFT + 1, n // 3)
    k = int(rng.integers(MIN_SHIFT, kmax)) * (1 if rng.random() < 0.5 else -1)
    out = {}
    for c, v in idx.items():
        w = v + k
        out[c] = w[(w >= 0) & (w < n)]
    return out


def load_session(f):
    z = np.load(f, allow_pickle=True)
    lab, L, c, t = z["lick_lab"], z["lickbins"].astype(float), z["lick_c"], z["lick_t"].astype(float)
    o = np.argsort(t, kind="stable"); lab, L = lab[o], L[:, o]
    Xw = L[:, :, (c >= -0.1) & (c < 0.0)].mean(2)
    return dict(lab=lab, L=L, c=c, X=Xw, area=z["area_group"].astype(str), n=len(lab),
                idx={k: np.where(lab == k)[0] for k in ("WH", "AH", "SL")})


# ------------------------------------------------------------------ part A: per session
def session(args):
    f, = args
    sid = f.stem; S = load_session(f); idx = S["idx"]
    if min(len(idx["WH"]), len(idx["AH"]), len(idx["SL"]) // 2) < MIN_EV:
        return [], []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|111".encode()))
    p = rng.permutation(idx["SL"]); sa, sb = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
    base = dict(WH=idx["WH"], AH=idx["AH"], SA=sa, SB=sb)

    def run(X, I):
        if min(len(I["WH"]), len(I["AH"]), len(I["SA"]), len(I["SB"])) < MIN_EV:
            return np.nan, np.nan
        return ccgp_ps(X[:, I["WH"]].T, X[:, I["AH"]].T, X[:, I["SA"]].T, X[:, I["SB"]].T)
    rows = []
    groups = [("Whole brain", np.ones(len(S["area"]), bool))] + [(g, S["area"] == g) for g in sorted(set(S["area"])) if g not in ("nan", "None")]
    shifts = [shifted(base, S["n"], rng) for _ in range(N_SHIFT)]
    for g, m in groups:
        if m.sum() < MIN_UNITS:
            continue
        X = S["X"][m]; cc, ps = run(X, base)
        nul = np.array([run(X, I) for I in shifts])
        rows.append(dict(session_id=sid, area=g, n_units=int(m.sum()), CCGP=cc, CCGP_null=np.nanmean(nul[:, 0]), PS=ps, PS_null=np.nanmean(nul[:, 1])))
    tc = []
    sh_tc = shifts[:N_SHIFT_TC]
    for k, cc in enumerate(S["c"]):
        Xk = S["L"][:, :, k]; r, _ = run(Xk, base); nl = np.nanmean([run(Xk, I)[0] for I in sh_tc])
        tc.append(dict(session_id=sid, t=float(cc * 1e3), CCGP=r, CCGP_null=nl))
    return rows, tc


# ------------------------------------------------------------------ part B: pseudo-population
DATA = {}


def pp_load(meta):
    for r in meta.itertuples():
        f = m105.OUT / f"{r.session_id}.npz"
        if f.exists():
            S = load_session(f)
            if min(len(S["idx"]["WH"]), len(S["idx"]["AH"]), len(S["idx"]["SL"]) // 2) >= MIN_EV:
                DATA[r.session_id] = S


def pp_iter(tree, region, rng):
    mice = list(tree)
    inst = [rng.choice(tree[m]) for m in rng.choice(mice, len(mice))]
    units = [np.where(DATA[s]["area"] == region)[0] if region != "Whole brain" else np.arange(len(DATA[s]["area"])) for s in inst]
    sizes = np.array([len(u) for u in units]); tot = sizes.sum()
    if tot < 5:
        return None
    pick = np.sort(rng.choice(tot, M, replace=tot < M)); off = np.r_[0, np.cumsum(sizes)]
    sel = [(s, units[j][pick[(pick >= off[j]) & (pick < off[j + 1])] - off[j]]) for j, s in enumerate(inst)]
    sel = [(s, u) for s, u in sel if len(u)]
    out = []
    for null in (False, True):
        cs, ps = [], []
        for _ in range(N_SPLIT):
            P = {c: np.zeros((N_PS, M), np.float32) for c in ("WH", "AH", "SA", "SB")}; col = 0
            for s, u in sel:
                S = DATA[s]; idx = S["idx"]
                pp = rng.permutation(idx["SL"]); I = dict(WH=idx["WH"], AH=idx["AH"], SA=pp[: len(pp) // 2], SB=pp[len(pp) // 2:])
                if null:
                    J = shifted(I, S["n"], rng); I = {c: (J[c] if len(J[c]) >= 2 else I[c]) for c in I}
                for c in P:
                    P[c][:, col:col + len(u)] = S["X"][np.ix_(u, rng.choice(I[c], N_PS))].T
                col += len(u)
            c_, p_ = ccgp_ps(P["WH"], P["AH"], P["SA"], P["SB"]); cs.append(c_); ps.append(p_)
        out += [np.mean(cs), np.mean(ps)]
    return out                                                          # CCGP, PS, CCGP_null, PS_null


def pp_job(args):
    g, region, tree, seed = args
    rng = np.random.default_rng(seed); res = []
    for b in range(B):
        r = pp_iter(tree, region, rng)
        if r is not None:
            res.append(dict(cohort=g[0], stage=g[1], area=region, it=b, CCGP=r[0], PS=r[1], CCGP_null=r[2], PS_null=r[3]))
    return res


# ------------------------------------------------------------------ compute
def compute(n_proc, pop):
    OUT.mkdir(parents=True, exist_ok=True)
    m61 = importlib.import_module("061_roc_prelick_learners")
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(f,) for f in files])
    pd.DataFrame([x for r, _ in res for x in r]).merge(meta, on="session_id").to_parquet(OUT / "sessions.parquet", index=False)
    pd.DataFrame([x for _, t in res for x in t]).merge(meta, on="session_id").to_parquet(OUT / "timecourse.parquet", index=False)
    print("part A done", flush=True)
    mt = m61.learner_filter(meta) if pop == "learners" else meta
    pp_load(mt); mt = mt[mt.session_id.isin(DATA)]
    trees = {g: {} for g in GROUPS}
    for r in mt.itertuples():
        trees[(r.cohort, r.stage)].setdefault(r.mouse_id, []).append(r.session_id)
    areas = sorted({a for s in DATA for a in set(DATA[s]["area"]) if a not in ("nan", "None")})
    regions = ["Whole brain"]
    for a in areas:
        ok = True
        for g in GROUPS:
            mice = {m for m, ss in trees[g].items() if any((DATA[s]["area"] == a).any() for s in ss)}
            ok &= len(mice) >= MIN_MICE
        if ok:
            regions.append(a)
    jobs = []
    for region in regions:
        for g in GROUPS:
            tree = {m: [s for s in ss if region == "Whole brain" or (DATA[s]["area"] == region).any()] for m, ss in trees[g].items()}
            tree = {m: ss for m, ss in tree.items() if ss}
            jobs.append((g, region, tree, zlib.crc32(f"{g}|{region}|111".encode())))
    with mp.get_context("fork").Pool(n_proc) as pool:
        P = [x for r in pool.imap_unordered(pp_job, jobs) for x in r]
    pd.DataFrame(P).to_parquet(OUT / f"pseudopop_{pop}.parquet", index=False)
    print("part B done", len(regions), "regions", flush=True)


# ------------------------------------------------------------------ figure
def boot_stats(P):
    P = P.assign(ex=P.CCGP - P.CCGP_null, psx=P.PS - P.PS_null)
    rows = []
    for a, d in P.groupby("area"):
        g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])].sort_values("it").ex.to_numpy() for k in GROUPS}
        n = min(len(v) for v in g.values())
        if n == 0:
            continue
        g = {k: v[:n] for k, v in g.items()}
        ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        bp = lambda v: min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        r = dict(area=a, n_boot=n)
        for k, v in g.items():
            r[f"{k[0]} {k[1]}"] = np.median(v); r[f"{k[0]} {k[1]} lo"], r[f"{k[0]} {k[1]} hi"] = ci(v)
        for name, v in {"R+ E-D0": g[GROUPS[1]] - g[GROUPS[0]], "R- E-D0": g[GROUPS[3]] - g[GROUPS[2]],
                        "D0 R+-R-": g[GROUPS[0]] - g[GROUPS[2]], "E R+-R-": g[GROUPS[1]] - g[GROUPS[3]],
                        "interaction": (g[GROUPS[1]] - g[GROUPS[0]]) - (g[GROUPS[3]] - g[GROUPS[2]])}.items():
            r[name] = np.median(v); r[f"{name} lo"], r[f"{name} hi"] = ci(v); r[f"{name} p"] = bp(v)
        rows.append(r)
    return pd.DataFrame(rows)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH = m88.COH
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors(); plt = m62.setup(); rng = np.random.default_rng(0)
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    A = flt(pd.read_parquet(OUT / "sessions.parquet")); A["ex"] = A.CCGP - A.CCGP_null; A["psx"] = A.PS - A.PS_null
    T = flt(pd.read_parquet(OUT / "timecourse.parquet")); T["ex"] = T.CCGP - T.CCGP_null
    P = pd.read_parquet(OUT / f"pseudopop_{pop}.parquet"); BS = boot_stats(P); BS.to_csv(OUT / f"pseudopop_stats_{pop}.csv", index=False)
    H = flt(pd.read_csv(m51.OUTROOT / "harmonised" / "sessions_areas.csv")); H = H[H.area == "whole_brain"][["session_id", "theta", "d_WS"]]
    Wb = A[A.area == "Whole brain"].merge(H, on="session_id", how="left"); Wb["cos_theta"] = np.cos(np.radians(Wb.theta))
    with plt.rc_context({"font.size": 5.4, "axes.labelsize": 5.4, "xtick.labelsize": 4.9, "ytick.labelsize": 4.9}):
        fig = plt.figure(figsize=(7.2, 4.6))
        gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], hspace=0.75, wspace=0.75, left=0.07, right=0.985, top=0.9, bottom=0.08)
        L = []
        ax = fig.add_subplot(gs[0, 0]); ax.set_xticks([]); ax.set_yticks([]); L.append(ax)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("0.85")
        ax.text(0.5, 0.5, "schematic:\ntrain WH vs SL-A,\ntest AH vs SL-B\n(and reverse)", ha="center", va="center", fontsize=5, color="0.55", transform=ax.transAxes)
        ax.set_box_aspect(1)
        # b time course
        ax = fig.add_subplot(gs[0, 1]); ax.set_box_aspect(1); L.append(ax)
        for coh in ("R+", "R-"):
            for st, ls in (("learning", (0, (2.5, 1.5))), ("expert", "-")):
                piv = T[(T.cohort == coh) & (T.stage == st)].pivot_table(index="session_id", columns="t", values="ex"); mu, se = piv.mean(0), piv.sem(0)
                ax.fill_between(piv.columns, mu - se, mu + se, color=COH[coh], alpha=0.13, lw=0)
                ax.plot(piv.columns, mu, color=COH[coh], ls=ls, lw=0.9, label=f"{coh.replace('-', '−')} {'D0' if st == 'learning' else 'E'} ({piv.shape[0]})")
        ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.4, ls=":"); ax.axvspan(-100, 0, color="#FDD49E", alpha=0.4, lw=0, zorder=0)
        ax.set_xlabel("time from lick (ms)"); ax.set_ylabel("CCGP − shift null"); ax.legend(frameon=False, fontsize=3.8, loc="upper left", handlelength=1.3)
        ax.set_title("Whole brain, per session", fontsize=5.4)
        # c per-session whole brain; d PS
        for j, (col, yl, tt) in enumerate((("ex", "CCGP − shift null", "Cue CCGP (−100–0 ms)"), ("psx", "parallelism − shift null", "Parallelism\ncos(WH−SL, AH−SL)"))):
            ax = fig.add_subplot(gs[0, 2 + j]); ax.set_box_aspect(1); L.append(ax)
            m88.mean_panel2(ax, A[A.area == "Whole brain"], col, yl, rng, f"111 {col}", tt, ref=0)
        # e relation to the CD angle
        ax = fig.add_subplot(gs[0, 4]); ax.set_box_aspect(1); L.append(ax)
        d = Wb.dropna(subset=["cos_theta", "ex"])
        for (c, s) in GROUPS:
            q = d[(d.cohort == c) & (d.stage == s)]
            ax.plot(q.cos_theta, q.ex, "o", ms=2.6, color=COH[c], mfc=COH[c] if s == "expert" else "white", mew=0.6, ls="")
        import statsmodels.formula.api as smf
        from scipy import stats
        mdl = smf.ols("ex ~ cos_theta", d).fit(); xx = np.linspace(d.cos_theta.min(), d.cos_theta.max(), 50)
        pr = mdl.get_prediction(pd.DataFrame({"cos_theta": xx})).summary_frame(alpha=0.05); rs, ps_ = stats.spearmanr(d.cos_theta, d.ex)
        ax.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color="0.5", alpha=0.2, lw=0)
        ax.plot(xx, pr["mean"], color="0.2", lw=0.9, ls="-" if mdl.pvalues["cos_theta"] < 0.05 else "--")
        ax.set_xlabel("cos θ (WH − SL vs reward-lick CD)"); ax.set_ylabel("CCGP − shift null")
        ax.set_title(f"Relation to the CD\nSpearman r = {rs:.2f}, {m62.fmt_p(ps_)}", fontsize=5.4)
        # f pseudo-population per area, sorted by R+ expert; g changes with CIs
        order = ["Whole brain"] + BS[BS.area != "Whole brain"].sort_values("R+ expert", ascending=False).area.tolist()
        y = np.arange(len(order))[::-1]
        ax = fig.add_subplot(gs[1, 0:3]); L.append(ax)
        off = {GROUPS[0]: 0.27, GROUPS[1]: 0.09, GROUPS[2]: -0.09, GROUPS[3]: -0.27}
        for i, a in enumerate(order):
            r = BS[BS.area == a].iloc[0]
            for k in GROUPS:
                key = f"{k[0]} {k[1]}"
                ax.errorbar(r[key], y[i] + off[k], xerr=[[r[key] - r[f"{key} lo"]], [r[f"{key} hi"] - r[key]]], fmt="o", ms=2.8, color=COH[k[0]],
                            mfc=COH[k[0]] if k[1] == "expert" else "white", lw=0.8, capsize=0)
        ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([m88.m83.m62_short(a) for a in order], fontsize=4.8)
        for tl, a in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(a, "k")); tl.set_weight("bold" if a == "Whole brain" else "normal")
        ax.set_xlabel(f"pseudo-population CCGP − shift null ({M} neurons; median, 95 % CI)")
        ax.set_title("Areas ranked by generalisation (R+ expert)", fontsize=5.4)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], marker="o", ls="", color=COH[c], mfc=COH[c] if s == "expert" else "white", ms=3) for c, s in GROUPS],
                  ["R+ day 0", "R+ expert", "R− day 0", "R− expert"], frameon=False, fontsize=4.4, loc="lower right", handletextpad=0.2)
        for j, (name, ttl) in enumerate((("R+ E-D0", "R+: expert − day 0"), ("E R+-R-", "Expert: R+ − R−"))):
            ax = fig.add_subplot(gs[1, 3 + j], sharey=None); L.append(ax)
            for i, a in enumerate(order):
                r = BS[BS.area == a].iloc[0]; sig = r[f"{name} p"] < 0.05
                col = COH["R+"] if name.startswith("R+") else "0.2"
                ax.errorbar(r[name], y[i], xerr=[[r[name] - r[f"{name} lo"]], [r[f"{name} hi"] - r[name]]], fmt="o", ms=2.8, color=col, mfc=col if sig else "white",
                            lw=0.8, capsize=0)
            ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
            ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel("Δ CCGP (median, 95 % CI)"); ax.set_title(f"{ttl}\nfilled: bootstrap p < 0.05", fontsize=5.4)
        m62.letter_row(fig, L[:5], "abcde", dx_in=0.28, dy_in=0.16); m62.letter_row(fig, L[5:], "fgh", dx_in=0.28, dy_in=0.16)
        fig.suptitle(f"Whisker licks join the auditory reward-lick code: cross-modal generalisation ({pop})", x=0.02, y=0.995, ha="left", fontsize=7, weight="bold")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"ccgp_cue_pub_{pop}.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("111")]); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    pd.set_option("display.width", 260)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "interaction_p_perm", "R+ L vs E p_MWU",
              "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print(BS[["area", "n_boot", "R+ learning", "R+ expert", "R- learning", "R- expert", "R+ E-D0", "R+ E-D0 p", "E R+-R-", "E R+-R- p", "interaction", "interaction p"]].round(3).to_string(index=False))


def main(a):
    if not a.replot:
        compute(a.n_proc, a.pop)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
