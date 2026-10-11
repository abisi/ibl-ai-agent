"""112 -- Cross-modal CCGP of the cued-lick code, publication figure v2 (user 2026-10-11): equalised pseudo-population (10 sessions x 20
neurons per group, mouse -> session -> trials -> neurons), process panels (schematic + definition, example decoders, within-modality
accuracies, per-cohort time courses with the linear-shift control, both decoding directions), and the relation to the reward-lick CD on
the same and on independent trials. 111 kept.
CCGP (cross-condition generalisation performance; Bernardi et al. 2020, Cell): a decoder of one variable trained in one condition is tested
in another; here the variable is cued vs spontaneous lick and the condition is the cue modality: train whisker hits (WH) vs spontaneous
licks (SL-A) -> test auditory hits (AH) vs other spontaneous licks (SL-B), and the reverse; CCGP = mean balanced accuracy of the two.
CCGP above chance = a single read-out of "cued lick" shared by the two modalities (an abstract, modality-invariant code).
Data: 105 lick-aligned rates (stable tested units; baseline-corrected spikes/s); window = mean of the 50-ms bins centred in [-100, 0) ms;
events in time order. Decoder: logistic regression (L2, C = 0.05, balanced class weights), units z-scored on the training events.
Linear-shift null (as 064 / 111): labels keep their time order, activity taken from event i + k (|k| in [5, n // 3], random sign, no wrap),
decoders retrained; reported value - null.
Per session (whole brain and area groups >= 10 units): CCGP (and the two directions), within-modality CV accuracies (WH vs SL, AH vs SL),
parallelism PS = cos(WH - SL-A, AH - SL-B); N_SHIFT shifts. Time course: CCGP and its shift null per 50-ms bin (whole brain).
Independent halves: WH, AH and SL events split at random in halves 1 / 2; PS from half 1, CCGP from half 2 (no shared events).
Pseudo-population (equalised), per cohort x stage group and region: sessions with >= N_NEUR units of the region; groups with >= MIN_MICE
such mice; per iteration: N_SESS draws of (mouse with replacement -> one of its sessions) -> per drawn session random SL halves and N_PS
pseudo-trials per class (same event indices for all its neurons) -> N_NEUR neurons without replacement; real and linear-shift null with the
same neurons; N_SPLIT trial draws averaged; B iterations. NOTE: B and N_SPLIT are pilot values.
Output: across_days/<ref>/ccgp_pub2/{sessions.parquet, timecourse.parquet, examples.parquet, pseudopop_<pop>.parquet, *_stats_<pop>.csv,
ccgp_cue_pub2_<pop>.{png,pdf,svg}}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../112_ccgp_cue_pub2.py [--n-proc 25] [--replot] [--pop learners]
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
m111 = importlib.import_module("111_ccgp_cue_pub")
m105, m51 = m111.m105, m111.m51
OUT = m51.OUTROOT / "ccgp_pub2"
GROUPS = m111.GROUPS
MIN_UNITS, MIN_EV, N_SHIFT, N_SHIFT_TC, C_REG = 10, 6, 20, 10, 0.05
N_SESS, N_NEUR, N_PS, N_SPLIT, B, MIN_MICE = 10, 20, 50, 5, 200, 3


def fit(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    mu, sd = m111.zfit(Xtr)
    clf = LogisticRegression(C=C_REG, max_iter=1000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    return lambda X: clf.decision_function((X - mu) / sd)


def bacc(dv, y):
    p = dv > 0
    return 0.5 * ((p[y == 1]).mean() + (~p[y == 0]).mean())


def ccgp_full(WH, AH, SA, SB, keep_dv=False):
    y = lambda a, b: np.r_[np.ones(len(a)), np.zeros(len(b))]
    f1 = fit(np.vstack([WH, SA]), y(WH, SA)); d1 = f1(np.vstack([AH, SB])); c1 = bacc(d1, y(AH, SB))
    f2 = fit(np.vstack([AH, SA]), y(AH, SA)); d2 = f2(np.vstack([WH, SB])); c2 = bacc(d2, y(WH, SB))
    mu, sd = m111.zfit(np.vstack([WH, AH, SA, SB])); z = lambda A: (A - mu) / sd
    v1, v2 = z(WH).mean(0) - z(SA).mean(0), z(AH).mean(0) - z(SB).mean(0)
    ps = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
    out = dict(CCGP=0.5 * (c1 + c2), dir_WtoA=c1, dir_AtoW=c2, PS=ps)
    if keep_dv:
        out.update(dv_AH=d1[: len(AH)].tolist(), dv_SB=d1[len(AH):].tolist())
    return out


def cv_within(X, a, b, seed):
    from sklearn.model_selection import StratifiedKFold
    idx = np.r_[a, b]; y = np.r_[np.ones(len(a)), np.zeros(len(b))]; acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y):
        f = fit(X[:, idx[tr]].T, y[tr]); acc.append(bacc(f(X[:, idx[te]].T), y[te]))
    return float(np.mean(acc))


def run_one(X, I, seed, keep_dv=False):
    if min(len(I["WH"]), len(I["AH"]), len(I["SA"]), len(I["SB"])) < MIN_EV:
        return None
    r = ccgp_full(X[:, I["WH"]].T, X[:, I["AH"]].T, X[:, I["SA"]].T, X[:, I["SB"]].T, keep_dv)
    sl = np.r_[I["SA"], I["SB"]]
    r["within_WH"], r["within_AH"] = cv_within(X, I["WH"], sl, seed), cv_within(X, I["AH"], sl, seed)
    return r


# ------------------------------------------------------------------ per session
def session(args):
    f, = args
    sid = f.stem; S = m111.load_session(f); idx = S["idx"]
    if min(len(idx["WH"]), len(idx["AH"]), len(idx["SL"]) // 2) < MIN_EV:
        return [], [], []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|112".encode())); seed = int(rng.integers(1e6))
    p = rng.permutation(idx["SL"]); base = dict(WH=idx["WH"], AH=idx["AH"], SA=np.sort(p[: len(p) // 2]), SB=np.sort(p[len(p) // 2:]))
    shifts = [m111.shifted(base, S["n"], rng) for _ in range(N_SHIFT)]
    rows, ex = [], []
    groups = [("Whole brain", np.ones(len(S["area"]), bool))] + [(g, S["area"] == g) for g in sorted(set(S["area"])) if g not in ("nan", "None")]
    for g, m in groups:
        if m.sum() < MIN_UNITS:
            continue
        X = S["X"][m]; r = run_one(X, base, seed, keep_dv=(g == "Whole brain"))
        if r is None:
            continue
        nul = [q for q in (run_one(X, I, seed) for I in shifts) if q is not None]
        row = dict(session_id=sid, area=g, n_units=int(m.sum()))
        for k in ("CCGP", "dir_WtoA", "dir_AtoW", "PS", "within_WH", "within_AH"):
            row[k] = r[k]; row[f"{k}_null"] = float(np.mean([q[k] for q in nul])) if nul else np.nan
        if g == "Whole brain":
            ex.append(dict(session_id=sid, dv_AH=r["dv_AH"], dv_SB=r["dv_SB"],
                           dv_AH_null=[], dv_SB_null=[]))
            # independent halves: PS from half 1, CCGP from half 2
            h = {}
            for c in ("WH", "AH", "SL"):
                q = rng.permutation(idx[c]); h[c] = (np.sort(q[: len(q) // 2]), np.sort(q[len(q) // 2:]))
            def halfset(j):
                s_ = h["SL"][j]; return dict(WH=h["WH"][j], AH=h["AH"][j], SA=s_[: len(s_) // 2], SB=s_[len(s_) // 2:])
            I1, I2 = halfset(0), halfset(1)
            if min(min(len(v) for v in I1.values()), min(len(v) for v in I2.values())) >= 3:
                r1 = ccgp_full(X[:, I1["WH"]].T, X[:, I1["AH"]].T, X[:, I1["SA"]].T, X[:, I1["SB"]].T)
                r2 = ccgp_full(X[:, I2["WH"]].T, X[:, I2["AH"]].T, X[:, I2["SA"]].T, X[:, I2["SB"]].T)
                row["PS_h1"], row["CCGP_h2"] = r1["PS"], r2["CCGP"]
        rows.append(row)
    tc = []
    for k, cc in enumerate(S["c"]):
        Xk = S["L"][:, :, k]
        if min(len(v) for v in base.values()) < MIN_EV:
            break
        r = ccgp_full(Xk[:, base["WH"]].T, Xk[:, base["AH"]].T, Xk[:, base["SA"]].T, Xk[:, base["SB"]].T)["CCGP"]
        nl = [ccgp_full(Xk[:, I["WH"]].T, Xk[:, I["AH"]].T, Xk[:, I["SA"]].T, Xk[:, I["SB"]].T)["CCGP"] for I in shifts[:N_SHIFT_TC]
              if min(len(v) for v in I.values()) >= MIN_EV]
        tc.append(dict(session_id=sid, t=float(cc * 1e3), CCGP=r, CCGP_null=float(np.mean(nl)) if nl else np.nan))
    return rows, tc, ex


# ------------------------------------------------------------------ equalised pseudo-population
DATA = {}


def pp_iter(tree, region, rng):
    mice = list(tree)
    draws = [(m, rng.choice(tree[m])) for m in rng.choice(mice, N_SESS)]
    neur = {}
    for j, (_, s) in enumerate(draws):
        u = np.where(DATA[s]["area"] == region)[0] if region != "Whole brain" else np.arange(len(DATA[s]["area"]))
        neur[j] = rng.choice(u, N_NEUR, replace=False)
    out = []
    for null in (False, True):
        acc = []
        for _ in range(N_SPLIT):
            P = {c: np.zeros((N_PS, N_SESS * N_NEUR), np.float32) for c in ("WH", "AH", "SA", "SB")}
            for j, (_, s) in enumerate(draws):
                S = DATA[s]; idx = S["idx"]
                pp = rng.permutation(idx["SL"]); I = dict(WH=idx["WH"], AH=idx["AH"], SA=pp[: len(pp) // 2], SB=pp[len(pp) // 2:])
                if null:
                    J = m111.shifted(I, S["n"], rng); I = {c: (J[c] if len(J[c]) >= 2 else I[c]) for c in I}
                tr = {c: rng.choice(I[c], N_PS) for c in P}                    # trials first, then the session's sampled neurons
                for c in P:
                    P[c][:, j * N_NEUR:(j + 1) * N_NEUR] = S["X"][np.ix_(neur[j], tr[c])].T
            acc.append(ccgp_full(P["WH"], P["AH"], P["SA"], P["SB"]))
        out.append({k: np.mean([a[k] for a in acc]) for k in ("CCGP", "PS", "dir_WtoA", "dir_AtoW")})
    return out


def pp_job(args):
    g, region, tree, seed = args
    rng = np.random.default_rng(seed); res = []
    for b in range(B):
        r, n = pp_iter(tree, region, rng)
        res.append(dict(cohort=g[0], stage=g[1], area=region, it=b, **r, **{f"{k}_null": v for k, v in n.items()},
                        n_mice=len(tree), n_sessions=sum(len(v) for v in tree.values())))
    return res


def compute(n_proc, pop):
    OUT.mkdir(parents=True, exist_ok=True)
    m61 = importlib.import_module("061_roc_prelick_learners")
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(f,) for f in files])
    pd.DataFrame([x for r in res for x in r[0]]).merge(meta, on="session_id").to_parquet(OUT / "sessions.parquet", index=False)
    pd.DataFrame([x for r in res for x in r[1]]).merge(meta, on="session_id").to_parquet(OUT / "timecourse.parquet", index=False)
    pd.DataFrame([x for r in res for x in r[2]]).merge(meta, on="session_id").to_parquet(OUT / "examples.parquet", index=False)
    print("per-session done", flush=True)
    mt = m61.learner_filter(meta) if pop == "learners" else meta
    for r in mt.itertuples():
        f = m105.OUT / f"{r.session_id}.npz"
        if f.exists():
            S = m111.load_session(f)
            if min(len(S["idx"]["WH"]), len(S["idx"]["AH"]), len(S["idx"]["SL"]) // 2) >= MIN_EV:
                DATA[r.session_id] = S
    mt = mt[mt.session_id.isin(DATA)]
    areas = ["Whole brain"] + sorted({a for s in DATA for a in set(DATA[s]["area"]) if a not in ("nan", "None")})
    jobs, kept = [], []
    for region in areas:
        trees = {}
        for g in GROUPS:
            d = mt[(mt.cohort == g[0]) & (mt.stage == g[1])]
            ok = [r for r in d.itertuples() if ((DATA[r.session_id]["area"] == region).sum() if region != "Whole brain" else len(DATA[r.session_id]["area"])) >= N_NEUR]
            tree = {}
            for r in ok:
                tree.setdefault(r.mouse_id, []).append(r.session_id)
            trees[g] = tree
        if all(len(t) >= MIN_MICE for t in trees.values()):
            kept.append(region)
            jobs += [(g, region, trees[g], zlib.crc32(f"{g}|{region}|112".encode())) for g in GROUPS]
    with mp.get_context("fork").Pool(n_proc) as pool:
        P = [x for r in pool.imap_unordered(pp_job, jobs) for x in r]
    pd.DataFrame(P).to_parquet(OUT / f"pseudopop_{pop}.parquet", index=False)
    print("pseudo-population done:", kept, flush=True)


# ------------------------------------------------------------------ figure
def schematic(ax_tr, ax_te):
    rng = np.random.default_rng(3)
    cl = {"WH": "#f7b519", "AH": "#2c2cdb", "SL": "#262626"}
    pts = {"SL": rng.normal([-1.0, -0.3], 0.35, (25, 2)), "WH": rng.normal([1.0, 0.6], 0.35, (25, 2)), "AH": rng.normal([1.1, -0.5], 0.35, (25, 2))}
    for ax, show, ttl in ((ax_tr, ("WH", "SL"), "train: WH vs SL-A"), (ax_te, ("AH", "SL"), "test: AH vs SL-B")):
        for k in ("WH", "AH", "SL"):
            a = 1.0 if k in show else 0.12
            ax.plot(*pts[k].T, "o", ms=2.2, color=cl[k], alpha=a, mew=0)
        ax.plot([0.15, -0.05], [-1.6, 1.6], color="#d94801", lw=1.0)                  # decision boundary learned on WH vs SL
        ax.text(0.3, 1.45, "decoder", color="#d94801", fontsize=4.6)
        ax.set_xlim(-2.2, 2.4); ax.set_ylim(-1.7, 1.8); ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("neuron 1", fontsize=4.6); ax.set_ylabel("neuron 2", fontsize=4.6); ax.set_title(ttl, fontsize=5)
        for k in show:
            m = pts[k].mean(0); ax.text(m[0], m[1] + 0.55, k if k != "SL" else "SL", color=cl[k], fontsize=5, ha="center", weight="bold")


def boot_stats(P):
    P = P.assign(ex=P.CCGP - P.CCGP_null)
    rows = []
    for a, d in P.groupby("area"):
        g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])].sort_values("it").ex.to_numpy() for k in GROUPS}
        n = min(len(v) for v in g.values()); g = {k: v[:n] for k, v in g.items()}
        ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)); bp = lambda v: min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        r = dict(area=a, n_boot=n)
        for k, v in g.items():
            r[f"{k[0]} {k[1]}"] = np.median(v); r[f"{k[0]} {k[1]} lo"], r[f"{k[0]} {k[1]} hi"] = ci(v); r[f"{k[0]} {k[1]} p"] = bp(v)
        for name, v in {"R+ E-D0": g[GROUPS[1]] - g[GROUPS[0]], "R- E-D0": g[GROUPS[3]] - g[GROUPS[2]], "D0 R+-R-": g[GROUPS[0]] - g[GROUPS[2]],
                        "E R+-R-": g[GROUPS[1]] - g[GROUPS[3]], "interaction": (g[GROUPS[1]] - g[GROUPS[0]]) - (g[GROUPS[3]] - g[GROUPS[2]])}.items():
            r[name] = np.median(v); r[f"{name} lo"], r[f"{name} hi"] = ci(v); r[f"{name} p"] = bp(v)
        rows.append(r)
    return pd.DataFrame(rows)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH = m88.COH
    import ephys_utilities.allen_utils.allen_utils as au
    import statsmodels.formula.api as smf
    from scipy import stats
    gcol = au.get_custom_area_groups_colors(); plt = m62.setup(); rng = np.random.default_rng(0)
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    A = flt(pd.read_parquet(OUT / "sessions.parquet"))
    for k in ("CCGP", "dir_WtoA", "dir_AtoW", "PS", "within_WH", "within_AH"):
        A[f"{k}_ex"] = A[k] - A[f"{k}_null"]
    W = A[A.area == "Whole brain"].copy()
    T = flt(pd.read_parquet(OUT / "timecourse.parquet")); E = flt(pd.read_parquet(OUT / "examples.parquet"))
    # per-area / equalised results: single-session decoding with 20-unit draws (114; user 2026-10-11 "keep single-session decoding"),
    # replacing the equalised pseudo-population of the first version of this figure (pseudo-population code kept above, not plotted)
    m114 = importlib.import_module("114_ccgp_cue_sessions_equalised")
    Q = flt(pd.read_parquet(m114.OUT / "sessions.parquet")); QS = m114.stats_table(Q, "ex", m62, rng)
    H = flt(pd.read_csv(m51.OUTROOT / "harmonised" / "sessions_areas.csv")); H = H[H.area == "whole_brain"][["session_id", "theta"]]
    W = W.merge(H, on="session_id", how="left"); W["cos_theta"] = np.cos(np.radians(W.theta))
    with plt.rc_context({"font.size": 5.2, "axes.labelsize": 5.2, "xtick.labelsize": 4.7, "ytick.labelsize": 4.7}):
        fig = plt.figure(figsize=(7.2, 7.6))
        gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1.35], hspace=0.85, wspace=0.85, left=0.07, right=0.985, top=0.94, bottom=0.05)
        L = []
        # ---- row 1: schematic + definition, example sessions, within accuracies
        sa = gs[0, 0:2].subgridspec(1, 2, wspace=0.35); a1, a2 = fig.add_subplot(sa[0]), fig.add_subplot(sa[1]); schematic(a1, a2); L.append(a1)
        a1.text(0.0, -0.3, "CCGP: train a cued- vs spontaneous-lick decoder with one cue modality,\ntest it with the other (and the reverse); "
                "CCGP = mean balanced accuracy;\nabove chance = a modality-invariant code", fontsize=4.3, color="0.25", va="top", transform=a1.transAxes)
        for j, coh in enumerate(("R+", "R-")):
            ax = fig.add_subplot(gs[0, 2 + j]); ax.set_box_aspect(1); L.append(ax)
            d = W[(W.cohort == coh) & (W.stage == "expert")]
            sid = d.iloc[(d.CCGP_ex - d.CCGP_ex.median()).abs().argsort().iloc[0]].session_id
            e = E[E.session_id == sid].iloc[0]
            bins = np.linspace(min(min(e.dv_AH), min(e.dv_SB)), max(max(e.dv_AH), max(e.dv_SB)), 20)
            ax.hist(e.dv_SB, bins, color="0.25", alpha=0.6, label="SL-B (test)"); ax.hist(e.dv_AH, bins, color="#2c2cdb", alpha=0.6, label="AH (test)")
            ax.axvline(0, color="#d94801", lw=0.8)
            ax.set_xlabel("decoder output (trained WH vs SL-A)"); ax.set_ylabel("events")
            r = W[W.session_id == sid].iloc[0]
            ax.set_title(f"{coh.replace('-', '−')} expert example\nWH→AH bal. acc. {r.dir_WtoA:.2f} (null {r.dir_WtoA_null:.2f})", fontsize=5, color=COH[coh])
            if j == 0:
                ax.legend(frameon=False, fontsize=4, loc="upper left")
        ax = fig.add_subplot(gs[0, 4]); ax.set_box_aspect(1); L.append(ax)
        WW = pd.concat([W.assign(v=W.within_WH_ex, k="WH vs SL"), W.assign(v=W.within_AH_ex, k="AH vs SL")])
        for i, (k, mk) in enumerate((("WH vs SL", "o"), ("AH vs SL", "s"))):
            for (c, s), x in zip(GROUPS, (0, 1, 2.4, 3.4)):
                v = WW[(WW.k == k) & (WW.cohort == c) & (WW.stage == s)].v
                ax.errorbar(x + (i - 0.5) * 0.25, v.mean(), v.sem(), fmt=mk, ms=3, color=COH[c], mfc=COH[c] if s == "expert" else "white", capsize=0, lw=0.8)
        ax.set_xticks([0, 1, 2.4, 3.4], ["D0", "E", "D0", "E"]); ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_ylabel("CV bal. acc. − shift null"); ax.set_title("Within modality\n(● WH vs SL, ■ AH vs SL)", fontsize=5)
        # ---- row 2: time courses per cohort with shift control, CCGP, directions
        for j, coh in enumerate(("R+", "R-")):
            ax = fig.add_subplot(gs[1, j]); ax.set_box_aspect(1); L.append(ax)
            for st, ls in (("learning", (0, (2.5, 1.5))), ("expert", "-")):
                d = T[(T.cohort == coh) & (T.stage == st)]
                for col, cc, lab in (("CCGP", COH[coh], "real"), ("CCGP_null", "0.55", "shift control")):
                    piv = d.pivot_table(index="session_id", columns="t", values=col); mu, se = piv.mean(0), piv.sem(0)
                    ax.fill_between(piv.columns, mu - se, mu + se, color=cc, alpha=0.13, lw=0)
                    ax.plot(piv.columns, mu, color=cc, ls=ls, lw=0.9, label=f"{lab}, {'D0' if st == 'learning' else 'E'} ({piv.shape[0]})")
            ax.axhline(0.5, color="0.7", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.4, ls=":"); ax.axvspan(-100, 0, color="#FDD49E", alpha=0.4, lw=0, zorder=0)
            ax.set_xlabel("time from lick (ms)"); ax.set_ylabel("CCGP (balanced accuracy)"); ax.legend(frameon=False, fontsize=3.6, loc="upper left", handlelength=1.3)
            ax.set_title(f"{coh.replace('-', '−')}: real vs linear-shift control", fontsize=5, color=COH[coh])
        ax = fig.add_subplot(gs[1, 2]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, W, "CCGP_ex", "CCGP − shift null", rng, "112 CCGP", "Cue CCGP (−100–0 ms)", ref=0)
        for j, (col, ttl) in enumerate((("dir_WtoA_ex", "Train WH → test AH"), ("dir_AtoW_ex", "Train AH → test WH"))):
            ax = fig.add_subplot(gs[1, 3 + j]); ax.set_box_aspect(1); L.append(ax)
            m88.mean_panel2(ax, W, col, "bal. acc. − shift null", rng, f"112 {col}", ttl, ref=0)
        # ---- row 3: parallelism, relation to the CD (same trials, independent halves), pseudo-pop whole brain
        ax = fig.add_subplot(gs[2, 0]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, W, "PS_ex", "cosine − shift null", rng, "112 PS", "Parallelism\ncos(WH−SL, AH−SL)", ref=0)

        def scat(ax, d, x, y, xl, yl, ttl):
            d = d.dropna(subset=[x, y])
            for (c, s) in GROUPS:
                q = d[(d.cohort == c) & (d.stage == s)]
                ax.plot(q[x], q[y], "o", ms=2.4, color=COH[c], mfc=COH[c] if s == "expert" else "white", mew=0.6, ls="")
            mdl = smf.ols(f"{y} ~ {x}", d).fit(); xx = np.linspace(d[x].min(), d[x].max(), 50)
            pr = mdl.get_prediction(pd.DataFrame({x: xx})).summary_frame(alpha=0.05); rs, ps_ = stats.spearmanr(d[x], d[y])
            ax.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color="0.5", alpha=0.2, lw=0)
            ax.plot(xx, pr["mean"], color="0.2", lw=0.9, ls="-" if mdl.pvalues[x] < 0.05 else "--")
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{ttl}\nSpearman r = {rs:.2f}, {m62.fmt_p(ps_)} (n = {len(d)})", fontsize=5)
        ax = fig.add_subplot(gs[2, 1]); ax.set_box_aspect(1); L.append(ax)
        scat(ax, W, "cos_theta", "CCGP_ex", "cos θ (WH − SL vs CD; 087)", "CCGP − shift null", "Same trials")
        ax = fig.add_subplot(gs[2, 2]); ax.set_box_aspect(1); L.append(ax)
        scat(ax, W, "PS_h1", "CCGP_h2", "parallelism, trial half 1", "CCGP, trial half 2", "Independent trial halves")
        Qw = Q[Q.area == "Whole brain"]
        ax = fig.add_subplot(gs[2, 3]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, Qw, "ex", "CCGP − shift null", rng, "112 eq CCGP", f"Equalised: {m114.N_SUB} units" + chr(10) + f"× {m114.R_SUB} draws per session", ref=0)
        ax = fig.add_subplot(gs[2, 4]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, Qw, "psx", "cosine − shift null", rng, "112 eq PS", f"Parallelism, {m114.N_SUB} units", ref=0)
        # ---- row 4: per area, single sessions with equalised unit count (114)
        S4 = QS.set_index("area")
        order = ["Whole brain"] + S4.drop("Whole brain").sort_values("R+ expert", ascending=False).index.tolist()
        y = np.arange(len(order))[::-1]
        ax = fig.add_subplot(gs[3, 0:3]); L.append(ax)
        off = {GROUPS[0]: 0.27, GROUPS[1]: 0.09, GROUPS[2]: -0.09, GROUPS[3]: -0.27}
        for i, a in enumerate(order):
            r = S4.loc[a]
            for k in GROUPS:
                key = f"{k[0]} {k[1]}"
                if r[f"n {key}"] >= m114.MIN_SESS:
                    ax.errorbar(r[key], y[i] + off[k], xerr=r[f"sem {key}"], fmt="o", ms=2.6, color=COH[k[0]],
                                mfc=COH[k[0]] if k[1] == "expert" else "white", lw=0.8, capsize=0, mew=0.6)
        ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([m88.m83.m62_short(a) for a in order], fontsize=4.8)
        for tl, a in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(a, "k")); tl.set_weight("bold" if a == "Whole brain" else "normal")
        ax.set_xlabel(f"CCGP − shift null (session mean ± SEM; {m114.N_SUB} units × {m114.R_SUB} draws)")
        ax.set_title("Areas ranked by generalisation (R+ expert; groups with ≥ 3 sessions)", fontsize=5.2)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], marker="o", ls="", color=COH[c], mfc=COH[c] if s == "expert" else "white", ms=3) for c, s in GROUPS],
                  ["R+ day 0", "R+ expert", "R− day 0", "R− expert"], frameon=False, fontsize=4.2, loc="lower right", handletextpad=0.2, ncol=2)
        for j, (ttl, a_, b_, test, col) in enumerate((("R+: expert − day 0", "R+ expert", "R+ learning", "D0vsE R+", COH["R+"]),
                                                      ("Expert: R+ − R−", "R+ expert", "R- expert", "RpvsRm expert", "0.2"))):
            ax = fig.add_subplot(gs[3, 3 + j]); L.append(ax)
            for i, a in enumerate(order):
                r = S4.loc[a]; p_ = r.get(f"p_MWU {test}", np.nan)
                if not np.isfinite(p_):
                    continue
                q_ = p_ if a == "Whole brain" else r.get(f"q_MWU {test}", np.nan)
                dv, se = r[a_] - r[b_], np.hypot(r[f"sem {a_}"], r[f"sem {b_}"])
                ax.errorbar(dv, y[i], xerr=se, fmt="o", ms=2.6, color=col, mfc=col if q_ < 0.05 else "white", lw=0.8, capsize=0, mew=0.6)
                if p_ < 0.05 and not q_ < 0.05:
                    ax.text(dv + se, y[i], " †", fontsize=5, va="center", color=col)
            ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5); ax.set_ylim(L[-1 - j - 1].get_ylim() if j else L[-2].get_ylim())
            ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel("Δ CCGP − shift null (± SEM)")
            ax.set_title(f"{ttl}" + chr(10) + "filled q < 0.05 (BH), † p < 0.05", fontsize=5)
        m62.letter_row(fig, L[:4], "abcd", dx_in=0.26, dy_in=0.14); m62.letter_row(fig, L[4:9], "efghi", dx_in=0.26, dy_in=0.14)
        m62.letter_row(fig, L[9:14], "jklmn", dx_in=0.26, dy_in=0.14); m62.letter_row(fig, L[14:], "opq", dx_in=0.26, dy_in=0.14)
        fig.suptitle(f"Whisker licks join the auditory reward-lick code: cross-modal generalisation ({pop})", x=0.02, y=0.997, ha="left", fontsize=7, weight="bold")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"ccgp_cue_pub2_{pop}.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("112")]); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    pd.set_option("display.width", 260)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "interaction_p_perm", "R+ L vs E p_MWU",
              "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print(QS[["area", "R+ learning", "R+ expert", "R- learning", "R- expert", "p_MWU D0vsE R+", "q_MWU D0vsE R+", "p_MWU RpvsRm expert",
              "p_Welch RpvsRm expert", "q_MWU RpvsRm expert", "p_int"]].round(3).to_string(index=False))


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
