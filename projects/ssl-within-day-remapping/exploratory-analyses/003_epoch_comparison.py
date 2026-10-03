"""Within-day and across-day changes of whisker-hit (WH) activity on one common footing.

Every session (day 0 and expert) is cut into the same phases (early / late halves, split at the midpoint between the two
middle auditory hits). Every epoch is estimated from the same fixed number of events per class (N_FIX WH, AH and SL,
random subsamples, N_SUB repeats) and the same number of units (N_UNITS random good + mua units per session), so that
all estimates share one noise level. Measures are anchored on each session's own spontaneous-lick (SL = 0) and
auditory-hit (AH = 1) activity, which removes most differences in neuron sampling between sessions:
  pos   WH - SL position on the session's cross-validated SL -> AH mean-difference axis (unit-norm axis per fold, one
        normalisation per session: pooled held-out SL mean = 0, AH mean = 1; d' >= 0.3), epoch means of the trial
        scores (WH fold-averaged, AH / SL held-out). Within an epoch WH is referenced to SL of the same epoch, so the
        session-wide drift of all classes along the axis cancels.
  dec   whole-session decoder (L2 logistic regression, C = 0.05, 5-fold CV over the session's AH and SL; WH never in
        training), read out per epoch: P(AH | WH) - P(AH | SL), minus the median of the same epoch readout under a
        whole-session linear-shift null (40 shifts). Epoch-trained decoders are not used: N_FIX events per class are too
        few to fit one.
  ddn   cross-validated distance difference of the epoch, d(WH, SL) - d(WH, AH) (057 estimator on the epoch's
        subsampled events), divided by the session's d(AH, SL) (all events).
Epochs: L-early, L-late (day 0), E-early, E-late (expert sessions). Contrasts (group means; sessions are not paired
across days):
  within-day   L-late - L-early
  across-day   E-early - L-early and E-late - L-late (phase-matched: the within-session drift cancels)
  carry-over   E-early - L-late (does the expert session start where day 0 ended?)
  remaining    E-late - L-late
Uncertainty: hierarchical bootstrap per cohort (mice with replacement, then each mouse's sessions with replacement),
2000 iterations, 95% percentile CI. Cohort difference of each contrast: permutation of cohort labels across mice (5000).
Mixed model (statsmodels MixedLM): value ~ C(cohort) * C(epoch), random intercept per mouse and variance component
per session; contrasts as Wald tests. Populations: all mice and learners.
Output: combined_results_ks4/_within_day<TAG>/epochs/
"""
import argparse
import importlib
import json
import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m57 = importlib.import_module("057_roc_prelick_lambda")
m001 = importlib.import_module("001_within_session_halves")
m002 = importlib.import_module("002_trial_slopes")
OUT = m51.RES / f"_within_day{m51.TAG}" / "epochs"
UNIT_SET = ("good", "mua")
N_FIX, N_SUB, N_UNITS, MIN_UNITS, K_SHIFT = 6, 20, 150, 30, 40
MEAS = [("pos", "WH − SL position on the session's\nSL → AH axis (0 = SL, 1 = AH)"),
        ("dec", "P(AH | WH) − P(AH | SL) − chance\n(whole-session decoder)"),
        ("ddn", "Δd / d(AH, SL) of the session")]
EPOCHS = ["L-early", "L-late", "E-early", "E-late"]
CONTRASTS = {"within-day": ("L-late", "L-early"), "across-day (early)": ("E-early", "L-early"),
             "across-day (late)": ("E-late", "L-late"), "carry-over": ("E-early", "L-late")}


def epoch_values(Xu, lab, ep_idx, S, P0, rng):
    """per epoch (fixed N_FIX per class, N_SUB subsamples): pos, dec (raw and null-corrected), ddn numerator"""
    out = {}
    for ep, idx_all in ep_idx.items():
        cls_idx = {c: idx_all[lab[idx_all] == c] for c in m51.CLASSES}
        if min(len(v) for v in cls_idx.values()) < N_FIX:
            return None
        acc = {k: [] for k in ("pos", "dec", "dec0", "dd")}
        for _ in range(N_SUB):
            sub = {c: rng.choice(v, N_FIX, replace=False) for c, v in cls_idx.items()}
            if np.isfinite(S["md"]).any():
                acc["pos"].append(np.nanmean(S["md"][sub["WH"]]) - np.nanmean(S["md"][sub["FA"]]))
            acc["dec"].append(np.nanmean(S["dec"][sub["WH"]]) - np.nanmean(S["dec"][sub["FA"]]))
            if P0 is not None:
                acc["dec0"].append(np.nanmedian([np.nanmean(p[sub["WH"]]) - np.nanmean(p[sub["FA"]]) for p in P0]))
            ii = np.sort(np.concatenate(list(sub.values())))
            r = m57.lam(Xu[ii].T, lab[ii], rng)
            if r:
                acc["dd"].append(r["d_WH_FA"] - r["d_WH_AH"])
        out[ep] = {k: (np.nanmean(v) if len(v) else np.nan) for k, v in acc.items()}
    return out


def session(args):
    sid, f, meta, W_s, seed = args
    rng = np.random.default_rng(seed)
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["trial_start"].astype(float)[o]
    if min((lab == c).sum() for c in m51.CLASSES) < 2 * N_FIX:     # both epochs need N_FIX of every class
        return None
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = np.where(K.quality_label.isin(UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR))[0]
    if len(ok) < MIN_UNITS:
        return None
    u = np.sort(rng.choice(ok, min(N_UNITS, len(ok)), replace=False))      # fixed unit budget per session
    Xu = X[u].T
    mu, sd = Xu.mean(0), Xu.std(0); good = sd > 0
    Xz = (Xu[:, good] - mu[good]) / sd[good]                               # z-scored over the session (for distances)
    sp = m001.split_halves(lab, t, "time")
    if sp is None:
        return None
    e, l = sp
    stage = "L" if meta["stage"] == "learning" else "E"
    ep_idx = {f"{stage}-early": np.where(e)[0], f"{stage}-late": np.where(l)[0]}
    S = m002.trial_scores(Xu, lab, seed)
    dprime = S.pop("dprime")
    # whole-session shift null for the decoder readout (activity shifted against the time-ordered labels)
    P0 = []
    for k in m001.shifts(len(lab))[:K_SHIFT]:
        Xs, ls, i = m001.shifted(Xu, lab, int(k))
        if min((ls == c).sum() for c in m51.CLASSES) < N_FIX:
            continue
        try:
            Ss = m002.trial_scores(Xs, ls, seed + int(k) + 1000)
        except ValueError:
            continue
        p = np.full(len(lab), np.nan); p[i] = Ss["dec"]; P0.append(p)
    V = epoch_values(Xz, lab, ep_idx, S, P0, rng)
    if V is None:
        return None
    r_all = m57.lam(Xz.T, lab, rng)
    dAF = r_all["d_AH_FA"] if r_all else np.nan
    rows = []
    for ep, v in V.items():
        rows.append(dict(meta, epoch=ep, n_units=len(u), dprime=dprime, d_AH_SL_session=dAF, pos=v["pos"],
                         dec_raw=v["dec"], dec=v["dec"] - v["dec0"] if np.isfinite(v["dec0"]) else np.nan,
                         dd=v["dd"], ddn=v["dd"] / dAF if dAF and dAF >= 0.01 else np.nan))
    return rows


# ------------------------------------------------------------------ statistics
def group_means(d, col):
    """epoch means: mean over each mouse's sessions, then over mice (mouse-level weighting, as the bootstrap)"""
    g = d.dropna(subset=[col]).groupby(["epoch", "mouse_id"])[col].mean().groupby("epoch").mean()
    return g.reindex(EPOCHS)


def contrasts_of(m):
    return {k: m[a] - m[b] for k, (a, b) in CONTRASTS.items()}


def hboot(d, col, B, rng):
    """hierarchical bootstrap: mice with replacement, then each sampled mouse's sessions with replacement"""
    d = d.dropna(subset=[col])
    by_mouse = {m: g for m, g in d.groupby("mouse_id")}
    mice = list(by_mouse)
    out = []
    for _ in range(B):
        parts = []
        for m in rng.choice(mice, len(mice), replace=True):
            g = by_mouse[m]; s = g.session_id.unique()
            pick = rng.choice(s, len(s), replace=True)
            parts.append(pd.concat([g[g.session_id == x] for x in pick]).assign(mouse_id=f"{m}_{len(parts)}"))
        out.append(group_means(pd.concat(parts), col))
    return pd.DataFrame(out)


def perm_cohort(D, col, n_perm, rng):
    """cohort difference (R+ - R-) of every contrast; null by permuting cohort labels across mice"""
    d = D.dropna(subset=[col])
    def diff(dd):
        a = contrasts_of(group_means(dd[dd.cohort == "R+"], col)); b = contrasts_of(group_means(dd[dd.cohort == "R-"], col))
        return {k: a[k] - b[k] for k in CONTRASTS}
    obs = diff(d)
    mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    mi = pd.Index(mice).get_indexer(d.mouse_id)
    null = {k: [] for k in CONTRASTS}
    for _ in range(n_perm):
        r = diff(d.assign(cohort=rng.permutation(mc)[mi]))
        for k in CONTRASTS:
            null[k].append(r[k])
    return {k: (obs[k], (1 + np.sum(np.abs(np.array(null[k])[np.isfinite(null[k])]) >= abs(obs[k]))) / (1 + len(null[k])))
            for k in CONTRASTS}


RHS = "C(cohort) * C(epoch, Treatment('L-early'))"


def mixed_model(D, col):
    """value ~ C(cohort) * C(epoch), random intercept per mouse, variance component per session; Wald contrasts"""
    import statsmodels.formula.api as smf
    d = D.dropna(subset=[col]).copy()
    try:
        fit = smf.mixedlm(f"{col} ~ {RHS}", d, groups="mouse_id",
                          vc_formula={"session": "0 + C(session_id)"}).fit(reml=True)
    except Exception as ex:
        return {"error": str(ex)}
    import patsy
    grid = pd.DataFrame([dict(cohort=c, epoch=e) for c in ("R+", "R-") for e in EPOCHS])
    Xg = patsy.dmatrix(RHS, grid, return_type="dataframe")
    Xg.index = [f"{c}|{e}" for c, e in zip(grid.cohort, grid.epoch)]
    res = {}
    fe = fit.fe_params.index
    for c in ("R+", "R-"):
        for k, (a, b) in CONTRASTS.items():
            L = (Xg.loc[f"{c}|{a}"] - Xg.loc[f"{c}|{b}"]).reindex(fe).fillna(0).to_numpy()
            tt = fit.t_test(L[None, :])
            res[f"{c} {k}"] = (float(tt.effect[0]), float(tt.pvalue))
    for k, (a, b) in CONTRASTS.items():
        L = ((Xg.loc[f"R+|{a}"] - Xg.loc[f"R+|{b}"]) - (Xg.loc[f"R-|{a}"] - Xg.loc[f"R-|{b}"])).reindex(fe).fillna(0).to_numpy()
        tt = fit.t_test(L[None, :])
        res[f"R+ - R- {k}"] = (float(tt.effect[0]), float(tt.pvalue))
    return res


def analyse(D, out, pop, B, n_perm):
    rng = np.random.default_rng(0)
    rows, boots = [], {}
    for col, _ in MEAS:
        for c in ("R+", "R-"):
            d = D[D.cohort == c]
            m = group_means(d, col); bt = hboot(d, col, B, rng); boots[(col, c)] = (m, bt)
            for ep in EPOCHS:
                v = bt[ep].dropna()
                rows.append(dict(measure=col, cohort=c, kind="epoch", name=ep, value=m[ep],
                                 lo=np.percentile(v, 2.5) if len(v) else np.nan, hi=np.percentile(v, 97.5) if len(v) else np.nan,
                                 n_sessions=int(d[d.epoch == ep][col].notna().sum()),
                                 n_mice=int(d[d.epoch == ep].dropna(subset=[col]).mouse_id.nunique())))
            cm = contrasts_of(m)
            for k, (a, b) in CONTRASTS.items():
                v = (bt[a] - bt[b]).dropna()
                rows.append(dict(measure=col, cohort=c, kind="contrast", name=k, value=cm[k],
                                 lo=np.percentile(v, 2.5) if len(v) else np.nan, hi=np.percentile(v, 97.5) if len(v) else np.nan,
                                 p_boot=min(1, 2 * min((v <= 0).mean(), (v >= 0).mean())) if len(v) else np.nan))
        pc = perm_cohort(D, col, n_perm, rng)
        for k, (v, p) in pc.items():
            rows.append(dict(measure=col, cohort="R+ - R-", kind="contrast", name=k, value=v, p_perm=p))
        mm = mixed_model(D, col)
        for k, v in mm.items():
            if k != "error":
                rows.append(dict(measure=col, cohort="mixed model", kind="contrast", name=k, value=v[0], p_mixed=v[1]))
    T = pd.DataFrame(rows); T.to_csv(out / "epoch_contrasts.csv", index=False)
    figure(T, out, pop)
    return T


def figure(T, out, pop):
    m62 = importlib.import_module("062_pub_convergence_figures")
    plt = m62.setup()
    COH = m62.COH
    fig, axs = plt.subplots(2, len(MEAS), figsize=(m62.W_IN, 5.0), gridspec_kw=dict(hspace=0.75, wspace=0.55))
    xe = {"L-early": 0, "L-late": 1, "E-early": 2.6, "E-late": 3.6}
    for j, (col, yl) in enumerate(MEAS):
        ax = axs[0, j]
        for k_, c in enumerate(("R+", "R-")):
            q = T[(T.measure == col) & (T.cohort == c) & (T.kind == "epoch")].set_index("name").reindex(EPOCHS)
            dx = (k_ - 0.5) * 0.18
            for seg in (("L-early", "L-late"), ("E-early", "E-late")):
                ax.plot([xe[s] + dx for s in seg], q.loc[list(seg), "value"], color=COH[c], lw=1.1)
            ax.plot([xe["L-late"] + dx, xe["E-early"] + dx], q.loc[["L-late", "E-early"], "value"], color=COH[c], lw=0.7,
                    ls=(0, (2, 2)))
            for ep in EPOCHS:
                ax.errorbar(xe[ep] + dx, q.loc[ep, "value"], [[q.loc[ep, "value"] - q.loc[ep, "lo"]], [q.loc[ep, "hi"] - q.loc[ep, "value"]]],
                            fmt="o", ms=3.5, color=COH[c], mfc="white" if ep.startswith("L") else COH[c], lw=0.9, capsize=0,
                            label=c.replace("-", "−") if ep == "L-early" else None)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(list(xe.values()), ["early", "late", "early", "late"])
        for x_, s in ((0.5, "Day 0"), (3.1, "Expert")):
            ax.annotate(s, xy=(x_, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -12), textcoords="offset points",
                        ha="center", va="top", fontsize=5.5)
        ax.set_ylabel(yl, fontsize=5.2); ax.set_title("Epoch means (95% hierarchical-bootstrap CI)", fontsize=5.4)
        if j == 0:
            ax.legend(frameon=False, fontsize=5)
        ax = axs[1, j]
        names = list(CONTRASTS)
        for k_, c in enumerate(("R+", "R-")):
            q = T[(T.measure == col) & (T.cohort == c) & (T.kind == "contrast")].set_index("name").reindex(names)
            x = np.arange(len(names)) + (k_ - 0.5) * 0.3
            for xi, (_, r) in zip(x, q.iterrows()):
                ax.errorbar(xi, r.value, [[r.value - r.lo], [r.hi - r.value]], fmt="o", ms=3.5, color=COH[c], lw=0.9,
                            capsize=0, mfc=COH[c] if (r.p_boot if np.isfinite(r.p_boot) else 1) < 0.05 else "white",
                            label=c.replace("-", "−") if xi == x[0] else None)
        pp = T[(T.measure == col) & (T.cohort == "R+ - R-")].set_index("name").reindex(names)
        for i, n_ in enumerate(names):
            ax.text(i, 1.01, m62.fmt_p(pp.loc[n_, "p_perm"]).replace("p = ", "").replace("p < ", "<"),
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=4.4)
        ax.axhline(0, color="0.3", lw=0.5)
        ax.set_xticks(range(len(names)), [n_.replace(" (", "\n(") for n_ in names], fontsize=4.6)
        ax.set_ylabel("Change", fontsize=5.2)
        ax.set_title("Contrasts (filled: bootstrap p < 0.05)\ntop: R+ vs R− (mouse permutation)", fontsize=5.0, pad=8)
    m62.letter_row(fig, list(axs[0]), "abc"); m62.letter_row(fig, list(axs[1]), "def")
    fig.suptitle(f"Within-day vs across-day change on a common footing (phase-matched halves, {N_FIX} events per class, "
                 f"{N_UNITS} units; SL reference; {pop})", x=0.06, y=0.995, ha="left", fontsize=6.8, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.12)
    m62.save(fig, out, "epoch_comparison"); plt.close(fig)


def main(a):
    global N_FIX, OUT
    N_FIX = a.n_fix
    if N_FIX != 6:
        OUT = OUT.parent / f"epochs_n{N_FIX}"
    t0 = time.time()
    m61 = importlib.import_module("061_roc_prelick_learners")
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0],
                                               stage=Ws.stage.iloc[0]), Ws[["electrode_group", "cluster_id", "quality_label"]], 500 + i))
    if a.test:
        jobs = jobs[:a.test]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.replot:
        D = pd.read_csv(OUT / "epoch_sessions.csv")
    else:
        rows = []
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            for res in pool.imap_unordered(session, jobs):
                if res:
                    rows += res
        D = pd.DataFrame(rows); D.to_csv(OUT / "epoch_sessions.csv", index=False)
    for pop in ["all", "learners"]:
        Dp = m61.learner_filter(D) if pop == "learners" else D
        out = OUT / pop; out.mkdir(exist_ok=True)
        analyse(Dp, out, pop, a.B, a.n_perm)
    json.dump(dict(script="003_epoch_comparison.py", ref=m51.REF, n_fix=N_FIX, n_sub=N_SUB, n_units=N_UNITS, k_shift=K_SHIFT,
                   B=a.B, n_perm=a.n_perm, n_sessions=int(D.session_id.nunique()), runtime_min=round((time.time() - t0) / 60, 1)),
              open(OUT / "provenance.json", "w"), indent=1)
    print("ALL DONE", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=60)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-fix", type=int, default=6, help="events per class per epoch (output in epochs_n<k> if != 6)")
    main(ap.parse_args())
