"""Exposure (count-based novelty) model + reward term for whisker-trial responses (steps 1, 3, 4, 5 of the plan).

Novelty (frequency form). Leaky counts over the trial sequence (passive-pre exposures prepended, then active trials):
    c_j <- alpha * c_j + 1[event activates j],  components W (whisker), A (auditory), C (catch/no-stim) and
    S (shared "task stimulus", activated by W and A).  Pretraining: c_A(0) = c_C(0) = c_S(0) = A0, c_W(0) = 0.
    p_mod(W) = (c_W + eps) / (c_W + c_A + c_C + 3 eps)      <- trial-mix normalisation
    p_sh(W)  = (c_S + eps) / (c_S + c_C + 2 eps)
    p(W)     = (1 - lam) p_mod(W) + lam p_sh(W);   N_w(t) = -log p(W) just before whisker trial t.
Reward term R(t): value V(t) of a Rescorla-Wagner model fit to whisker-trial licking per session
    P(lick) = sigmoid(kappa (V - theta)); V <- V + eta (r - V) after licked whisker trials (r = 1 in R+, 0 in R-);
    V(0) = v0 (generalisation from auditory).  Used for the cue windows (response before the outcome).
Response model (whisker trials of one region/window, all sessions):
    y = mouse intercept + bN N_w + bR_group R + nuisance + e     (within-session demeaning = mouse fixed effects)
    nuisance: auditory and catch trends of the same region/window (per-session linear fit on clock time, evaluated at
    the whisker trial; diff-in-diff), pre-stimulus rate (z_base), pupil, whisking, jaw (+ missing indicators),
    log time since last whisker stimulus, previous-trial lick and reward, current lick (hit vs miss).
Models (nested; grid search over nonlinear parameters; leave-one-mouse-out CV of within-mouse R^2):
    nuis | exp_count / exp_trial / exp_clock (exp decay in whisker exposures / active-trial index / clock time)
    | freq (lam = 0) | freq_shared (lam free) | + reward (bR per cohort) | freq with group-specific alpha
Sub-commands
    recover  step 1: parameter recovery on the real trial sequences (simulate -> refit)
    rminus   step 3: R- only, novelty vs exponential / count-based alternatives
    joint    step 4: joint R-/R+ fit, bR per cohort (mouse-level cluster bootstrap CI + cohort-label permutation)
    map      step 5: per-region joint fits -> derived parameters (bootstrap CIs)
Output: combined_results_ks4/_novelty_model/model/
"""
import argparse
import itertools
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib                                                          # noqa: E402
m = importlib.import_module("029_roc_rpe_v2")
BASE = m.RES / "_novelty_model"
OUT = BASE / "model"
EPS = 1.0
ALPHAS = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0]
LAMS = [0.0, 0.25, 0.5, 0.75, 0.9]
A0S = [30.0, 100.0, 300.0, 1000.0]
TAUS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
NUIS = ["aud_trend", "catch_trend", "z_base", "pupil", "whisking", "jaw", "pupil_na", "video_na", "log_dt_whisker",
        "prev_lick", "prev_reward", "lick"]
MIN_UNITS = 5


# ------------------------------------------------------------------ novelty and RL models
def event_sequence(C):
    """events of one session: passive-pre block (interleaved W/A) then active trials; returns codes, active mask"""
    n_pw, n_pa = int(C.n_passive_pre_whisker.iloc[0]), int(C.n_auditory_before_all.iloc[0])
    pre = []
    if n_pw + n_pa:
        order = np.argsort(np.r_[np.linspace(0, 1, n_pw, endpoint=False), np.linspace(0, 1, n_pa, endpoint=False) + 1e-6])
        pre = list(np.r_[np.zeros(n_pw, int), np.ones(n_pa, int)][order])
    code = {"whisker": 0, "auditory": 1, "catch": 2}
    act = [code.get(t, 2) for t in C.ttype]
    return np.array(pre + act, int), np.r_[np.zeros(len(pre), bool), np.ones(len(act), bool)]


def novelty_series(ev, is_act, alpha, lam, A0):
    cW, cA, cC, cS = 0.0, A0, A0, A0
    N = np.empty(len(ev))
    for i, e in enumerate(ev):
        pm = (cW + EPS) / (cW + cA + cC + 3 * EPS)
        ps = (cS + EPS) / (cS + cC + 2 * EPS)
        N[i] = -np.log((1 - lam) * pm + lam * ps)
        cW *= alpha; cA *= alpha; cC *= alpha; cS *= alpha
        if e == 0:
            cW += 1; cS += 1
        elif e == 1:
            cA += 1; cS += 1
        else:
            cC += 1
    return N[is_act]


def fit_rw(lick, rewarded):
    """RW value on whisker trials; returns V(t) before each whisker trial and parameters"""
    lick = np.asarray(lick, float); r = np.asarray(rewarded, float)

    def values(eta, v0):
        V = np.empty(len(lick)); v = v0
        for t in range(len(lick)):
            V[t] = v
            if lick[t]:
                v += eta * (r[t] - v)
        return V

    def nll(p):
        eta, v0, kap, th = p
        z = kap * (values(eta, v0) - th)
        return -np.sum(lick * z - np.logaddexp(0, z))
    best = None
    for x0 in [(0.1, 0.5, 5, 0.5), (0.3, 0.8, 3, 0.3), (0.05, 0.3, 8, 0.2)]:
        res = optimize.minimize(nll, x0, bounds=[(1e-3, 1), (0, 1), (0.1, 30), (-1, 2)], method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return values(*best.x[:2]), dict(eta=best.x[0], v0=best.x[1], kappa=best.x[2], theta=best.x[3], nll=best.fun)


# ------------------------------------------------------------------ data assembly
def trend_at(g_src, g_tgt, col):
    ok = np.isfinite(g_src[col].to_numpy())
    if ok.sum() < 8:
        return np.zeros(len(g_tgt))
    b = np.polyfit(g_src.clock_min.to_numpy()[ok], g_src[col].to_numpy()[ok], 1)
    return np.polyval(b, g_tgt.clock_min.to_numpy())


def assemble(R, C, region, window, level="area_group"):
    """whisker-trial table for one region/window with nuisance covariates, novelty inputs and RL value"""
    Rr = R[(R.region == region) & ((R.level == level) | (region == "all")) & (R.n_units >= MIN_UNITS)]
    D = C.merge(Rr[["session_id", "trial", "n_units", "z_base", f"z_{window}"]], on=["session_id", "trial"])
    D = D.rename(columns={f"z_{window}": "y"})
    parts = []
    for sid, g in D.groupby("session_id"):
        w = g[g.ttype == "whisker"].copy()
        if len(w) < 20:
            continue
        a = g[(g.ttype == "auditory") & (g.aud_rank >= 5)]; c = g[g.ttype == "catch"]
        w["aud_trend"] = trend_at(a, w, "y"); w["catch_trend"] = trend_at(c, w, "y")
        w["aud_mean_late"] = a.y.iloc[int(len(a) * 2 / 3):].mean() if len(a) else np.nan
        parts.append(w)
    if not parts:
        return None
    W = pd.concat(parts, ignore_index=True)
    W["pupil_na"] = W.pupil.isna().astype(float); W["video_na"] = W.whisking.isna().astype(float)
    for c in ["pupil", "whisking", "jaw"]:
        W[c] = W[c].fillna(0.0)
    W["log_dt_whisker"] = np.log(W.time_since_whisker.fillna(W.time_since_whisker.max()).clip(lower=1.0))
    W["prev_lick"] = W.prev_lick.fillna(0).astype(float); W["prev_reward"] = W.prev_reward.fillna(0).astype(float)
    W["is_Rplus"] = (W.cohort == "R+").astype(float)
    W["whisker_idx"] = W.groupby("session_id").cumcount()
    W["n_whisker_all"] = W.n_whisker_before_all
    return W


def add_rl(W, C):
    vals = {}
    for sid, g in C[C.ttype == "whisker"].groupby("session_id"):
        V, _ = fit_rw(g.lick.to_numpy(), g.reward.to_numpy())
        vals[sid] = pd.Series(V, index=g.trial.to_numpy())
    W["V"] = [vals[s].get(t, np.nan) for s, t in zip(W.session_id, W.trial)]
    return W


class NoveltyCache:
    def __init__(self, C, sessions):
        self.seq = {s: event_sequence(C[C.session_id == s].sort_values("trial")) for s in sessions}
        self.types = {s: C[C.session_id == s].sort_values("trial").ttype.to_numpy() for s in sessions}
        self.cache = {}

    def N(self, sid, alpha, lam, A0):
        k = (sid, alpha, lam, A0)
        if k not in self.cache:
            ev, act = self.seq[sid]
            self.cache[k] = novelty_series(ev, act, alpha, lam, A0)[self.types[sid] == "whisker"]
        return self.cache[k]


# ------------------------------------------------------------------ fitting
def demean(W, cols):
    X = W[cols].to_numpy(float)
    return X - W.groupby("session_id")[cols].transform("mean").to_numpy(float)


class Design:
    """Precomputed, within-session demeaned design (mouse fixed effects). Any subset of sessions = row mask, a
    mouse-level bootstrap = session weights, a cohort-label permutation = relabelling the demeaned value column."""

    def __init__(self, W, NC):
        self.W, self.NC = W.reset_index(drop=True), NC
        self.sess = self.W.session_id.to_numpy()
        self.Xn = demean(self.W, NUIS)
        self.y = self.W.y.to_numpy(float) - self.W.groupby("session_id").y.transform("mean").to_numpy(float)
        self.Vdm = demean(self.W, ["V"])[:, 0] if "V" in self.W else np.zeros(len(self.W))
        self.lab = self.W.is_Rplus.to_numpy(float)
        self.cache = {}

    def nov(self, spec):
        if spec not in self.cache:
            r = regressor(self.W, self.NC, spec)
            self.cache[spec] = None if r is None else demean(self.W.assign(_n=r), ["_n"])[:, 0]
        return self.cache[spec]

    def X(self, spec, reward, lab=None):
        cols, names = [self.Xn], list(NUIS)
        n = self.nov(spec)
        if n is not None:
            cols.append(n[:, None]); names.append("bN")
        if reward:
            lab = self.lab if lab is None else lab
            cols.append(np.column_stack([self.Vdm * lab, self.Vdm * (1 - lab)])); names += ["bR_Rplus", "bR_Rminus"]
        return np.column_stack(cols), names


def regressor(W, NC, spec):
    """spec: ('freq', alpha, lam, A0[, alpha_Rminus]) | ('exp_count'|'exp_trial'|'exp_clock', tau) | ('none',)"""
    kind = spec[0]
    if kind == "none":
        return None
    if kind == "freq":
        out = np.empty(len(W)); wi = W.whisker_idx.to_numpy()
        for sid, idx in W.groupby("session_id").indices.items():
            a = spec[1] if (len(spec) < 5 or W.cohort.iloc[idx[0]] == "R+") else spec[4]
            out[idx] = NC.N(sid, a, spec[2], spec[3])[wi[idx]]
        return out
    x = {"exp_count": W.n_whisker_all, "exp_trial": W.trial, "exp_clock": W.clock_min}[kind].to_numpy(float)
    return np.exp(-x / spec[1])


def wls(X, y, w=None):
    if w is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, float(np.sum((y - X @ beta) ** 2))
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta, float(np.sum(w * (y - X @ beta) ** 2))


def spec_grid(kind):
    if kind == "none":
        return [("none",)]
    if kind.startswith("exp"):
        return [(kind, t) for t in TAUS]
    if kind == "freq":
        return [("freq", a, 0.0, A) for a, A in itertools.product(ALPHAS, A0S)]
    if kind == "freq_shared":
        return [("freq", a, l, A) for a, l, A in itertools.product(ALPHAS, LAMS, A0S)]
    if kind == "freq_groupalpha":
        return [("freq", a, 0.0, A, a2) for a, a2, A in itertools.product(ALPHAS, ALPHAS, A0S)]
    raise ValueError(kind)


def fit_model(D, kind, reward, w=None, lab=None):
    """grid search; w = per-row weights (0 = excluded session, k = bootstrap multiplicity)"""
    best = None
    for spec in spec_grid(kind):
        X, names = D.X(spec, reward, lab)
        beta, sse = wls(X, D.y, w)
        if best is None or sse < best[3]:
            best = (spec, beta, names, sse)
    return best


def lomo_cv(D, kind, reward):
    out = {}
    for s in np.unique(D.sess):
        test = D.sess == s
        spec, beta, names, _ = fit_model(D, kind, reward, w=(~test).astype(float))
        X, _ = D.X(spec, reward)
        y = D.y[test]; e = y - X[test] @ beta
        out[s] = 1 - np.sum(e ** 2) / np.sum(y ** 2) if np.sum(y ** 2) > 0 else np.nan
    return pd.Series(out)


def boot_weights(D, rng):
    s = np.unique(D.sess)
    cnt = pd.Series(rng.choice(s, len(s), replace=True)).value_counts()
    return pd.Series(D.sess).map(cnt).fillna(0).to_numpy(float), cnt


def derived(D, spec, beta, names, sess_w=None):
    """half-life (whisker exposures), initial amplitude, asymptote vs auditory; sess_w = bootstrap multiplicities"""
    if spec[0] != "freq":
        return {}
    bN = beta[names.index("bN")]
    hl, amp, asym, ww = [], [], [], []
    for sid, g in D.W.groupby("session_id"):
        wt = 1.0 if sess_w is None else float(sess_w.get(sid, 0))
        if wt == 0:
            continue
        a = spec[1] if (len(spec) < 5 or g.cohort.iloc[0] == "R+") else spec[4]
        N = D.NC.N(sid, a, spec[2], spec[3])
        n_inf = np.median(N[int(len(N) * 2 / 3):])
        rel = (N - n_inf) / (N[0] - n_inf) if N[0] != n_inf else np.zeros_like(N)
        hl.append(int(np.argmax(rel < 0.5)) if (rel < 0.5).any() else np.nan)
        amp.append(bN * (N[0] - n_inf))
        late = g.whisker_idx >= int(g.whisker_idx.max() * 2 / 3)
        asym.append(g.y[late].mean() - g.aud_mean_late.iloc[0]); ww.append(wt)
    ww = np.array(ww)
    wmean = lambda v: np.nansum(np.array(v) * ww) / np.sum(ww[np.isfinite(v)])       # noqa: E731
    return dict(half_life_exposures=float(np.nanmedian(np.repeat(hl, ww.astype(int)))), initial_amplitude=wmean(amp),
                asymptote_minus_auditory=wmean(asym))


def load():
    R = pd.read_parquet(BASE / "trial_region_table.parquet")
    C = pd.read_parquet(BASE / "trial_covariates.parquet").sort_values(["session_id", "trial"])
    return R, C


# ------------------------------------------------------------------ step 1: parameter recovery
def resid_sd(W, NC):
    D = Design(W, NC); X, _ = D.X(("none",), False)
    return float(np.std(D.y - X @ wls(X, D.y)[0]))


def cmd_recover(a):
    OUT.mkdir(parents=True, exist_ok=True)
    R, C = load()
    W = add_rl(assemble(R, C, "all", "late"), C)
    NC = NoveltyCache(C, W.session_id.unique())
    reg_sd = []
    for reg in R[R.level == "area_group"].region.unique():
        Wr = assemble(R, C, reg, "late")
        if Wr is not None and Wr.session_id.nunique() >= 8:
            reg_sd.append(resid_sd(Wr.assign(V=0.0), None))
    noises = {"all_units": resid_sd(W, NC), "region_median": float(np.median(reg_sd)),
              "region_p90": float(np.percentile(reg_sd, 90))}
    print("residual SD:", {k: round(v, 3) for k, v in noises.items()}, flush=True)
    rng = np.random.default_rng(0)
    truths = [dict(alpha=al, lam=lm, A0=100.0, bN=bn, bRp=br) for al, lm, bn, br in
              itertools.product([0.95, 0.99, 1.0], [0.0, 0.5], [0.05, 0.15], [0.0, 0.2])]
    D0 = Design(W, NC)
    rows = []
    for (nk, sd), tr in itertools.product(noises.items(), truths):
        spec_t = ("freq", tr["alpha"], tr["lam"], tr["A0"])
        mu = tr["bN"] * regressor(W, NC, spec_t) + tr["bRp"] * W.V.to_numpy() * W.is_Rplus.to_numpy()
        dt = derived(D0, spec_t, np.r_[np.zeros(len(NUIS)), tr["bN"]], NUIS + ["bN"])
        for rep in range(a.reps):
            D = Design(W.assign(y=mu + rng.normal(0, sd, len(W))), NC)
            D.cache = dict(D0.cache)                                       # novelty columns do not depend on y
            spec, beta, names, _ = fit_model(D, "freq_shared", True)
            D0.cache.update(D.cache)
            df = derived(D, spec, beta, names)
            rows.append(dict(noise=nk, noise_sd=sd, **{f"true_{k}": v for k, v in tr.items()},
                             fit_alpha=spec[1], fit_lam=spec[2], fit_A0=spec[3], fit_bN=beta[names.index("bN")],
                             fit_bRp=beta[names.index("bR_Rplus")], fit_bRm=beta[names.index("bR_Rminus")],
                             true_half_life=dt["half_life_exposures"], fit_half_life=df["half_life_exposures"],
                             true_amp=dt["initial_amplitude"], fit_amp=df["initial_amplitude"], rep=rep))
        print("done", nk, tr, flush=True)
    P = pd.DataFrame(rows); P.to_csv(OUT / "recovery.csv", index=False)
    summ = P.groupby("noise").apply(lambda g: pd.Series(dict(
        alpha_exact=(g.fit_alpha == g.true_alpha).mean(), lam_exact=(g.fit_lam == g.true_lam).mean(),
        lam_within_025=(abs(g.fit_lam - g.true_lam) <= 0.25).mean(),
        r_half_life=g[["true_half_life", "fit_half_life"]].corr(method="spearman").iloc[0, 1],
        r_amp=g[["true_amp", "fit_amp"]].corr().iloc[0, 1], r_bRp=g[["true_bRp", "fit_bRp"]].corr().iloc[0, 1],
        bias_bRp=(g.fit_bRp - g.true_bRp).mean(), sd_bRp=(g.fit_bRp - g.true_bRp).std(),
        sd_bRm_null=g.fit_bRm.std())))
    summ.to_csv(OUT / "recovery_summary.csv"); print(summ.round(3).to_string(), flush=True)


# ------------------------------------------------------------------ steps 3/4: model comparison
def compare(D, models):
    cv = pd.DataFrame({name: lomo_cv(D, kind, rew) for name, (kind, rew) in models.items()})
    cv["cohort"] = D.W.groupby("session_id").cohort.first().reindex(cv.index)
    fits = {}
    for name, (kind, rew) in models.items():
        spec, beta, names, sse = fit_model(D, kind, rew)
        fits[name] = dict(spec=str(spec), sse=sse, mean_cvR2=cv[name].mean(),
                          **{n: b for n, b in zip(names, beta) if n.startswith("b")}, **derived(D, spec, beta, names))
    F = pd.DataFrame(fits).T
    rows = []
    for name in models:
        for base in ["nuis", "exp_count", "freq"]:
            if name == base or base not in cv:
                continue
            dlt = (cv[name] - cv[base]).dropna()
            rows.append(dict(model=name, vs=base, mean_cvR2_gain=dlt.mean(), median_gain=dlt.median(), n=len(dlt),
                             p_wilcoxon=stats.wilcoxon(dlt).pvalue if (dlt != 0).any() else np.nan,
                             p_ttest=stats.ttest_1samp(dlt, 0).pvalue))
    return cv, F, pd.DataFrame(rows)


def cmd_fit(a, joint):
    OUT.mkdir(parents=True, exist_ok=True)
    R, C = load()
    res = []
    for window in a.windows:
        W = add_rl(assemble(R, C, a.region, window), C)
        if not joint:
            W = W[W.cohort == "R-"]
        NC = NoveltyCache(C, W.session_id.unique()); D = Design(W, NC)
        models = {"nuis": ("none", False), "exp_count": ("exp_count", False), "exp_trial": ("exp_trial", False),
                  "exp_clock": ("exp_clock", False), "freq": ("freq", False), "freq_shared": ("freq_shared", False)}
        if joint:
            models.update({"freq_shared+reward": ("freq_shared", True), "freq_groupalpha": ("freq_groupalpha", False)})
        cv, F, G = compare(D, models)
        tag = f"{'joint' if joint else 'rminus'}_{a.region.replace(' ', '_')}_{window}"
        cv.to_csv(OUT / f"cv_{tag}.csv"); F.to_csv(OUT / f"fits_{tag}.csv"); G.to_csv(OUT / f"gains_{tag}.csv", index=False)
        print(f"== {tag}: {W.session_id.nunique()} sessions, {len(W)} whisker trials")
        print(G.round(4).to_string())
        print(F[[c for c in ["spec", "mean_cvR2", "bN", "bR_Rplus", "bR_Rminus", "half_life_exposures", "initial_amplitude",
                             "asymptote_minus_auditory"] if c in F]].to_string(), flush=True)
        if joint:
            spec, beta, names, _ = fit_model(D, "freq_shared", True)
            iP, iM = names.index("bR_Rplus"), names.index("bR_Rminus")
            obs = beta[iP] - beta[iM]
            rng = np.random.default_rng(1)
            sess = D.W.groupby("session_id").cohort.first()
            perm = []
            for _ in range(a.n_perm):
                lab = pd.Series(rng.permutation((sess == "R+").to_numpy(float)), index=sess.index)
                X, _ = D.X(spec, True, lab=pd.Series(D.sess).map(lab).to_numpy())
                bp = wls(X, D.y)[0]; perm.append(bp[iP] - bp[iM])
            boot = []
            X, _ = D.X(spec, True)
            for _ in range(a.n_boot):
                w, _ = boot_weights(D, rng); bb = wls(X, D.y, w)[0]
                boot.append({n: b for n, b in zip(names, bb) if n.startswith("b")})
            B = pd.DataFrame(boot)
            res.append(dict(window=window, spec=str(spec), **{n: b for n, b in zip(names, beta) if n.startswith("b")},
                            bR_diff=obs, p_perm_bR_diff=(1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + len(perm)),
                            **{f"{c}_ci_lo": B[c].quantile(.025) for c in B}, **{f"{c}_ci_hi": B[c].quantile(.975) for c in B}))
    if joint:
        P = pd.DataFrame(res); P.to_csv(OUT / f"joint_inference_{a.region.replace(' ', '_')}.csv", index=False)
        print(P.round(4).T.to_string(), flush=True)


def cmd_map(a):
    OUT.mkdir(parents=True, exist_ok=True)
    R, C = load()
    regions = [r for r in R[R.level == "area_group"].region.unique() if r != "unassigned"]
    rows = []
    for window in a.windows:
        for reg in regions:
            W = assemble(R, C, reg, window)
            if W is None:
                continue
            nc = W.groupby("cohort").session_id.nunique()
            if nc.get("R+", 0) < a.min_sess or nc.get("R-", 0) < a.min_sess:
                continue
            W = add_rl(W, C); NC = NoveltyCache(C, W.session_id.unique()); D = Design(W, NC)
            spec, beta, names, _ = fit_model(D, "freq_shared", True)
            row = dict(window=window, region=reg, n_sess_Rplus=nc.get("R+", 0), n_sess_Rminus=nc.get("R-", 0),
                       median_units=W.groupby("session_id").n_units.first().median(), spec=str(spec), lam=spec[2],
                       alpha=spec[1], **{n: b for n, b in zip(names, beta) if n.startswith("b")},
                       **derived(D, spec, beta, names))
            cvs = {k: lomo_cv(D, kind, rew) for k, (kind, rew) in
                   {"nuis": ("none", False), "freq_shared": ("freq_shared", False), "freq_shared+reward": ("freq_shared", True)}.items()}
            for k1, k0 in [("freq_shared", "nuis"), ("freq_shared+reward", "freq_shared")]:
                dl = (cvs[k1] - cvs[k0]).dropna()
                row[f"cv_gain_{k1}_vs_{k0}"] = dl.mean()
                row[f"p_wilcoxon_{k1}_vs_{k0}"] = stats.wilcoxon(dl).pvalue if (dl != 0).any() else np.nan
                row[f"p_ttest_{k1}_vs_{k0}"] = stats.ttest_1samp(dl, 0).pvalue
            rng = np.random.default_rng(2); boots = []
            for _ in range(a.n_boot):
                w, cnt = boot_weights(D, rng)
                sp, bb, nn, _ = fit_model(D, "freq_shared", True, w=w)
                boots.append(dict(lam=sp[2], alpha=sp[1], **{n: b for n, b in zip(nn, bb) if n.startswith("b")},
                                  **derived(D, sp, bb, nn, sess_w=cnt)))
            B = pd.DataFrame(boots)
            for c in ["lam", "bN", "bR_Rplus", "bR_Rminus", "half_life_exposures", "initial_amplitude", "asymptote_minus_auditory"]:
                if c in B:
                    row[f"{c}_lo"], row[f"{c}_hi"] = B[c].quantile(.025), B[c].quantile(.975)
            rows.append(row)
            print("mapped", window, reg, flush=True)
    M = pd.DataFrame(rows); M.to_csv(OUT / "region_map.csv", index=False)
    print(M.round(3).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("recover"); r.add_argument("--reps", type=int, default=5)
    for name in ["rminus", "joint"]:
        p = sp.add_parser(name); p.add_argument("--region", default="all"); p.add_argument("--windows", nargs="+", default=["early", "late"])
        p.add_argument("--n-perm", type=int, default=500); p.add_argument("--n-boot", type=int, default=200)
    mp = sp.add_parser("map"); mp.add_argument("--windows", nargs="+", default=["early", "late"])
    mp.add_argument("--n-boot", type=int, default=100); mp.add_argument("--min-sess", type=int, default=6)
    a = ap.parse_args()
    {"recover": cmd_recover, "rminus": lambda x: cmd_fit(x, False), "joint": lambda x: cmd_fit(x, True), "map": cmd_map}[a.cmd](a)
