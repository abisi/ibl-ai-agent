"""Refit of the exposure / reward models (after 037 review):
1. Lick model v2: value separated from motivation. Whisker-trial licking per cohort:
       logit P(lick_t) = session intercept + kV * V_t + kS * satiety_t + kE * engagement_t
       V: Rescorla-Wagner value of the whisker stimulus, updated after whisker licks (r = 1 R+, 0 R-),
          learning rate eta and initial value v0 shared within a cohort (grid), V_0 = v0 in every session
       satiety_t: rewards consumed before trial t (all trial types) / 100
       engagement_t: logit of the local auditory hit rate (auditory trials within +-5 min, excluding the first 5)
   Fit per cohort: grid over (eta, v0), logistic regression (no penalty) for the rest; best log-likelihood.
2. Response models refit separately on whisker MISSES (no lick: clean exposure test) and whisker HITS (reward
   contrast), without the current-trial lick as a nuisance covariate. Models: nuisance | exp_count | freq |
   freq_shared | freq_shared + reward (bR per cohort on V). LOMO CV (within-mouse R^2), bootstrap over mice (CIs),
   cohort-label permutation for bR(R+) - bR(R-).
3. Session subsets: all | passive-pre whisker exposures before the active block (pre) | none (few sessions:
   descriptive).
Output: combined_results_ks4/_novelty_model/refit/
"""
import argparse
import importlib
import itertools
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
nm = importlib.import_module("037_novelty_model")
BASE = nm.BASE
OUT = BASE / "refit"
ETAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
V0S = [0.0, 0.25, 0.5, 0.75, 1.0]
REGIONS = ["all", "Somatosensory-whisker", "Retrosplenial areas", "Motor areas", "Thalamus", "Hippocampus"]
NUIS_NOLICK = [c for c in nm.NUIS if c != "lick"]


# ------------------------------------------------------------------ 1. lick model v2
def rw_values(lick, rew, eta, v0):
    V = np.empty(len(lick)); v = v0
    for t in range(len(lick)):
        V[t] = v
        if lick[t]:
            v += eta * (rew[t] - v)
    return V


def lick_covariates(C):
    rows = []
    for sid, g in C.groupby("session_id"):
        g = g.sort_values("trial")
        g = g.assign(satiety=np.r_[0, np.cumsum(g.reward.to_numpy())[:-1]] / 100)
        a = g[(g.ttype == "auditory") & (g.aud_rank >= 5)]
        eng = []
        for t in g.clock_min.to_numpy():
            near = a[(a.clock_min - t).abs() <= 5]
            p = (near.lick.sum() + 0.5) / (len(near) + 1)
            eng.append(np.log(p / (1 - p)))
        rows.append(g.assign(engagement=eng))
    return pd.concat(rows)


def fit_lick_model(C):
    from sklearn.linear_model import LogisticRegression
    C = lick_covariates(C)
    out, params = {}, []
    for coh, gc in C.groupby("cohort"):
        w = gc[gc.ttype == "whisker"].sort_values(["session_id", "trial"])
        S = pd.get_dummies(w.session_id).to_numpy(float)
        best = None
        for eta, v0 in itertools.product(ETAS, V0S):
            V = np.concatenate([rw_values(g.lick.to_numpy(), g.reward.to_numpy(), eta, v0)
                                for _, g in w.groupby("session_id", sort=True)])
            X = np.column_stack([S, V, w.satiety, w.engagement])
            lr = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000).fit(X, w.lick)
            p = np.clip(lr.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
            ll = np.sum(w.lick * np.log(p) + (1 - w.lick) * np.log(1 - p))
            if best is None or ll > best[0]:
                best = (ll, eta, v0, lr.coef_[0][-3:], V, p)
        ll, eta, v0, (kV, kS, kE), V, p = best
        # null: same model without V (is value needed at all?)
        X0 = np.column_stack([S, w.satiety, w.engagement])
        lr0 = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000).fit(X0, w.lick)
        p0 = np.clip(lr0.predict_proba(X0)[:, 1], 1e-9, 1 - 1e-9)
        ll0 = np.sum(w.lick * np.log(p0) + (1 - w.lick) * np.log(1 - p0))
        params.append(dict(cohort=coh, eta=eta, v0=v0, kV=kV, k_satiety=kS, k_engagement=kE, loglik=ll,
                           loglik_noV=ll0, LR_stat=2 * (ll - ll0), n_trials=len(w), n_sessions=w.session_id.nunique()))
        out.update({(s, t): (v, q) for s, t, v, q in zip(w.session_id, w.trial, V, p)})
    V = pd.DataFrame([dict(session_id=s, trial=t, V=v, p_lick_fit=q) for (s, t), (v, q) in out.items()])
    return V, pd.DataFrame(params), C


# ------------------------------------------------------------------ 2. refits
def run_one(args):
    subset, outcome, region, window = args
    R, C = nm.load()
    Vdf = pd.read_parquet(OUT / "lick_model_values.parquet")
    W = nm.assemble(R, C, region, window)
    if W is None:
        return None
    W = W.merge(Vdf[["session_id", "trial", "V"]], on=["session_id", "trial"], how="left")
    W = W[W.lick == (1 if outcome == "hit" else 0)]
    if subset != "all":
        pre = W.n_passive_pre_whisker > 0
        W = W[pre if subset == "pre" else ~pre]
    W = W[W.groupby("session_id").trial.transform("size") >= 8].copy()
    W["whisker_idx"] = W.whisker_idx                                    # index among ALL whisker trials (novelty)
    nc = W.groupby("cohort").session_id.nunique()
    if nc.get("R+", 0) < 4 or nc.get("R-", 0) < 4:
        return None
    NC = nm.NoveltyCache(C, W.session_id.unique())
    old = nm.NUIS
    nm.NUIS = NUIS_NOLICK                                                # current lick is constant within outcome
    try:
        D = nm.Design(W, NC)
        models = {"nuis": ("none", False), "exp_count": ("exp_count", False), "freq": ("freq", False),
                  "freq_shared": ("freq_shared", False), "freq_shared+reward": ("freq_shared", True)}
        cv = pd.DataFrame({k: nm.lomo_cv(D, kind, rew) for k, (kind, rew) in models.items()})
        row = dict(subset=subset, outcome=outcome, region=region, window=window, n_sess_Rplus=nc.get("R+", 0),
                   n_sess_Rminus=nc.get("R-", 0), n_trials=len(W))
        for k1, k0 in [("freq_shared", "nuis"), ("freq_shared", "exp_count"), ("freq_shared+reward", "freq_shared"),
                       ("freq_shared+reward", "nuis")]:
            dl = (cv[k1] - cv[k0]).dropna()
            row[f"gain_{k1}_vs_{k0}"] = dl.mean()
            row[f"pW_{k1}_vs_{k0}"] = stats.wilcoxon(dl).pvalue if (dl != 0).any() else np.nan
            row[f"pT_{k1}_vs_{k0}"] = stats.ttest_1samp(dl, 0).pvalue
        spec, beta, names, _ = nm.fit_model(D, "freq_shared", True)
        row.update(spec=str(spec), lam=spec[2], **{n: b for n, b in zip(names, beta) if n.startswith("b")},
                   **nm.derived(D, spec, beta, names))
        iP, iM = names.index("bR_Rplus"), names.index("bR_Rminus")
        rng = np.random.default_rng(3)
        sess = D.W.groupby("session_id").cohort.first()
        X, _ = D.X(spec, True)
        perm = []
        for _ in range(500):
            lab = pd.Series(rng.permutation((sess == "R+").to_numpy(float)), index=sess.index)
            Xp, _ = D.X(spec, True, lab=pd.Series(D.sess).map(lab).to_numpy())
            bp = nm.wls(Xp, D.y)[0]; perm.append(bp[iP] - bp[iM])
        row["bR_diff"] = beta[iP] - beta[iM]
        row["p_perm_bR_diff"] = (1 + np.sum(np.abs(perm) >= abs(row["bR_diff"]))) / 501
        boots = []
        for _ in range(200):
            w, cnt = nm.boot_weights(D, rng); bb = nm.wls(X, D.y, w)[0]
            boots.append(dict(**{n: b for n, b in zip(names, bb) if n.startswith("b")},
                              amp=nm.derived(D, spec, bb, names, sess_w=cnt)["initial_amplitude"]))
        B = pd.DataFrame(boots)
        for c in ["bN", "bR_Rplus", "bR_Rminus", "amp"]:
            row[f"{c}_lo"], row[f"{c}_hi"] = B[c].quantile(.025), B[c].quantile(.975)
        # within-mouse V variance (is the reward regressor informative?)
        row["V_within_sd"] = float(np.std(D.Vdm))
        return row
    finally:
        nm.NUIS = old


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    R, C = nm.load()
    if a.step in ("lick", "all"):
        V, P, Cl = fit_lick_model(C)
        V.to_parquet(OUT / "lick_model_values.parquet", index=False); P.to_csv(OUT / "lick_model_params.csv", index=False)
        print(P.round(4).to_string(), flush=True)
        within = V.merge(C[["session_id", "trial", "cohort"]]).groupby(["cohort", "session_id"]).V.std()
        print("within-session SD of V:", within.groupby("cohort").describe().round(3).to_string(), flush=True)
        lick_figure(V, Cl)
    if a.step in ("refit", "all"):
        jobs = list(itertools.product(["all", "pre", "none"], ["miss", "hit"], REGIONS, ["early", "late"]))
        rows = []
        with ProcessPoolExecutor(a.workers) as pool:
            for r in pool.map(run_one, jobs):
                if r is not None:
                    rows.append(r); print("done", r["subset"], r["outcome"], r["region"], r["window"], flush=True)
        T = pd.DataFrame(rows); T.to_csv(OUT / "refit_results.csv", index=False)
        pd.set_option("display.width", 300)
        cols = ["subset", "outcome", "region", "window", "n_sess_Rplus", "n_sess_Rminus", "lam", "initial_amplitude",
                "amp_lo", "amp_hi", "bR_Rplus", "bR_Rplus_lo", "bR_Rplus_hi", "bR_Rminus", "bR_Rminus_lo", "bR_Rminus_hi",
                "p_perm_bR_diff", "gain_freq_shared_vs_nuis", "pW_freq_shared_vs_nuis", "pT_freq_shared_vs_nuis",
                "gain_freq_shared+reward_vs_freq_shared", "pW_freq_shared+reward_vs_freq_shared",
                "pT_freq_shared+reward_vs_freq_shared", "V_within_sd"]
        print(T[cols].round(3).to_string(), flush=True)


def lick_figure(V, Cl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    D = Cl[Cl.ttype == "whisker"].merge(V, on=["session_id", "trial"])
    D["b"] = (D.n_whisker_before // 10).astype(int)
    D = D[D.b < 15]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for coh, col in [("R+", "#00B400"), ("R-", "#C800C8")]:
        g = D[D.cohort == coh]
        for ax, (y, ls, lab) in zip(axs, [("lick", "-", "observed"), ("p_lick_fit", "--", "model"), ("V", "-", "value V")]):
            ps = g.groupby(["session_id", "b"])[y].mean().unstack()
            mu, se = ps.mean(), ps.std() / np.sqrt(ps.notna().sum()); ok = ps.notna().sum() >= 5
            x = (mu.index[ok] + 0.5) * 10
            ax.plot(x, mu[ok], color=col, ls=ls, lw=1, label=f"{coh.replace('-', chr(8722))} {lab}")
            ax.fill_between(x, (mu - se)[ok], (mu + se)[ok], color=col, alpha=0.15, lw=0)
    axs[0].set_title("P(lick) on whisker trials, observed"); axs[1].set_title("model prediction")
    axs[2].set_title("fitted value V (Rescorla-Wagner)")
    for ax in axs:
        ax.set_xlabel("whisker trial"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout(); fig.savefig(OUT / "lick_model_check.png", dpi=220)
    print("saved", OUT / "lick_model_check.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["lick", "refit", "all"])
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
