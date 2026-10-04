"""124 -- Continuous behavioural learning scores vs the neural pre/post change at a FORCED behavioural split, every session
(user 2026-10-01: correlate how much the split hit/miss decoding effect beats placebo with a measure of performance, to
separate gradual / step learners from non-lickers and non-learners without the binary "has an LT" split).
Forced split k* (every session): the MAP change point of the joint whisker + no-stim (FA) Bayesian change-point model
(L6 of ssl-learning-trial-identification 005; computed for all sessions in 028), whether or not it passes the L6 / L6
lenient thresholds; whisker-trial index of the curve-aligned list.
Behavioural scores (whisker day 0, full session -- LT identification is never trimmed):
  log10 BF L6     evidence for a change (naive -> learned [-> end segment]) vs no change, joint whisker + FA RAW outcomes
                  (Beta(1,1) segment rates integrated out); thresholds: 0 (L6 lenient), 0.5 (L6 strict);
  log10 BF L5     same, whisker outcomes only;
  step size       Delta(whisker - FA) across k*: raw lick rates in the WIN whisker trials after vs before (no-stim trials by
                  start time);
  max slope       largest rise (R+) / fall (R-) of the whisker - FA HMM curve (sigma = 1, 024) over SLOPE_W whisker trials,
                  per trial;
  net change      mean whisker - FA curve over the last third minus the first third of the session;
  linear slope    OLS slope of the whisker - FA curve vs whisker trial (per 10 trials).
Neural measures at k* (sessions decodable in 122 / 123; A1-trimmed neural data):
  122 delta       post - pre size-matched balanced accuracy at the split nearest k* (122 splits every 4 whisker trials,
                  nearest within 2); percentile among the session's splits >= 10 trials away (placebo);
                  hit vs miss 5-50 / 5-100 ms, whisker vs auditory -100..0 ms pre-lick;
  123 margin      change of the class-residualised single-trial margin (WIN decoded trials after vs before k*), percentile
                  among all splits >= 10 trials away; hit vs miss 5-50 / 5-100 ms.
  excess          (user 2026-10-01) real two-decoder change at the forced split MINUS the session's mean placebo change
                  (splits >= 10 trials away), in accuracy units -- how much the pre/post decoder difference beats placebo;
                  figure 124_score_vs_excess.
Stats per cohort (mouse = unit): Spearman AND Pearson, permutation p (5000 label shuffles of the behavioural score);
partial Spearman controlling for the size-matched class count of the split (122 rows). Scores vs scores: Spearman
matrix per cohort (are BF and slope measures redundant?).
Figures (figures/): 124_score_vs_neural.{pdf,png,svg} (rows: neural percentile measures; columns: behavioural scores;
scatter + OLS line + 95% CI band, solid if p < 0.05; BF thresholds marked), 124_score_agreement.{pdf,png}.
Tables: 124_per_session.csv, 124_correlations.csv, 124_score_agreement.csv
Run (haas): python 124_behaviour_score_vs_split_decoding.py
"""

from __future__ import annotations

import pickle
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
COL = {"R+": "#00B400", "R-": "#C800C8"}
WIN, SLOPE_W, EXCLUDE_NEAR, N_PERM = 20, 10, 10, 5000
FS_L, FS_M, FS_S = 8, 7, 6
SCORES = [("bf_l6", "log10 BF, joint CP (L6)"), ("bf_l5", "log10 BF, whisker CP (L5)"),
          ("step", "step size Δ(W−FA) at CP"), ("max_slope", f"max slope W−FA ({SLOPE_W} trials)"),
          ("net", "net change W−FA (last − first third)"), ("lin_slope", "linear slope W−FA (/10 trials)")]
NEURAL = [("pct_hitmiss_5-50ms", "hit/miss 5-50 ms\npercentile vs placebo"),
          ("pct_hitmiss_5-100ms", "hit/miss 5-100 ms\npercentile vs placebo"),
          ("pct_modality_lick_-100-0ms", "whisker vs auditory pre-lick\npercentile vs placebo"),
          ("mpct_5-50ms", "single-trial margin 5-50 ms\npercentile vs other splits"),
          ("mpct_5-100ms", "single-trial margin 5-100 ms\npercentile vs other splits")]


def raw_rate(t_tr, y_tr, t0, t1):
    s = (t_tr >= t0) & (t_tr < t1)
    return y_tr[s].mean() if s.any() else np.nan


def behaviour_scores():
    art = LTP / "artifacts"
    inp = pickle.load(open(art / "028_chain_all" / "001_inputs.pkl", "rb"))
    cp6 = pickle.load(open(art / "028_chain_all" / "005_cp_posteriors.pkl", "rb"))
    cp5 = pickle.load(open(art / "028_chain_all" / "004_cp_posteriors.pkl", "rb"))
    curves = {p["session_id"]: p for p in pickle.load(open(art / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    lt = pd.read_csv(art / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")
    rows = []
    for sid, d in inp.items():
        r6 = cp6.get(sid)
        tw, yw, tn, yn = (np.asarray(d[k], float) for k in ("w_start", "w_outcomes", "n_start", "n_outcomes"))
        n = len(tw)
        k = int(r6["map_k"]) if r6 is not None and np.isfinite(r6.get("map_k", np.nan)) else np.nan
        row = dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=d["reward_group"],
                   learning_category=d["learning_category"], n_whisker=n, k_star=k,
                   bf_l6=r6["log10_bf"] if r6 is not None else np.nan,
                   bf_l5=cp5[sid]["log10_bf"] if cp5.get(sid) is not None else np.nan,
                   l6_category=lt.loc[sid, "L6 category"] if sid in lt.index else None,
                   **{f"has_{c}": sid in lt.index and pd.notna(lt.loc[sid, c]) for c in
                      ("L6 joint CP", "L6 lenient", "L5 whisker CP", "lenient cascade", "L0 stored")})
        if np.isfinite(k) and 0 < k < n:
            t_lo = tw[max(0, k - WIN)]
            t_k = tw[k]
            t_hi = tw[min(n - 1, k + WIN - 1)] + 1e-6
            pre = raw_rate(tw, yw, t_lo, t_k) - raw_rate(tn, yn, t_lo, t_k)
            post = raw_rate(tw, yw, t_k, t_hi) - raw_rate(tn, yn, t_k, t_hi)
            row["step"] = post - pre
        c = np.asarray(curves[sid]["curves"]["w-fa"], float) if sid in curves else None
        if c is not None and len(c) > SLOPE_W + 1:
            dif = (c[SLOPE_W:] - c[:-SLOPE_W]) / SLOPE_W
            row["max_slope"] = dif.max() if d["reward_group"] == "R+" else dif.min()
            th = max(1, len(c) // 3)
            row["net"] = c[-th:].mean() - c[:th].mean()
            row["lin_slope"] = np.polyfit(np.arange(len(c)), c, 1)[0] * 10
        rows.append(row)
    return pd.DataFrame(rows)


def neural_122(B):
    d = pd.read_parquet(OUT / f"122_lt_placebo_whole_brain{os.environ.get('SSL_PLACEBO_TAG', '')}.parquet")
    d = d[d.skipped_reason.isna() & (d.split_k >= 0)]
    kstar = B.set_index("session_id").k_star
    rows = []
    for (sid, dec, w), g in d.groupby(["session_id", "decoding", "window"]):
        k = kstar.get(sid, np.nan)
        if not np.isfinite(k):
            continue
        dist = np.abs(g.split_k.to_numpy() - k)
        i = int(np.argmin(dist))
        if dist[i] > 2:
            continue
        real = g.delta_matched.to_numpy()[i]
        pl = g.delta_matched.to_numpy()[dist >= EXCLUDE_NEAR]
        pl = pl[np.isfinite(pl)]
        if len(pl) < 5 or not np.isfinite(real):
            continue
        key = f"{dec}_{w}"
        rows.append(dict(session_id=sid, measure=key, delta=real, pct=float(np.mean(pl < real) + 0.5 * np.mean(pl == real)),
                         excess=real - float(np.mean(pl)), excess_z=(real - float(np.mean(pl))) / float(np.std(pl)) if np.std(pl) > 0 else np.nan,
                         matched_n=int(min(g.matched_n_pos.to_numpy()[i], g.matched_n_neg.to_numpy()[i]))))
    return pd.DataFrame(rows)


def neural_123(B):
    D = pickle.load(open(OUT / "123_singletrial_scores_all.pkl", "rb"))
    kstar = B.set_index("session_id").k_star
    rows = []
    for (sid, w), out in D["res"].items():
        k = kstar.get(sid, np.nan)
        if not np.isfinite(k):
            continue
        m = D["meta"][sid]
        y, z = m["y"], np.asarray(out["real"]["margin"], float).copy()
        ok = np.isfinite(z)
        for cls in (True, False):
            sel = ok & (y == cls)
            if sel.any():
                z[sel] -= z[sel].mean()
        z[~ok] = 0
        cw_t = m["cw_t"]
        if int(k) >= len(cw_t):
            continue
        c = int(np.sum(m["t"] < cw_t[int(k)]))

        def ch(cc):
            lo, hi = max(0, cc - WIN), min(len(z), cc + WIN)
            return z[cc:hi].mean() - z[lo:cc].mean() if (cc - lo >= 5 and hi - cc >= 5) else np.nan
        allc = np.array([ch(j) for j in range(len(z))])
        real = allc[c] if 0 < c < len(z) else np.nan
        far = np.abs(np.arange(len(z)) - c) >= EXCLUDE_NEAR
        pl = allc[far & np.isfinite(allc)]
        if len(pl) < 5 or not np.isfinite(real):
            continue
        rows.append(dict(session_id=sid, measure=f"m_{w}", delta=real, pct=float(np.mean(pl < real) + 0.5 * np.mean(pl == real))))
    return pd.DataFrame(rows)


def perm_p(x, y, f, rng):
    obs = f(x, y)
    null = np.array([f(rng.permutation(x), y) for _ in range(N_PERM)])
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS_M})
    rng = np.random.default_rng(0)
    B = behaviour_scores()
    N1, N2 = neural_122(B), neural_123(B)
    W = B.copy()
    for _, g in N1.groupby("measure"):
        key = g.measure.iloc[0]
        W = W.merge(g[["session_id", "pct", "delta", "excess", "excess_z", "matched_n"]].rename(
            columns={"pct": f"pct_{key}", "delta": f"delta_{key}", "excess": f"exc_{key}", "excess_z": f"excz_{key}",
                     "matched_n": f"n_{key}"}), on="session_id", how="left")
    for _, g in N2.groupby("measure"):
        key = g.measure.iloc[0][2:]
        W = W.merge(g[["session_id", "pct", "delta"]].rename(columns={"pct": f"mpct_{key}", "delta": f"mdelta_{key}"}),
                    on="session_id", how="left")
    W.to_csv(OUT / "124_per_session.csv", index=False)
    crow = []
    corr_figure(W, NEURAL, crow, rng, "124_score_vs_neural", "neural pre/post change at the forced behavioural split")
    EXC = [(f"exc_{k}", f"{lab}\nexcess Δacc vs placebo") for k, lab in
           (("hitmiss_5-50ms", "hit/miss 5-50 ms"), ("hitmiss_5-100ms", "hit/miss 5-100 ms"),
            ("modality_lick_-100-0ms", "whisker vs auditory pre-lick"))]
    corr_figure(W, EXC, crow, rng, "124_score_vs_excess",
                "EXCESS of the two-decoder change (post − pre, separate size-matched decoders) at the forced split over the "
                "session's mean placebo change")
    C = pd.DataFrame(crow)
    C.to_csv(OUT / "124_correlations.csv", index=False)
    score_agreement(B)
    pd.set_option("display.width", 220)
    print(C[C.neural.str.startswith("exc_")].round(3).to_string(index=False))


def corr_figure(W, NEURAL, crow, rng, name, what):
    fig, axes = plt.subplots(len(NEURAL), len(SCORES), figsize=(11.7, 1.95 * len(NEURAL) + 0.7))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.06, wspace=0.35, hspace=0.55)
    for r, (nk, nl) in enumerate(NEURAL):
        for c, (sk, sl) in enumerate(SCORES):
            ax = axes[r, c]
            for rg in ("R+", "R-"):
                ckey = nk.split("_", 1)[1]
                cov = [f"n_{ckey}"] if nk.startswith(("pct_", "exc_")) else []
                g = W[(W.reward_group == rg)][[sk, nk] + cov].dropna()
                if len(g) < 6:
                    continue
                x, y = g[sk].to_numpy(float), g[nk].to_numpy(float)
                rs, ps = stats.spearmanr(x, y)
                rp, pp = stats.pearsonr(x, y)
                pperm = perm_p(x, y, lambda a, b: stats.spearmanr(a, b)[0], rng)
                rec = dict(neural=nk, score=sk, cohort=rg, n=len(g), spearman=rs, p_spearman=ps, pearson=rp,
                           p_pearson=pp, p_perm_spearman=pperm)
                if cov:
                    rpar, ppar = partial_spearman(x, y, g[cov[0]].to_numpy(float))
                    rec.update(partial_spearman_classn=rpar, p_partial=ppar)
                crow.append(rec)
                ax.scatter(x, y, s=8, color=COL[rg], alpha=0.75, lw=0)
                xs = np.linspace(x.min(), x.max(), 50)
                sl_, ic = np.polyfit(x, y, 1)
                X = np.c_[np.ones_like(x), x]
                res = y - X @ np.array([ic, sl_])
                s2 = res @ res / max(len(x) - 2, 1)
                cov = s2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.einsum("ij,jk,ik->i", np.c_[np.ones_like(xs), xs], cov, np.c_[np.ones_like(xs), xs]))
                tq = stats.t.ppf(0.975, max(len(x) - 2, 1))
                yh = ic + sl_ * xs
                ax.fill_between(xs, yh - tq * se, yh + tq * se, color=COL[rg], alpha=0.15, lw=0)
                ax.plot(xs, yh, color=COL[rg], lw=1.1, ls="-" if pp < 0.05 else "--")
                ax.text(0.02 if rg == "R+" else 0.52, 1.02, f"ρ {rs:.2f} p {pperm:.2f}", transform=ax.transAxes,
                        fontsize=FS_S - 0.5, color=COL[rg])
            ax.axhline(0 if nk.startswith("exc_") else 0.5, color="0.7", lw=0.5, ls=":")
            if sk.startswith("bf"):
                for th, ls in ((0, ":"), (0.5, "--")):
                    ax.axvline(th, color="0.5", lw=0.6, ls=ls)
            ax.tick_params(labelsize=FS_S)
            if r == len(NEURAL) - 1:
                ax.set_xlabel(sl, fontsize=FS_S)
            if c == 0:
                ax.set_ylabel(nl, fontsize=FS_S)
    fig.suptitle(f"Behavioural learning scores (whisker day 0) vs {what} (MAP joint change point, every session).\n"
                 "Spearman ρ with permutation p; line solid if Pearson p < 0.05; BF lines: 0 (lenient), 0.5 (strict)",
                 fontsize=FS_M, y=0.995)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=250)
    plt.close(fig)


def score_agreement(B):
    keys = [k for k, _ in SCORES]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.3, top=0.85, wspace=0.6)
    agr = []
    for ax, rg in zip(axes, ("R+", "R-")):
        g = B[B.reward_group == rg][keys]
        R = g.corr(method="spearman").to_numpy()
        im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=FS_S - 0.5)
                agr.append(dict(cohort=rg, a=keys[i], b=keys[j], spearman=R[i, j]))
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=FS_S)
        ax.set_yticklabels(keys, fontsize=FS_S)
        ax.set_title(f"{'R+' if rg == 'R+' else 'R−'} (n = {len(g)}): Spearman between scores", fontsize=FS_M, color=COL[rg])
    fig.colorbar(im, ax=axes, shrink=0.7)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"124_score_agreement.{ext}", dpi=250)
    plt.close(fig)
    pd.DataFrame(agr).to_csv(OUT / "124_score_agreement.csv", index=False)


if __name__ == "__main__":
    main()
