"""106 -- Convergence (R+) vs divergence (R-) of whisker representations across days (user 2026-10-11: "do 1, 2, 3"). Exploratory.
Data: 105 per-trial rates (stable tested units; baseline-corrected spikes/s). Learners rule (--pop all to keep all); unit = session.
1  Go / no-go remapping, stimulus window [50, 150) ms (truncated at the first lick):
   a  population: go axis u = unit(mean_z AH - mean_z CR), 5-fold CV over AH + CR events, units z-scored on the training folds; held-out
      AH and CR and every whisker trial (averaged over folds) projected, per neuron (/ sqrt(N)); d_W = p_W - p_CR (all whisker trials),
      d_WH, d_WM (hits, misses), d_AC = p_AH - p_CR (axis separation)
   b  per neuron: go-selective neurons = AH vs CR-A significant (label permutation, N_PERM, p < 0.05; CR split in random halves A / B);
      SI_go = (|s_W,CR-B| - |s_W,AH|) / (|s_W,CR-B| + |s_W,AH|), cross-validated |s| (product of random halves, N_SPLIT);
      +1: whisker go-like (differs from no-go, like AH); -1: no-go-like. For all whisker trials and for misses only.
2  Decoding (logistic regression, L2, C = C_REG, units z-scored on the training events, per 50-ms window); 2026-10-11: accuracy =
   balanced 5-fold CV test accuracy on the training classes; chance = the same with the training labels shuffled (N_SHUF shuffles,
   mean); transfer = fraction of test events decoded as the positive class, minus the mean fraction from the shuffled-label decoders:
   lick-aligned: train AH vs SL (5-fold CV accuracy on AH / SL), test WH: fraction of WH decoded as AH; reverse: train WH vs SL, test AH
   stimulus-aligned: train AH vs CR, test whisker trials (valid windows only, i.e. before the first lick): fraction decoded as AH ("go")
3  Subspace overlap, lick-aligned windows [-300, 0] ms: trajectories relative to spontaneous licks, D_X = mean_X - mean_SL (units x
   time, z-scored units); AH trials split in halves A / B; U = top K PCs of D_AH-A (uncentred); alignment VA(X) = ||U' D_X||^2 / ||D_X||^2;
   normalised alignment of WH = VA(WH) / VA(AH-B) (1 = WH as aligned as held-out AH); N_SPLIT_SUB random splits averaged.
Output: across_days/<ref>/convergence_divergence/{sessions.parquet, timecourses.parquet, figure_<pop>.{png,pdf}, stats_<pop>.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../106_convergence_divergence.py [--n-proc 25] [--replot] [--pop learners]
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
m80 = importlib.import_module("080_single_unit_geometry")
m51 = m105.m51
OUT = m51.OUTROOT / "convergence_divergence"
N_PERM, N_SPLIT, N_SPLIT_SUB, K, C_REG, N_SHUF = 500, 50, 20, 3, 0.1, 20


def zfit(A):
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
    return mu, sd


def go_axis(X, ia, ic, iw, seed):
    from sklearn.model_selection import StratifiedKFold
    y_idx = np.r_[ia, ic]; y = np.r_[np.ones(len(ia)), np.zeros(len(ic))]
    hold = np.full(len(y_idx), np.nan); pw = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(y_idx, y):
        mu, sd = zfit(X[:, y_idx[tr]].T); Z = (X.T - mu) / sd
        d = Z[y_idx[tr][y[tr] == 1]].mean(0) - Z[y_idx[tr][y[tr] == 0]].mean(0); u = d / max(np.linalg.norm(d), 1e-12)
        hold[te] = Z[y_idx[te]] @ u; pw.append(Z[iw] @ u)
    sN = np.sqrt(X.shape[0])
    return hold[y == 1].mean() / sN, hold[y == 0].mean() / sN, np.mean(pw, 0) / sN


def perm_sel(X, ia, ib, rng):
    obs = m80.auc_sel(X, ia, ib); idx = np.r_[ia, ib]; na = len(ia); cnt = np.zeros(len(X))
    for _ in range(N_PERM):
        p = rng.permutation(idx); cnt += np.abs(m80.auc_sel(X, p[:na], p[na:])) >= np.abs(obs) - 1e-12
    return obs, (1 + cnt) / (1 + N_PERM)


def cv_abs(X, ia, ib, rng):
    acc = np.zeros(len(X))
    for _ in range(N_SPLIT):
        pa, pb = rng.permutation(ia), rng.permutation(ib)
        acc += m80.auc_sel(X, pa[: len(pa) // 2], pb[: len(pb) // 2]) * m80.auc_sel(X, pa[len(pa) // 2:], pb[len(pb) // 2:])
    return np.sqrt(np.clip(acc / N_SPLIT, 0, None))


def si(X, iw, ia, icB, rng):
    a, b = cv_abs(X, iw, icB, rng), cv_abs(X, iw, ia, rng); den = a + b
    return np.where(den > 0, (a - b) / np.where(den > 0, den, 1), np.nan)


def _fit_eval(Xb, idx, y, test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y):
        mu, sd = zfit(Xb[:, idx[tr]].T)
        clf = LogisticRegression(C=C_REG, max_iter=500, class_weight="balanced").fit((Xb[:, idx[tr]].T - mu) / sd, y[tr])
        acc.append(balanced_accuracy_score(y[te], clf.predict((Xb[:, idx[te]].T - mu) / sd)))
    mu, sd = zfit(Xb[:, idx].T)
    clf = LogisticRegression(C=C_REG, max_iter=500, class_weight="balanced").fit((Xb[:, idx].T - mu) / sd, y)
    return float(np.mean(acc)), float(clf.predict((Xb[:, test].T - mu) / sd).mean())


def transfer(Xb, tr_pos, tr_neg, test, seed):
    """Xb units x events for one window; returns balanced CV accuracy, its shuffle mean, transfer fraction, its shuffle mean"""
    idx = np.r_[tr_pos, tr_neg]; y = np.r_[np.ones(len(tr_pos)), np.zeros(len(tr_neg))]
    if min(len(tr_pos), len(tr_neg)) < 5 or len(test) < 2:
        return dict(acc=np.nan, acc_null=np.nan, frac=np.nan, frac_null=np.nan)
    acc, fr = _fit_eval(Xb, idx, y, test, seed)
    rng = np.random.default_rng(seed); nul = [_fit_eval(Xb, idx, rng.permutation(y), test, seed) for _ in range(N_SHUF)]
    return dict(acc=acc, acc_null=float(np.mean([n[0] for n in nul])), frac=fr, frac_null=float(np.mean([n[1] for n in nul])))


def subspace(L, lab, c, rng):
    w = (c >= -0.3) & (c <= 0.0)
    Lw = L[:, :, w]; mu = Lw.reshape(len(Lw), -1).mean(1); sd = Lw.reshape(len(Lw), -1).std(1); sd[sd == 0] = 1
    Z = (Lw - mu[:, None, None]) / sd[:, None, None]
    sl, wh, ah = (np.where(lab == k)[0] for k in ("SL", "WH", "AH"))
    if min(len(sl), len(wh), len(ah)) < 6:
        return np.nan, np.nan
    msl = Z[:, sl].mean(1); D_wh = Z[:, wh].mean(1) - msl; out_w, out_a = [], []
    va = lambda U, D: (np.linalg.norm(U.T @ D) ** 2) / max(np.linalg.norm(D) ** 2, 1e-12)
    for _ in range(N_SPLIT_SUB):
        p = rng.permutation(ah); A, B = p[: len(p) // 2], p[len(p) // 2:]
        U = np.linalg.svd(Z[:, A].mean(1) - msl, full_matrices=False)[0][:, :K]
        out_a.append(va(U, Z[:, B].mean(1) - msl)); out_w.append(va(U, D_wh))
    return float(np.mean(out_w)), float(np.mean(out_a))


def session(args):
    f, = args
    sid = f.stem; z = np.load(f, allow_pickle=True)
    rng = np.random.default_rng(zlib.crc32(f"{sid}|106".encode())); seed = int(rng.integers(1e6))
    lab, hit = z["stim_lab"], z["is_hit"]; X = z["stimwin"].astype(float)
    ok = np.isfinite(X).all(0)
    iw, ia, ic = (np.where(ok & (lab == k))[0] for k in ("W", "AH", "CR"))
    r = dict(session_id=sid, n_units=X.shape[0])
    if min(len(iw), len(ia), len(ic)) >= 6:
        pa, pc, pw = go_axis(X, ia, ic, iw, seed)
        h = hit[iw]
        r.update(d_AC=pa - pc, d_W=pw.mean() - pc, d_WH=pw[h].mean() - pc if h.sum() else np.nan, d_WM=pw[~h].mean() - pc if (~h).sum() else np.nan)
        p = rng.permutation(ic); cA, cB = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
        s, pv = perm_sel(X, ia, cA, rng); sel = pv < 0.05
        r["n_go_neurons"] = int(sel.sum())
        if sel.sum() >= 3:
            r["SI_go"] = np.nanmean(si(X[sel], iw, ia, cB, rng))
            wm = iw[~hit[iw]]
            r["SI_go_WM"] = np.nanmean(si(X[sel], wm, ia, cB, rng)) if len(wm) >= 6 else np.nan
    # decoding over time
    tc = []
    Sb, sc = z["stimbins"].astype(float), z["stim_c"]
    for k, cc in enumerate(sc):
        okb = np.isfinite(Sb[:, :, k]).all(0)
        o = transfer(Sb[:, :, k], np.where(okb & (lab == "AH"))[0], np.where(okb & (lab == "CR"))[0], np.where(okb & (lab == "W"))[0], seed)
        tc.append(dict(session_id=sid, align="stim", test="W as go (train AH vs CR)", t=cc * 1e3, **o))
    Lb, llab, lc = z["lickbins"].astype(float), z["lick_lab"], z["lick_c"]
    a_, s_, w_ = (np.where(llab == k)[0] for k in ("AH", "SL", "WH"))
    for k, cc in enumerate(lc):
        tc.append(dict(session_id=sid, align="lick", test="WH as AH (train AH vs SL)", t=cc * 1e3, **transfer(Lb[:, :, k], a_, s_, w_, seed)))
        tc.append(dict(session_id=sid, align="lick", test="AH as WH (train WH vs SL)", t=cc * 1e3, **transfer(Lb[:, :, k], w_, s_, a_, seed)))
    vw, vb = subspace(Lb, llab, lc, rng)
    r.update(VA_WH=vw, VA_AHB=vb, VA_norm=vw / vb if vb and vb > 0 else np.nan)
    return r, tc


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(f,) for f in files])
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    pd.DataFrame([r for r, _ in res]).merge(meta, on="session_id").to_parquet(OUT / "sessions.parquet", index=False)
    pd.DataFrame([x for _, t in res for x in t]).merge(meta, on="session_id").to_parquet(OUT / "timecourses.parquet", index=False)
    print("computed", len(res), flush=True)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2")
    m62, m61 = m88.m62, m88.m61; COH = m88.COH
    plt = m62.setup(); rng = np.random.default_rng(0)
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    S = flt(pd.read_parquet(OUT / "sessions.parquet")); T = flt(pd.read_parquet(OUT / "timecourses.parquet"))
    GR = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
    fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.62))
    gs = fig.add_gridspec(2, 5, hspace=0.95, wspace=0.6, left=0.05, right=0.98, top=0.86, bottom=0.1)
    panels1 = [("d_W", "whisker − CR projection\non go axis (per neuron)", "Whisker trials on the go axis\n(AH − CR, 50–150 ms)"),
               ("d_WM", "whisker miss − CR projection\n(per neuron)", "Whisker misses on the go axis"),
               ("SI_go", "SI_go (−1 no-go … +1 go-like)", "Go-neurons: whisker trials\ngo- vs no-go-like"),
               ("SI_go_WM", "SI_go, misses only", "Go-neurons: whisker misses"),
               ("VA_norm", "WH alignment / held-out AH", "Subspace overlap: WH pre-lick\nin the AH − SL subspace")]
    for j, (col, yl, ttl) in enumerate(panels1):
        ax = fig.add_subplot(gs[0, j]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, S, col, yl, rng, f"106 {col}", ttl, ref=0 if col != "VA_norm" else None)
    T["acc_ex"] = T.acc - T.acc_null; T["frac_ex"] = T.frac - T.frac_null
    DEC = [("lick", "WH as AH (train AH vs SL)", (-100, 0), "AH vs SL decoder; WH as AH"),
           ("stim", "W as go (train AH vs CR)", (75, 150), "AH vs CR decoder; whisker as go"),
           ("lick", "AH as WH (train WH vs SL)", (-100, 0), "WH vs SL decoder; AH as WH")]

    def tc_panel(spec, align, test, col, ttl, ylab):
        sub = spec.subgridspec(2, 2, hspace=0.45, wspace=0.12); axs = []
        d0 = T[(T["align"] == align) & (T.test == test)]
        for j, g in enumerate(GR):
            ax = fig.add_subplot(sub[j // 2, j % 2]); axs.append(ax)
            d = d0[(d0.cohort == g[0]) & (d0.stage == g[1])]
            piv = d.pivot_table(index="session_id", columns="t", values=col); mu, se = piv.mean(0), piv.sem(0)
            ax.fill_between(piv.columns, mu - se, mu + se, color=COH[g[0]], alpha=0.2, lw=0); ax.plot(piv.columns, mu, color=COH[g[0]], lw=0.9)
            ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.4, ls=":")
            ax.set_title(f"{g[0].replace('-', '−')} {'D0' if g[1] == 'learning' else 'E'} ({piv.shape[0]})", color=COH[g[0]], fontsize=4.6, pad=1)
            ax.tick_params(labelsize=4.2)
            if j < 2:
                ax.set_xticklabels([])
        yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
        for j, a in enumerate(axs):
            a.set_ylim(*yl)
            if j % 2:
                a.set_yticklabels([])
        axs[2].set_xlabel("from lick (ms)" if align == "lick" else "from stimulus (ms)", fontsize=4.5); axs[0].set_ylabel(ylab, fontsize=4.5)
        bb = spec.get_position(fig); fig.text(bb.x0, bb.y1 + 0.035, ttl, fontsize=5.0)
    plt.close(fig)                                                   # first pass only fills the "106 " stats rows
    # decoding figure: rows = decoders; columns: accuracy over chance (time), transfer over chance (time), window summaries
    fig = plt.figure(figsize=(m62.W_IN * 1.1, m62.W_IN * 0.85))
    gs2 = fig.add_gridspec(3, 4, hspace=0.95, wspace=0.6, left=0.06, right=0.98, top=0.9, bottom=0.07)
    for i, (align, test, win, ttl) in enumerate(DEC):
        tc_panel(gs2[i, 0], align, test, "acc_ex", f"{ttl}\nbalanced accuracy − shuffle", "acc − chance")
        tc_panel(gs2[i, 1], align, test, "frac_ex", f"{ttl}\ntransfer − shuffle", "fraction − chance")
        d = T[(T["align"] == align) & (T.test == test) & (T.t >= win[0]) & (T.t < win[1])].groupby(["session_id", "mouse_id", "cohort", "stage"])[["acc", "acc_null", "acc_ex", "frac_ex"]].mean().reset_index()
        for j, (col, yl, tt) in enumerate((("acc_ex", "balanced acc − chance", f"accuracy, {win[0]}–{win[1]} ms"),
                                            ("frac_ex", "transfer − chance", f"transfer, {win[0]}–{win[1]} ms"))):
            ax = fig.add_subplot(gs2[i, 2 + j]); ax.set_box_aspect(1)
            m88.mean_panel2(ax, d, col, yl, rng, f"106 {test} {col}", tt, ref=0)
    fig.suptitle(f"Decoding: accuracy and transfer above label-shuffle chance ({pop})", x=0.02, y=0.995, ha="left", fontsize=7, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"decoding_{pop}.{ext}", dpi=300)
    plt.close(fig)
    # main figure: row 1 as before, row 2 = the three transfer-over-chance summaries + 2 accuracy summaries
    fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.62))
    gs = fig.add_gridspec(2, 5, hspace=0.95, wspace=0.6, left=0.05, right=0.98, top=0.86, bottom=0.1)
    for j, (col, yl, ttl) in enumerate(panels1):
        ax = fig.add_subplot(gs[0, j]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, S, col, yl, rng, f"106b {col}", ttl, ref=0 if col != "VA_norm" else None)
    for j, (align, test, win, ttl) in enumerate(DEC):
        d = T[(T["align"] == align) & (T.test == test) & (T.t >= win[0]) & (T.t < win[1])].groupby(["session_id", "mouse_id", "cohort", "stage"])[["acc_ex", "frac_ex"]].mean().reset_index()
        ax = fig.add_subplot(gs[1, j]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, d, "frac_ex", "transfer − chance", rng, f"106b {test} frac", f"{ttl}\n{win[0]}–{win[1]} ms", ref=0)
        if j < 2:
            ax = fig.add_subplot(gs[1, 3 + j]); ax.set_box_aspect(1)
            m88.mean_panel2(ax, d, "acc_ex", "balanced acc − chance", rng, f"106b {test} acc", f"{ttl.split(';')[0]} accuracy\n{win[0]}–{win[1]} ms", ref=0)
    fig.suptitle(f"Convergence (R+) vs divergence (R−) of whisker representations ({pop})", x=0.02, y=0.995, ha="left", fontsize=7, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure_{pop}.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("106 ")]); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning", "n_R− expert",
              "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print(T.groupby(["align", "test"])[["acc", "acc_null", "frac", "frac_null"]].mean().round(3))


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
