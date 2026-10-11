"""108 -- Cross-condition generalisation performance (CCGP; Bernardi et al. 2020, Cell) of lick and whisker coding (user 2026-10-11).
Exploratory. Auditory misses are almost absent (median 0-2 per session), so the modality x lick design is not possible; the 2 x 2 design is
stimulus (whisker / none) x lick (yes / no) within trials: WH (whisker hit), WM (whisker miss), FA (false alarm: no-stimulus trial with a
lick), CR (correct rejection). Data: 105 stimulus window [50, 150) ms after trial start, truncated at the first lick (baseline-corrected
spikes/s; stable tested units); sessions with >= MIN_EV events of each of the four classes.
Decoder: logistic regression (L2, C = 0.1, balanced class weights), units z-scored on the training events; performance = balanced accuracy.
  CCGP_lick   train WH vs WM -> test FA vs CR, and train FA vs CR -> test WH vs WM; mean of the two (lick coded the same with or without
              a stimulus = abstract go)
  CCGP_whisk  train WH vs FA -> test WM vs CR, and the reverse; mean (whisker coded the same with or without a lick; low = gated by the
              decision)
  within      5-fold CV balanced accuracy of each of the four training pairs (reference: how decodable each variable is at all)
  PS          parallelism score: cosine between the two coding vectors of a variable (lick: mean WH - mean WM vs mean FA - mean CR;
              whisker: WH - FA vs WM - CR), z-scored units
  AUD         one-sided transfer: train WH vs WM -> test AH vs CR (no auditory no-lick class): balanced accuracy
Chance: each accuracy recomputed with the training labels shuffled (N_SHUF = 20), mean; reported as value - chance.
Statistics as 106 (session = unit; MWU within cohort and between cohorts; cohort x stage mouse permutation). Learners rule (--pop all).
Output: across_days/<ref>/ccgp/{sessions.parquet, ccgp_<pop>.{png,pdf}, stats_<pop>.csv}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../108_ccgp.py [--n-proc 25] [--replot] [--pop learners]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m105 = importlib.import_module("105_trial_rates")
m51 = m105.m51
OUT = m51.OUTROOT / "ccgp"
MIN_EV, N_SHUF, C_REG = 6, 20, 0.1


def zfit(A):
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
    return mu, sd


def fit_test(X, pos, neg, tpos, tneg, y_override=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    idx = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))] if y_override is None else y_override
    mu, sd = zfit(X[:, idx].T)
    clf = LogisticRegression(C=C_REG, max_iter=500, class_weight="balanced").fit((X[:, idx].T - mu) / sd, y)
    ti = np.r_[tpos, tneg]; ty = np.r_[np.ones(len(tpos)), np.zeros(len(tneg))]
    return balanced_accuracy_score(ty, clf.predict((X[:, ti].T - mu) / sd))


def cv_within(X, pos, neg, seed, y_shuf=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    idx = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))] if y_shuf is None else y_shuf; acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y):
        mu, sd = zfit(X[:, idx[tr]].T)
        clf = LogisticRegression(C=C_REG, max_iter=500, class_weight="balanced").fit((X[:, idx[tr]].T - mu) / sd, y[tr])
        acc.append(balanced_accuracy_score(y[te], clf.predict((X[:, idx[te]].T - mu) / sd)))
    return float(np.mean(acc))


def cosine(X, a, b, c, d):
    mu, sd = zfit(X[:, np.r_[a, b, c, d]].T); Z = (X.T - mu) / sd
    v1 = Z[a].mean(0) - Z[b].mean(0); v2 = Z[c].mean(0) - Z[d].mean(0)
    return float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))


def session(args):
    f, = args
    sid = f.stem; z = np.load(f, allow_pickle=True)
    lab, hit, X = z["stim_lab"], z["is_hit"], z["stimwin"].astype(float)
    ok = np.isfinite(X).all(0)
    wh, wm = np.where(ok & (lab == "W") & hit)[0], np.where(ok & (lab == "W") & ~hit)[0]
    fa, cr, ah = (np.where(ok & (lab == k))[0] for k in ("FA", "CR", "AH"))
    n = dict(n_WH=len(wh), n_WM=len(wm), n_FA=len(fa), n_CR=len(cr), n_AH=len(ah), n_units=X.shape[0])
    if min(len(wh), len(wm), len(fa), len(cr)) < MIN_EV:
        return dict(session_id=sid, **n)
    rng = np.random.default_rng(zlib.crc32(f"{sid}|108".encode())); seed = int(rng.integers(1e6))
    pairs = {"lick_W": (wh, wm), "lick_N": (fa, cr), "whisk_L": (wh, fa), "whisk_N": (wm, cr)}
    r = dict(session_id=sid, **n)

    def both(p1, p2, shuf=False):
        (a, b), (c, d) = pairs[p1], pairs[p2]
        y1 = rng.permutation(np.r_[np.ones(len(a)), np.zeros(len(b))]) if shuf else None
        y2 = rng.permutation(np.r_[np.ones(len(c)), np.zeros(len(d))]) if shuf else None
        return 0.5 * (fit_test(X, a, b, c, d, y1) + fit_test(X, c, d, a, b, y2))
    for name, (p1, p2) in {"CCGP_lick": ("lick_W", "lick_N"), "CCGP_whisk": ("whisk_L", "whisk_N")}.items():
        r[name] = both(p1, p2); r[f"{name}_null"] = float(np.mean([both(p1, p2, True) for _ in range(N_SHUF)]))
    for k, (a, b) in pairs.items():
        r[f"within_{k}"] = cv_within(X, a, b, seed)
        r[f"within_{k}_null"] = float(np.mean([cv_within(X, a, b, seed, rng.permutation(np.r_[np.ones(len(a)), np.zeros(len(b))])) for _ in range(N_SHUF)]))
    r["PS_lick"] = cosine(X, wh, wm, fa, cr); r["PS_whisk"] = cosine(X, wh, fa, wm, cr)
    if len(ah) >= MIN_EV:
        r["AUD"] = fit_test(X, wh, wm, ah, cr)
        r["AUD_null"] = float(np.mean([fit_test(X, wh, wm, ah, cr, rng.permutation(np.r_[np.ones(len(wh)), np.zeros(len(wm))])) for _ in range(N_SHUF)]))
    return r


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(f,) for f in files])
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    D = pd.DataFrame(res).merge(meta, on="session_id"); D.to_parquet(OUT / "sessions.parquet", index=False)
    print("computed", len(D), "sessions;", D.CCGP_lick.notna().sum(), "with all four classes", flush=True)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61
    plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(OUT / "sessions.parquet"); D = m61.learner_filter(D) if pop == "learners" else D
    D = D.dropna(subset=["CCGP_lick"])
    for k in ("CCGP_lick", "CCGP_whisk", "AUD", "within_lick_W", "within_lick_N", "within_whisk_L", "within_whisk_N"):
        if k in D:
            D[f"{k}_ex"] = D[k] - D[f"{k}_null"]
    P = [("CCGP_lick_ex", "CCGP − chance", "Lick CCGP\n(train WH/WM, test FA/CR, and back)"), ("PS_lick", "cosine", "Lick parallelism\n(WH−WM vs FA−CR)"),
         ("AUD_ex", "balanced acc − chance", "Whisker lick decoder\n→ AH vs CR"), ("CCGP_whisk_ex", "CCGP − chance", "Whisker CCGP\n(train WH/FA, test WM/CR, and back)"),
         ("PS_whisk", "cosine", "Whisker parallelism\n(WH−FA vs WM−CR)"),
         ("within_lick_W_ex", "CV acc − chance", "Within: WH vs WM"), ("within_lick_N_ex", "CV acc − chance", "Within: FA vs CR"),
         ("within_whisk_L_ex", "CV acc − chance", "Within: WH vs FA"), ("within_whisk_N_ex", "CV acc − chance", "Within: WM vs CR")]
    fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.62))
    gs = fig.add_gridspec(2, 5, hspace=0.95, wspace=0.6, left=0.05, right=0.98, top=0.86, bottom=0.1)
    for i, (col, yl, ttl) in enumerate(P):
        ax = fig.add_subplot(gs[i // 5, i % 5]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, D, col, yl, rng, f"108 {col}", ttl, ref=0)
    fig.suptitle(f"Cross-condition generalisation, stimulus (whisker / none) × lick, 50–150 ms after trial start ({pop}; "
                 f"sessions with ≥ {MIN_EV} WH, WM, FA, CR)", x=0.02, y=0.995, ha="left", fontsize=6.6, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ccgp_{pop}.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("108")]); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning", "n_R− expert",
              "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())


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
