"""Pseudo-population decoding with a hierarchical bootstrap: do whisker hits look like auditory hits before the lick,
more in R+ than R- after learning?

Data: 051 per-session pre-lick rates (100 ms before the corrected first lick, baseline-corrected; trials active,
perf != 6, warm-up cut, A1 trim); units quality good or mua with mean raw pre-lick rate >= 0.1 Hz. Sessions need
>= MIN_TRIALS trials of each class (WH, AH, FA).
Hierarchical bootstrap, per cohort x stage group, per iteration:
  1. mice: sample the group's mice with replacement;
  2. sessions: for each sampled mouse, sample one of its sessions (several only for expert mice);
  3. neurons: sample M neurons from the pooled units of the sampled sessions (whole brain, or one area group) without
     replacement (with replacement if the pool is smaller than M);
  4. trials: in every sampled session, split each class's trials at random into a training and a test half, and draw
     N_PS pseudo-trials per class from each half; the same trial indices are used for all neurons of a session (keeps
     within-session correlations; correlations across sessions are absent by construction).
Step 4 is repeated N_SPLIT times per iteration (same mice / sessions / neurons); numerators and denominators of the
ratio readouts are averaged over splits before dividing; values clipped to [-1, 2].
Within-session null (chance), identical for both references (false alarms and spontaneous licks; user 2026-10-03):
in every iteration the same mice / sessions / neurons are also run with linearly shifted event indices in every sampled
session: the event labels keep their time order and each label takes the activity of the event k positions later
(one random k per session and iteration, |k| in [5, n // 3], no wrap-around; events whose shifted index falls outside
the session are dropped). This keeps the temporal autocorrelation of event types (e.g. bouts of licking, engagement
drifts) and of neural activity while breaking their alignment. Readouts are reported chance-corrected (bacc - null;
ratios component-wise: numerator and denominator each minus their null value, then the ratio) and raw.
Readouts (units z-scored on the training pseudo-trials):
  transfer  L2 logistic regression (C = 0.05) trained on AH vs FA training pseudo-trials, applied to test pseudo-trials:
            (mean P(AH | WH_test) - mean P(AH | FA_test)) / (mean P(AH | AH_test) - mean P(AH | FA_test))
  lambda    cross-validated projection: (WH_test - FA_test) . (AH_train - FA_train) / (AH_test - FA_test) . (AH_train - FA_train)
  bacc      balanced accuracy of AH vs FA on test pseudo-trials
  transfer_bin  yes/no version of transfer: (P(WH decoded AH) - FPR) / (TPR - FPR) (component-wise chance-corrected)
  num_prob  numerator of transfer alone, mean P(AH | WH_test) - mean P(AH | FA_test), minus its null value
Statistics: per group, bootstrap distribution of the readouts (B iterations); expert - learning change per cohort and
the interaction [E-L](R+) - [E-L](R-) with 95% percentile CIs and a two-sided bootstrap p (2 x the fraction of the
bootstrap distribution on the other side of 0). Whole brain (M = M_PERM): permutation test of the interaction,
cohort labels permuted across mice (same label for a mouse at both stages), each permutation re-running the bootstrap
with B_PERM iterations; p = (1 + #|null| >= |obs|) / (1 + n_perm).
Areas: area groups with >= MIN_MICE mice in both stages of a cohort (within-cohort inclusion); interaction only if both
cohorts qualify.
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/pseudopop/<population>/
"""
import argparse
import importlib
import json
import pathlib
import sys
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
BASE = m51.OUTROOT
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
MIN_TRIALS = 6
N_SPLIT = 10                # trial splits per bootstrap iteration (num / den averaged before the ratio)
CLIP = (-1.0, 2.0)
MIN_MICE = 3
N_PS = 100
C_REG = 0.05
UNIT_SET = ("good", "mua")
DATA = {}


# ------------------------------------------------------------------ data
def load(pop):
    W = pd.read_parquet(BASE / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    if pop == "learners":
        m61 = importlib.import_module("061_roc_prelick_learners")
        W = m61.learner_filter(W)                     # non-learners: day-0 sessions dropped, expert kept
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    S, info = {}, []
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        order = np.argsort(z["trial_start"])                       # events in time order (needed by the shift null)
        lab = z["cls"][order]
        if min((lab == c).sum() for c in m51.CLASSES) < MIN_TRIALS:
            continue
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "quality_label", "area_group",
                                                      "cohort", "stage", "mouse_id"]], on=["electrode_group", "cluster_id"],
                    how="left")
        if K.cohort.isna().all():
            continue
        ok = K.quality_label.isin(UNIT_SET).to_numpy() & (z["raw"].mean(1) >= m51.MIN_FR)
        if ok.sum() < 5:
            continue
        S[r.session_id] = dict(X=z["rates"][ok][:, order].astype(np.float32), area=K.area_group.to_numpy()[ok].astype(str),
                               idx={c: np.where(lab == c)[0] for c in m51.CLASSES}, n=len(lab))
        info.append(dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                         stage=K.stage.dropna().iloc[0], n_units=int(ok.sum())))
    return S, pd.DataFrame(info)


def group_tree(info, coh_of_mouse=None):
    """{group: {mouse: [sessions]}}; coh_of_mouse overrides cohorts (permutation)"""
    T = {g: {} for g in GROUPS}
    for r in info.itertuples():
        c = coh_of_mouse[r.mouse_id] if coh_of_mouse is not None else r.cohort
        T[(c, r.stage)].setdefault(r.mouse_id, []).append(r.session_id)
    return T


# ------------------------------------------------------------------ one bootstrap iteration
def sample_neurons(tree, region, M, rng):
    """steps 1-3: mice -> one session per sampled mouse -> M neurons from the pooled units; returns [(session, unit idx)]"""
    S = DATA["S"]
    mice = list(tree)
    if len(mice) == 0:
        return None
    inst = [rng.choice(tree[m]) for m in rng.choice(mice, len(mice))]
    units = [np.where(S[s]["area"] == region)[0] if region != "all" else np.arange(len(S[s]["X"])) for s in inst]
    sizes = np.array([len(u) for u in units]); tot = sizes.sum()
    if tot < 5:
        return None
    pick = np.sort(rng.choice(tot, M, replace=tot < M))
    off = np.r_[0, np.cumsum(sizes)]
    sel = []
    for j, s in enumerate(inst):
        k = pick[(pick >= off[j]) & (pick < off[j + 1])] - off[j]
        if len(k):
            sel.append((s, units[j][k]))
    return sel


NULL = "shift"   # decoders: linear-shift null for both references (user 2026-10-03; label permutation reserved for ROC)
MIN_SHIFT = 5


def null_idx(s, rng):
    """event indices per class under the within-session null. shift: labels stay in time order, activity taken from
    event i + k (random k, |k| in [MIN_SHIFT, n // 3], no wrap-around); perm: auditory-hit / reference labels permuted
    (whisker hits unchanged)"""
    S = DATA["S"]; idx = S[s]["idx"]; n = S[s]["n"]
    if NULL == "perm":
        pool = np.r_[idx["AH"], idx["FA"]]; p = rng.permutation(pool)
        return {"WH": idx["WH"], "AH": p[:len(idx["AH"])], "FA": p[len(idx["AH"]):]}
    kmax = max(MIN_SHIFT + 1, n // 3)
    k = int(rng.integers(MIN_SHIFT, kmax)) * (1 if rng.random() < 0.5 else -1)
    out = {}
    for c, v in idx.items():
        w = v + k
        out[c] = w[(w >= 0) & (w < n)]
    return out


def sample_trials(sel, M, rng, null=False):
    """step 4: per sampled session, random train / test halves per class, N_PS pseudo-trials per class from each half
    (same trial indices for all neurons of a session); null=True uses the within-session null indices"""
    S = DATA["S"]
    tr = {c: np.zeros((N_PS, M), np.float32) for c in m51.CLASSES}
    te = {c: np.zeros((N_PS, M), np.float32) for c in m51.CLASSES}
    col = 0
    for s, u in sel:
        X = S[s]["X"][u]
        idx = null_idx(s, rng) if null else S[s]["idx"]
        for c in m51.CLASSES:
            src = idx[c] if len(idx[c]) >= 2 else S[s]["idx"][c]
            ii = rng.permutation(src); h = len(ii) // 2
            tr[c][:, col:col + len(u)] = X[:, rng.choice(ii[:h], N_PS)].T
            te[c][:, col:col + len(u)] = X[:, rng.choice(ii[h:], N_PS)].T
        col += len(u)
    return tr, te


def readouts(tr, te):
    """numerators / denominators (averaged over trial splits before the ratio) and balanced accuracy"""
    from sklearn.linear_model import LogisticRegression
    A = np.vstack([tr["AH"], tr["FA"]]); y = np.r_[np.ones(N_PS), np.zeros(N_PS)]
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
    z = lambda x: (x - mu) / sd
    clf = LogisticRegression(C=C_REG, max_iter=2000).fit(z(A), y)
    pA, pF, pW = (clf.predict_proba(z(te[c]))[:, 1] for c in ["AH", "FA", "WH"])
    ax_ = z(tr["AH"]).mean(0) - z(tr["FA"]).mean(0)
    return dict(t_num=pW.mean() - pF.mean(), t_den=pA.mean() - pF.mean(),
                l_num=(z(te["WH"]).mean(0) - z(te["FA"]).mean(0)) @ ax_ / len(ax_),
                l_den=(z(te["AH"]).mean(0) - z(te["FA"]).mean(0)) @ ax_ / len(ax_),
                bacc=((pA > 0.5).mean() + (pF <= 0.5).mean()) / 2,
                b_num=(pW > 0.5).mean() - (pF > 0.5).mean(), b_den=(pA > 0.5).mean() - (pF > 0.5).mean())


def run_boot(tree, region, M, B, seed):
    rng = np.random.default_rng(seed)
    out = np.full((B, len(READOUTS)), np.nan)
    for b in range(B):
        sel = sample_neurons(tree, region, M, rng)
        if sel is None:
            continue
        R = [readouts(*sample_trials(sel, M, rng)) for _ in range(N_SPLIT)]
        R0 = [readouts(*sample_trials(sel, M, rng, null=True)) for _ in range(N_SPLIT)]    # same neurons, null labels
        m = {k: np.mean([r[k] for r in R]) for k in R[0]}
        m0 = {k: np.mean([r[k] for r in R0]) for k in R0[0]}
        ratio = lambda a, d, thr=0.0: np.clip(a / d, *CLIP) if d > thr else np.nan
        t, l = ratio(m["t_num"], m["t_den"]), ratio(m["l_num"], m["l_den"])
        # chance-corrected (component-wise: numerator and denominator minus their null value, then ratio)
        tc = ratio(m["t_num"] - m0["t_num"], m["t_den"] - m0["t_den"], 0.02)
        lc = ratio(m["l_num"] - m0["l_num"], m["l_den"] - m0["l_den"], 0.0)
        bc = ratio(m["b_num"] - m0["b_num"], m["b_den"] - m0["b_den"], 0.02)
        out[b] = [tc, lc, m["bacc"] - m0["bacc"], t, l, m["bacc"], bc, m["t_num"] - m0["t_num"],
                  ratio(m["b_num"], m["b_den"], 0.02), m["t_num"]]
    return out


READOUTS = ["transfer", "lambda", "bacc", "transfer_raw", "lambda_raw", "bacc_raw",   # first three chance-corrected
            "transfer_bin", "num_prob", "transfer_bin_raw", "num_prob_raw"]
# added 2026-10-03 (Fig 5 equivalents of the single-session readouts): transfer_bin = yes/no transfer
# [(P(WH decoded AH) - FPR) / (TPR - FPR)], num_prob = P(AH | WH) - P(AH | ref) (chance-corrected: minus the shift null)


# ------------------------------------------------------------------ jobs
def job_region(args):
    region, M, B, seed = args
    tree = DATA["tree"]
    res = {}
    for gi, g in enumerate(GROUPS):
        res[g] = run_boot(tree[g], region, M, B, seed + gi)
    return region, M, res


def job_perm(args):
    k, M, B = args
    rng = np.random.default_rng(10_000 + k)
    info = DATA["info"]
    mice = info.drop_duplicates("mouse_id")[["mouse_id", "cohort"]]
    perm = dict(zip(mice.mouse_id, rng.permutation(mice.cohort.to_numpy())))
    tree = group_tree(info, perm)
    means = {}
    for gi, g in enumerate(GROUPS):
        r = run_boot(tree[g], "all", M, B, 20_000 + 10 * k + gi)
        means[g] = np.nanmean(r, 0)
    return k, (means[GROUPS[1]] - means[GROUPS[0]]) - (means[GROUPS[3]] - means[GROUPS[2]])


def eligible_regions(info):
    S = DATA["S"]
    rows = []
    for r in info.itertuples():
        for a in np.unique(S[r.session_id]["area"]):
            if a not in ("nan", "None"):
                rows.append(dict(region=a, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage,
                                 n=int((S[r.session_id]["area"] == a).sum())))
    A = pd.DataFrame(rows)
    A = A[A.n >= 1]
    nm = A.groupby(["region", "cohort", "stage"]).mouse_id.nunique().unstack(["cohort", "stage"]).fillna(0)
    ok = {c: {g for g in nm.index if nm.loc[g].get((c, "learning"), 0) >= MIN_MICE and nm.loc[g].get((c, "expert"), 0) >= MIN_MICE}
          for c in ["R+", "R-"]}
    return ok, nm


def summarize(res, label):
    """per readout: group means / CIs, per-cohort change, interaction (bootstrap CI and p)"""
    rows = []
    for k, name in enumerate(READOUTS):
        v = {g: res[g][:, k] if g in res else np.full(1, np.nan) for g in GROUPS}
        row = dict(label, readout=name)
        for g in GROUPS:
            x = v[g][np.isfinite(v[g])]
            row[f"mean {g[0]} {g[1]}"] = x.mean() if len(x) else np.nan
            row[f"lo {g[0]} {g[1]}"], row[f"hi {g[0]} {g[1]}"] = (np.percentile(x, [2.5, 97.5]) if len(x) else (np.nan, np.nan))
        n = min(len(v[g]) for g in GROUPS)
        dp = v[GROUPS[1]][:n] - v[GROUPS[0]][:n]; dm = v[GROUPS[3]][:n] - v[GROUPS[2]][:n]
        for nm_, d in [("dR+", dp), ("dR-", dm), ("interaction", dp - dm)]:
            d = d[np.isfinite(d)]
            if len(d) < 20:
                continue
            row[f"{nm_}"] = d.mean(); row[f"{nm_} lo"], row[f"{nm_} hi"] = np.percentile(d, [2.5, 97.5])
            row[f"{nm_} p_boot"] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ figure
def figure(T, P, out, pop, Ms):
    m62 = importlib.import_module("062_pub_convergence_figures")
    plt = m62.setup(); COH = m62.COH
    fig = plt.figure(figsize=(m62.W_IN, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.87, bottom=0.1)
    wb = T[T.region == "all"]
    axs = []
    for j, ro in enumerate(["transfer", "lambda"]):
        ax = fig.add_subplot(gs[0, j]); axs.append(ax)
        d = wb[(wb.readout == ro) & (wb.M == P["M_PERM"])].iloc[0]
        for c, x0 in [("R+", 0), ("R-", 1.6)]:
            m = [d[f"mean {c} {s}"] for s in ["learning", "expert"]]
            lo = [d[f"lo {c} {s}"] for s in ["learning", "expert"]]; hi = [d[f"hi {c} {s}"] for s in ["learning", "expert"]]
            xs = [x0, x0 + 0.8]
            ax.plot(xs, m, color=COH[c], lw=1.2)
            for x, mm, l, h, s in zip(xs, m, lo, hi, ["learning", "expert"]):
                ax.plot([x, x], [l, h], color=COH[c], lw=1)
                ax.plot(x, mm, "o", ms=4, color=COH[c], mfc="white" if s == "learning" else COH[c])
        ax.axhline(0, color=m62.CL["FA"], lw=0.5, ls=(0, (2, 2))); ax.axhline(1, color=m62.CL["AH"], lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0, 0.8, 1.6, 2.4], ["L", "E", "L", "E"]); ax.set_xlim(-0.4, 2.8)
        for x, c in [(0.4, "R+"), (2.0, "R-")]:
            ax.text(x, -0.16, c.replace("-", "−"), transform=ax.get_xaxis_transform(), ha="center", color=COH[c],
                    weight="bold", fontsize=6.5)
        pi = P["perm"].get(ro, np.nan)
        ax.set_title(f"{'Decoder transfer' if ro == 'transfer' else 'λ'} (M = {P['M_PERM']})\n"
                     f"interaction {d['interaction']:+.2f} [{d['interaction lo']:+.2f}, {d['interaction hi']:+.2f}]\n"
                     f"boot {m62.fmt_p(d['interaction p_boot'])}, perm {m62.fmt_p(pi)}", fontsize=5.4)
        ax.set_ylabel("Transfer − chance (ref = 0, AH = 1)" if ro == "transfer" else "λ − chance")
    # interaction vs M
    for j, ro in enumerate(["transfer", "lambda"]):
        ax = fig.add_subplot(gs[0, 2 + j]); axs.append(ax)
        d = wb[wb.readout == ro].sort_values("M")
        for nm_, col in [("dR+", COH["R+"]), ("dR-", COH["R-"]), ("interaction", "0.15")]:
            ax.fill_between(d.M, d[f"{nm_} lo"], d[f"{nm_} hi"], color=col, alpha=0.15, lw=0)
            ax.plot(d.M, d[nm_], "o-", color=col, ms=3, lw=1,
                    label={"dR+": "Δ R+", "dR-": "Δ R−", "interaction": "Δ R+ − Δ R−"}[nm_])
        ax.axhline(0, color="0.5", lw=0.5); ax.set_xscale("log"); ax.set_xticks(Ms, [str(m) for m in Ms])
        ax.set_xlabel("Neurons per pseudo-population (M)"); ax.set_ylabel("Expert − learning")
        ax.set_title(f"{'Transfer' if ro == 'transfer' else 'λ'}: change vs population size", fontsize=5.8)
        if j == 0:
            ax.legend(frameon=False, fontsize=4.8, loc="upper left")
    # areas
    ar = T[(T.region != "all") & (T.readout == "transfer") & (T.M == P["M_AREA"])]
    import ephys_utilities.allen_utils.allen_utils as au
    order = [g for g in au.get_area_group_custom_order() if g in set(ar.region)]
    for j, ro in enumerate(["transfer", "lambda"]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); axs.append(ax)
        a = T[(T.region != "all") & (T.readout == ro) & (T.M == P["M_AREA"])].set_index("region")
        x = np.arange(len(order))
        for k, (c, nm_) in enumerate([("R+", "dR+"), ("R-", "dR-")]):
            for i, g in enumerate(order):
                if g not in a.index or g not in P["ok"][c] or not np.isfinite(a.loc[g].get(nm_, np.nan)):
                    continue
                r = a.loc[g]; xx = i + (k - 0.5) * 0.28
                sig = r[f"{nm_} p_boot"] < 0.05
                ax.plot([xx, xx], [r[f"{nm_} lo"], r[f"{nm_} hi"]], color=COH[c], lw=0.8)
                ax.plot(xx, r[nm_], "o", ms=3.5, color=COH[c], mfc=COH[c] if sig else "white", mew=0.8)
        top = ax.get_ylim()[1]
        for i, g in enumerate(order):
            if g in a.index and g in P["ok"]["R+"] and g in P["ok"]["R-"] and np.isfinite(a.loc[g].get("interaction p_boot", np.nan)):
                ax.text(i, top, m62.fmt_p(a.loc[g, "interaction p_boot"]).replace("p ", ""), ha="center", va="bottom",
                        fontsize=4.4, rotation=90)
        ax.axhline(0, color="0.3", lw=0.5)
        ax.set_xticks(x, [g.replace(" areas", "").replace("Somatosensory-", "SS-").replace("Lateral septal complex", "LSX")
                          .replace("Posterior parietal", "PPC").replace("Retrosplenial", "RSP") for g in order], rotation=40, ha="right")
        ax.set_ylabel("Expert − learning (95% CI)")
        ax.set_title(f"Areas, {'transfer' if ro == 'transfer' else 'λ'} (M = {P['M_AREA']}; filled: bootstrap p < 0.05; "
                     "top: interaction p)", fontsize=5.8, pad=22)
    m62.letter_row(fig, axs[:4], "abcd"); m62.letter_row(fig, axs[4:], "ef")
    fig.suptitle(f"Hierarchical-bootstrap pseudo-populations (mice → sessions → neurons → trials; {pop}; "
                 f"{m51.REF.upper()} reference, chance-corrected by {'linear shift' if NULL == 'shift' else 'label permutation'})", x=0.08, y=0.985,
                 ha="left", fontsize=7.5, weight="bold")
    m62.save(fig, out, "pseudopop_summary"); plt.close(fig)


def main(a):
    t0 = time.time()
    out = BASE / "pseudopop" / a.population; out.mkdir(parents=True, exist_ok=True)
    S, info = load(a.population)
    DATA["S"], DATA["info"], DATA["tree"] = S, info, group_tree(info)
    ok, nm = eligible_regions(info)
    regions = sorted(ok["R+"] | ok["R-"])
    print(f"[064] {len(S)} sessions, {info.mouse_id.nunique()} mice; groups "
          f"{info.groupby(['cohort', 'stage']).agg(s=('session_id', 'size'), m=('mouse_id', 'nunique')).to_dict('index')}; "
          f"areas R+ {len(ok['R+'])}, R- {len(ok['R-'])}", flush=True)
    jobs = [("all", M, a.B, 1000 * i) for i, M in enumerate(a.Ms)] + \
           [(g, M, a.B_area, 50_000 + 1000 * j + i) for j, g in enumerate(regions) for i, M in enumerate([a.M_area])]
    rows, raw = [], {}
    with mp.get_context("fork").Pool(a.n_proc) as pool:   # fork: workers inherit the loaded data (DATA)
        for region, M, res in pool.imap_unordered(job_region, jobs):
            rows += summarize(res, dict(region=region, M=M))
            raw[f"{region}|{M}"] = {f"{g[0]} {g[1]}": res[g].tolist() for g in GROUPS}
            print(f"[064] {region} M={M} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        T = pd.DataFrame(rows)
        obs = T[(T.region == "all") & (T.M == a.M_perm)].set_index("readout").interaction
        null = np.array([v for _, v in sorted(pool.imap_unordered(job_perm, [(k, a.M_perm, a.B_perm) for k in range(a.n_perm)]),
                                              key=lambda x: x[0])])
    perm = {ro: float((1 + np.sum(np.abs(null[:, k]) >= abs(obs[ro]))) / (1 + len(null)))
            for k, ro in enumerate(READOUTS)}
    T["perm_p_interaction"] = [perm.get(r.readout) if (r.region == "all" and r.M == a.M_perm) else np.nan for r in T.itertuples()]
    T.to_csv(out / "pseudopop_stats.csv", index=False)
    np.save(out / "perm_null.npy", null)
    json.dump(raw, open(out / "bootstrap_samples.json", "w"))
    P = dict(M_PERM=a.M_perm, M_AREA=a.M_area, perm=perm, ok=ok)
    figure(T, P, out, a.population, a.Ms)
    json.dump(dict(script="064_pseudopop_hbootstrap.py", population=a.population, B=a.B, B_area=a.B_area, Ms=a.Ms,
                   M_area=a.M_area, n_perm=a.n_perm, B_perm=a.B_perm, M_perm=a.M_perm, n_pseudo_trials=N_PS,
                   min_trials_per_class=MIN_TRIALS, min_mice_area=MIN_MICE, unit_set=UNIT_SET, min_fr=m51.MIN_FR,
                   C=C_REG, areas_ok={c: sorted(v) for c, v in ok.items()}, perm_p=perm,
                   runtime_min=round((time.time() - t0) / 60, 1)), open(out / "provenance.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(T[T.region == "all"].round(3).to_string(index=False))
    print("perm p:", perm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="all", choices=["all", "learners"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--B-area", type=int, default=500)
    ap.add_argument("--Ms", type=int, nargs="*", default=[100, 200, 400, 800, 2000])
    ap.add_argument("--M-area", type=int, default=100)
    ap.add_argument("--M-perm", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--B-perm", type=int, default=50)
    ap.add_argument("--n-proc", type=int, default=48)
    main(ap.parse_args())
