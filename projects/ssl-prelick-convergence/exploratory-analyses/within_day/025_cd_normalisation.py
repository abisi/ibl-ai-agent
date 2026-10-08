"""025 -- How the normalisation of the within-day reward-lick coding direction (CD) shapes the drift, and length vs angle.

User question (2026-10-08): is the CD normalised to unit length, does that matter for the projection, how does the
projection relate to the length of the whisker-hit (WH) displacement and its angle to the CD, and how would the drift
behave with other normalisations or none?

Current estimator (002 trial_scores, 'md'): units z-scored on the training folds (AH + SL events); per fold the axis
Delta_f = mean_z(AH) - mean_z(SL) is scaled to unit length u_f = Delta_f / |Delta_f|; held-out AH / SL events and
fold-averaged WH events are projected, p = z(x) . u_f; one affine session normalisation c = (p - pbar_SL) / (pbar_AH -
pbar_SL) (held-out means; SL = 0, AH = 1); session kept if held-out d' >= 0.3.
Variants (same folds, same events, same sessions; only the scale or the unit preprocessing changes):
  affine     current c (dimensionless; 1 = the session's held-out AH - SL separation along the axis)
  unit       p - pbar_SL: distance along the unit axis, in z units (grows ~ sqrt(N units) for a fixed per-unit effect)
  per_unit   unit / sqrt(N): root-mean-square per unit
  raw_dot    z(x) . Delta_f - pbar_SL (axis not normalised: scale |Delta_f|, which includes estimation noise ~ N / n events)
  dprime     unit / pooled held-out SD of the AH and SL projections (noise units)
  affine_hz  as affine, but on raw rates (Hz, centred, not z-scored): high-rate units dominate the axis
Within a session, affine / unit / per_unit / dprime differ only by one positive factor, so per-session drift signs are
identical; they change how sessions are weighted in group means and tests. raw_dot also weights folds by |Delta_f|;
affine_hz changes the axis itself.
Drift (as 011): per session and class, OLS slope of the trial scores against normalised time tau; D_ref = beta_WH - beta_ref
(ref = SL, AH). Tests per variant: day 0 R+ vs R- and expert R+ vs R- (Mann-Whitney U and Welch t), and the effect
size as rank-biserial r (comparable across variants).
Length vs angle (cross-validated, early half tau < 0.5 vs late half, units z-scored on the session's AH + SL events,
N_SPLIT random halvings of AH and SL into A / B): u = unit(mean AH_A - mean SL_A); v = mean WH - mean SL;
projection p = mean_i (w_i - SL_B) . u; squared length |v|^2 = mean_{i != j} (w_i - SL_A) . (w_j - SL_B) (unbiased);
orthogonal o^2 = |v|^2 - p^2; cos theta = p / |v|; separation s = (AH_B - SL_B) . u; c = p / s; rho = |v| / |Delta| with
|Delta|^2 = (AH_A - SL_A) . (AH_B - SL_B). Sessions with >= 3 WH per half. Late - early per group; day 0 R+ vs R-.
Schematics: 2-D geometry and simulated sessions (WH growing at a fixed angle, rotating at a fixed length, moving
orthogonally; N = 25 / 100 / 400 units) with every variant.
Population: learners (report); all mice also written. Output: within_day/<ref>/cd_normalisation/.
"""
import argparse
import importlib
import json
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
m51 = importlib.import_module("051_roc_prelick")
m002 = importlib.import_module("002_trial_slopes")
OUT = m51.WITHIN / "cd_normalisation"
VARS = ["affine", "unit", "per_unit", "raw_dot", "dprime", "affine_hz"]
VLAB = {"affine": "SL = 0, AH = 1 (current)", "unit": "Unit axis (z units)", "per_unit": "Unit axis / √N",
        "raw_dot": "Not normalised (· Δ)", "dprime": "Noise units (d′)", "affine_hz": "SL = 0, AH = 1, Hz (no z-score)"}
CLS = ["WH", "AH", "FA"]
N_SPLIT, MIN_HALF = 20, 3


# ------------------------------------------------------------------------------------------------ estimators
def variant_scores(X, Xhz, lab, seed=0, min_dprime=None):
    """X, Xhz: (events, units); returns {variant: per-event score}, info"""
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = np.where(lab == "WH")[0]; ia = np.where(m)[0]
    y = (lab[m] == "AH").astype(int)
    N = X.shape[1]
    U, R, H = (np.full(len(lab), np.nan) for _ in range(3))
    Uw, Rw, Hw = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(ia, y):
        A = X[ia[tr]]; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
        z = lambda q: (q - mu) / sd
        d = z(A[y[tr] == 1]).mean(0) - z(A[y[tr] == 0]).mean(0)
        u = d / max(np.linalg.norm(d), 1e-12)
        U[ia[te]] = z(X[ia[te]]) @ u; Uw.append(z(X[w]) @ u)
        R[ia[te]] = z(X[ia[te]]) @ d; Rw.append(z(X[w]) @ d)
        Ah = Xhz[ia[tr]]; mh = Ah.mean(0)
        dh = (Ah[y[tr] == 1] - mh).mean(0) - (Ah[y[tr] == 0] - mh).mean(0)
        uh = dh / max(np.linalg.norm(dh), 1e-12)
        H[ia[te]] = (Xhz[ia[te]] - mh) @ uh; Hw.append((Xhz[w] - mh) @ uh)
    U[w], R[w], H[w] = np.mean(Uw, 0), np.mean(Rw, 0), np.mean(Hw, 0)
    isA, isS = lab == "AH", lab == "FA"
    sA, sS = U[isA], U[isS]
    sp = np.sqrt(0.5 * (sA.var(ddof=1) + sS.var(ddof=1)))
    sep = sA.mean() - sS.mean()
    dprime = sep / sp if sp > 0 else np.nan
    if not np.isfinite(dprime) or dprime < (m002.MIN_DPRIME if min_dprime is None else min_dprime):
        return None
    S = dict(affine=(U - sS.mean()) / sep, unit=U - sS.mean(), per_unit=(U - sS.mean()) / np.sqrt(N),
             raw_dot=R - R[isS].mean(), dprime=(U - sS.mean()) / sp)
    sepH = H[isA].mean() - H[isS].mean()
    S["affine_hz"] = (H - H[isS].mean()) / sepH if sepH > 0 else np.full(len(lab), np.nan)
    return S, dict(n_units=N, dprime=dprime, sep_unit=sep, sd_proj=sp, sep_raw=R[isA].mean() - R[isS].mean(), sep_hz=sepH)


def halves_geometry(Z, lab, tau, rng):
    """cross-validated length / projection / angle of the WH displacement from SL, early vs late half"""
    iA, iS = np.where(lab == "AH")[0], np.where(lab == "FA")[0]
    out = {}
    for half, sel in (("early", tau < 0.5), ("late", tau >= 0.5)):
        iw = np.where((lab == "WH") & sel)[0]
        if len(iw) < MIN_HALF:
            return None
        acc = []
        for _ in range(N_SPLIT):
            a, s = rng.permutation(iA), rng.permutation(iS)
            aA, aB, sA, sB = a[:len(a) // 2], a[len(a) // 2:], s[:len(s) // 2], s[len(s) // 2:]
            mAA, mAB, mSA, mSB = Z[aA].mean(0), Z[aB].mean(0), Z[sA].mean(0), Z[sB].mean(0)
            d = mAA - mSA
            u = d / max(np.linalg.norm(d), 1e-12)
            W = Z[iw]
            p = ((W - mSB) @ u).mean()
            G = (W - mSA) @ (W - mSB).T
            n = len(iw)
            L2 = (G.sum() - np.trace(G)) / (n * (n - 1))
            D2 = (mAA - mSA) @ (mAB - mSB)
            sep = (mAB - mSB) @ u
            acc.append((p, L2, D2, sep))
        p, L2, D2, sep = np.mean(acc, 0)
        L = np.sqrt(max(L2, 1e-12))
        out.update({f"proj_{half}": p, f"len2_{half}": L2, f"len_{half}": L, f"orth2_{half}": L2 - p ** 2,
                    f"theta_{half}": np.degrees(np.arccos(np.clip(p / L, -1, 1))), f"c_{half}": p / sep if sep > 0 else np.nan,
                    f"rho_{half}": L / np.sqrt(D2) if D2 > 0 else np.nan, f"n_wh_{half}": len(iw)})
    return out


def session(args):
    sid, f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["trial_start"].astype(float)[o]
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return None
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    if ok.sum() < m002.MIN_UNITS:
        return None
    r = variant_scores(X[ok].T, raw[ok].T, lab, seed)
    if r is None:
        return None
    S, info = r
    tau = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    row = dict(meta, **info, **{f"n_{c}": int((lab == c).sum()) for c in CLS})
    for v in VARS:
        for c in CLS:
            k = (lab == c) & np.isfinite(S[v])
            if k.sum() >= (m002.MIN_WH if c == "WH" else m002.MIN_EV):
                row[f"{v}_{c}_slope"], row[f"{v}_{c}_start"] = np.polyfit(tau[k], S[v][k], 1)
        for ref in ("FA", "AH"):
            if f"{v}_WH_slope" in row and f"{v}_{ref}_slope" in row:
                row[f"{v}_D_{ref}"] = row[f"{v}_WH_slope"] - row[f"{v}_{ref}_slope"]
    Zs = X[ok].T
    m_ = np.isin(lab, ["AH", "FA"])
    mu, sd = Zs[m_].mean(0), Zs[m_].std(0); sd[sd == 0] = 1
    g = halves_geometry((Zs - mu) / sd, lab, tau, np.random.default_rng(seed))
    if g:
        row.update(g)
    return row


# ------------------------------------------------------------------------------------------------ statistics
def rank_biserial(a, b):
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def tests(D):
    rows = []
    cols = [f"{v}_D_{r}" for v in VARS for r in ("FA", "AH")] + [f"{v}_WH_slope" for v in VARS]
    for q in ("proj", "len", "len2", "orth2", "theta", "c", "rho"):
        if f"{q}_early" in D:
            D[f"{q}_change"] = D[f"{q}_late"] - D[f"{q}_early"]
            cols.append(f"{q}_change")
    for col in cols:
        for st in ("learning", "expert"):
            a = D[(D.cohort == "R+") & (D.stage == st)][col].dropna()
            b = D[(D.cohort == "R-") & (D.stage == st)][col].dropna()
            r = dict(measure=col, stage=st, n_Rp=len(a), n_Rm=len(b), mean_Rp=a.mean(), sem_Rp=a.sem(), mean_Rm=b.mean(),
                     sem_Rm=b.sem())
            if len(a) >= 3 and len(b) >= 3:
                r.update(p_MWU=stats.mannwhitneyu(a, b).pvalue, p_Welch=stats.ttest_ind(a, b, equal_var=False).pvalue,
                         rank_biserial=rank_biserial(a, b), p_Rp_vs0=stats.wilcoxon(a).pvalue, p_Rm_vs0=stats.wilcoxon(b).pvalue)
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ simulations
def simulate(scenario, N, rng, n_ev=(40, 40, 40), noise=1.0):
    """toy session: N units, SL at 0, AH at a fixed per-unit separation; WH displacement in the plane (e1 = AH axis,
    e2 = orthogonal); returns scores of every variant (no z-scoring difference: unit noise SD = 1)"""
    e1 = np.ones(N) / np.sqrt(N)
    e2 = np.r_[np.ones(N // 2), -np.ones(N - N // 2)] / np.sqrt(N)
    e2 -= (e2 @ e1) * e1; e2 /= np.linalg.norm(e2)
    sepN = 0.25 * np.sqrt(N)                                # per-unit effect 0.25 SD -> |Delta| grows with sqrt(N)
    nW, nA, nS = n_ev
    tau = np.linspace(0, 1, nW)
    if scenario == "grow":
        Lw, th = sepN * (0.2 + 0.6 * tau), np.radians(35) * np.ones(nW)
    elif scenario == "rotate":
        Lw, th = sepN * 0.7 * np.ones(nW), np.radians(75 - 50 * tau)
    else:
        Lw, th = sepN * np.sqrt((0.35) ** 2 + (0.2 + 0.6 * tau) ** 2), np.arctan2(0.2 + 0.6 * tau, 0.35)
    W = (Lw * np.cos(th))[:, None] * e1 + (Lw * np.sin(th))[:, None] * e2 + noise * rng.standard_normal((nW, N))
    A = sepN * e1 + noise * rng.standard_normal((nA, N))
    S_ = noise * rng.standard_normal((nS, N))
    X = np.vstack([W, A, S_])
    lab = np.array(["WH"] * nW + ["AH"] * nA + ["FA"] * nS)
    r = variant_scores(X, X, lab, 0, min_dprime=-np.inf)
    return r, tau, Lw, np.degrees(th)


# ------------------------------------------------------------------------------------------------ figures
def fig_schematic(plt, out):
    m62 = importlib.import_module("062_pub_convergence_figures")
    CL = m62.CL
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.62))
    gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.55, left=0.06, right=0.98, top=0.92, bottom=0.08)
    # a: geometry
    ax = fig.add_subplot(gs[0, 0])
    S0, A0, W0 = np.r_[0, 0], np.r_[1, 0], np.r_[0.55, 0.42]
    ax.annotate("", A0, S0, arrowprops=dict(arrowstyle="->", color=CL["AH"], lw=1.2))
    ax.annotate("", W0, S0, arrowprops=dict(arrowstyle="->", color=CL["WH"], lw=1.2))
    ax.plot([W0[0], W0[0]], [0, W0[1]], ":", color="0.4", lw=0.8)
    ax.plot([0, W0[0]], [-0.06, -0.06], color="#d94801", lw=2)
    for P, c, t in ((S0, CL["FA"], "SL"), (A0, CL["AH"], "AH"), (W0, CL["WH"], "WH")):
        ax.plot(*P, "o", color=c, ms=4); ax.text(P[0] + 0.03, P[1] + 0.04, t, fontsize=5.5, color=c)
    th = np.linspace(0, np.arctan2(W0[1], W0[0]), 30)
    ax.plot(0.18 * np.cos(th), 0.18 * np.sin(th), color="0.3", lw=0.7); ax.text(0.2, 0.05, "θ", fontsize=6)
    ax.text(0.27, -0.16, "p = |v| cos θ", fontsize=5, color="#d94801", ha="center")
    ax.text(0.6, 0.22, "o = |v| sin θ", fontsize=5, color="0.3")
    ax.text(0.15, 0.3, "v", fontsize=6, color="#b07e00")
    ax.text(0.5, 0.04, "Δ", fontsize=6, color=CL["AH"])
    ax.set_xlim(-0.1, 1.15); ax.set_ylim(-0.25, 0.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Geometry of one session", fontsize=6)
    # b: equations
    ax = fig.add_subplot(gs[0, 1:3]); ax.axis("off")
    eq = [r"$u = \Delta/|\Delta|,\ \ \Delta = \bar z_{AH} - \bar z_{SL}$ (training folds)",
          r"$p(x) = (z(x) - \bar z_{SL})\cdot u = |v|\cos\theta$",
          r"current:  $c = p\,/\,(\bar p_{AH} - \bar p_{SL}) = \rho\cos\theta$,  $\rho = |v|/|\Delta|$",
          r"not normalised:  $(z(x)-\bar z_{SL})\cdot\Delta = |\Delta|\,p$",
          r"per unit:  $p/\sqrt{N}$;   noise units:  $p/\sigma_{p}$  ($= c\cdot d'$)",
          r"drift  $D_{ref} = \beta_{WH} - \beta_{ref}$,  $\beta$ = slope of the score vs $\tau$",
          r"same session: $D^{unit} = |\Delta_p|\,D^{c}$,  $D^{d'} = d'\,D^{c}$  (same sign, other weights)"]
    for i, e in enumerate(eq):
        ax.text(0, 1 - i * 0.15, e, fontsize=5.6, va="top", transform=ax.transAxes)
    ax.set_title("Scores and drifts", fontsize=6, loc="left")
    # c: scenarios in 2-D
    ax = fig.add_subplot(gs[0, 3])
    tt = np.linspace(0, 1, 5)
    sc = {"grow": (0.2 + 0.6 * tt, np.radians(35) * np.ones(5)), "rotate": (0.7 * np.ones(5), np.radians(75 - 50 * tt)),
          "orth": (np.sqrt(0.35 ** 2 + (0.2 + 0.6 * tt) ** 2), np.arctan2(0.2 + 0.6 * tt, 0.35))}
    col = {"grow": "#d94801", "rotate": "#6a51a3", "orth": "0.45"}
    lab_ = {"grow": "grows, fixed θ", "rotate": "rotates, fixed |v|", "orth": "moves orthogonally"}
    for k, (L, a) in sc.items():
        x, y = L * np.cos(a), L * np.sin(a)
        ax.plot(x, y, "-o", color=col[k], ms=2, lw=0.9, label=lab_[k])
        ax.annotate("", (x[-1], y[-1]), (x[-2], y[-2]), arrowprops=dict(arrowstyle="->", color=col[k], lw=0.9))
    ax.plot(0, 0, "o", color=CL["FA"], ms=4); ax.plot(1, 0, "o", color=CL["AH"], ms=4)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_aspect("equal"); ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.1, 0.85)
    ax.set_xlabel("along CD (AH = 1)", fontsize=5); ax.set_ylabel("orthogonal", fontsize=5)
    ax.legend(fontsize=4.3, frameon=False, loc="upper right"); ax.tick_params(labelsize=4.5)
    ax.set_title("Three within-session WH paths", fontsize=6)
    # d-g: simulated scores
    axs = [fig.add_subplot(gs[1, j]) for j in range(4)]
    nb = 8
    for k in ("grow", "rotate", "orth"):
        C = []
        for rep in range(30):
            r, tau, Lw, thd = simulate(k, 100, np.random.default_rng(1000 + rep))
            b = np.minimum((tau * nb).astype(int), nb - 1)
            C.append([r[0]["affine"][:len(tau)][b == i].mean() for i in range(nb)])
        C = np.array(C)
        xb = (np.arange(nb) + 0.5) / nb
        axs[0].errorbar(xb, C.mean(0), yerr=C.std(0) / np.sqrt(len(C)), fmt="-o", ms=2, lw=0.9, color=col[k], label=lab_[k])
        axs[0].plot(tau, Lw * np.cos(np.radians(thd)) / 2.5, ":", color=col[k], lw=0.8)
        axs[1].plot(tau, Lw / 2.5, color=col[k], lw=1)
        axs[1].plot(tau, thd / 90, color=col[k], lw=1, ls="--")
    axs[0].legend(fontsize=4, frameon=False, loc="upper left")
    axs[0].set_title("Projection c of WH trials\n(N = 100 units, simulated)", fontsize=5.6)
    axs[0].set_xlabel("τ (time in session)", fontsize=5); axs[0].set_ylabel("c (SL = 0, AH = 1)", fontsize=5)
    axs[1].set_title("Ground truth: ρ = |v|/|Δ| (solid)\nand θ / 90° (dashed)", fontsize=5.6)
    axs[1].set_xlabel("τ", fontsize=5)
    # scaling with N: slope of WH for the 'grow' scenario per variant
    Ns = [25, 100, 400]
    res = {v: [] for v in VARS[:5]}
    for N in Ns:
        vals = {v: [] for v in VARS[:5]}
        for rep in range(20):
            r, tau, _, _ = simulate("grow", N, np.random.default_rng(100 + rep))
            S, _ = r
            for v in VARS[:5]:
                vals[v].append(np.polyfit(tau, S[v][:len(tau)], 1)[0])
        for v in VARS[:5]:
            res[v].append(np.mean(vals[v]))
    for v in VARS[:5]:
        rel = np.array(res[v]) / res[v][1]
        axs[2].plot(Ns, rel, "-o", ms=2.5, lw=0.9, label=VLAB[v])
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xticks(Ns, [str(n) for n in Ns])
    axs[2].set_xlabel("Number of units N", fontsize=5); axs[2].set_ylabel("WH slope / slope at N = 100", fontsize=5)
    axs[2].legend(fontsize=4, frameon=False, loc="upper left")
    axs[2].set_title("Same geometry, more units:\nthe scale of each variant", fontsize=5.6)
    # number of trials: raw_dot inflated by axis noise
    nev = [12, 20, 40, 80]
    rr = {v: [] for v in ("affine", "unit", "raw_dot")}
    for n in nev:
        vals = {v: [] for v in rr}
        for rep in range(100):
            r, tau, _, _ = simulate("grow", 100, np.random.default_rng(200 + rep), n_ev=(40, n, n))
            S, _ = r
            for v in rr:
                vals[v].append(np.polyfit(tau, S[v][:len(tau)], 1)[0])
        for v in rr:
            rr[v].append(np.mean(vals[v]))
    for v in rr:
        axs[3].plot(nev, np.array(rr[v]) / rr[v][-1], "-o", ms=2.5, lw=0.9, label=VLAB[v])
    axs[3].set_xscale("log"); axs[3].set_xticks(nev, [str(n) for n in nev])
    axs[3].set_xlabel("AH / SL events per class", fontsize=5); axs[3].set_ylabel("WH slope / slope at 80 events", fontsize=5)
    axs[3].legend(fontsize=4, frameon=False)
    axs[3].set_title("Same geometry, fewer events:\nnoisy axis estimate", fontsize=5.6)
    from matplotlib.ticker import NullFormatter, NullLocator
    for a in axs[2:]:
        a.xaxis.set_minor_locator(NullLocator()); a.xaxis.set_minor_formatter(NullFormatter())
    axs[2].yaxis.set_minor_formatter(NullFormatter())
    axs[2].set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    for a in axs:
        a.tick_params(labelsize=4.5)
    for a, L in zip([fig.axes[0], fig.axes[1], fig.axes[2]] + axs, "abcdefg"):
        bb = a.get_position()
        fig.text(bb.x0 - 0.035, bb.y1 + 0.03, L, fontsize=8, fontweight="bold", va="bottom")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cd_normalisation_schematic.{ext}", dpi=300)
    plt.close(fig)


def fig_geometry_schematic(plt, out):
    """explains every quantity of cd_normalisation_geometry (illustration, not data)"""
    m62 = importlib.import_module("062_pub_convergence_figures")
    CL = m62.CL
    OR, GR = "#d94801", "0.35"
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.4))
    gs = fig.add_gridspec(1, 4, wspace=0.18, left=0.02, right=0.99, top=0.86, bottom=0.04, width_ratios=[1.25, 0.9, 0.9, 1.25])

    def base(ax, xl=(-0.12, 1.2), yl=(-0.32, 0.85)):
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")
        ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=CL["AH"], lw=1.1))
        ax.plot(0, 0, "o", color=CL["FA"], ms=4.5, zorder=5); ax.plot(1, 0, "o", color=CL["AH"], ms=4.5, zorder=5)
        ax.text(-0.02, -0.08, "SL", fontsize=5.5, ha="center", va="top", color=CL["FA"])
        ax.text(1.0, -0.08, "AH", fontsize=5.5, ha="center", va="top", color=CL["AH"])

    def wh(ax, W, filled=True, lab=None, th_r=0.16, show_th=True, col_th=GR):
        ax.annotate("", W, (0, 0), arrowprops=dict(arrowstyle="->", color=CL["WH"], lw=1.1, ls="-" if filled else "--"))
        ax.plot(*W, "o", ms=4.5, mfc=CL["WH"] if filled else "white", mec=CL["WH"], mew=1, zorder=5)
        ax.plot([W[0], W[0]], [0, W[1]], ":", color=GR, lw=0.8)
        if show_th:
            a = np.linspace(0, np.arctan2(W[1], W[0]), 30)
            ax.plot(th_r * np.cos(a), th_r * np.sin(a), color=col_th, lw=0.7, ls="-" if filled else "--")
        if lab:
            ax.text(W[0] + 0.04, W[1] + 0.03, lab, fontsize=5.2, color="#b07e00")

    # a: one half, every quantity
    ax = fig.add_subplot(gs[0, 0]); base(ax)
    W = np.r_[0.55, 0.5]
    wh(ax, W, lab="WH (mean of the half)")
    ax.plot([0, W[0]], [-0.035, -0.035], color=OR, lw=2.2, solid_capstyle="butt")
    ax.text(W[0] / 2, -0.12, "p = |v| cos θ", fontsize=5, color=OR, ha="center", va="top")
    ax.annotate("", (1, -0.2), (0, -0.2), arrowprops=dict(arrowstyle="<->", color=CL["AH"], lw=0.7))
    ax.text(0.5, -0.23, "separation s = (AH − SL) · u  (≈ |Δ|)", fontsize=4.8, color=CL["AH"], ha="center", va="top")
    ax.text(W[0] + 0.03, W[1] / 2, "o = |v| sin θ\no² = |v|² − p²", fontsize=4.8, color=GR, va="center")
    ax.text(0.17, 0.33, "|v|", fontsize=5.5, color="#b07e00")
    ax.text(0.19, 0.05, "θ", fontsize=5.8, color=GR)
    ax.text(0.78, 0.03, "u (CD)", fontsize=5, color=CL["AH"])
    ax.text(-0.1, 0.84, "c = p / s   (0 = SL, 1 = AH)\nρ = |v| / |Δ|   (length relative to AH)\ncos θ = p / |v|", fontsize=5,
            va="top", bbox=dict(fc="white", ec="0.75", lw=0.5, boxstyle="round,pad=0.3"))
    ax.set_title("Quantities of one session half", fontsize=6)

    # b: R+-like pattern (day 0): projection grows, angle shrinks
    ax = fig.add_subplot(gs[0, 1]); base(ax)
    E, L = np.r_[0.2, 0.55], np.r_[0.5, 0.55]
    wh(ax, E, filled=False, lab="early", th_r=0.12)
    wh(ax, L, filled=True, lab="late", th_r=0.2)
    ax.plot([0, E[0]], [-0.03, -0.03], color=OR, lw=1.6, alpha=0.5)
    ax.plot([0, L[0]], [-0.07, -0.07], color=OR, lw=1.6)
    ax.text(0.5, -0.2, "p ↑, θ ↓, c ↑\n|v| ↑ a little, o ≈ same", fontsize=4.9, ha="center", va="top")
    ax.set_title("R+ day-0 pattern\n(turns toward the CD)", fontsize=5.6, color=m62.COH["R+"])

    # c: R- like pattern (day 0): length grows orthogonally, projection flat
    ax = fig.add_subplot(gs[0, 2]); base(ax)
    E, L = np.r_[0.25, 0.35], np.r_[0.25, 0.65]
    wh(ax, E, filled=False, lab="early", th_r=0.12)
    wh(ax, L, filled=True, lab="late", th_r=0.2)
    ax.plot([0, E[0]], [-0.05, -0.05], color=OR, lw=1.6)
    ax.text(0.5, -0.2, "p ≈ same, θ ↑, c ≈ same\n|v| ↑, o² ↑ (orthogonal)", fontsize=4.9, ha="center", va="top")
    ax.set_title("R− day-0 pattern\n(grows orthogonally)", fontsize=5.6, color=m62.COH["R-"])

    # d: how each quantity is estimated
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    txt = [r"$\bf{Estimation}$ (per session; units z-scored on AH + SL)",
           "Halves: early τ < 0.5, late τ ≥ 0.5 (≥ 3 WH each);",
           "AH and SL trials split at random into A / B (20 splits)",
           r"$u = (\bar z_{AH,A} - \bar z_{SL,A})\,/\,|\cdot|$",
           r"$p = \mathrm{mean}_i\,(w_i - \bar z_{SL,B})\cdot u$",
           r"$|v|^2 = \mathrm{mean}_{i\neq j}\,(w_i - \bar z_{SL,A})\cdot(w_j - \bar z_{SL,B})$",
           r"$|\Delta|^2 = (\bar z_{AH,A} - \bar z_{SL,A})\cdot(\bar z_{AH,B} - \bar z_{SL,B})$",
           r"$s = (\bar z_{AH,B} - \bar z_{SL,B})\cdot u$",
           r"$o^2 = |v|^2 - p^2,\ \ \theta = \arccos(p/|v|)$",
           r"$c = p/s,\ \ \rho = |v|/|\Delta|$",
           "Independent halves in every product: no noise bias in",
           "|v|², |Δ|²; axis noise shrinks p (θ biased up), equally",
           "in both halves. Units: p, |v| in z (SD) units; o² in z²."]
    for i, t in enumerate(txt):
        ax.text(0, 1 - i * 0.077, t, fontsize=4.9, va="top", transform=ax.transAxes)
    for a, Lt in zip(fig.axes, "abcd"):
        bb = a.get_position()
        fig.text(bb.x0, 0.97, Lt, fontsize=8, fontweight="bold", va="top")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cd_normalisation_geometry_schematic.{ext}", dpi=300)
    plt.close(fig)


def fig_data(plt, D, T, out, pop):
    m62 = importlib.import_module("062_pub_convergence_figures")
    COH = m62.COH
    fig, axs = plt.subplots(3, 6, figsize=(m62.W_IN, m62.W_IN * 0.62))
    for i, ref in enumerate(("FA", "AH")):
        for j, v in enumerate(VARS):
            ax = axs[i, j]
            col = f"{v}_D_{ref}"
            for k, st in enumerate(("learning", "expert")):
                for c, dx in (("R+", -0.12), ("R-", 0.12)):
                    x = D[(D.cohort == c) & (D.stage == st)][col].dropna()
                    ax.errorbar(k + dx, x.mean(), yerr=x.sem(), fmt="o", ms=2.6, color=COH[c], lw=0.8, capsize=0)
                r = T[(T.measure == col) & (T.stage == st)]
                if len(r) and np.isfinite(r.iloc[0].get("p_MWU", np.nan)):
                    r = r.iloc[0]
                    ax.text(k, 1.0, f"{m62.fmt_p(r.p_MWU)}\nr={r.rank_biserial:+.2f}", transform=ax.get_xaxis_transform(),
                            fontsize=3.9, ha="center", va="bottom")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_xticks([0, 1], ["day 0", "expert"], fontsize=4.5); ax.set_xlim(-0.5, 1.5)
            ax.tick_params(labelsize=4.3)
            if i == 0:
                ax.set_title(VLAB[v], fontsize=4.8, pad=12)
            if j == 0:
                ax.set_ylabel(f"D = β_WH − β_{'SL' if ref == 'FA' else 'AH'}\n(mean ± s.e.m.)", fontsize=5)
    # row 3: per-session reweighting: D_AH current vs unit / dprime / raw_dot / hz; separation and N
    pairs = [("unit", "sep_unit", "Separation |Δ_p| (z)"), ("dprime", "dprime", "d′"), ("raw_dot", "sep_raw", "|Δ|·|Δ_p|"),
             ("affine_hz", None, None)]
    for j, (v, sc, sl) in enumerate(pairs):
        ax = axs[2, j]
        for c in ("R+", "R-"):
            d = D[(D.cohort == c) & (D.stage == "learning")]
            ax.scatter(d["affine_D_AH"], d[f"{v}_D_AH"], s=3, color=COH[c], lw=0, alpha=0.8)
        ok = D[["affine_D_AH", f"{v}_D_AH"]].dropna()
        rs = stats.spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])[0]
        sg = np.mean(np.sign(ok.iloc[:, 0]) == np.sign(ok.iloc[:, 1]))
        ax.set_title(f"{VLAB[v]}\nSpearman {rs:.2f}, same sign {100 * sg:.0f} %", fontsize=4.5)
        ax.set_xlabel("D_AH, current (c units)", fontsize=4.6); ax.tick_params(labelsize=4.3)
        ax.axhline(0, color="0.8", lw=0.4); ax.axvline(0, color="0.8", lw=0.4)
    ax = axs[2, 4]
    for c in ("R+", "R-"):
        d = D[D.cohort == c]
        ax.scatter(d.n_units, d.sep_unit, s=3, color=COH[c], lw=0, alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Units per session", fontsize=4.6); ax.set_ylabel("|Δ_p| (z units)", fontsize=4.6)
    ax.set_title(f"Separation grows with N\nSpearman {stats.spearmanr(D.n_units, D.sep_unit)[0]:.2f}", fontsize=4.5)
    ax.tick_params(labelsize=4.3)
    ax = axs[2, 5]
    for c in ("R+", "R-"):
        d = D[D.cohort == c]
        ax.scatter(d.n_units, d.dprime, s=3, color=COH[c], lw=0, alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Units per session", fontsize=4.6); ax.set_ylabel("held-out d′", fontsize=4.6)
    ax.set_title(f"d′ vs N\nSpearman {stats.spearmanr(D.n_units, D.dprime)[0]:.2f}", fontsize=4.5)
    ax.tick_params(labelsize=4.3)
    fig.text(0.01, 0.99, f"Within-day drift under six normalisations ({pop}); p: R+ vs R− Mann-Whitney, r: rank-biserial",
             fontsize=5.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cd_normalisation_drifts.{ext}", dpi=300)
    plt.close(fig)


def fig_geometry(plt, D, T, out, pop):
    m62 = importlib.import_module("062_pub_convergence_figures")
    COH = m62.COH
    meas = [("proj", "Projection p (z units)"), ("len", "Length |v| (z units)"), ("orth2", "Orthogonal o² (z²)"),
            ("theta", "Angle θ to CD (°)"), ("c", "c = p / separation"), ("rho", "ρ = |v| / |Δ|")]
    fig, axs = plt.subplots(2, 6, figsize=(m62.W_IN, m62.W_IN * 0.4))
    for j, (q, ttl) in enumerate(meas):
        for i, st in enumerate(("learning", "expert")):
            ax = axs[i, j]
            for c, dx in (("R+", -0.1), ("R-", 0.1)):
                d = D[(D.cohort == c) & (D.stage == st)][[f"{q}_early", f"{q}_late"]].dropna()
                m, s = d.mean(), d.sem()
                ax.errorbar(np.r_[0, 1] + dx, m.to_numpy(), yerr=s.to_numpy(), fmt="-o", ms=2.4, lw=0.9, color=COH[c],
                            label=f"{c.replace('-', '−')} (n={len(d)})")
            r = T[(T.measure == f"{q}_change") & (T.stage == st)]
            if len(r) and np.isfinite(r.iloc[0].get("p_MWU", np.nan)):
                r = r.iloc[0]
                ax.set_title(f"{ttl}\n{'day 0' if st == 'learning' else 'expert'}: Δ R+ vs R− {m62.fmt_p(r.p_MWU)}",
                             fontsize=4.5)
            ax.set_xticks([0, 1], ["early", "late"], fontsize=4.5); ax.set_xlim(-0.4, 1.4); ax.tick_params(labelsize=4.3)
            if j == 0:
                ax.legend(fontsize=3.9, frameon=False)
    fig.text(0.01, 0.99, f"Cross-validated WH geometry relative to SL, early vs late half ({pop}; ≥ {MIN_HALF} WH per half)",
             fontsize=5.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cd_normalisation_geometry.{ext}", dpi=300)
    plt.close(fig)


def main(a):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    m61 = importlib.import_module("061_roc_prelick_learners")
    m62 = importlib.import_module("062_pub_convergence_figures")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.replot:
        D = pd.read_csv(OUT / "sessions.csv")
    else:
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
                                                   stage=Ws.stage.iloc[0]), Ws[["electrode_group", "cluster_id", "quality_label"]], i))
        if a.test:
            jobs = jobs[:a.test]
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            D = pd.DataFrame([r for r in pool.imap_unordered(session, jobs) if r])
        D.to_csv(OUT / "sessions.csv", index=False)
    plt = m62.setup()
    fig_schematic(plt, OUT)
    fig_geometry_schematic(plt, OUT)
    for pop in ("learners", "all"):
        Dp = m61.learner_filter(D) if pop == "learners" else D
        o = OUT / pop; o.mkdir(exist_ok=True)
        T = tests(Dp.copy())
        T.to_csv(o / "tests.csv", index=False)
        Dq = Dp.copy(); tests(Dq)
        fig_data(plt, Dq, T, o, pop)
        fig_geometry(plt, Dq, T, o, pop)
        print(pop, T[T.measure.str.contains("_D_|_change")].round(4).to_string(), flush=True)
    json.dump(dict(script="025_cd_normalisation.py", ref=m51.REF, units=m51.UNITS, n_sessions=int(D.session_id.nunique()),
                   variants=VLAB, n_split=N_SPLIT, min_half=MIN_HALF, min_wh=m002.MIN_WH, min_ev=m002.MIN_EV,
                   min_dprime=m002.MIN_DPRIME), open(OUT / "provenance.json", "w"), indent=1)
    print("ALL DONE", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
