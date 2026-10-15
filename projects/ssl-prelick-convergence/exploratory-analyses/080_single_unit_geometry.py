"""080 -- Single-neuron (ROC) counterpart of the population geometry (user 2026-10-07: "an ROC-type single-cell analysis that
complements the geometrical view").

The population quantities are sums over units: with per-unit effects d_W,u (WH vs SL) and d_A,u (AH vs SL), the
cross-validated dot product a = (1/N) sum_u d_W,u d_A,u (Eq. 7), lambda = slope through the origin of d_W on d_A across
units (Eq. 9), cos theta = their uncentred across-unit correlation and rho = the ratio of their spreads (Eq. 10). The ROC
selectivity s = 2 AUC - 1 (Eq. 2) is a bounded, rank-based per-unit effect, so the across-unit relation of s_WS (WH vs SL;
> 0: WH > SL) and s_AS (AH vs SL; > 0: AH > SL) is the single-neuron picture of the geometry.

Cross-validation (as the population distances): in each of N_SPLIT = 50 splits, the events of each class are halved at
random and both selectivities are computed on half a and on half b; products are taken across halves, so per-unit noise
does not inflate them: ww_u = <s_WS^a s_WS^b>, aa_u = <s_AS^a s_AS^b>, wa_u = <(s_WS^a s_AS^b + s_WS^b s_AS^a) / 2>.
For a set of units (session, or session x area group), C_XY = mean_u xy_u and
  alignment  = C_WA / sqrt(C_WW C_AA)   (cos theta analogue; requires C_WW, C_AA > 0)
  gain       = C_WA / C_AA              (lambda analogue: slope of s_WS on s_AS through the origin)
  ratio      = sqrt(C_WW / C_AA)        (rho analogue)
  residual   = C_WW - C_WA^2 / C_AA     (WH-specific selectivity orthogonal to the AH-SL pattern; o^2 analogue)
Effect-size variant (exact correspondence with the geometry): the same with d = mean difference / within-class SD
(Eq. 4; the decided scaling) instead of s.
Per-unit contributions: wa_u by functional type (051 significance on all events: converging = WH vs SL and AH vs SL both
significant with the same sign; WH-only; AH-only; neither) -> mean per-unit contribution wa_u per type and session (share of C_WA stored too, unstable when C_WA ~ 0).
Units: quality good / mua (PRELICK_UNITS=stable: stable only, via 053), mean raw pre-lick rate >= 0.1 Hz; sessions with
>= 8 events per class (>= 4 per half) and >= 10 units; area groups >= 5 units. Statistics (unit = session): learning vs
expert per cohort (MWU, Welch), expert R+ vs R-, learning x cohort interaction (Eq. 15; 10,000 mouse-level cohort
permutations; 062 dots_panel); per area group the same interaction, Benjamini-Hochberg over area groups, and the ANOVA
gain ~ cohort x stage x area (062 area_anova).
Output: across_days/<ref>[_units]/single_unit_geometry/: units.parquet (per-unit products, join keys mouse_id, session_id,
electrode_group, cluster_id), sessions.csv, areas.csv; <pop>/single_unit_geometry.{png,pdf,svg},
single_unit_geometry_effectsize.{png,pdf,svg}, stats.csv, area_tests.csv.
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
OUT = m51.OUTROOT / "single_unit_geometry"
N_SPLIT, MIN_EV, MIN_UNITS, MIN_AREA_UNITS, CAA_MIN = 50, 8, 10, 5, 1e-3
UNIT_SET = ("good", "mua")
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
TYPES = ["converging", "WH-only", "AH-only", "neither"]
SEL_WS, SEL_AS = "whisker_hit_vs_fa_prelick@all", "auditory_hit_vs_fa_prelick@all"


def auc_sel(X, ib, ia):
    """s = 2 AUC - 1 of every unit (rows of X) for events ib (positive class) vs ia; ties count half (rank average)"""
    Y = X[:, np.r_[ib, ia]]
    R = stats.rankdata(Y, axis=1)
    nb, na = len(ib), len(ia)
    U = R[:, :nb].sum(1) - nb * (nb + 1) / 2
    return 2 * U / (nb * na) - 1


def session(args):
    f, meta, K, seed = args
    rng = np.random.default_rng(seed)
    z = np.load(f, allow_pickle=True)
    X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
    idx = {c: np.where(lab == c)[0] for c in ("WH", "AH", "FA")}
    if min(len(v) for v in idx.values()) < MIN_EV:
        return None
    ok = (raw.mean(1) >= m51.MIN_FR) & K.quality_label.isin(UNIT_SET).to_numpy() & (X.std(1) > 0)
    if ok.sum() < MIN_UNITS:
        return None
    X, K = X[ok], K[ok].reset_index(drop=True)
    sd = np.sqrt(np.mean([X[:, v].var(1, ddof=1) for v in idx.values()], 0))        # within-class SD (Eq. 4)
    sd[sd == 0] = 1
    acc = {k: np.zeros(len(X)) for k in ("ww", "aa", "wa", "ww_d", "aa_d", "wa_d")}
    for _ in range(N_SPLIT):
        h = {}
        for c, v in idx.items():
            p = rng.permutation(v); k = len(p) // 2
            h[c] = (p[:k], p[k:])
        s = {}
        for hh in (0, 1):
            s[("W", hh)] = auc_sel(X, h["WH"][hh], h["FA"][hh])
            s[("A", hh)] = auc_sel(X, h["AH"][hh], h["FA"][hh])
            s[("dW", hh)] = (X[:, h["WH"][hh]].mean(1) - X[:, h["FA"][hh]].mean(1)) / sd
            s[("dA", hh)] = (X[:, h["AH"][hh]].mean(1) - X[:, h["FA"][hh]].mean(1)) / sd
        for sfx, w, a in (("", "W", "A"), ("_d", "dW", "dA")):
            acc["ww" + sfx] += s[(w, 0)] * s[(w, 1)]
            acc["aa" + sfx] += s[(a, 0)] * s[(a, 1)]
            acc["wa" + sfx] += (s[(w, 0)] * s[(a, 1)] + s[(w, 1)] * s[(a, 0)]) / 2
    full_w = (X[:, idx["WH"]].mean(1) - X[:, idx["FA"]].mean(1)) / sd                  # all events, for the scatter
    full_a = (X[:, idx["AH"]].mean(1) - X[:, idx["FA"]].mean(1)) / sd
    U = K.assign(**meta, **{k: v / N_SPLIT for k, v in acc.items()}, dW_full=full_w, dA_full=full_a,
                 n_WH=len(idx["WH"]), n_AH=len(idx["AH"]), n_SL=len(idx["FA"]))
    return U


def unit_types(U):
    sig = lambda c: U[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    sw, sa = sig(f"sig:{SEL_WS}"), sig(f"sig:{SEL_AS}")
    same = np.sign(U[f"sel:{SEL_WS}"]) == np.sign(U[f"sel:{SEL_AS}"])
    return np.select([sw & sa & same, sw & ~sa, sa & ~sw], ["converging", "WH-only", "AH-only"], "neither")


def metrics(d, sfx=""):
    ww, aa, wa = d["ww" + sfx].mean(), d["aa" + sfx].mean(), d["wa" + sfx].mean()
    ok = aa > CAA_MIN
    r = dict(C_WW=ww, C_AA=aa, C_WA=wa, n_units=len(d),
             gain=np.clip(wa / aa, -1, 3) if ok else np.nan,
             align=np.clip(wa / np.sqrt(ww * aa), -1, 1) if ok and ww > 0 else np.nan,
             ratio=np.sqrt(ww / aa) if ok and ww > 0 else np.nan,
             resid=ww - wa ** 2 / aa if ok else np.nan)
    return {(k + sfx if k != "n_units" else k): v for k, v in r.items() if not (sfx and k == "n_units")}


def summarise(U):
    rows, arows = [], []
    for sid, d in U.groupby("session_id"):
        meta = d.iloc[0][["session_id", "mouse_id", "cohort", "stage"]].to_dict()
        r = dict(meta, **metrics(d), **metrics(d, "_d"))
        tot = d.wa.sum()
        for t in TYPES:
            q = d[d.type == t]
            r[f"frac_{t}"] = len(q) / len(d)
            r[f"share_{t}"] = q.wa.sum() / tot if tot > 0 else np.nan
            r[f"wa_{t}"] = q.wa.mean() if len(q) else np.nan
        sel = d[[f"sel:{SEL_WS}", f"sel:{SEL_AS}"]].dropna()
        r["r_full"] = stats.pearsonr(sel.iloc[:, 0], sel.iloc[:, 1])[0] if len(sel) >= 10 else np.nan
        rows.append(r)
        for g, q in d.groupby("area_group"):
            if len(q) >= MIN_AREA_UNITS:
                arows.append(dict(meta, region=g, **metrics(q), **metrics(q, "_d")))
    return pd.DataFrame(rows), pd.DataFrame(arows)


def compute(n_proc):
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    W["type"] = unit_types(W)
    cols = ["mouse_id", "electrode_group", "cluster_id", "quality_label", "area_group", "type",
            f"sel:{SEL_WS}", f"sel:{SEL_AS}", f"sig:{SEL_WS}", f"sig:{SEL_AS}"]
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        Ws = W[W.session_id == r.session_id]
        K = K.merge(Ws[cols].drop_duplicates(["electrode_group", "cluster_id"]), on=["electrode_group", "cluster_id"], how="left")
        meta = dict(session_id=r.session_id, cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0])
        K["mouse_id"] = Ws.mouse_id.iloc[0]
        jobs.append((f, meta, K, i))
    out = []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for res in pool.imap_unordered(session, jobs):
            if res is not None:
                out.append(res)
    return pd.concat(out, ignore_index=True)


def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full_like(p, np.nan)
    if ok.any():
        pv = p[ok]; o = np.argsort(pv); r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        q[np.where(ok)[0][o]] = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    return q


def area_tests(A, col):
    rows = []
    Q = m62.area_qualify(A, col=col)
    rng = np.random.default_rng(0)
    for g in sorted(Q["R+"] & Q["R-"]):
        d = A[A.region == g].dropna(subset=[col])
        mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
        mi = pd.Index(mice).get_indexer(d.mouse_id); st = d.stage.to_numpy(); v = d[col].to_numpy()

        def stat(coh):
            gm = lambda c, s: v[(coh == c) & (st == s)].mean()
            return (gm("R+", "expert") - gm("R+", "learning")) - (gm("R-", "expert") - gm("R-", "learning"))
        obs = stat(d.cohort.to_numpy())
        null = np.array([stat(rng.permutation(mc)[mi]) for _ in range(5000)])
        null = null[np.isfinite(null)]
        r = dict(measure=col, region=g, interaction=obs, p_perm=(1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))
        for c in ("R+", "R-"):
            for s in ("learning", "expert"):
                q = d[(d.cohort == c) & (d.stage == s)][col]
                r[f"mean {c} {s}"], r[f"sem {c} {s}"], r[f"n {c} {s}"] = q.mean(), q.sem(), len(q)
        rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        T["q_BH"] = bh(T.p_perm)
    return T


def area_panel(ax, T, gcol, title):
    import ephys_utilities.allen_utils.allen_utils as au
    order = [g for g in au.get_area_group_custom_order() if g in set(T.region)]
    T = T.set_index("region")
    y = np.arange(len(order))[::-1]
    for k, c in enumerate(("R+", "R-")):
        for i, g in enumerate(order):
            r = T.loc[g]
            dlt = r[f"mean {c} expert"] - r[f"mean {c} learning"]
            se = np.sqrt(r[f"sem {c} expert"] ** 2 + r[f"sem {c} learning"] ** 2)
            ax.errorbar(dlt, y[i] + 0.15 - 0.3 * k, xerr=se, fmt="o", ms=3, color=m62.COH[c], lw=0.8, capsize=0, mfc="white")
    for i, g in enumerate(order):
        r = T.loc[g]
        ax.text(1.02, y[i], f"{r.p_perm:.3f}".lstrip("0") + f" ({r.q_BH:.2f})".replace("0.", "."), transform=ax.get_yaxis_transform(),
                va="center", fontsize=4.3, color="#D62728" if r.p_perm < 0.05 else "0.35")
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [m62_short(g) for g in order], fontsize=4.8)
    for tl, g in zip(ax.get_yticklabels(), order):
        tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel("Expert − day 0 (mean ± s.e.m.)", fontsize=5)
    ax.set_title(title, fontsize=5.2)


def m62_short(g):
    return (g.replace(" areas", "").replace("Somatosensory-", "Ss-").replace("Lateral septal complex", "LSX")
            .replace("Posterior parietal", "PPC").replace("Retrosplenial", "RSP").replace("Hippocampus", "HPF")
            .replace("Amygdala and hypothalamus", "Amyg."))


def figure(plt, U, S, A, out, pop, sfx=""):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    rng = np.random.default_rng(0)
    lab = "ROC selectivity s" if not sfx else "effect size d (within-class SD)"
    fig = plt.figure(figsize=(m62.W_IN, 6.4))
    gs = fig.add_gridspec(3, 4, hspace=0.8, wspace=0.95, left=0.08, right=0.95, top=0.91, bottom=0.07)
    axs1 = []
    xw, xa = (f"sel:{SEL_WS}", f"sel:{SEL_AS}") if not sfx else ("dW_full", "dA_full")
    for j, (c, st) in enumerate([(c, s) for c in ("R+", "R-") for s in ("learning", "expert")]):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax); ax.set_box_aspect(1)
        d = U[(U.cohort == c) & (U.stage == st)][[xa, xw]].dropna()
        lim = 1.0 if not sfx else 2.0
        ax.hexbin(d[xa], d[xw], gridsize=40, extent=(-lim, lim, -lim, lim), bins="log", cmap="Greys", mincnt=1, lw=0)
        Sg = S[(S.cohort == c) & (S.stage == st)]
        g_ = Sg["gain" + sfx].mean()
        xs = np.linspace(-lim, lim, 10)
        ax.plot(xs, g_ * xs, color=m62.COH[c], lw=1.2)
        ax.axhline(0, color="0.7", lw=0.4); ax.axvline(0, color="0.7", lw=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"AH vs SL ({'s' if not sfx else 'd'})", fontsize=5)
        if j == 0:
            ax.set_ylabel(f"WH vs SL ({'s' if not sfx else 'd'})", fontsize=5)
        ax.set_title(f"{c.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'} ({len(d)} units)\n"
                     f"mean gain {g_:.2f}, alignment {Sg['align' + sfx].mean():.2f}", color=m62.COH[c], fontsize=5)
    axs2 = []
    for j, (col, ttl) in enumerate([("align", "Alignment (cos θ analogue)"), ("gain", "Gain (λ analogue)"),
                                    ("ratio", "Spread ratio (ρ analogue)"), ("resid", "WH-specific residual (o² analogue)")]):
        ax = fig.add_subplot(gs[1, j]); axs2.append(ax); ax.set_box_aspect(1)
        m62.dots_panel(ax, S, col + sfx, "Cross-validated, per session" if j == 0 else "", rng, f"080{sfx} {col}", ttl,
                       ref=[(0, "0.6")] if col in ("gain", "resid") else [])
        ax.set_title(ax.get_title(), fontsize=5.0)
    axs3 = []
    if not sfx:
        for j, (col, ttl) in enumerate([("wa_converging", "Per-unit contribution to C_WA:\nconverging neurons"),
                                        ("wa_WH-only", "Per-unit contribution to C_WA:\nWH-only neurons")]):
            ax = fig.add_subplot(gs[2, j]); axs3.append(ax); ax.set_box_aspect(1)
            m62.dots_panel(ax, S, col, "Mean cross-validated s_WS·s_AS" if j == 0 else "", rng, f"080 {col}", ttl, ref=[(0, "0.6")])
            ax.set_title(ax.get_title(), fontsize=5.0)
    else:
        for j, (col, ttl) in enumerate([("C_WA_d", "Dot product a (Eq. 7)"), ("C_WW_d", "d(WH, SL) analogue")]):
            ax = fig.add_subplot(gs[2, j]); axs3.append(ax); ax.set_box_aspect(1)
            m62.dots_panel(ax, S, col, "Per unit, cross-validated" if j == 0 else "", rng, f"080d {col}", ttl, ref=[(0, "0.6")])
            ax.set_title(ax.get_title(), fontsize=5.0)
    for j, col in enumerate(("gain", "align")):
        T = area_tests(A, col + sfx)
        ax = fig.add_subplot(gs[2, 2 + j]); axs3.append(ax)
        if len(T):
            T.to_csv(out / f"area_tests_{col}{sfx}.csv", index=False)
            area_panel(ax, T, gcol, f"{'Gain' if col == 'gain' else 'Alignment'} per area group\n(p interaction (q))")
    m62.letter_row(fig, axs1, "abcd"); m62.letter_row(fig, axs2, "efgh"); m62.letter_row(fig, axs3, "ijkl")
    fig.suptitle(f"Single-neuron view of the population geometry: {lab}, WH vs SL against AH vs SL ({pop})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold")
    m62.save(fig, out, f"single_unit_geometry{'_effectsize' if sfx else ''}"); plt.close(fig)


def main(a):
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "units.parquet"
    if a.replot and f.exists():
        U0 = pd.read_parquet(f)
    else:
        U0 = compute(a.n_proc)
        U0.to_parquet(f, index=False)
    S0, A0 = summarise(U0)
    S0.to_csv(OUT / "sessions.csv", index=False); A0.to_csv(OUT / "areas.csv", index=False)
    for pop in ("learners", "all"):
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        U, S, A = flt(U0), flt(S0), flt(A0)
        out = OUT / pop; out.mkdir(exist_ok=True)
        m62.STATS.clear()
        figure(plt, U, S, A, out, pop)
        figure(plt, U, S, A, out, pop, "_d")
        T = pd.DataFrame(m62.STATS); T.to_csv(out / "stats.csv", index=False)
        an = m62.area_anova(A.rename(columns={}), sorted(set(A.region)), col="gain")
        pd.DataFrame([dict(term=k, **v) for k, v in (an or {}).items()]).to_csv(out / "area_anova_gain.csv", index=False)
        cols = [c for c in T.columns if c in ("measure", "interaction_p_perm") or c.startswith("mean_")]
        with pd.option_context("display.width", 250):
            print(pop); print(T[cols].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
