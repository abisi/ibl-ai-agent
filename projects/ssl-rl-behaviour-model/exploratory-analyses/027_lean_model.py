"""027 -- lean v6 model (user 2026-10-24 "Ok remove those"): cohort-level refits of the final model with the parameters that do nothing
or are not identifiable removed, with and without the absorption lag.
Removed (fixed): g = 0 (fitted 0 in both cohorts); R+ alpha- (in R+ every outcome is +1 and V <= 1, so delta >= 0 and alpha- never acts);
R+ theta = -0.5 (fitted -0.05 < 0, so T never falls below theta, the bias is linear and theta only trades off with c); optionally R+ d_ctx = 0.
Question: is the lag (abs) needed, or was the inst fit (R- alpha- = 0.0001 vs 0.23 with abs) a local optimum? Every fit is started from the
023c inst and abs optima besides the default and random starts.
Steps 4-5 (user 2026-10-24 "yes go, do 1-5"): +gw replaces the +ctx step by a whisker -> no-stimulus transfer g_NW; +hom rewarded outcome
scaled by thirst (homeostatic RL). --variants selects variants; results are merged into fits/027_fits.csv.
Output: fits/027_fits.csv, tables/027_latents/<variant>.parquet
Run (haas): OMP_NUM_THREADS=1 python 027_lean_model.py --n-jobs 16
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd

import common as C
import rl7_model as M

F019 = importlib.import_module("019_fit_v4")
F023 = importlib.import_module("023_v6_prediction_check")
COHORTS = ["R+", "R-"]
OPT = "+fa+dis+kap+off+ctx"
REF = {"inst": f"sess-thr-inst-noeng{OPT}", "abs": f"sess-thr-abs-noeng{OPT}"}
LEAN = {"R+": dict(alpha_neg=0.01, theta=-0.5), "R-": {}}
# variant -> (model, extra fixed parameters per cohort); g = 0 always
VARIANTS = {
    "lean-abs": (REF["abs"], LEAN),
    "lean-inst": (REF["inst"], LEAN),
    "lean-inst-noctxR+": (REF["inst"], {"R+": dict(alpha_neg=0.01, theta=-0.5, d_ctx=0.0), "R-": {}}),
}
for lag in ("abs", "inst"):
    base = f"sess-thr-{lag}-noeng+fa+dis+kap+off"
    VARIANTS[f"lean-{lag}-gw"] = (base + "+gw", LEAN)
    VARIANTS[f"lean-{lag}-hom"] = (base + "+ctx+hom", LEAN)
    VARIANTS[f"lean-{lag}-gw-hom"] = (base + "+gw+hom", LEAN)
# +stim: alpha_A, alpha_W (both free in both cohorts)
VARIANTS["lean-abs-stim"] = (REF["abs"] + "+stim", {"R+": dict(theta=-0.5), "R-": {}})
VARIANTS["lean-inst-stim"] = (REF["inst"] + "+stim", {"R+": dict(theta=-0.5), "R-": {}})
VARIANTS["lean-abs-stim-hom"] = (REF["abs"] + "+hom+stim", {"R+": dict(theta=-0.5), "R-": {}})
VARIANTS["lean-abs-stim-gw-hom"] = ("sess-thr-abs-noeng+fa+dis+kap+off+gw+hom+stim", {"R+": dict(theta=-0.5), "R-": {}})
# without the engaged / disengaged HMM: is the end-of-session drop in licking absorbed by thirst alone?
VARIANTS["lean-abs-stim-nodis"] = (REF["abs"].replace("+dis", "") + "+stim", {"R+": dict(theta=-0.5), "R-": {}})
VARIANTS["lean-abs-stim-hom-nodis"] = (REF["abs"].replace("+dis", "") + "+hom+stim", {"R+": dict(theta=-0.5), "R-": {}})
# minimal (literature form, Berditchevskaia et al. 2016 / Guitart-Masip et al. 2012): z = beta Q + go bias b0 (= -c) + linear satiety
# s_B * T(water drunk, per-session rho_s) + u_s; no thirst scaling of value (kappa = 0), no threshold, no lag, no ctx; with / without HMM
MIN = "sess-lin-inst-noeng+fa"
for dis in ("+dis", ""):
    for stim in ("", "+stim"):
        fx = {cs: dict(kappa=0.0) | ({} if stim or cs == "R-" else dict(alpha_neg=0.01)) for cs in COHORTS}
        VARIANTS[f"min{dis.replace('+', '-')}{stim.replace('+', '-')}"] = (f"{MIN}{dis}+kap+off{stim}", fx)
# simple (2026-10-29): same 10 free parameters in both cohorts; no absorption lag (inst), linear thirst-driven licking (lin), kappa = 1,
# no HMM, no ctx step, no auditory -> no-stimulus transfer (g_N = 0), stimulus-specific rates; unrewarded licks teach 0 (o_neg = 0);
# simple-m1: the same with the old -1 coding
SIMPLE = "sess-lin-inst-noeng+fa+off+stim"
VARIANTS["simple"] = (SIMPLE, {cs: dict(g_N=0.0, o_neg=0.0) for cs in COHORTS})
VARIANTS["simple-m1"] = (SIMPLE, {cs: dict(g_N=0.0, o_neg=-1.0) for cs in COHORTS})
# simple-gen (user 2026-10-29, option b): + whisker generalisation, the whisker value starts at g_W0 * V_A on the first whisker trial
# (with o_neg = 0 and V_W starting at 0, R- whisker updates have delta = 0 and alpha_W has no effect); 11 free parameters in both cohorts
VARIANTS["simple-gen"] = (SIMPLE + "+gen", {cs: dict(g_N=0.0, o_neg=0.0) for cs in COHORTS})
FIRST = ["lean-abs", "lean-inst", "lean-inst-noctxR+"]
# likelihood profiles of the whisker learning rate (lean-abs; 2026-10-24: the 023c R- optimum alpha- = 0.23 was local, lean-abs found
# alpha- = 0.0008 with a better likelihood): R+ alpha+ and R- alpha- fixed on a grid, everything else refitted; one cohort per variant
ONLY = {}
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5]
for cs, q in (("R+", "alpha_pos"), ("R-", "alpha_neg")):
    for v in GRID:
        nm = f"prof-{cs}-{q}-{v:g}"
        VARIANTS[nm] = (REF["abs"], {c: dict(LEAN[c]) | ({q: v} if c == cs else {}) for c in COHORTS})
        ONLY[nm] = cs
for v in GRID:                                          # final model (+stim): whisker learning rate alpha_W (alpha_neg slot) in R+
    nm = f"profS-R+-alpha_W-{v:g}"
    VARIANTS[nm] = (REF["abs"] + "+stim", {"R+": dict(theta=-0.5, alpha_neg=v), "R-": {}})
    ONLY[nm] = "R+"
for cs in COHORTS:                                      # simple-gen model: whisker learning rate alpha_W profile per cohort
    for v in GRID:
        nm = f"profSimple-{cs}-alpha_W-{v:g}"
        VARIANTS[nm] = (SIMPLE + "+gen", {c: dict(g_N=0.0, o_neg=0.0) | ({"alpha_neg": v} if c == cs else {}) for c in COHORTS})
        ONLY[nm] = cs
# shared whisker generalisation (user 2026-10-29): g_W0 is the same in both cohorts (identical pretraining, same onset bias in the data).
# Cohort likelihoods add and g_W0 is their only shared parameter, so the joint optimum is argmin_g [nll_R+(g) + nll_R-(g)] with all
# other parameters refitted per cohort at each fixed g (profile, both cohorts per variant); 035 picks the minimum
G_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
NOLAT = set()
for v in G_GRID:
    nm = f"profG-g_W0-{v:g}"
    VARIANTS[nm] = (SIMPLE + "+gen", {c: dict(g_N=0.0, o_neg=0.0, g_W0=v) for c in COHORTS})
    NOLAT.add(nm)


def shared_g(g):
    """simple-gen with g_W0 fixed at the shared value g (simple-genS) and its alpha_W profiles per cohort (profSimpleS-*)"""
    VARIANTS["simple-genS"] = (SIMPLE + "+gen", {c: dict(g_N=0.0, o_neg=0.0, g_W0=g) for c in COHORTS})
    for cs in COHORTS:
        for v in GRID:
            nm = f"profSimpleS-{cs}-alpha_W-{v:g}"
            VARIANTS[nm] = (SIMPLE + "+gen", {c: dict(g_N=0.0, o_neg=0.0, g_W0=g) | ({"alpha_neg": v} if c == cs else {})
                                               for c in COHORTS})
            ONLY[nm] = cs


# core model (user 2026-10-30, "ok with these conclusions"): simple-gen + engagement HMM (absorbing within a session, h_back fixed);
# dropped for good: absorption lag, kappa, theta, ctx step, -1 coding. Add-ons for the single-mouse fits (one at a time, then combined):
# +rec spontaneous recovery, +fast two-rate whisker learning, +pers lick history, +gw whisker -> no-stimulus transfer, +hab whisker
# habituation (full overnight recovery; +habr partial). g_W0 per cohort in the screen (fit 027l / 027m); shared for the chosen model
CORE = "sess-lin-inst-noeng+fa+dis+off+stim+gen"
FX_CORE = dict(g_N=0.0, o_neg=0.0, h_back=-50.0)


def core_name(*extra):
    opts = set(CORE.split("+")[1:]) | set(extra)
    return CORE.split("+")[0] + "".join(f"+{o}" for o in M.OPTS if o in opts)


VARIANTS["core"] = (CORE, {c: dict(FX_CORE) for c in COHORTS})
for nm, ex in (("rec", ("rec",)), ("fast", ("fast",)), ("pers", ("pers",)), ("gw", ("gw",)), ("hab", ("hab",)), ("habr", ("hab", "habr"))):
    VARIANTS[f"core-{nm}"] = (core_name(*ex), {c: dict(FX_CORE) for c in COHORTS})


def add_core_variant(tag, extra, g_shared=None):
    """register core + extra options (e.g. the combination chosen after the screen), optionally with g_W0 fixed (shared)"""
    fx = dict(FX_CORE) | ({} if g_shared is None else dict(g_W0=g_shared))
    VARIANTS[tag] = (core_name(*extra), {c: dict(fx) for c in COHORTS})
    return tag


LAT_DIR = C.TAB / "027_latents"
D = None
FIXED0 = dict(M.FIXED)


def configure(variant, cs):
    """patch rl7 so that free_params drops g and the variant's fixed parameters (workers are forked; set per job)"""
    name, fixed = VARIANTS[variant]
    fx = dict(g=0.0, **fixed[cs])
    M.FIXED.clear(); M.FIXED.update(FIXED0)        # pool workers are reused: no fixed value may leak from a previous job
    M.FIXED.update(fx)
    return name, [k for k in M.free_params(name) if k not in fx]


def job(args):
    variant, cs = args
    name, fp = configure(variant, cs)
    orig = M.free_params
    M.free_params = lambda n: fp
    try:
        A = M.arrays(F023.subset(D, cs))
        ref = pd.read_csv(C.FITS / "023c_fits.csv")
        starts = [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                  for _, r in ref[(ref.model.isin(REF.values())) & (ref.cohort_set == cs)].iterrows()]
        if (C.FITS / "027_fits.csv").exists():          # also from the lean optima
            lf = pd.read_csv(C.FITS / "027_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[lf.variant.isin(["lean-abs", "lean-inst"]) & (lf.cohort_set == cs)].iterrows()]
        if "+hom" in name and (C.FITS / "027b_fits.csv").exists():
            lf = pd.read_csv(C.FITS / "027b_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[lf.variant.isin(["lean-abs-hom", "lean-abs-gw-hom"]) & (lf.cohort_set == cs)].iterrows()]
        if "+stim" in name and (C.FITS / "027c_fits.csv").exists():
            lf = pd.read_csv(C.FITS / "027c_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[(lf.variant == "lean-abs-stim") & (lf.cohort_set == cs)].iterrows()]
        if "+hom" in name and "+stim" in name and (C.FITS / "027d_fits.csv").exists():
            lf = pd.read_csv(C.FITS / "027d_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[(lf.variant == "lean-abs-stim-hom") & (lf.cohort_set == cs)].iterrows()]
        if name.startswith(SIMPLE) and (C.FITS / "027f_fits.csv").exists():   # both codings' optima (a fast and a slow alpha_W start)
            lf = pd.read_csv(C.FITS / "027f_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[lf.variant.isin(["simple", "simple-m1", "simple-gen"]) & (lf.cohort_set == cs)].iterrows()]
        if "+gen" in name and (C.FITS / "027h_fits.csv").exists():
            lf = pd.read_csv(C.FITS / "027h_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[(lf.variant == "simple-gen") & (lf.cohort_set == cs)].iterrows()]
        if "+gen" in name and (C.FITS / "027j_fits.csv").exists():      # shared-g profile optima
            lf = pd.read_csv(C.FITS / "027j_fits.csv")
            starts += [[float(np.clip(r.get(f"p_{k}", M.DEFAULT[k]), *M.BOUNDS[k])) for k in fp]
                       for _, r in lf[lf.variant.str.startswith("profG-") & (lf.cohort_set == cs)].iterrows()]
        if name.startswith(CORE.split("+gen")[0].replace("+stim", "")):     # core family: simple optima + HMM of the final, then core optima
            hm = {}
            starts = []                         # only the core-family starts (the older optima have other structures)
            if (C.FITS / "027c_fits.csv").exists():
                lc = pd.read_csv(C.FITS / "027c_fits.csv")
                q = lc[(lc.variant == "lean-abs-stim") & (lc.cohort_set == cs)]
                if len(q):
                    hm = {k: float(q.iloc[0][f"p_{k}"]) for k in ("h0", "h1")}
                    starts.append([float(np.clip(q.iloc[0].get(f"p_{k}", M.DEFAULT[k]) if k not in ("g_W0",) else 0.12,
                                                 *M.BOUNDS[k])) for k in fp])
            core_files = [(f.name, None) for f in sorted(C.FITS.glob("027[l-z]_fits.csv"))]   # core-family fits (screen, combos, profiles)
            for f, vs in [("027h_fits.csv", ["simple-gen"]), ("027k_fits.csv", ["simple-genS"])] + core_files:
                if not (C.FITS / f).exists():
                    continue
                lf = pd.read_csv(C.FITS / f)
                lf = lf[(lf.cohort_set == cs) & (lf.variant.isin(vs) if vs else lf.model.str.startswith(CORE.split("+")[0] + "+fa+dis+off"))]
                for _, r in lf.iterrows():
                    src = set(r.free.split()) if isinstance(r.get("free"), str) else set()
                    st = []
                    for k in fp:
                        v = r.get(f"p_{k}", np.nan)
                        if k in hm and k not in src:
                            v = hm[k]
                        if k not in src and k not in hm and k not in ("g_W0",):
                            v = M.DEFAULT[k] if pd.isna(v) or k in ("r_W", "alpha_F", "pi_L", "nu_W", "lam_W", "r_H", "g_NW") else v
                        st.append(float(np.clip(M.DEFAULT[k] if pd.isna(v) else v, *M.BOUNDS[k])))
                    starts.append(st)
        if "gw" in name:                       # the ctx step expressed as transfer: start g_NW from 0.3 with d_ctx dropped
            starts += [[s_[i] if k != "g_NW" else 0.3 for i, k in enumerate(fp)] for s_ in starts]
        rng = np.random.default_rng(zlib.crc32(f"{variant}{cs}".encode()))
        p, nll = M.fit(name, A, np.ones(len(A[0]), np.int64), rng, n_random=5, extra_starts=starts)
    finally:
        M.free_params = orig
    k = len(fp)
    return dict(variant=variant, model=name, cohort_set=cs, k=k, n=len(A[0]), nll=nll, bic=2 * nll + k * np.log(len(A[0])),
                free=" ".join(fp), **{f"p_{q}": p[q] for q in M.P_ALL})


def latents(F, variant):
    out = []
    for cs in COHORTS:
        r = F[(F.variant == variant) & (F.cohort_set == cs)].iloc[0]
        d = F023.subset(D, cs)
        p = {q: r[f"p_{q}"] if pd.notna(r.get(f"p_{q}", np.nan)) else M.FIXED[q] for q in M.P_ALL}
        L = M.latents(r.model, p, M.arrays(d))
        o = d[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group"]].copy()
        o["p_lick"] = L["p_lick"]; o["p_engaged"] = L["p_engaged"]; o["lrho_smooth"] = L["lrho_smooth"]
        for i, q in enumerate(M.LATENTS):
            o[q] = L["lat"][:, i]
        out.append(o)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=FIRST)
    ap.add_argument("--out", default="027_fits.csv", help="fits file in fits/ (merged by variant if it exists)")
    ap.add_argument("--shared-g", type=float, default=None, help="register simple-genS / profSimpleS-* with g_W0 fixed at this value")
    ap.add_argument("--core-extra", nargs="*", default=[],
                    help="tag=opt1,opt2[,g=0.12][,gprof]: register core + options (g_W0 fixed if g=; gprof adds the shared-g profile "
                         "profC-<tag>-g-<v> over G_GRID, keyword profiles-c)")
    a = ap.parse_args()
    if a.shared_g is not None:
        shared_g(a.shared_g)
    for e in a.core_extra:
        tag, opts = e.split("=", 1)
        o = [x for x in opts.split(",") if x]
        g = [float(x[2:]) for x in o if x.startswith("g=")]
        ex = [x for x in o if not x.startswith("g=") and x != "gprof"]
        add_core_variant(tag, ex, g[0] if g else None)
        if "gprof" in o:
            for v in G_GRID:
                NOLAT.add(add_core_variant(f"profC-{tag}-g-{v:g}", ex, v))
    D = F019.load()
    groups = {"profiles": [v for v in ONLY if v.startswith("prof-")],          # keywords expand to profile sets, mixable with names
              "profiles-simple": [v for v in ONLY if v.startswith("profSimple-")],
              "profiles-g": sorted(v for v in NOLAT if v.startswith("profG-")),
              "profiles-c": sorted(v for v in NOLAT if v.startswith("profC-")),
              "profiles-simpleS": [v for v in ONLY if v.startswith("profSimpleS-")],
              "profiles-stim": [v for v in ONLY if v.startswith("profS-")]}
    a.variants = [w for v in a.variants for w in groups.get(v, [v])]
    jobs = [(v, cs) for v in a.variants for cs in COHORTS if not (v.endswith("R+") and cs == "R-") and ONLY.get(v, cs) == cs]
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        F = pd.DataFrame(pool.map(job, jobs, chunksize=1))
    out = C.FITS / a.out
    if out.exists():
        old = pd.read_csv(out)
        F = pd.concat([old[~old.variant.isin(a.variants)], F], ignore_index=True)
    else:
        ref = pd.read_csv(C.FITS / "023c_fits.csv")
        ref = ref[ref.model.isin(REF.values())].assign(variant=lambda e: "023c-" + e.model.str.split("-").str[2])
        F = pd.concat([ref, F], ignore_index=True)
    F.to_csv(out, index=False)
    show = ["variant", "cohort_set", "k", "nll", "bic", "p_alpha_pos", "p_alpha_neg", "p_beta", "p_c", "p_s_B", "p_theta", "p_tau_a",
            "p_kappa", "p_d_ctx", "p_g_NW"]
    print(F[show].round(4).to_string(), flush=True)
    LAT_DIR.mkdir(exist_ok=True)
    for v in a.variants:
        if not v.endswith("R+") and v not in ONLY and v not in NOLAT:
            latents(F, v).to_parquet(LAT_DIR / f"{v}.parquet")
