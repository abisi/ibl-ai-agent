"""rl7_model -- model v6 (model_v6_design.md, 2026-10-23): v5 learning core (overlap transfer g, asymmetric alpha) with the poster
motivation block (Matteucci et al. "Mousemind-Q"; model_v3_design.md 2.2): thirst with per-session capacity, stimulus-independent
thirst-driven lick bias, optional absorption lag and v5 engagement. v6b options (023 check: abrupt session-end collapse and R- auditory
dragged down with FA by one shared thirst): +fa learned no-stimulus value, +dis absorbing disengaged state.

  z_t = beta * T_t * V_t - c + B(T_t)        P(lick | engaged) = 0.01 + 0.98 * e_t * sigmoid(z_t)
  T_t = max(0, 1 - rho_s * S_t); S_t water drunk this session (inst) or absorbed so far (abs, time constant tau_a trials)
  B: lin s_B * T | thr s_B * softplus(20 (T - theta)) / 20
  capacity: pop one rho per cohort | sess log rho_s ~ N(mu_rho, sd_rho^2), integrated per session by Gauss-Hermite quadrature
  +fa   no-stimulus trials have a value V_N = w_N + g_N * w_A (w_N starts at w_N0 per mouse); an FA lick is an unrewarded outcome
        (o = -1), w_N += alpha_N * (o - V_N). Without +fa, V_N = 0.
  +dis  hidden state engaged / disengaged (P(lick | disengaged) = 0.01; re-engagement probability sigmoid(h_back) per trial, -50 =
        absorbing within the session); P(engaged) = 1 at session start,
        filtered on each trial's lick (Bayes), then hazard h_t = sigmoid(h0 + h1 * (1 - T_t)) of disengaging before the next trial.
Learning on licked stimulus trials: delta = o - V(s), w += alpha+- * delta * x_s, x_A = (1, g), x_W = (g, 1); outcomes auditory +1,
whisker +1 (R+) / -1 (R-). Weights, Rbar and satiety depend only on observed licks and rewards, so sessions are conditionally independent
given the shared parameters; the quadrature over rho_s and the engaged-state filter (per node) are exact.
  +kap  stimulus drive beta * T^kappa * V (kappa fitted; 1 without +kap, 0 = stimulus licking independent of thirst)
  +off  per-session lick-propensity offset u_s ~ N(0, sd_u^2) added to z (integrated jointly with rho_s: 7 x 5 nodes)
  +ctx  context step at whisker onset: w_N += d_ctx at the mouse's first whisker trial (needs +fa)
  +gw   whisker -> no-stimulus transfer (027, replaces the +ctx step by a mechanism): V_N = w_N + g_N * w_A + g_NW * w_W (needs +fa)
  +hom  homeostatic outcome (027; Keramati & Gutkin 2014): a rewarded lick teaches o = Tbar = max(0, 1 - exp(mu_rho) * S) instead of 1
        (thirst at the mouse's typical capacity, so the weights stay independent of the session's rho_s); unrewarded licks keep o = -1
  +stim stimulus-specific learning rates (027): alpha_pos slot = alpha_A (auditory updates), alpha_neg slot = alpha_W (whisker updates),
        whatever the sign of delta (a single whisker learning rate in both cohorts)
  o_neg (parameter, fixed; 2026-10-29) outcome of an unrewarded lick (R- whisker, FA): -1 (default) or 0 (simple model)
  +gen  whisker generalisation (2026-10-29): at the mouse's first whisker trial w_W = g_W0 * V_A (whisker value starts at a fraction of
        the auditory value; data: first whisker trial licked 0.55 vs 0.27 false alarms before it, both cohorts)
Single-mouse add-ons (2026-10-30; user: "improve single-mouse fits, especially R-"), each one parameter set:
  +rec  spontaneous recovery: at each new session w_W += r_W * (W0 - w_W), W0 = the whisker value at onset (Rescorla 2004)
  +fast two-rate whisker learning: V_W = w_W + w_F, the fast part w_F learns from the same delta at alpha_F and is reset to 0 at each
        session start (fast memory not consolidated; Smith et al. 2006)
  +pers lick history: z += pi_L * (lick on the previous trial), reset at session start (choice-history kernel, Lau & Glimcher 2005)
  +hab  whisker habituation: z_W += H, H = nu_W at whisker onset, H *= (1 - lam_W) after every whisker trial, and H recovers fully
        overnight (+habr: H += r_H * (nu_W - H) at session start, r_H free); stimulus-specific, outcome-independent (Thompson & Spencer
        1966; Rankin et al. 2009)
Model name: <pop|sess>-<lin|thr>-<inst|abs>-<noeng|eng>[+fa][+dis][+kap][+off][+ctx][+gw][+hom][+stim][+gen][+rec][+fast][+pers][+hab][+habr]
"""
from __future__ import annotations

import numpy as np
from numba import njit
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize

P_ALL = ["alpha_pos", "alpha_neg", "beta", "g", "c", "mu_rho", "sd_rho", "s_B", "theta", "tau_a", "e_min", "r_e", "alpha_R", "R0",
         "w_N0", "alpha_N", "g_N", "h0", "h1", "kappa", "sd_u", "d_ctx", "h_back", "g_NW", "o_neg", "g_W0", "r_W", "alpha_F", "pi_L",
         "nu_W", "lam_W", "r_H"]
BOUNDS = dict(alpha_pos=(1e-4, 1.0), alpha_neg=(1e-4, 1.0), beta=(0.1, 100.0), g=(0.0, 1.0), c=(-10.0, 10.0), mu_rho=(-9.0, -1.0),
              sd_rho=(0.0, 2.0), s_B=(0.0, 20.0), theta=(-0.5, 1.0), tau_a=(0.1, 500.0), e_min=(0.0, 1.0), r_e=(0.01, 2.0),
              alpha_R=(1e-3, 1.0), R0=(0.0, 1.0), w_N0=(-3.0, 3.0), alpha_N=(1e-4, 1.0), g_N=(0.0, 1.0), h0=(-50.0, 2.0), h1=(0.0, 20.0),
              kappa=(0.0, 3.0), sd_u=(0.0, 3.0), d_ctx=(-5.0, 5.0), h_back=(-50.0, 2.0), g_NW=(0.0, 1.0), o_neg=(-2.0, 0.0), g_W0=(0.0, 1.0),
              r_W=(0.0, 1.0), alpha_F=(1e-4, 1.0), pi_L=(-5.0, 5.0), nu_W=(-5.0, 10.0), lam_W=(1e-4, 1.0), r_H=(0.0, 1.0))
FIXED = dict(alpha_pos=0.01, alpha_neg=0.01, beta=5.0, g=0.0, c=2.0, mu_rho=-5.0, sd_rho=0.0, s_B=0.0, theta=-0.5, tau_a=0.1, e_min=1.0,
             r_e=1.0, alpha_R=0.1, R0=0.0, w_N0=0.0, alpha_N=1e-4, g_N=0.0, h0=-50.0, h1=0.0, kappa=1.0, sd_u=0.0, d_ctx=0.0,
             h_back=-50.0, g_NW=0.0, o_neg=-1.0, g_W0=0.0, r_W=0.0, alpha_F=0.0, pi_L=0.0, nu_W=0.0, lam_W=0.1, r_H=1.0)
DEFAULT = dict(alpha_pos=0.01, alpha_neg=0.01, beta=5.0, g=0.05, c=2.5, mu_rho=-4.7, sd_rho=0.5, s_B=3.0, theta=0.2, tau_a=20.0,
               e_min=0.3, r_e=0.2, alpha_R=0.1, R0=0.5, w_N0=0.0, alpha_N=0.01, g_N=0.3, h0=-6.0, h1=4.0, kappa=0.3, sd_u=0.7, d_ctx=-0.5, h_back=-4.0, g_NW=0.3, o_neg=-1.0, g_W0=0.5,
               r_W=0.3, alpha_F=0.05, pi_L=0.5, nu_W=1.0, lam_W=0.05, r_H=0.7)
LOG = {"alpha_pos", "alpha_neg", "beta", "tau_a", "r_e", "alpha_R", "alpha_N", "alpha_F", "lam_W"}
CORE = ["alpha_pos", "alpha_neg", "beta", "g", "c", "mu_rho", "s_B"]
K_NODES = 11
K_RHO_2D, K_U = 7, 5
OPTS = ("fa", "dis", "kap", "off", "ctx", "gw", "hom", "stim", "gen", "rec", "fast", "pers", "hab", "habr")   # canonical order in names
P_DIS = 0.01
LATENTS = ["V", "V_A", "V_W", "w_A", "w_W", "S", "eng", "V_N", "V_F", "H_W"]


def model_names():
    return [f"{a}-{b}-{s}-{e}" for a in ("pop", "sess") for b in ("lin", "thr") for s in ("inst", "abs") for e in ("noeng", "eng")]


def split(name):
    base, *opts = name.split("+")
    return base.split("-"), set(opts)


def parts(name):
    (cap, bias, ab, en), opts = split(name)
    return (int(cap == "sess"), int(bias == "thr"), int(ab == "abs"), int(en == "eng"), int("fa" in opts), int("dis" in opts),
            int("ctx" in opts), int("hom" in opts), int("stim" in opts))


def free_params(name):
    (cap, bias, ab, en), opts = split(name)
    p = list(CORE)
    p += ["sd_rho"] if cap == "sess" else []
    p += ["theta"] if bias == "thr" else []
    p += ["tau_a"] if ab == "abs" else []
    p += ["e_min", "r_e", "alpha_R", "R0"] if en == "eng" else []
    p += ["w_N0", "alpha_N", "g_N"] if "fa" in opts else []
    p += ["h0", "h1", "h_back"] if "dis" in opts else []
    p += ["kappa"] if "kap" in opts else []
    p += ["sd_u"] if "off" in opts else []
    p += ["d_ctx"] if "ctx" in opts else []
    p += ["g_NW"] if "gw" in opts else []
    p += ["g_W0"] if "gen" in opts else []
    p += ["r_W"] if "rec" in opts else []
    p += ["alpha_F"] if "fast" in opts else []
    p += ["pi_L"] if "pers" in opts else []
    p += ["nu_W", "lam_W"] if "hab" in opts else []
    p += ["r_H"] if "habr" in opts else []
    return p


def full_vector(name, x):
    p = dict(FIXED)
    p.update(dict(zip(free_params(name), x)))
    return np.array([p[k] for k in P_ALL], float), p


def nodes(name, P):
    """quadrature nodes of the session-level random effects: log rho_s (and, with +off, the lick-propensity offset u_s), flattened,
    with log prior weights"""
    off = "off" in split(name)[1] and P[20] > 0
    if name.startswith("pop") or P[6] <= 0:
        lr, wr = np.array([P[5]]), np.ones(1)
    else:
        x, wr = hermegauss(K_RHO_2D if off else K_NODES)
        lr = P[5] + P[6] * x
    if off:
        x, wu = hermegauss(K_U)
        u = P[20] * x
    else:
        u, wu = np.zeros(1), np.ones(1)
    LR, UU = np.meshgrid(lr, u, indexing="ij")
    W = np.outer(wr / wr.sum(), wu / wu.sum())
    return LR.ravel(), UU.ravel(), np.log(W.ravel())


@njit(cache=True)
def _softplus(x):
    return x if x > 30.0 else np.log1p(np.exp(x))


@njit(cache=True)
def _p(beta, c, V, T, thr, sB, theta, e, kappa, u):
    B = sB * _softplus(20.0 * (T - theta)) / 20.0 if thr == 1 else sB * T
    z = beta * T ** kappa * V - c + B + u
    s = 1.0 / (1.0 + np.exp(-z)) if z > -50.0 else 0.0
    return 0.01 + 0.98 * e * s


@njit(cache=True)
def _hazard(h0, h1, T):
    return 1.0 / (1.0 + np.exp(-(h0 + h1 * (1.0 - T))))


@njit(cache=True)
def kernel(tt, lick, rplus, sess_start, mouse_start, P, thr, abs_on, eng_on, fa_on, dis_on, ctx_on, hom_on, stim_on, lrho, uoff, lw, mask, keep, out_pk, out_lat,
           out_pi):
    ap, an, beta, g, c = P[0], P[1], P[2], P[3], P[4]
    sB, theta, tau, emin, re, aR, R0 = P[7], P[8], P[9], P[10], P[11], P[12], P[13]
    wN0, aN, gN, h0, h1, kap, dctx = P[14], P[15], P[16], P[17], P[18], P[19], P[21]
    hb = 1.0 / (1.0 + np.exp(-P[22]))
    gNW = P[23]; rho_bar = np.exp(P[5]); oneg = P[24]; gW0 = P[25]
    rW, aF, piL, nuW, lamW, rH = P[26], P[27], P[28], P[29], P[30], P[31]
    K = lrho.shape[0]
    rho = np.exp(lrho)
    llk = np.zeros(K)
    pi = np.ones(K)
    total = 0.0
    wA = 0.0; wW = 0.0; wN = wN0; R = 0.0; U = 0.0; Rbar = 0.0; seen_w = False; wF = 0.0; W0 = 0.0; hN = 0.0; hL = 0.0
    dec = np.exp(-1.0 / tau)
    for t in range(tt.shape[0]):
        if mouse_start[t] == 1 or sess_start[t] == 1:
            if t > 0:
                m = np.max(lw + llk)
                total += m + np.log(np.sum(np.exp(lw + llk - m)))
            llk[:] = 0.0
            pi[:] = 1.0
            R = 0.0; U = 0.0
            Rbar = R0 if eng_on == 1 else 0.0
            wF = 0.0; hL = 0.0                 # +fast / +pers: fast whisker memory and lick history do not survive the night
            if seen_w and mouse_start[t] == 0:
                wW += rW * (W0 - wW)           # +rec: spontaneous recovery of the whisker value toward its start
                hN += rH * (nuW - hN)          # +hab: the habituated whisker drive recovers overnight
        if mouse_start[t] == 1:
            wA = 0.0; wW = 0.0; wN = wN0; seen_w = False; W0 = 0.0; hN = 0.0
        k = tt[t]
        if k == 1 and not seen_w:
            seen_w = True
            if gW0 > 0.0:                  # generalisation: the whisker value starts at a fraction of the auditory value
                wW = gW0 * (wA + g * wW)
            W0 = wW; hN = nuW
            if ctx_on == 1:
                wN += dctx
        VA = wA + g * wW; VW = wW + g * wA + wF
        VN = wN + gN * wA + gNW * wW if fa_on == 1 else 0.0
        V = VA if k == 0 else (VW if k == 1 else VN)
        S = R - U if abs_on == 1 else R
        e = emin + (1.0 - emin) * (1.0 - np.exp(-Rbar / re)) if eng_on == 1 else 1.0
        l = lick[t]
        xz = piL * hL + (hN if k == 1 else 0.0)     # +pers lick history, +hab whisker drive
        for j in range(K):
            T = max(0.0, 1.0 - rho[j] * S)
            pe = _p(beta, c, V, T, thr, sB, theta, e, kap, uoff[j] + xz)
            p = pi[j] * pe + (1.0 - pi[j]) * P_DIS if dis_on == 1 else pe
            if keep:
                out_pk[t, j] = p
                out_pi[t, j] = pi[j]
            if mask[t] == 1:
                llk[j] += np.log(p) if l == 1 else np.log(1.0 - p)
            if dis_on == 1:
                le = pe if l == 1 else 1.0 - pe
                ld = P_DIS if l == 1 else 1.0 - P_DIS
                post = pi[j] * le / (pi[j] * le + (1.0 - pi[j]) * ld)
                pi[j] = post * (1.0 - _hazard(h0, h1, T)) + (1.0 - post) * hb
        if keep:
            out_lat[t, 0] = V; out_lat[t, 1] = VA; out_lat[t, 2] = VW; out_lat[t, 3] = wA; out_lat[t, 4] = wW
            out_lat[t, 5] = S; out_lat[t, 6] = e; out_lat[t, 7] = VN; out_lat[t, 8] = wF; out_lat[t, 9] = hN
        water = 0.0
        if l == 1:
            if k < 2:
                if k == 0:
                    o = 1.0; water = 1.0
                else:
                    o = 1.0 if rplus[t] == 1 else oneg; water = 1.0 if rplus[t] == 1 else 0.0
                if hom_on == 1 and water > 0.0:
                    o = max(0.0, 1.0 - rho_bar * S)
                d = o - V
                if stim_on == 1:
                    a = ap if k == 0 else an
                else:
                    a = ap if d > 0 else an
                wA += a * d * (1.0 if k == 0 else g)
                wW += a * d * (g if k == 0 else 1.0)
                if k == 1:
                    wF += aF * d
            elif fa_on == 1:
                wN += aN * (oneg - VN)
        if k == 1:
            hN *= 1.0 - lamW
        hL = 1.0 * l
        R += water
        U = U * dec + water
        Rbar += aR * (water - Rbar)
    m = np.max(lw + llk)
    total += m + np.log(np.sum(np.exp(lw + llk - m)))
    return total


@njit(cache=True)
def sim_kernel(tt, rplus, sess_start, mouse_start, P, thr, abs_on, eng_on, fa_on, dis_on, ctx_on, hom_on, stim_on, rho_t, uoff_t, u, u2, lick, p_out):
    """simulate licks; rho_t per trial (constant within session), learning on simulated licks, engaged state drawn"""
    ap, an, beta, g, c = P[0], P[1], P[2], P[3], P[4]
    sB, theta, tau, emin, re, aR, R0 = P[7], P[8], P[9], P[10], P[11], P[12], P[13]
    wN0, aN, gN, h0, h1, kap, dctx = P[14], P[15], P[16], P[17], P[18], P[19], P[21]
    hb = 1.0 / (1.0 + np.exp(-P[22]))
    gNW = P[23]; rho_bar = np.exp(P[5]); oneg = P[24]; gW0 = P[25]
    rW, aF, piL, nuW, lamW, rH = P[26], P[27], P[28], P[29], P[30], P[31]
    wA = 0.0; wW = 0.0; wN = wN0; R = 0.0; U = 0.0; Rbar = 0.0; engaged = True; seen_w = False; wF = 0.0; W0 = 0.0; hN = 0.0; hL = 0.0
    dec = np.exp(-1.0 / tau)
    for t in range(tt.shape[0]):
        if mouse_start[t] == 1 or sess_start[t] == 1:
            R = 0.0; U = 0.0; engaged = True
            Rbar = R0 if eng_on == 1 else 0.0
            wF = 0.0; hL = 0.0                 # +fast / +pers: fast whisker memory and lick history do not survive the night
            if seen_w and mouse_start[t] == 0:
                wW += rW * (W0 - wW)           # +rec: spontaneous recovery of the whisker value toward its start
                hN += rH * (nuW - hN)          # +hab: the habituated whisker drive recovers overnight
        if mouse_start[t] == 1:
            wA = 0.0; wW = 0.0; wN = wN0; seen_w = False; W0 = 0.0; hN = 0.0
        k = tt[t]
        if k == 1 and not seen_w:
            seen_w = True
            if gW0 > 0.0:                  # generalisation: the whisker value starts at a fraction of the auditory value
                wW = gW0 * (wA + g * wW)
            W0 = wW; hN = nuW
            if ctx_on == 1:
                wN += dctx
        VA = wA + g * wW; VW = wW + g * wA + wF
        VN = wN + gN * wA + gNW * wW if fa_on == 1 else 0.0
        V = VA if k == 0 else (VW if k == 1 else VN)
        S = R - U if abs_on == 1 else R
        T = max(0.0, 1.0 - rho_t[t] * S)
        e = emin + (1.0 - emin) * (1.0 - np.exp(-Rbar / re)) if eng_on == 1 else 1.0
        p = _p(beta, c, V, T, thr, sB, theta, e, kap, uoff_t[t] + piL * hL + (hN if k == 1 else 0.0)) if engaged else P_DIS
        p_out[t] = p
        l = 1 if u[t] < p else 0
        lick[t] = l
        if dis_on == 1:
            if engaged and u2[t] < _hazard(h0, h1, T):
                engaged = False
            elif not engaged and u2[t] < hb:
                engaged = True
        water = 0.0
        if l == 1:
            if k < 2:
                if k == 0:
                    o = 1.0; water = 1.0
                else:
                    o = 1.0 if rplus[t] == 1 else oneg; water = 1.0 if rplus[t] == 1 else 0.0
                if hom_on == 1 and water > 0.0:
                    o = max(0.0, 1.0 - rho_bar * S)
                d = o - V
                if stim_on == 1:
                    a = ap if k == 0 else an
                else:
                    a = ap if d > 0 else an
                wA += a * d * (1.0 if k == 0 else g)
                wW += a * d * (g if k == 0 else 1.0)
                if k == 1:
                    wF += aF * d
            elif fa_on == 1:
                wN += aN * (oneg - VN)
        if k == 1:
            hN *= 1.0 - lamW
        hL = 1.0 * l
        R += water
        U = U * dec + water
        Rbar += aR * (water - Rbar)


def _modes(name):
    return parts(name)[1:]


def loglik(name, x, A, mask):
    P, _ = full_vector(name, x)
    lr, uo, lw = nodes(name, P)
    e = np.empty((0, 0))
    return kernel(*A, P, *_modes(name), lr, uo, lw, mask, False, e, e, e)


def latents(name, p, A):
    """per-trial node probabilities, causal predictive p_lick, causal and smoothed posterior mean log rho, P(engaged), latents"""
    P = np.array([p.get(k, FIXED[k]) for k in P_ALL], float)
    lr, uo, lw = nodes(name, P)
    n = len(A[0])
    pk = np.empty((n, len(lr))); lat = np.empty((n, len(LATENTS))); pik = np.empty((n, len(lr)))
    kernel(*A, P, *_modes(name), lr, uo, lw, np.ones(n, np.int64), True, pk, lat, pik)
    lick = A[1]
    ll = np.where(lick[:, None] == 1, np.log(pk), np.log(1 - pk))
    sid = np.cumsum(A[3] | A[4]) - 1
    tot = np.zeros((sid.max() + 1, len(lr)))
    np.add.at(tot, sid, ll)
    c = np.cumsum(ll, axis=0) - ll                                        # log-lik of all trials before t
    cum = c - c[np.flatnonzero(np.r_[1, np.diff(sid)])][sid]              # ... restricted to t's session
    wc = lw + cum; wc = np.exp(wc - wc.max(1, keepdims=True)); wc /= wc.sum(1, keepdims=True)
    ws = lw + tot[sid]; ws = np.exp(ws - ws.max(1, keepdims=True)); ws /= ws.sum(1, keepdims=True)
    return dict(pk=pk, lat=lat, lrho=lr, p_lick=(wc * pk).sum(1), p_engaged=(wc * pik).sum(1), lrho_causal=wc @ lr, u_smooth=ws @ uo,
                lrho_smooth=ws @ lr, w_smooth=ws)


def simulate(name, p, A, rng, w_smooth=None):
    """simulate; per session draw log rho from the smoothed posterior (w_smooth, per trial rows) or from the prior"""
    tt, _, rplus, ss, ms = A
    P = np.array([p.get(k, FIXED[k]) for k in P_ALL], float)
    lr, uo, lw = nodes(name, P)
    sid = np.cumsum(ss | ms) - 1
    first = np.flatnonzero(np.r_[1, np.diff(sid)])
    W = w_smooth[first] if w_smooth is not None else np.tile(np.exp(lw), (len(first), 1))
    j = np.array([rng.choice(len(lr), p=w / w.sum()) for w in W])
    rho_t = np.exp(lr[j][sid]); uoff_t = uo[j][sid]
    lick = np.empty(len(tt), np.int64); pp = np.empty(len(tt))
    sim_kernel(tt, rplus, ss, ms, P, *_modes(name), rho_t, uoff_t, rng.uniform(size=len(tt)), rng.uniform(size=len(tt)), lick, pp)
    return lick, pp


def fit(name, A, mask, rng, n_random=5, extra_starts=()):
    fp = free_params(name)
    lo = np.array([BOUNDS[k][0] for k in fp]); hi = np.array([BOUNDS[k][1] for k in fp])
    islog = np.array([k in LOG for k in fp])
    blo = np.where(islog, np.log(np.where(islog, lo, 1.0)), lo); bhi = np.where(islog, np.log(np.where(islog, hi, 1.0)), hi)

    def to_x(y):
        return np.where(islog, np.exp(y), y)

    def nll(y):
        v = -loglik(name, to_x(y), A, mask)
        return v if np.isfinite(v) else 1e12
    starts = [[DEFAULT[k] for k in fp]] + [list(s) for s in extra_starts]
    for _ in range(n_random):
        starts.append([float(np.exp(rng.uniform(np.log(BOUNDS[k][0]), np.log(BOUNDS[k][1])))) if k in LOG else rng.uniform(*BOUNDS[k])
                       for k in fp])
    best = None
    for s in starts:
        s = np.clip(s, lo, hi)
        y0 = np.clip(np.where(islog, np.log(np.where(islog, s, 1.0)), s), blo, bhi)
        r = minimize(nll, y0, method="L-BFGS-B", bounds=list(zip(blo, bhi)), options=dict(maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    return full_vector(name, to_x(best.x))[1], float(best.fun)


def nested_of(name):
    """(sub-model, extra values that make `name` reproduce it)"""
    (cap, bias, ab, en), opts = split(name)
    suf = "".join(f"+{o}" for o in OPTS if o in opts)
    out = []
    if cap == "sess":
        out.append((f"pop-{bias}-{ab}-{en}{suf}", dict(sd_rho=0.0)))
    if ab == "abs":
        out.append((f"{cap}-{bias}-inst-{en}{suf}", dict(tau_a=0.1)))
    if en == "eng":
        out.append((f"{cap}-{bias}-{ab}-noeng{suf}", dict(e_min=1.0, R0=0.0)))
    base = f"{cap}-{bias}-{ab}-{en}"
    neutral = dict(fa=dict(w_N0=0.0, alpha_N=1e-4, g_N=0.0), dis=dict(h0=-50.0, h1=0.0, h_back=-50.0), kap=dict(kappa=1.0), off=dict(sd_u=0.0),
                   ctx=dict(d_ctx=0.0), gw=dict(g_NW=0.0), gen=dict(g_W0=0.0), rec=dict(r_W=0.0), fast=dict(alpha_F=0.0),
                   pers=dict(pi_L=0.0), hab=dict(nu_W=0.0), habr=dict(r_H=1.0))
    for o in opts:
        if (o == "fa" and ("ctx" in opts or "gw" in opts)) or o in ("hom", "stim"):
            continue
        out.append((base + "".join(f"+{q}" for q in OPTS if q in opts and q != o), neutral[o]))
    return out


def arrays(d):
    tt = d.trial_type.map({"auditory": 0, "whisker": 1, "no_stim": 2}).to_numpy(np.int64)
    lick = d.lick_flag.to_numpy(np.int64).copy()
    rplus = (d.reward_group == "R+").to_numpy(np.int64)
    ms = (d.mouse_id != d.mouse_id.shift()).to_numpy(np.int64)
    ss = (d.session_id != d.session_id.shift()).to_numpy(np.int64)
    return tt, lick, rplus, ss, ms
