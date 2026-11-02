"""VENDORED in ibl-ai-agent/projects/ssl-paper-figures (2026-11-02) from haas ~/code/behaviour_analysis, where it was
added on 2026-10-01 but never committed to github.com/abisi/behaviour_analysis. Move it into behaviour_analysis with one
commit, then import it from there. Uses learning_utils.identify_learning_trial_* from the canonical repo (rules checked
identical to the old copy).

Exact learning curves (grid HMM) -- port from ibl-ai-agent/projects/ssl-learning-trial-identification (2026-10-01).

NEW FILE: nothing in the existing behaviour_analysis code is changed. This is an alternative to the PyMC fit in
`beh_plotting_functions.plot_single_mouse_session_learning_curve` (same model, exact inference, no MCMC noise):

    x_t = x_{t-1} + N(0, sigma^2),   p_t = invlogit(x_t),   y_t ~ Bernoulli(p_t)

The latent x is discretised on X_GRID (181 points, -9..9 in logit) and forward-backward recursions give the exact
smoothed posterior of p_t given all trials for a FIXED sigma (default 1.0 -- the value chosen for the project; the
stored PyMC model has tau ~ Gamma(10, 10), i.e. sigma ~ 1). Why: the stored fits use pm.sample(10, tune=10) -- 40
unconverged samples (R-hat 1.4-2.0), and the FA curve was interpolated on an even grid instead of at the real no-stim
trial times (off by up to 0.29). Here:
  - whisker, no-stim (and auditory) curves are fitted separately on their own trials;
  - the FA (no-stim) posterior is interpolated IN TIME onto the whisker trials (mixture of the neighbouring trials'
    marginals, linear in start time) -> p_chance;
  - bands = posterior quantiles (conf_int, default 80%);
  - learning trial: YOUR rule (learning_utils.identify_learning_trial_rewarded / _nonrewarded, params as in
    beh_plotting_functions) applied to the exact curves -- the "L1 stored rule, exact" definition of the project.

Main entry points:
  fit_exact(outcomes, sigma)                         -> marginals (n x G)
  summarize(marg, conf_int)                          -> p_mean, p_low, p_high
  interp_marginals(marg, t_src, t_dst)               -> marginals at other trial times
  session_learning_curves_exact(session_table, ...)  -> dict per trial type + learning trial (one session)
  plot_single_mouse_session_learning_curve_exact(combine_bhv_data, color_palette, saving_path, params)
      drop-in sibling of plot_single_mouse_session_learning_curve: every whisker session (all days, not only day 0),
      saves <mouse>_<behavior>_<day>_<trial_type>_learning_curve_exact.h5 (same columns as the stored H5, p_samples=None,
      + method/sigma) and a figure <mouse>_<behavior>_<day>_learning_curve_exact.{pdf,png,svg} in
      <saving_path>/<behavior>_<day>/learning_curve/.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

X_GRID = np.linspace(-9.0, 9.0, 181)
P_GRID = 1.0 / (1.0 + np.exp(-X_GRID))
DEFAULT_SIGMA = 1.0


def _transition(sigma):
    d = X_GRID[None, :] - X_GRID[:, None]
    T = np.exp(-0.5 * (d / sigma) ** 2)
    return T / T.sum(1, keepdims=True)


def forward_backward(y, sigma=DEFAULT_SIGMA):
    """Smoothed marginals gamma[t, g] (n x G) and log marginal likelihood. Flat initial state on the grid."""
    y = np.asarray(y).astype(int)
    n, G = len(y), len(X_GRID)
    T = _transition(sigma)
    lik = np.where(y[:, None] == 1, P_GRID[None, :], 1 - P_GRID[None, :])
    alpha = np.empty((n, G))
    c = np.empty(n)
    a = np.full(G, 1.0 / G) * lik[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * lik[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.ones(G)
    gam = np.empty((n, G))
    gam[-1] = alpha[-1]
    for t in range(n - 2, -1, -1):
        beta = T @ (lik[t + 1] * beta) / c[t + 1]
        g = alpha[t] * beta
        gam[t] = g / g.sum()
    return gam, float(np.log(c).sum())


def fit_exact(outcomes, sigma=DEFAULT_SIGMA):
    return forward_backward(outcomes, sigma)[0]


def quantiles(marg, qs):
    cdf = np.cumsum(marg, axis=1)
    return [P_GRID[np.clip((cdf < q).sum(1), 0, len(P_GRID) - 1)] for q in qs]


def summarize(marg, conf_int=80):
    lo, hi = quantiles(marg, [(100 - conf_int) / 200, 1 - (100 - conf_int) / 200])
    return marg @ P_GRID, lo, hi


def interp_marginals(marg, t_src, t_dst):
    """Linear-in-time mixture of the source trials' marginals at t_dst (edges held constant)."""
    t_src, t_dst = np.asarray(t_src, float), np.asarray(t_dst, float)
    idx = np.searchsorted(t_src, t_dst)
    out = np.empty((len(t_dst), marg.shape[1]))
    for k, (i, t) in enumerate(zip(idx, t_dst)):
        if i <= 0:
            out[k] = marg[0]
        elif i >= len(t_src):
            out[k] = marg[-1]
        else:
            f = (t - t_src[i - 1]) / (t_src[i] - t_src[i - 1])
            out[k] = (1 - f) * marg[i - 1] + f * marg[i]
    return out


def learning_trial(outcomes, p_mean, p_low, p_high, p_chance, reward_group):
    """Your rule (learning_utils) on the exact curves; params as hardcoded in beh_plotting_functions."""
    from learning_utils import identify_learning_trial_nonrewarded, identify_learning_trial_rewarded
    outcomes = np.asarray(outcomes)
    if not (outcomes == 1).any():
        return "non-learner", np.nan
    first_hit = int(np.argmax(outcomes == 1))
    above = np.where(p_low > p_chance)[0]
    below = np.where(p_high < p_chance)[0]
    above, below = above[above >= first_hit], below[below >= first_hit]
    if reward_group == 1:
        params = {"n_trials_for_expert": 20, "min_perf_for_expert": 0.8, "n_consec_trials_for_learner": 5}
        return identify_learning_trial_rewarded(above, p_mean, p_low, params)
    params = {"n_trials_for_expert": 20, "max_perf_for_expert": 0.2, "n_consec_trials_for_learner": 5}
    return identify_learning_trial_nonrewarded(below, above, p_mean, p_low, params, True)


def session_learning_curves_exact(session_table, sigma=DEFAULT_SIGMA, conf_int=80):
    """One session (already filtered to active, non-early-lick trials, time-ordered). Columns used: trial_type,
    lick_flag, start_time, reward_group. Whisker curve on whisker trials; no-stim and auditory curves on their own
    trials, interpolated in time onto the whisker trials."""
    st = session_table.sort_values("start_time").reset_index(drop=True)
    out = {}
    w = st[st.trial_type == "whisker_trial"]
    tw = w.start_time.to_numpy(float)
    for tt in ("whisker_trial", "no_stim_trial", "auditory_trial"):
        d = st[st.trial_type == tt]
        if len(d) < 3:
            continue
        y = d.lick_flag.to_numpy().astype(int)
        g = fit_exact(y, sigma)
        pm, lo, hi = summarize(g, conf_int)
        rec = dict(trial_type=tt, outcomes=y, start_time=d.start_time.to_numpy(float), p_mean=pm, p_low=lo, p_high=hi,
                   marginals=g)
        if tt != "whisker_trial" and len(tw):
            gi = interp_marginals(g, rec["start_time"], tw)
            rec["at_whisker"] = dict(zip(("p_mean", "p_low", "p_high"), summarize(gi, conf_int)))
        out[tt] = rec
    if "whisker_trial" in out and "no_stim_trial" in out:
        wc = out["whisker_trial"]
        p_chance = out["no_stim_trial"]["at_whisker"]["p_mean"]
        rg = st.reward_group.values[0]
        cat, lt = learning_trial(wc["outcomes"], wc["p_mean"], wc["p_low"], wc["p_high"], p_chance, rg)
        wc.update(p_chance=p_chance, learning_trial=lt, mouse_cat=cat)
    return out


def plot_single_mouse_session_learning_curve_exact(combine_bhv_data, color_palette, saving_path, params):
    """Sibling of plot_single_mouse_session_learning_curve: exact curves for every whisker session (all days).
    params: sigma (default 1.0), conf_int (default 80), days (optional list; default all)."""
    import matplotlib.pyplot as plt
    import behavior_analysis_utils as bhv_utils
    sigma = params.get("sigma", DEFAULT_SIGMA)
    conf_int = params.get("conf_int", 80)
    data = bhv_utils.keep_active_trials(combine_bhv_data)
    if params.get("days") is not None:
        data = data[data.day.isin(params["days"])]
    for session_id in data.session_id.unique():
        st = data[data.session_id == session_id].reset_index(drop=True)
        behavior, day, mouse_id = st.behavior.values[0], st.day.values[0], st.mouse_id.values[0]
        if behavior in ("free_licking", "fl", "auditory"):
            continue
        if "early_lick" in st:
            st = st[st.early_lick == 0]
        res = session_learning_curves_exact(st, sigma, conf_int)
        if "whisker_trial" not in res:
            continue
        rg = st.reward_group.values[0]
        out_dir = os.path.join(saving_path, f"{behavior}_{day}", "learning_curve")
        os.makedirs(out_dir, exist_ok=True)
        for tt, rec in res.items():
            row = dict(mouse_id=mouse_id, reward_group=rg, behavior=behavior, day=day, trial_type=tt,
                       outcomes=rec["outcomes"], p_samples=None, p_mean=rec["p_mean"], p_low=rec["p_low"],
                       p_high=rec["p_high"], ci_level=conf_int, p_chance=rec.get("p_chance"),
                       learning_trial=rec.get("learning_trial", np.nan), mouse_cat=rec.get("mouse_cat"),
                       method="exact_grid_hmm", sigma=sigma)
            pd.DataFrame([row]).to_hdf(os.path.join(out_dir, f"{mouse_id}_{behavior}_{day}_{tt}_learning_curve_exact.h5"),
                                       key="df", mode="w")
        wc = res["whisker_trial"]
        x = np.arange(len(wc["p_mean"]))
        col = color_palette["whisker_reward"] if rg == 1 else color_palette["whisker_no_reward"]
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
        for tt, c in (("auditory_trial", color_palette.get("auditory", "#1f5fbf")), ("no_stim_trial", "dimgrey")):
            if tt in res:
                a = res[tt]["at_whisker"]
                ax.fill_between(x, a["p_low"], a["p_high"], color=c, alpha=0.2, lw=0)
                ax.plot(x, a["p_mean"], color=c, lw=1.5)
        ax.fill_between(x, wc["p_low"], wc["p_high"], color=col, alpha=0.2, lw=0)
        ax.plot(x, wc["p_mean"], color=col, lw=1.9)
        o = wc["outcomes"]
        ax.vlines(x[o == 1], 1.04, 1.1, color=col, lw=0.8)
        ax.vlines(x[o == 0], -0.1, -0.04, color=col, lw=0.8)
        if np.isfinite(wc.get("learning_trial", np.nan)):
            ax.axvline(wc["learning_trial"], color="k", lw=0.8)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xlabel("whisker trial")
        ax.set_ylabel("P(lick)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("pdf", "png", "svg"):
            fig.savefig(os.path.join(out_dir, f"{mouse_id}_{behavior}_{day}_learning_curve_exact.{ext}"))
        plt.close(fig)
