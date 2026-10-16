"""
rl_joint.py — Fit with one shared parameter set for the auditory pretraining.

Why: R+ and R- mice are trained identically until day 0 (the group only defines what happens to whisker licks from day 0 on). The
previous fits gave each group its own fixed learning parameters (b, a, v, tr) and its own initial weights, so the simulated pretraining of
R+ and R- mice differed although the mice could not yet differ. Here:

  * pretraining parameters — the learning parameters (b, a, v) and all initial weights Q_* — are one set, common to R+ and R-;
  * pretraining is simulated ONCE (same trial sequence) and both groups start the transfer from that same end-of-pretraining state;
  * by default the transfer phase uses the same parameters too, so R+ and R- differ only through the whisker reward contingency
    (R_if_lick = +1 / -1 on whisker trials);
  * optionally (--free-transfer), named parameters are allowed to differ between groups in the transfer phase only (e.g. tr, or
    b a v tr), as a nested model to test whether a group-specific learning rule is needed.

Objective: mean over the two groups of each group's mse_global (the model module's compute_mse: pretraining auditory + no-stim curves,
transfer auditory + whisker + no-stim curves). The simulated pretraining curve is compared with each group's own pretraining target.

Modes
  mean        fit the group-mean targets (targets/<dir>/target_Rplus.npy, target_Rminus.npy) jointly
  individual  per mouse: learning parameters b, a, v, tr fixed to the joint mean fit (same values for every mouse, whatever its group),
              initial weights fitted per mouse (as fit_individuals_fast*.py, but without group-specific base parameters)

Entry points: 002_parameter_recovery.py (selftest), 003_joint_fit_mean.py (mean), 004_joint_fit_individual.py (individual).
Vendored from github.com/abisi/rl_model_auditory_whisker Lila/CODE/joint_fit.py (2026-10-16); outputs to common.FITS.
"""

import argparse
import copy
import importlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import common as C

MODELS = {
    'independent': ('rl_independent', 'IndependentModel'),
    'shared':      ('rl_shared',      'SharedModel'),
    'mixed':       ('rl_mixed',       'MixedModel'),
}
DYN_KEYS = ['b', 'a', 'v', 'tr']
GROUPS = ['R+', 'R-']
OUT = C.FITS
# starting points: the group-specific fixed values of the previous fits (PARAM_GRIDS / BASE_PARAMS), and their average
START_DYN = {
    'independent': [dict(b=0.0010, a=0.45, v=7.5, tr=0.4), dict(b=0.0012, a=0.40, v=6.5, tr=0.4)],
    'shared':      [dict(b=0.0010, a=0.70, v=6.0, tr=0.5), dict(b=0.0010, a=0.60, v=8.0, tr=0.5)],
    'mixed':       [dict(b=0.0010, a=0.60, v=6.0, tr=0.4), dict(b=0.0015, a=0.50, v=5.0, tr=0.3)],
}


def load_model(name):
    mod = importlib.import_module(MODELS[name][0])
    return mod, getattr(mod, MODELS[name][1])


def weight_keys(mod):
    return [k for k in mod.Params.__dataclass_fields__ if k.startswith('Q_')]


class Layout:
    """maps a log10 parameter vector to {common params} + {group: transfer overrides}"""

    def __init__(self, keys, free_transfer=()):
        self.common = list(keys)
        self.free = [k for k in free_transfer]
        self.names = self.common + [f'{k}|{g}' for g in GROUPS for k in self.free]

    def split(self, x):
        v = dict(zip(self.names, 10 ** np.asarray(x, float)))
        common = {k: v[k] for k in self.common}
        over = {g: {k: v[f'{k}|{g}'] for k in self.free} for g in GROUPS}
        return common, over

    def x0(self, common, over=None):
        over = over or {}
        return np.log10([common[k] for k in self.common] + [over.get(g, {}).get(k, common[k]) for g in GROUPS for k in self.free])


def simulate_joint(mod, Model, common, over, seed=0):
    """one pretraining (common parameters), then one transfer per group from the same end-of-pretraining state"""
    keymap = {'auditory': 'p_aud', 'whisker': 'p_whisk', 'nostim': 'p_ns'}
    rng = np.random.RandomState(seed)
    p = mod.Params(**{k: common[k] for k in mod.Params.__dataclass_fields__ if k in common})
    m = Model(p=p)
    pre = m.run_session(mod.make_pretrain_trials(mod.N_PRE, rng=rng))
    sim_pre = {s: np.array([r[keymap[s]] for r in pre]) for s in ('auditory', 'nostim')}
    out = {}
    for i, g in enumerate(GROUPS):
        mg = copy.deepcopy(m)
        for k, val in over.get(g, {}).items():
            setattr(mg.p, k, val)
        tr = mg.run_session(mod.make_transfer_trials(g, mod.N_TRANS, rng=np.random.RandomState(seed + 1 + i)))
        out[g] = (sim_pre, {s: np.array([r[keymap[s]] for r in tr]) for s in ('auditory', 'whisker', 'nostim')})
    return out


def joint_loss(x, mod, Model, layout, targets, fixed=None):
    try:
        common, over = layout.split(x)
        common = {**(fixed or {}), **common}
        sims = simulate_joint(mod, Model, common, over)
        vals = [mod.compute_mse(*sims[g], targets[g])['mse_global'] for g in GROUPS if g in targets]
        v = float(np.mean(vals))
        return v if np.isfinite(v) else 1e6
    except (FloatingPointError, ValueError, OverflowError):
        return 1e6


def fit(mod, Model, layout, targets, starts, fixed=None, maxiter=3000):
    best = None
    for x0 in starts:
        r = minimize(joint_loss, x0, args=(mod, Model, layout, targets, fixed), method='Nelder-Mead',
                     options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-7, adaptive=True))
        if best is None or r.fun < best.fun:
            best = r
    return best


def summarise(mod, Model, layout, x, targets, fixed=None):
    common, over = layout.split(x)
    common = {**(fixed or {}), **common}
    sims = simulate_joint(mod, Model, common, over)
    mse = {g: mod.compute_mse(*sims[g], targets[g]) for g in GROUPS if g in targets}
    return dict(common=common, transfer_overrides=over, mse=mse, loss=float(np.mean([m['mse_global'] for m in mse.values()])))


def load_mean_targets(d):
    return {g: np.load(Path(d) / f"target_{g.replace('+', 'plus').replace('-', 'minus')}.npy", allow_pickle=True).item() for g in GROUPS}


def mean_starts(mod, name, layout):
    base = {k: getattr(mod.Params(), k) for k in weight_keys(mod)}
    dyn = START_DYN[name] + [{k: float(np.mean([s[k] for s in START_DYN[name]])) for k in DYN_KEYS}]
    return [layout.x0({**base, **d}) for d in dyn]


def run_mean(a):
    mod, Model = load_model(a.model)
    targets = load_mean_targets(a.targets)
    layout = Layout(DYN_KEYS + weight_keys(mod), a.free_transfer)
    r = fit(mod, Model, layout, targets, mean_starts(mod, a.model, layout), maxiter=a.maxiter)
    res = summarise(mod, Model, layout, r.x, targets)
    res.update(model=a.model, free_transfer=list(a.free_transfer), converged=bool(r.success), n_eval=int(r.nfev), n_params=len(layout.names))
    tag = '_'.join(['003', a.model, 'mean'] + ([f'free-{"-".join(a.free_transfer)}'] if a.free_transfer else []))
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    plot_fit(mod, Model, res, targets, OUT / f'{tag}.png')
    print(json.dumps({k: res[k] for k in ('loss', 'common', 'transfer_overrides', 'converged')}, indent=1, default=float))


def _fit_mouse(args):
    name, mid, target, group, base, maxiter = args
    mod, Model = load_model(name)
    wk = weight_keys(mod)
    layout = Layout(wk)
    fixed = {k: base[k] for k in DYN_KEYS}
    tg = {group: target}
    starts = [layout.x0({k: getattr(mod.Params(), k) for k in wk}), layout.x0({k: base[k] for k in wk})]
    r = fit(mod, Model, layout, tg, starts, fixed=fixed, maxiter=maxiter)
    res = summarise(mod, Model, layout, r.x, tg, fixed=fixed)
    return dict(mouse_id=mid, group=group, converged=bool(r.success), mse_global=res['loss'],
                **{k: res['mse'][group][k] for k in res['mse'][group]}, **{k: res['common'][k] for k in wk}, **fixed)


def run_individual(a):
    import pandas as pd
    base = json.loads(Path(a.base).read_text())
    if base.get('transfer_overrides') and any(base['transfer_overrides'].values()):
        raise SystemExit('individual mode needs a base fit without group-specific transfer parameters (same learning rule for every mouse)')
    base = base['common']
    jobs = []
    for p in sorted(Path(a.targets).glob('target_*.npy')):
        t = np.load(p, allow_pickle=True).item()
        jobs.append((a.model, t['mouse_id'], t, t['group'], base, a.maxiter))
    with mp.get_context('spawn').Pool(a.n_jobs) as pool:
        rows = pool.map(_fit_mouse, jobs)
    df = pd.DataFrame(rows).sort_values(['group', 'mouse_id'])
    df.to_csv(OUT / f'004_{a.model}_individual_common_dynamics.csv', index=False)
    print(df.groupby('group')[['mse_global', 'converged']].mean())


def run_selftest(a):
    """parameter recovery: simulate targets from known common parameters, refit from the default starting points"""
    mod, Model = load_model(a.model)
    layout = Layout(DYN_KEYS + weight_keys(mod))
    truth = {**{k: getattr(mod.Params(), k) for k in weight_keys(mod)}, **START_DYN[a.model][0]}
    sims = simulate_joint(mod, Model, truth, {})
    targets = {g: {'auditory': np.r_[sims[g][0]['auditory'], sims[g][1]['auditory']], 'no_stim': np.r_[sims[g][0]['nostim'], sims[g][1]['nostim']],
                   'whisker': np.r_[np.full(mod.N_PRE, np.nan), sims[g][1]['whisker']]} for g in GROUPS}
    print('loss at truth:', joint_loss(layout.x0(truth), mod, Model, layout, targets))
    pre_rp, pre_rm = sims['R+'][0]['auditory'], sims['R-'][0]['auditory']
    print('pretraining identical across groups:', bool(np.allclose(pre_rp, pre_rm)))
    r = fit(mod, Model, layout, targets, mean_starts(mod, a.model, layout)[1:2], maxiter=a.maxiter)
    print('loss after fit:', r.fun, '| evals', r.nfev)
    fitted, _ = layout.split(r.x)
    for k in DYN_KEYS:
        print(f'  {k}: truth {truth[k]:.4g}  fitted {fitted[k]:.4g}')


def plot_fit(mod, Model, res, targets, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sims = simulate_joint(mod, Model, res['common'], res['transfer_overrides'])
    xp, xt = np.arange(mod.N_PRE) - mod.N_PRE, np.arange(mod.N_TRANS)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for i, g in enumerate(GROUPS):
        sp, st = sims[g]
        for j, (sk, tk, title) in enumerate((('auditory', 'auditory', 'Auditory'), ('whisker', 'whisker', 'Whisker'), ('nostim', 'no_stim', 'No stim (FA)'))):
            ax = axs[i, j]
            y = targets[g][tk][:mod.N_PRE + mod.N_TRANS]
            ax.plot(np.r_[xp, xt], y, color='k', lw=2, label='data')
            if sk in sp:
                ax.plot(xp, sp[sk], color='tab:blue', lw=1.5, label='model (common pretraining)')
            ax.plot(xt, st[sk], color='tab:green' if g == 'R+' else 'tab:purple', lw=1.5, label='model transfer')
            ax.axvline(0, color='r', ls='--', lw=1); ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{g} — {title}')
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{res['model']} | common pretraining | loss {res['loss']:.4f} | free transfer: {res['free_transfer'] or 'none'}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


