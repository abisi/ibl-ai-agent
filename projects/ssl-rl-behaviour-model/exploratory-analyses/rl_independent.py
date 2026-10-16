"""Vendored from github.com/abisi/rl_model_auditory_whisker (Lila/CODE/independant_model.py, branch shared-pretraining, 2026-10-16).
Changes: output / target paths from common.py (combined_results_ks4/ssl-rl-behaviour-model), no directory creation at import.
"""
"""
indep_model.py — Modèle RL multiplicatif indépendant (standalone).
Contexte (C) apprenable : Q_C_exc et Q_C_inh inclus dans le grid search.
Nelder-Mead refinement : warm-start depuis le top-k du grid search.
"""

import numpy as np
import itertools
import time
import os
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import erf
from scipy.optimize import minimize
from tqdm import tqdm

import common as C

from rl_data_utils import common_x

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR     = C.FITS / 'grid_independent'
TARGETS_DIR = C.TARGETS / 'final_mean_withoutfree'

# ─────────────────────────────────────────────────────────────────────────────
# Stimuli — x = [A, W, NS, C]
# ─────────────────────────────────────────────────────────────────────────────

STIM = {
    'auditory': np.array([1, 0, 0, 1]),
    'whisker':  np.array([0, 1, 0, 1]),
    'nostim':   np.array([0, 0, 1, 1]),
}

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Params:
    b:        float = 0.003
    a:        float = 0.62
    v:        float = 6.0
    tr:       float = 0.2
    # Initial weights — needed so refit_capped.py can pass Q values via
    # Params(**{...}) consistently with SharedModel and MixedModel.
    Q_A_exc:  float = 0.2
    Q_A_inh:  float = 0.35
    Q_NS_exc: float = 0.05
    Q_NS_inh: float = 0.55
    Q_W_exc:  float = 0.002
    Q_W_inh:  float = 0.9
    Q_C_exc:  float = 0.5
    Q_C_inh:  float = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de simulation
# ─────────────────────────────────────────────────────────────────────────────

N_SEEDS = 1
N_PRE   = 1000
N_TRANS = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres de base — meilleurs fits par groupe
# ─────────────────────────────────────────────────────────────────────────────

BASE_PARAMS = {
    'R+': dict(
        b=0.0010, a=0.45, v=7.5, tr=0.4,
        Q_A_exc=0.2,   Q_A_inh=0.35,
        Q_NS_exc=0.05, Q_NS_inh=0.55,
        Q_W_exc=0.002, Q_W_inh=0.9,
        Q_C_exc=0.5,   Q_C_inh=0.5,
    ),
    'R-': dict(
        b=0.0012, a=0.40, v=6.5, tr=0.4,
        Q_A_exc=0.2,   Q_A_inh=0.4,
        Q_NS_exc=0.05, Q_NS_inh=0.55,
        Q_W_exc=0.002, Q_W_inh=1.0,
        Q_C_exc=0.5,   Q_C_inh=0.5,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Grilles de paramètres
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    'R+': dict(
        b=  [BASE_PARAMS['R+']['b']],
        a=  [BASE_PARAMS['R+']['a']],
        v=  [BASE_PARAMS['R+']['v']],
        tr= [BASE_PARAMS['R+']['tr']],
        Q_A_exc=  [0.1,  0.15,  0.22],
        Q_A_inh=  [0.25, 0.35,  0.45],
        Q_NS_exc= [0.03, 0.05,  0.07],
        Q_NS_inh= [0.45, 0.55,  0.65],
        Q_W_exc=  [0.003, 0.008, 0.01],
        Q_W_inh=  [0.6,  0.8,   1.0],
        Q_C_exc=  [0.3,  0.5,   0.7],
        Q_C_inh=  [0.3,  0.5,   0.7],
    ),
    'R-': dict(
        b=  [BASE_PARAMS['R-']['b']],
        a=  [BASE_PARAMS['R-']['a']],
        v=  [BASE_PARAMS['R-']['v']],
        tr= [BASE_PARAMS['R-']['tr']],
        Q_A_exc=  [0.15,  0.19,  0.23],
        Q_A_inh=  [0.3,   0.4,   0.5],
        Q_NS_exc= [0.03,  0.05,  0.07],
        Q_NS_inh= [0.40,  0.50,  0.6],
        Q_W_exc=  [0.0005, 0.0008, 0.001],
        Q_W_inh=  [0.9,   1.0,   1.1],
        Q_C_exc=  [0.3,   0.5,   0.7],
        Q_C_inh=  [0.3,   0.5,   0.7],
    ),
}

# Keys of initial weights to optimise (b, a, v, tr are fixed)
WEIGHT_KEYS_INDEP = [
    'Q_A_exc', 'Q_A_inh',
    'Q_NS_exc', 'Q_NS_inh',
    'Q_W_exc', 'Q_W_inh',
    'Q_C_exc', 'Q_C_inh',
]

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des targets
# ─────────────────────────────────────────────────────────────────────────────

def load_target(group):
    tag  = group.replace('+', 'plus').replace('-', 'minus')
    path = TARGETS_DIR / f'target_{tag}.npy'
    if not path.exists():
        raise FileNotFoundError(
            f'Target non trouvée : {path}\n'
            f'Lance la cellule "Save group mean curves" dans data_analysis_bis.ipynb.'
        )
    return np.load(path, allow_pickle=True).item()

# ─────────────────────────────────────────────────────────────────────────────
# Learning rate asymétrique
# ─────────────────────────────────────────────────────────────────────────────

def lrate(delta, v, stim, whisker_reward=True, tr=0.2):
    if stim == 'auditory':
        return v * delta if delta > 0 else delta
    elif stim == 'nostim':
        return v * delta if delta < 0 else delta
    elif stim == 'whisker':
        return (v * tr * delta if delta > 0 else delta) if whisker_reward \
               else (v * delta if delta < 0 else delta)

# ─────────────────────────────────────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────────────────────────────────────

class IndependentModel:

    def __init__(self, p=None):
        self.p     = p or Params()
        self.sqrt2 = np.sqrt(2)
        self._init_weights()   # reads Q values from self.p, consistent with SharedModel/MixedModel

    def _init_weights(self):
        """Initialise Q_exc / Q_inh depuis self.p.
        Appelé par __init__ — permet à refit_capped.py de passer les poids
        via Params sans appeler set_init_weights() explicitement.
        """
        p = self.p
        self.Q_exc = np.array([max(1e-9, p.Q_A_exc),
                               max(1e-9, p.Q_W_exc),
                               max(1e-9, p.Q_NS_exc),
                               max(1e-9, p.Q_C_exc)])
        self.Q_inh = np.array([max(1e-9, p.Q_A_inh),
                               max(1e-9, p.Q_W_inh),
                               max(1e-9, p.Q_NS_inh),
                               max(1e-9, p.Q_C_inh)])

    def set_init_weights(self, Q_A_exc, Q_A_inh,
                         Q_NS_exc, Q_NS_inh,
                         Q_W_exc,  Q_W_inh,
                         Q_C_exc=0.5, Q_C_inh=0.5):
        """Conservé pour compatibilité avec fit_individuals.py."""
        self.Q_exc = np.array([max(1e-9, Q_A_exc),
                               max(1e-9, Q_W_exc),
                               max(1e-9, Q_NS_exc),
                               max(1e-9, Q_C_exc)])
        self.Q_inh = np.array([max(1e-9, Q_A_inh),
                               max(1e-9, Q_W_inh),
                               max(1e-9, Q_NS_inh),
                               max(1e-9, Q_C_inh)])

    def init_phase2(self):
        pass

    def dQ(self, x):
        return float((self.Q_exc - self.Q_inh) @ x)

    def p_lick(self, x):
        return float(0.5 * (1 + erf(self.dQ(x) / self.sqrt2)))

    def _update(self, x, R, whisker_reward, stim):
        p  = self.p
        dq = self.dQ(x)
        pl = float(0.5 * (1 + erf(dq / self.sqrt2)))
        lr = lrate(R - p.a * dq, p.v, stim, whisker_reward, tr=p.tr)
        self.Q_exc = np.clip(self.Q_exc * (1 + p.b * lr * pl * x), 1e-9, None)
        self.Q_inh = np.clip(self.Q_inh * (1 - p.b * lr * pl * x), 1e-9, None)

    def run_session(self, trials):
        out = []
        for t in trials:
            x, R, wr, stim = t['x'], t['R_if_lick'], t['whisker_reward'], t['stim']
            plick = self.p_lick(x)
            self._update(x, R, wr, stim)
            out.append({**t, 'p_lick': plick,
                        'p_aud':   self.p_lick(STIM['auditory']),
                        'p_whisk': self.p_lick(STIM['whisker']),
                        'p_ns':    self.p_lick(STIM['nostim']),
                        'Q_exc':   self.Q_exc.copy(),
                        'Q_inh':   self.Q_inh.copy()})
        return out

# ─────────────────────────────────────────────────────────────────────────────
# Trials
# ─────────────────────────────────────────────────────────────────────────────

def make_pretrain_trials(n=N_PRE, rng=None):
    rng = rng or np.random
    trials = []
    for _ in range(n):
        if rng.rand() < 0.5:
            trials.append({'x': STIM['auditory'].copy(), 'R_if_lick':  1,
                           'stim': 'auditory', 'whisker_reward': True})
        else:
            trials.append({'x': STIM['nostim'].copy(),   'R_if_lick': -1,
                           'stim': 'nostim',   'whisker_reward': True})
    return trials

def make_transfer_trials(group='R+', n=N_TRANS, rng=None):
    rng = rng or np.random
    wr  = group == 'R+'
    R_w = 1 if wr else -1
    trials = []
    for _ in range(n):
        r = rng.rand()
        if r < 0.50:
            trials.append({'x': STIM['whisker'].copy(),  'R_if_lick': R_w,
                           'stim': 'whisker',  'whisker_reward': wr})
        elif r < 0.90:
            trials.append({'x': STIM['nostim'].copy(),   'R_if_lick': -1,
                           'stim': 'nostim',   'whisker_reward': wr})
        else:
            trials.append({'x': STIM['auditory'].copy(), 'R_if_lick':  1,
                           'stim': 'auditory', 'whisker_reward': wr})
    return trials

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(params, group):
    # Pass all params including Q weights via Params — consistent with refit_capped.py
    p = Params(**{k: params[k] for k in Params.__dataclass_fields__ if k in params})

    pre_acc   = {s: [] for s in ['auditory', 'nostim']}
    trans_acc = {s: [] for s in ['auditory', 'whisker', 'nostim']}
    key_map   = {'auditory': 'p_aud', 'whisker': 'p_whisk', 'nostim': 'p_ns'}

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        mod = IndependentModel(p=p)   # _init_weights reads Q values from p
        pre   = mod.run_session(make_pretrain_trials(N_PRE,  rng=rng))
        mod.init_phase2()
        trans = mod.run_session(make_transfer_trials(group, N_TRANS, rng=rng))

        for s, k in key_map.items():
            if s in pre_acc:
                pre_acc[s].append(np.array([r[k] for r in pre]))
            if s in trans_acc:
                trans_acc[s].append(np.array([r[k] for r in trans]))

    return (
        {s: np.nanmean(pre_acc[s],   axis=0) for s in pre_acc},
        {s: np.nanmean(trans_acc[s], axis=0) for s in trans_acc},
    )

# ─────────────────────────────────────────────────────────────────────────────
# MSE — publique, importable depuis d'autres scripts
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(sim_pre, sim_trans, target):
    def mse1(y_sim, y_data):
        valid = ~np.isnan(y_sim) & ~np.isnan(y_data)
        if valid.sum() == 0:
            return np.nan
        return float(np.mean((y_sim[valid] - y_data[valid]) ** 2))

    mse = {
        'mse_pre_aud':     mse1(sim_pre['auditory'],   target['auditory'][:N_PRE]),
        'mse_pre_ns':      mse1(sim_pre['nostim'],      target['no_stim'][:N_PRE]),
        'mse_trans_aud':   mse1(sim_trans['auditory'],  target['auditory'][N_PRE:N_PRE+N_TRANS]),
        'mse_trans_whisk': mse1(sim_trans['whisker'],   target['whisker'][N_PRE:N_PRE+N_TRANS]),
        'mse_trans_ns':    mse1(sim_trans['nostim'],    target['no_stim'][N_PRE:N_PRE+N_TRANS]),
    }
    vp = [v for v in [mse['mse_pre_aud'], mse['mse_pre_ns']] if not np.isnan(v)]
    vt = [v for v in [mse['mse_trans_aud'], mse['mse_trans_whisk'], mse['mse_trans_ns']] if not np.isnan(v)]
    mse['mse_pre']    = float(np.mean(vp))      if vp      else np.inf
    mse['mse_trans']  = float(np.mean(vt))      if vt      else np.inf
    mse['mse_global'] = float(np.mean(vp + vt)) if (vp+vt) else np.inf
    return mse

# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    combo, keys, target, group = args
    params = dict(zip(keys, combo))
    try:
        sp, st = _simulate(params, group)
        mse    = compute_mse(sp, st, target)
    except Exception as e:
        print(f'  ⚠ Worker error: {e} | params: {params}')
        mse = {k: np.inf for k in ['mse_pre', 'mse_trans', 'mse_global',
                                    'mse_pre_aud', 'mse_pre_ns',
                                    'mse_trans_aud', 'mse_trans_whisk', 'mse_trans_ns']}
    return {**params, **mse}

# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(target, group='R+', n_jobs=None):
    pg     = PARAM_GRIDS[group]
    keys   = list(pg.keys())
    combos = list(itertools.product(*pg.values()))
    n      = len(combos)
    cores  = n_jobs or os.cpu_count()
    print(f'\nGrid search — Independent | {group} | {n:,} combos | {cores} cores')
    for k, v in pg.items():
        print(f'  {k}: {v}')

    args = [(c, keys, target, group) for c in combos]
    t0   = time.time()
    with mp.Pool(processes=cores) as pool:
        records = list(tqdm(pool.imap_unordered(_worker, args),
                            total=n, desc=f'Independent|{group}'))

    df = pd.DataFrame(records).sort_values('mse_global').reset_index(drop=True)
    print(f'  Done in {(time.time()-t0)/60:.1f} min | '
          f'best mse_global={df.mse_global.iloc[0]:.5f}')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Nelder-Mead refinement — warm-start depuis top-k du grid search
# Optimise uniquement les poids initiaux (WEIGHT_KEYS_INDEP).
# b, a, v, tr sont fixés (lus depuis la ligne du grid df).
# Bounds : [0, 10] via pénalité (Nelder-Mead ne supporte pas les bornes natives).
# adaptive=True : coefficients dépendants de la dimension (Gao & Han 2012),
#   utile ici car les poids couvrent des échelles très différentes (1e-3 à ~1).
# ─────────────────────────────────────────────────────────────────────────────

def refine_fit_nelder(
    grid_df,
    target,
    group,
    weight_keys=None,
    top_k=5,
    lo=0.0,
    hi=10.0,
    maxiter=2000,
    tol=1e-3,
):
    """
    Raffine le meilleur fit du grid search par Nelder-Mead.

    Paramètres
    ----------
    grid_df    : DataFrame sorti par grid_search()
    target     : dict de courbes cibles
    group      : 'R+' ou 'R-'
    weight_keys: liste des clés de poids initiaux à optimiser
                 (défaut : WEIGHT_KEYS_INDEP)
    top_k      : nombre de points de départ (top-k du grid)
    lo, hi     : bornes [0, 10] appliquées via pénalité
    maxiter    : iterations max par run Nelder-Mead
    tol        : xatol et fatol (tolérance lâche adaptée au paysage bruité)

    Retourne
    --------
    dict des paramètres optimisés + mse_global
    """
    if weight_keys is None:
        weight_keys = WEIGHT_KEYS_INDEP

    # Paramètres fixes (b, a, v, tr) lus depuis le meilleur point du grid
    fixed_keys  = [k for k in grid_df.columns if k not in weight_keys
                   and k not in ['mse_pre', 'mse_trans', 'mse_global',
                                 'mse_pre_aud', 'mse_pre_ns',
                                 'mse_trans_aud', 'mse_trans_whisk', 'mse_trans_ns']]
    fixed_params = grid_df.iloc[0][fixed_keys].to_dict()

    def objective(x):
        # Pénalité pour hors-bornes
        penalty = np.sum(np.maximum(lo - x, 0) + np.maximum(x - hi, 0))
        if penalty > 0:
            return 1e6 + penalty * 1e3
        params = {**fixed_params, **dict(zip(weight_keys, x))}
        try:
            sp, st = _simulate(params, group)
            return compute_mse(sp, st, target)['mse_global']
        except Exception:
            return 1e6

    top_k_rows  = grid_df.nsmallest(top_k, 'mse_global')
    best_result = None
    best_mse    = np.inf

    print(f'\nNelder-Mead refinement — Independent | {group} | top_k={top_k}')
    for rank, (_, row) in enumerate(top_k_rows.iterrows()):
        x0  = np.array([row[k] for k in weight_keys])
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'xatol':    tol,
                'fatol':    tol,
                'maxiter':  maxiter,
                'adaptive': True,   # coefficients Gao & Han — robuste aux échelles mixtes
                'disp':     False,
            },
        )
        print(f'  start #{rank+1} | grid mse={row["mse_global"]:.5f} '
              f'→ NM mse={res.fun:.5f} | nfev={res.nfev}')
        if res.fun < best_mse:
            best_mse    = res.fun
            best_result = res

    best_params = {**fixed_params,
                   **dict(zip(weight_keys, best_result.x)),
                   'mse_global': best_mse}
    print(f'  Best after refinement : mse_global={best_mse:.5f}')
    print(f'  Grid best was         : mse_global={grid_df.mse_global.iloc[0]:.5f}')
    return best_params

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — best fit
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_fit(df, target, group, top_n=5):
    cx_pre   = np.arange(N_PRE)   - N_PRE
    cx_trans = np.arange(N_TRANS)
    cx       = np.concatenate([cx_pre, cx_trans])

    panels = [('auditory', 'auditory', 'Auditory'),
              ('whisker',  'whisker',  'Whisker'),
              ('nostim',   'no_stim',  'No stim (FA)')]
    colors = [cm.Blues(0.4 + 0.6 * (top_n - i) / top_n) for i in range(top_n)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for col, (sim_key, data_key, title) in enumerate(panels):
        ax = axes[col]
        if sim_key == 'whisker':
            ax.plot(cx_trans, target[data_key][N_PRE:], color='black', lw=2.5,
                    zorder=20, label='Data')
        else:
            ax.plot(cx, target[data_key], color='black', lw=2.5,
                    zorder=20, label='Data')

        for rank, (_, row) in enumerate(df.head(top_n).iterrows()):
            try:
                sp, st = _simulate(row.to_dict(), group)
            except Exception as e:
                print(f'  rank {rank+1}: {e}')
                continue
            c     = colors[rank]
            label = f'#{rank+1} mse_g={row["mse_global"]:.4f}' if col == 1 else None
            if sim_key in sp and sim_key != 'whisker':
                ax.plot(cx_pre,   sp[sim_key],  color=c, lw=1.8, alpha=0.9)
            if sim_key in st:
                ax.plot(cx_trans, st[sim_key], color=c, lw=1.8, alpha=0.9,
                        label=label)

        ax.axvline(0, color='red', lw=1.2, ls='--', alpha=0.6)
        ax.axhline(0.5, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Trials (rel. to transfer onset)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('P(lick)', fontsize=11)
    axes[1].legend(fontsize=7, frameon=False, loc='upper left')
    best = df.iloc[0]
    axes[2].text(0.98, 0.02,
                 f'Best:\nmse_pre={best["mse_pre"]:.4f}\n'
                 f'mse_trans={best["mse_trans"]:.4f}\n'
                 f'mse_global={best["mse_global"]:.4f}',
                 transform=axes[2].transAxes, fontsize=8, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    fig.suptitle(f'Independent | Group {group} | Top {top_n} fits',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    tag  = group.replace('+', 'plus').replace('-', 'minus')
    path = OUT_DIR / f'best_fit_Independent_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved → {path}')
    plt.show()


def plot_refined_fit(best_params, target, group):
    """Plot the Nelder-Mead refined fit against data."""
    cx_pre   = np.arange(N_PRE)   - N_PRE
    cx_trans = np.arange(N_TRANS)
    cx       = np.concatenate([cx_pre, cx_trans])

    panels = [('auditory', 'auditory', 'Auditory'),
              ('whisker',  'whisker',  'Whisker'),
              ('nostim',   'no_stim',  'No stim (FA)')]

    sp, st = _simulate(best_params, group)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for col, (sim_key, data_key, title) in enumerate(panels):
        ax = axes[col]
        if sim_key == 'whisker':
            ax.plot(cx_trans, target[data_key][N_PRE:], color='black', lw=2.5,
                    zorder=20, label='Data')
        else:
            ax.plot(cx, target[data_key], color='black', lw=2.5,
                    zorder=20, label='Data')

        if sim_key in sp and sim_key != 'whisker':
            ax.plot(cx_pre,   sp[sim_key], color='steelblue', lw=2, alpha=0.9)
        if sim_key in st:
            ax.plot(cx_trans, st[sim_key], color='steelblue', lw=2, alpha=0.9,
                    label=f'NM fit  mse={best_params["mse_global"]:.4f}')

        ax.axvline(0, color='red', lw=1.2, ls='--', alpha=0.6)
        ax.axhline(0.5, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Trials (rel. to transfer onset)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('P(lick)', fontsize=11)
    axes[1].legend(fontsize=7, frameon=False, loc='upper left')
    fig.suptitle(f'Independent | Group {group} | Nelder-Mead refined fit',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    tag  = group.replace('+', 'plus').replace('-', 'minus')
    path = OUT_DIR / f'refined_fit_Independent_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved → {path}')
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f'Output folder : {OUT_DIR.resolve()}')

    for group in ['R+', 'R-']:
        target = load_target(group)

        # 1. Grid search
        df  = grid_search(target, group=group)
        tag = group.replace('+', 'plus').replace('-', 'minus')
        df.to_csv(OUT_DIR / f'grid_Independent_{tag}.csv', index=False)
        print(f'CSV → {OUT_DIR / f"grid_Independent_{tag}.csv"}')
        plot_best_fit(df, target, group)

        # 2. Nelder-Mead refinement warm-started from grid top-k
        best_params = refine_fit_nelder(df, target, group, top_k=5)
        pd.DataFrame([best_params]).to_csv(
            OUT_DIR / f'refined_Independent_{tag}.csv', index=False)
        print(f'Refined CSV → {OUT_DIR / f"refined_Independent_{tag}.csv"}')
        plot_refined_fit(best_params, target, group)