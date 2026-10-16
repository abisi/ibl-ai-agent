"""Vendored from github.com/abisi/rl_model_auditory_whisker (Lila/CODE/mixed_model.py, branch shared-pretraining, 2026-10-16).
Changes: output / target paths from common.py (combined_results_ks4/ssl-rl-behaviour-model), no directory creation at import.
"""
"""
mixed_model.py — Modèle RL multiplicatif Mixed (standalone).
Grid search : conditions initiales Q_*  —  b, a, v, tr fixés au meilleur fit.
Contexte (C) apprenable : Q_C_exc et Q_C_inh inclus dans le grid search.
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
from tqdm import tqdm

import common as C

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR     = C.FITS / 'grid_mixed'
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
    b:         float = 0.001
    a:         float = 0.6
    v:         float = 6.0
    tr:        float = 0.4
    Q_A_exc:   float = 0.2
    Q_A_inh:   float = 0.2
    Q_NS_exc:  float = 0.05
    Q_NS_inh:  float = 0.6
    Q_W_exc:   float = 0.25
    Q_W_inh:   float = 0.7
    Q_C_exc:   float = 0.5   # ← nouveau
    Q_C_inh:   float = 0.5   # ← nouveau
    Q_in_A:    float = 0.5
    Q_in_W:    float = 0.01
    Q_exc_out: float = 0.2
    Q_inh_out: float = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de simulation
# ─────────────────────────────────────────────────────────────────────────────

N_SEEDS = 1
N_PRE   = 1000
N_TRANS = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Grilles de paramètres
# b, a, v, tr fixés — sweep Q_* (3^12 = 531 441 combos)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    'R+': dict(
        # fixed
        b=  [0.0010],
        a=  [0.6   ],
        v=  [6.0   ],
        tr= [0.4   ],
        # swept — voie directe A, NS, W, C
        Q_A_exc=   [0.05, 0.1,  0.2  ],
        Q_A_inh=   [0.1,  0.2,  0.35 ],
        Q_NS_exc=  [0.01, 0.05, 0.1  ],
        Q_NS_inh=  [0.4,  0.6,  0.8  ],
        Q_W_exc=   [0.1,  0.2,  0.35 ],
        Q_W_inh=   [0.5,  0.7,  0.9  ],
        Q_C_exc=   [0.3,  0.5,  0.7  ],  # ← nouveau
        Q_C_inh=   [0.3,  0.5,  0.7  ],  # ← nouveau
        # swept — nœud partagé
        Q_in_A=    [0.3,  0.5,  0.7  ],
        Q_in_W=    [0.001, 0.01, 0.05 ],
        Q_exc_out= [0.1,  0.2,  0.35 ],
        Q_inh_out= [0.3,  0.5,  0.7  ],
    ),
    'R-': dict(
        # fixed
        b=  [0.0015],
        a=  [0.5   ],
        v=  [5.0   ],
        tr= [0.3   ],
        # swept — voie directe A, NS, W, C
        Q_A_exc=   [0.1,  0.2,  0.3  ],
        Q_A_inh=   [0.1,  0.2,  0.4  ],
        Q_NS_exc=  [0.001, 0.005, 0.01],
        Q_NS_inh=  [0.1,  0.2,  0.4  ],
        Q_W_exc=   [0.02, 0.05, 0.1  ],
        Q_W_inh=   [0.5,  0.7,  0.9  ],
        Q_C_exc=   [0.3,  0.5,  0.7  ],  # ← nouveau
        Q_C_inh=   [0.3,  0.5,  0.7  ],  # ← nouveau
        # swept — nœud partagé
        Q_in_A=    [0.3,  0.5,  0.8  ],
        Q_in_W=    [0.05, 0.1,  0.2  ],
        Q_exc_out= [0.1,  0.2,  0.35 ],
        Q_inh_out= [0.3,  0.5,  0.7  ],
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des targets
# ─────────────────────────────────────────────────────────────────────────────

def load_target(group):
    tag  = group.replace('+', 'plus').replace('-', 'minus')
    path = TARGETS_DIR / f'target_{tag}.npy'
    if not path.exists():
        raise FileNotFoundError(f'Target non trouvée : {path}')
    return np.load(path, allow_pickle=True).item()

# ─────────────────────────────────────────────────────────────────────────────
# Learning rate asymétrique
# ─────────────────────────────────────────────────────────────────────────────

def lrate(delta, v, stim, whisker_reward=True, tr=0.4):
    if stim == 'auditory':
        return v * delta if delta > 0 else delta
    elif stim == 'nostim':
        return v * delta if delta < 0 else delta
    elif stim == 'whisker':
        return (v * tr * delta if delta > 0 else delta) if whisker_reward \
               else (v * delta if delta < 0 else delta)

# ─────────────────────────────────────────────────────────────────────────────
# Modèle Mixed
# Architecture : voie directe [A, W, NS, C] + nœud partagé A/W
#
#   dQ(x) = (Q_exc - Q_inh) @ [A, W, NS, C]
#           + (Q_exc_out - Q_inh_out) * (Q_in @ [A, W])
#
# Q_C_exc / Q_C_inh font partie de la voie directe et sont apprenables.
# ─────────────────────────────────────────────────────────────────────────────

class MixedModel:
    def __init__(self, p=None):
        self.p     = p or Params()
        self.sqrt2 = np.sqrt(2)
        self._init_weights()

    def _init_weights(self):
        p = self.p
        # Voie directe : [A, W, NS, C]
        self.Q_exc = np.array([max(1e-9, p.Q_A_exc),
                               max(1e-9, p.Q_W_exc),
                               max(1e-9, p.Q_NS_exc),
                               max(1e-9, p.Q_C_exc)])  # ← Q_C apprenable
        self.Q_inh = np.array([max(1e-9, p.Q_A_inh),
                               max(1e-9, p.Q_W_inh),
                               max(1e-9, p.Q_NS_inh),
                               max(1e-9, p.Q_C_inh)])  # ← Q_C apprenable
        # Nœud partagé A/W
        self.Q_in      = np.array([max(1e-9, p.Q_in_A), max(1e-9, p.Q_in_W)])
        self.Q_exc_out = max(1e-9, p.Q_exc_out)
        self.Q_inh_out = max(1e-9, p.Q_inh_out)

    @staticmethod
    def _x_node(x):
        """Composantes nœud partagé : [A, W]"""
        return np.array([x[0], x[1]])

    def dQ(self, x):
        direct = float((self.Q_exc - self.Q_inh) @ x)
        S      = float(self.Q_in @ self._x_node(x))
        node   = (self.Q_exc_out - self.Q_inh_out) * S
        return direct + node

    def p_lick(self, x):
        return float(0.5 * (1 + erf(self.dQ(x) / self.sqrt2)))

    def _update_direct(self, x, lr, pl):
        p = self.p
        # x[3]=C=1 à chaque trial → Q_C mis à jour à chaque trial
        self.Q_exc = np.clip(self.Q_exc * (1 + p.b * lr * pl * x), 1e-9, None)
        self.Q_inh = np.clip(self.Q_inh * (1 - p.b * lr * pl * x), 1e-9, None)

    def _update_node(self, x, lr, pl):
        p   = self.p
        x_n = self._x_node(x)
        S   = float(self.Q_in @ x_n)
        self.Q_exc_out = max(1e-9, self.Q_exc_out * (1 + p.b * lr * pl * S))
        self.Q_inh_out = max(1e-9, self.Q_inh_out * (1 - p.b * lr * pl * S))
        self.Q_in      = np.clip(self.Q_in * (1 + p.b * lr * pl * x_n), 1e-9, None)

    def _update(self, x, R, whisker_reward, stim):
        p  = self.p
        dq = self.dQ(x)
        pl = float(0.5 * (1 + erf(dq / self.sqrt2)))
        lr = lrate(R - p.a * dq, p.v, stim, whisker_reward, tr=p.tr)
        # Voie directe mise à jour à chaque trial (C=1 toujours actif)
        self._update_direct(x, lr, pl)
        # Nœud partagé mis à jour seulement sur trials A et W
        if stim in ('auditory', 'whisker'):
            self._update_node(x, lr, pl)

    def run_session(self, trials):
        out = []
        for t in trials:
            x, R, wr, stim = t['x'], t['R_if_lick'], t['whisker_reward'], t['stim']
            plick = self.p_lick(x)
            self._update(x, R, wr, stim)
            out.append({**t,
                        'p_lick':  plick,
                        'p_aud':   self.p_lick(STIM['auditory']),
                        'p_whisk': self.p_lick(STIM['whisker']),
                        'p_ns':    self.p_lick(STIM['nostim'])})
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
    p = Params(**{k: params[k] for k in Params.__dataclass_fields__ if k in params})

    pre_acc   = {s: [] for s in ['auditory', 'nostim']}
    trans_acc = {s: [] for s in ['auditory', 'whisker', 'nostim']}
    key_map   = {'auditory': 'p_aud', 'whisker': 'p_whisk', 'nostim': 'p_ns'}

    for seed in range(N_SEEDS):
        rng   = np.random.RandomState(seed)
        m     = MixedModel(p=p)
        pre   = m.run_session(make_pretrain_trials(N_PRE,  rng=rng))
        trans = m.run_session(make_transfer_trials(group, N_TRANS, rng=rng))

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
# MSE
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(sim_pre, sim_trans, target):
    def mse1(y_sim, y_data):
        n     = min(len(y_sim), len(y_data))
        valid = ~np.isnan(y_sim[:n]) & ~np.isnan(y_data[:n])
        if valid.sum() == 0:
            return np.nan
        return float(np.mean((y_sim[:n][valid] - y_data[:n][valid]) ** 2))

    mse = {
        'mse_pre_aud':     mse1(sim_pre['auditory'],   target['auditory'][:N_PRE]),
        'mse_pre_ns':      mse1(sim_pre['nostim'],     target['no_stim'][:N_PRE]),
        'mse_trans_aud':   mse1(sim_trans['auditory'], target['auditory'][N_PRE:N_PRE+N_TRANS]),
        'mse_trans_whisk': mse1(sim_trans['whisker'],  target['whisker'][N_PRE:N_PRE+N_TRANS]),
        'mse_trans_ns':    mse1(sim_trans['nostim'],   target['no_stim'][N_PRE:N_PRE+N_TRANS]),
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
    print(f'\nGrid search — Mixed | {group} | {n:,} combos | {cores} cores')
    for k, v in pg.items():
        print(f'  {k}: {v}')

    args = [(c, keys, target, group) for c in combos]
    t0   = time.time()
    with mp.Pool(processes=cores) as pool:
        records = list(tqdm(pool.imap_unordered(_worker, args),
                            total=n, desc=f'Mixed|{group}'))

    df = pd.DataFrame(records).sort_values('mse_global').reset_index(drop=True)
    print(f'  Done in {(time.time()-t0)/60:.1f} min | '
          f'best mse_global={df.mse_global.iloc[0]:.5f}')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — best fit
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_fit(df, target, group, top_n=5):
    cx_pre   = np.arange(N_PRE)   - N_PRE
    cx_trans = np.arange(N_TRANS)

    panels = [('auditory', 'auditory', 'Auditory'),
              ('whisker',  'whisker',  'Whisker'),
              ('nostim',   'no_stim',  'No stim (FA)')]
    colors = [cm.Blues(0.4 + 0.6 * (top_n - i) / top_n) for i in range(top_n)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for col, (sim_key, data_key, title) in enumerate(panels):
        ax = axes[col]
        if sim_key == 'whisker':
            ax.plot(cx_trans, target[data_key][N_PRE:N_PRE+N_TRANS],
                    color='black', lw=2.5, zorder=20, label='Data')
        else:
            ax.plot(np.concatenate([cx_pre, cx_trans]),
                    target[data_key][:N_PRE+N_TRANS],
                    color='black', lw=2.5, zorder=20, label='Data')

        for rank, (_, row) in enumerate(df.head(top_n).iterrows()):
            try:
                sp, st = _simulate(row.to_dict(), group)
            except Exception as e:
                print(f'  rank {rank+1}: {e}')
                continue
            c     = colors[rank]
            label = f'#{rank+1} mse_g={row["mse_global"]:.4f}' if col == 1 else None
            if sim_key in sp and sim_key != 'whisker':
                ax.plot(cx_pre,   sp[sim_key], color=c, lw=1.8, alpha=0.9)
            if sim_key in st:
                ax.plot(cx_trans, st[sim_key], color=c, lw=1.8, alpha=0.9, label=label)

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
    fig.suptitle(f'Mixed | Group {group} | Top {top_n} fits', fontsize=12, fontweight='bold')
    plt.tight_layout()
    tag  = group.replace('+', 'plus').replace('-', 'minus')
    path = OUT_DIR / f'best_fit_Mixed_{tag}.png'
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
        print(f'\n  {group}:')
        for k, v in PARAM_GRIDS[group].items():
            print(f'    {k}: {v}')

    for group in ['R+', 'R-']:
        target = load_target(group)
        df     = grid_search(target, group=group)
        tag    = group.replace('+', 'plus').replace('-', 'minus')
        path   = OUT_DIR / f'grid_Mixed_{tag}.csv'
        df.to_csv(path, index=False)
        print(f'CSV → {path}')
        plot_best_fit(df, target, group)