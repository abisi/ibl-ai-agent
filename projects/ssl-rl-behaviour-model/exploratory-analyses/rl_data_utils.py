"""Vendored from github.com/abisi/rl_model_auditory_whisker (Lila/data_utils.py: free licking removed; trim with
cumcount for recent pandas), 2026-10-16."""
"""
data_utils.py — chargement, smoothing, interpolation des courbes comportementales.
108 souris, 243 637 trials. Auditory → whisker transfer.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

BIN_SIZE  = 100
N_POINTS  = 1000   # valeur par défaut — peut être surchargée via paramètre n_points

TRIAL_TYPES = {
    'auditory': {'color': 'royalblue',   'label': 'Auditory'},
    'whisker':  {'color': 'forestgreen', 'label': 'Whisker'},
    'no_stim':  {'color': 'black',       'label': 'False alarm'},
}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path='trials_data_final_filtered.csv'):
    d = pd.read_csv(path)

    # Retirer les free_licking
    if 'bhv_type' in d.columns:
        before = len(d)
        d = d[d['bhv_type'] != 'free_licking'].reset_index(drop=True)
        print(f'Free_licking retirés : {before - len(d)}')

    # Retirer les trials whisker avant le transfer
    d = d[~((d['trial_type'] == 'whisker') & (d['day'] < 0))].reset_index(drop=True)

    groups = (
        d[['mouse_id', 'reward_group']]
        .drop_duplicates()
        .set_index('mouse_id')['reward_group']
        .to_dict()
    )
    print(f'Souris: {d.mouse_id.nunique()} | Trials: {len(d)} | '
          f'Jours: {sorted(d.day.unique())}')
    return d, groups


# ─────────────────────────────────────────────────────────────────────────────
# Filtrage intra-session
# ─────────────────────────────────────────────────────────────────────────────

def filter_pretrial_whisker_per_session(mouse_data):
    keep_masks = []
    for day, session in mouse_data.groupby('day', sort=True):
        whisker_trials = session[session['trial_type'] == 'whisker']
        if len(whisker_trials) == 0:
            keep_masks.append(session)
        else:
            first_whisker_idx = whisker_trials['trial_idx'].min()
            keep_masks.append(session[session['trial_idx'] >= first_whisker_idx])
    return pd.concat(keep_masks).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window
# ─────────────────────────────────────────────────────────────────────────────

def smooth_lick_sliding(cum_trials, lick_flags, all_cum_trials, step=1,
                        causal=True, min_trials=10):
    """
    causal=True  → fenêtre [c - BIN_SIZE, c]
    causal=False → fenêtre symétrique [c - BIN_SIZE//2, c + BIN_SIZE//2]
    min_trials   → nombre minimum de trials dans la fenêtre, sinon NaN (default=10)
    """
    centers = all_cum_trials[::step]
    vals    = []
    for c in centers:
        if causal:
            mask = (cum_trials >= c - BIN_SIZE) & (cum_trials <= c)
        else:
            half = BIN_SIZE // 2
            mask = (cum_trials >= c - half) & (cum_trials <= c + half)
        vals.append(lick_flags[mask].mean() if mask.sum() >= min_trials else np.nan)
    return centers, np.array(vals)

# ─────────────────────────────────────────────────────────────────────────────
# Interpolation par souris
# ─────────────────────────────────────────────────────────────────────────────

def _stretch(t_c, sm, mask, n):
    if mask.sum() < 5:
        return np.full(n, np.nan)
    f    = interp1d(t_c[mask], sm[mask], bounds_error=False, fill_value=np.nan)
    grid = np.linspace(t_c[mask].min(), t_c[mask].max(), n)
    return f(grid)


def get_mouse_curves(mouse_data, trim_last=10, n_points=N_POINTS):
    """
    Retourne un dict {trial_type: array(2*n_points,)} pour une souris.
    L'axe X est normalisé : -n_points → 0 (pretraining) / 0 → +n_points (transfer).

    :param n_points: nombre de points interpolés de chaque côté du whisker onset.
    """
    mouse_data = mouse_data.sort_values(['day', 'trial_idx']).reset_index(drop=True)
    mouse_data = filter_pretrial_whisker_per_session(mouse_data)

    if trim_last > 0:
        # last trim_last trials of each day dropped (cumcount instead of groupby.apply, which drops the
        # grouping column in recent pandas)
        mouse_data = mouse_data[mouse_data.groupby('day').cumcount(ascending=False) >= trim_last].reset_index(drop=True)

    mouse_data['cum_trial'] = np.arange(len(mouse_data))
    all_cum = mouse_data['cum_trial'].values

    # ── Étape 1 : smooth dans l'espace cumulatif brut (avant centrage) ──
    # La fenêtre [c-BIN_SIZE//2, c+BIN_SIZE//2] ne peut pas traverser
    # le whisker onset car le référentiel n'existe pas encore ici.
    smoothed = {}
    for tt in TRIAL_TYPES:
        sub = mouse_data[mouse_data['trial_type'] == tt]
        if len(sub) < 10:
            smoothed[tt] = (None, None)
            continue
        t_c, sm = smooth_lick_sliding(sub['cum_trial'].values,
                                      sub['lick_flag'].values,
                                      all_cum, step=5)
        smoothed[tt] = (t_c, sm)

    # ── Étape 2 : centrage sur day 0 (whisker onset) ──
    day0 = mouse_data[mouse_data['day'] == 0]['cum_trial']
    if len(day0) == 0:
        return None
    ref = day0.min()

    # ── Étape 3 : interpolation pré/trans séparément ──
    curves = {}
    for tt in TRIAL_TYPES:
        t_c, sm = smoothed[tt]
        if t_c is None:
            curves[tt] = np.full(2 * n_points, np.nan)
            continue

        # Centrer l'axe
        t_c_rel = t_c - ref
        valid   = ~np.isnan(sm)

        if valid.sum() < 5:
            curves[tt] = np.full(2 * n_points, np.nan)
            continue

        pre_m  = valid & (t_c_rel < 0)
        post_m = valid & (t_c_rel >= 0)

        pre_interp  = _stretch(t_c_rel, sm, pre_m,  n_points)
        post_interp = _stretch(t_c_rel, sm, post_m, n_points)
        curves[tt]  = np.concatenate([pre_interp, post_interp])

    return curves


def get_all_interpolated_curves(d, groups, group_name, trim_last=10, n_points=N_POINTS):
    """
    Retourne { mouse_id: { trial_type: array(2*n_points,) } }
    pour toutes les souris du groupe.
    """
    mice = sorted([m for m, g in groups.items() if g == group_name])
    all_curves = {}
    for mid in mice:
        c = get_mouse_curves(d[d['mouse_id'] == mid].copy(),
                             trim_last=trim_last, n_points=n_points)
        if c is not None:
            all_curves[mid] = c
    return all_curves


# ─────────────────────────────────────────────────────────────────────────────
# Moyenne de groupe
# ─────────────────────────────────────────────────────────────────────────────

def get_mean_curves(d, groups, group_name, trim_last=10, n_points=N_POINTS):
    """
    Retourne { trial_type: array(2*n_points,) } — moyenne sur toutes les souris du groupe.
    """
    all_curves = get_all_interpolated_curves(d, groups, group_name,
                                             trim_last=trim_last, n_points=n_points)
    mean_curves = {}
    for tt in TRIAL_TYPES:
        stack = np.array([all_curves[m][tt] for m in all_curves])
        mean_curves[tt] = np.nanmean(stack, axis=0)
    return mean_curves


def common_x(n_points=N_POINTS):
    return np.linspace(-n_points, n_points, 2 * n_points)