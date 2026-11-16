"""001 -- Publication-style figures of the evoked kinematics from the 000 cache (no recomputation).

Inputs: the 000 cache (cache/000v1_<session>.npz / _trials.parquet).

Tongue reach (side view): each first reach after the stimulus centred on the spout of its session (spout = median
position of the spout keypoint, likelihood >= 0.5; axes = head frame: forward, up). Distance tongue tip -> spout every
5 ms from the reach onset. Per-mouse means (thin), mean across mice step by step (dots coloured by time).
Whisker sweep (top view): skeleton = three tracked points joined in order, whisker_base -> top_particle (mid-shaft
particle) -> whisker_tip; a frame counts when base and particle likelihood >= 0.1, tip >= 0.5 and both segment
lengths are within 0.6-1.6 x their session medians (000). Each mouse is centred on its own whisker base at baseline
(-100 .. 0 ms) and drawn in the top-camera image orientation (nose pointing down the image, whisker to the left, as in
the video: forward = image down, lateral = image left).
Time courses: cohorts overlaid (R+ #00B400, R- #C800C8), mean +- SEM across mice.

Outputs (combined_results_ks4/ssl-evoked-kinematics/figures/, png 300 dpi + pdf):
  001_<stage>_examples.png       grid of example mice (3 R+, 3 R-): tongue reaches and whisker sweeps, W and A hits
  001_<stage>_tongue.png         2 x 2 spout-centred reaches + distance to spout vs time
  001_<stage>_whisker.png        2 x 2 whisker skeleton sequence (mouse-centred, top-view orientation)
  001_<stage>_timecourses.png    whisker / auditory hits, cohorts overlaid
Run (haas, after 000): python 001_trajectory_figures.py
"""
import importlib.util
import pathlib
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('k000', HERE / '000_evoked_kinematics.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'sans-serif'})
FIG = k.OUT / 'figures'
STEP_MS = k.REACH_GRID * 1000
SK_SHOW = (-0.05, 0.0, 0.05, 0.1, 0.2, 0.3)
TMAX_REACH = 80  # ms shown for the reach
CMAP = plt.cm.viridis
MOD_LABEL = {'whisker': 'whisker hit', 'auditory': 'auditory hit'}


def wrap(text, fig):
    """wrap a suptitle to the figure width (~ 16 characters per inch at 7 pt)"""
    return '\n'.join(textwrap.fill(t, int(fig.get_figwidth() * 16)) for t in text.split('\n'))


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=300)
    fig.savefig(FIG / f'{name}.pdf')
    plt.close(fig)


def img(p):
    """head frame (forward, lateral) -> top-camera image orientation (x = -lateral, y = forward; y axis inverted)."""
    return np.stack([-p[..., 1], p[..., 0]], -1)


# ------------------------------------------------------------------ data
def load_stage(stage):
    sess = k.sessions_and_cohorts()
    sess = sess[sess.stage == stage]
    sk_grid = None
    per = {}  # (cohort, mod) -> mouse -> dict(reach=[session arrays], skel=[...])
    for r in sess.itertuples():
        f = k.CACHE / f'000v1_{r.session}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        t = pd.read_parquet(k.CACHE / f'000v1_{r.session}_trials.parquet')
        for mod, tt in k.HIT.items():
            e = per.setdefault((r.cohort, mod), {}).setdefault(r.mouse, {'reach': [], 'skel': [], 'sessions': []})
            e['sessions'].append(r.session)
            key = f'reach|{tt}'
            if key in d.files and 'spout_head' in d.files and 'jaw_rest_x' in t:
                tr = t[t.trial_type == tt]
                jr = np.c_[tr.jaw_rest_x.to_numpy(), tr.jaw_rest_y.to_numpy()]
                if len(jr) == len(d[key]):
                    rs = d[key] + jr[:, None, :] - d['spout_head'][None, None, :]   # origin = spout
                    rs = rs[np.isfinite(rs[:, :, 0]).any(1)]
                    if len(rs):
                        e['reach'].append(rs)
            key = f'skel|{tt}'
            if key in d.files:
                sk = d[key]
                good = np.isfinite(sk[:, :, :, 0]).all(2).mean(1) >= 0.8
                if good.sum() >= 3:
                    e['skel'].append(sk[good])
        if 'sk_grid' in d.files:
            sk_grid = d['sk_grid']
    return per, sk_grid


def reach_mean(arrs, min_frac=0.5):
    a = np.concatenate(arrs)
    frac = np.isfinite(a[:, :, 0]).mean(0)
    with np.errstate(all='ignore'):
        m = np.nanmean(a, 0)
    m[frac < min_frac] = np.nan
    return m


def skel_center(sk, g2):
    """trials x time x 3 x 2 -> mean skeleton centred on its baseline whisker base."""
    with np.errstate(all='ignore'):
        m = np.nanmean(sk, 0)
    return m - np.nanmean(m[g2 < 0, 0, :], 0)


# ------------------------------------------------------------------ drawing helpers
def draw_reach_steps(ax, m, size=14, lw=0.8):
    ok = np.isfinite(m[:, 0]) & (STEP_MS <= TMAX_REACH)
    if not ok.any():
        return None
    ax.plot(m[ok, 0], m[ok, 1], '-', color='0.15', lw=lw, zorder=3)
    return ax.scatter(m[ok, 0], m[ok, 1], c=STEP_MS[ok], cmap=CMAP, vmin=0, vmax=TMAX_REACH, s=size,
                      edgecolor='k', lw=0.3, zorder=4)


def draw_spout(ax):
    ax.plot(0, 0, marker='D', color='orange', mec='k', mew=0.6, ms=6, zorder=5)
    ax.annotate('spout', (0, 0), xytext=(5, 4), textcoords='offset points', fontsize=6)


def draw_skeleton(ax, m, g2, lw=2.0, ms=4.5, alpha=1.0):
    """skeletons at SK_SHOW times: base (square) - particle (circle) - tip (triangle)."""
    p = img(m)
    for ci, t in enumerate(SK_SHOW):
        i = int(np.argmin(np.abs(g2 - t)))
        col = CMAP(ci / (len(SK_SHOW) - 1))
        ax.plot(p[i, :, 0], p[i, :, 1], '-', color=col, lw=lw, alpha=alpha, solid_capstyle='round', zorder=3)
        for j, mk in enumerate(('s', 'o', '^')):
            ax.plot(p[i, j, 0], p[i, j, 1], mk, color=col, mec='k', mew=0.4, ms=ms, alpha=alpha, zorder=4)


def skel_legend(fig, y=0.005):
    h = [Line2D([], [], color=CMAP(ci / (len(SK_SHOW) - 1)), lw=2, label=f'{t * 1000:+.0f} ms')
         for ci, t in enumerate(SK_SHOW)]
    h += [Line2D([], [], ls='', marker=mk, color='0.5', mec='k', label=lab)
          for mk, lab in (('s', 'base'), ('o', 'particle'), ('^', 'tip'))]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, y), frameon=False, ncol=len(h), handlelength=1.2,
               columnspacing=0.8)


def square_limits(axes, pts, pad=0.6):
    pts = np.vstack([p for p in pts if p is not None and len(p)])
    pts = pts[np.isfinite(pts).all(1)]
    lo, hi = np.percentile(pts, 1, 0) - pad, np.percentile(pts, 99, 0) + pad
    half = max(hi - lo) / 2
    mid = (lo + hi) / 2
    for a in axes:
        a.set_xlim(mid[0] - half, mid[0] + half)
        a.set_ylim(mid[1] - half, mid[1] + half)
        a.set_aspect('equal', adjustable='box')


# ------------------------------------------------------------------ figures
def fig_examples(per, g2, stage, n_per=3):
    picks = []
    for c in k.COHORTS:
        sc = []
        for mouse in per.get((c, 'whisker'), {}):
            n = [sum(len(a) for a in per[(c, mod)].get(mouse, {}).get(x, [])) for mod in k.MODS for x in ('reach', 'skel')]
            sc.append((min(n), mouse))
        picks += [(c, m) for _, m in sorted(sc, reverse=True)[:n_per]]
    fig, axs = plt.subplots(len(picks), 4, figsize=(7.2, 1.85 * len(picks)), squeeze=False)
    tongue_axes, whisk_axes, tpts, wpts = [], [], [], []
    rng = np.random.default_rng(0)
    for r, (c, mouse) in enumerate(picks):
        for j, mod in enumerate(k.MODS):
            e = per[(c, mod)].get(mouse, {'reach': [], 'skel': []})
            ax = axs[r, j]
            tongue_axes.append(ax)
            if e['reach']:
                a = np.concatenate(e['reach'])
                for tr in a[rng.permutation(len(a))[:60]]:
                    ok = np.isfinite(tr[:, 0]) & (STEP_MS <= TMAX_REACH)
                    ax.plot(tr[ok, 0], tr[ok, 1], '-', color=k.COHORT_COLOR[c], lw=0.3, alpha=0.15)
                m = reach_mean(e['reach'])
                sc_t = draw_reach_steps(ax, m)
                tpts.append(m)
                ax.text(0.02, 0.97, f'n = {len(a)}', transform=ax.transAxes, va='top', fontsize=5.5)
            draw_spout(ax)
            ax = axs[r, 2 + j]
            whisk_axes.append(ax)
            if e['skel']:
                sk = np.concatenate(e['skel'])
                with np.errstate(all='ignore'):
                    b0 = np.nanmean(np.nanmean(sk, 0)[g2 < 0, 0, :], 0)
                for tr in sk[rng.permutation(len(sk))[:40]]:
                    p = img(tr[(g2 >= 0) & (g2 <= 0.3), 2] - b0)
                    ax.plot(p[:, 0], p[:, 1], '-', color=k.COHORT_COLOR[c], lw=0.3, alpha=0.15)
                m = skel_center(sk, g2)
                draw_skeleton(ax, m, g2, lw=1.4, ms=3)
                wpts.append(img(m).reshape(-1, 2))
                ax.text(0.02, 0.97, f'n = {len(sk)}', transform=ax.transAxes, va='top', fontsize=5.5)
            if r == 0:
                axs[0, j].set_title(f'tongue reach\n{MOD_LABEL[mod]}')
                axs[0, 2 + j].set_title(f'whisker sweep\n{MOD_LABEL[mod]}')
        axs[r, 0].set_ylabel(f'{mouse} ({c})\nup (mm)', color=k.COHORT_COLOR[c])
    square_limits(tongue_axes, tpts + [np.zeros((1, 2))], pad=1.0)
    square_limits(whisk_axes, wpts, pad=1.5)
    for ax in whisk_axes:
        ax.invert_yaxis()
    for ax in axs[-1, :2]:
        ax.set_xlabel('forward from spout (mm)')
    for ax in axs[-1, 2:]:
        ax.set_xlabel('image x (mm, lateral = left)')
    for ax in axs[:, 2]:
        ax.set_ylabel('image y (mm, forward = down)')
    skel_legend(fig)
    fig.suptitle(wrap(f'{k.STAGE_LABEL[stage]}: example mice. Tongue: single first reaches (thin; spout-centred) and '
                 f'mean every 5 ms (0-{TMAX_REACH} ms, colour = time). Whisker: single-trial tip paths 0-300 ms '
                 f'(thin) and mean skeleton at successive times', fig), fontsize=7)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    save(fig, f'001_{stage}_examples')


def fig_tongue(per, stage):
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], wspace=0.35, hspace=0.35)
    axes, pts, sc = [], [], None
    dist = {}
    for i, c in enumerate(k.COHORTS):
        for j, mod in enumerate(k.MODS):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
            mm, dd = [], []
            for mouse, e in per.get((c, mod), {}).items():
                if not e['reach']:
                    continue
                m = reach_mean(e['reach'])
                if np.isfinite(m[:, 0]).sum() < 3:
                    continue
                mm.append(m)
                a = np.concatenate(e['reach'])
                with np.errstate(all='ignore'):
                    dm = np.nanmean(np.hypot(a[:, :, 0], a[:, :, 1]), 0)
                dm[np.isfinite(a[:, :, 0]).mean(0) < 0.5] = np.nan
                dd.append(dm)
                ok = np.isfinite(m[:, 0]) & (STEP_MS <= TMAX_REACH)
                ax.plot(m[ok, 0], m[ok, 1], '-', color=k.COHORT_COLOR[c], lw=0.4, alpha=0.3)
            if mm:
                arr = np.array(mm)
                n = np.isfinite(arr[:, :, 0]).sum(0)
                with np.errstate(all='ignore'):
                    g = np.nanmean(arr, 0)
                    se = np.nanstd(arr, 0, ddof=1) / np.sqrt(np.maximum(n, 1))[:, None]
                g[n < 0.5 * len(mm)] = np.nan
                ok = np.isfinite(g[:, 0]) & (STEP_MS <= TMAX_REACH)
                ax.errorbar(g[ok, 0], g[ok, 1], xerr=se[ok, 0], yerr=se[ok, 1], fmt='none', ecolor='0.55', lw=0.5,
                            zorder=2)
                sc = draw_reach_steps(ax, g, size=18)
                pts.append(arr.reshape(-1, 2))
                dist[(c, mod)] = np.array(dd)
            draw_spout(ax)
            nre = sum(len(a) for e in per.get((c, mod), {}).values() for a in e['reach'])
            ax.set_title(f'{c} · {MOD_LABEL[mod]}', color=k.COHORT_COLOR[c])
            ax.text(0.02, 0.97, f'{len(mm)} mice, {nre} reaches', transform=ax.transAxes, va='top', fontsize=5.5)
            ax.set_xlabel('forward from spout (mm)')
            ax.set_ylabel('up from spout (mm)')
    square_limits(axes, pts + [np.zeros((1, 2))], pad=0.4)
    for j, mod in enumerate(k.MODS):
        ax = fig.add_subplot(gs[j, 2])
        for c in k.COHORTS:
            d = dist.get((c, mod))
            if d is None:
                continue
            n = np.isfinite(d).sum(0)
            with np.errstate(all='ignore'):
                m = np.nanmean(d, 0)
                se = np.nanstd(d, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
            ok = (n >= 0.5 * len(d)) & (STEP_MS <= TMAX_REACH)
            ax.fill_between(STEP_MS[ok], m[ok] - se[ok], m[ok] + se[ok], color=k.COHORT_COLOR[c], alpha=0.25, lw=0)
            ax.plot(STEP_MS[ok], m[ok], '.-', color=k.COHORT_COLOR[c], lw=1, ms=3, label=f'{c} ({len(d)} mice)')
        ax.set_title(f'tongue tip to spout · {MOD_LABEL[mod]}')
        ax.set_xlabel('time from reach onset (ms)')
        ax.set_ylabel('distance (mm)')
        ax.legend(frameon=False)
    if sc is not None:
        cax = fig.add_axes([0.08, 0.05, 0.25, 0.012])
        fig.colorbar(sc, cax=cax, orientation='horizontal', label='time from reach onset (ms)')
    fig.suptitle(wrap(f'{k.STAGE_LABEL[stage]}: first tongue reach after the stimulus, spout-centred (side view). '
                 f'Thin = mice, dots = mean across mice every 5 ms (+- SEM)', fig), fontsize=7.5)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.07, right=0.98)
    save(fig, f'001_{stage}_tongue')


def fig_whisker(per, g2, stage):
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 5.8))
    axes, pts = [], []
    for i, c in enumerate(k.COHORTS):
        for j, mod in enumerate(k.MODS):
            ax = axs[i, j]
            axes.append(ax)
            ms = []
            for mouse, e in per.get((c, mod), {}).items():
                if e['skel']:
                    m = skel_center(np.concatenate(e['skel']), g2)
                    if np.isfinite(m).mean() >= 0.8:
                        ms.append(m)
                        p = img(m[(g2 >= 0) & (g2 <= 0.3), 2])
                        ax.plot(p[:, 0], p[:, 1], '-', color=k.COHORT_COLOR[c], lw=0.4, alpha=0.25)
            if ms:
                with np.errstate(all='ignore'):
                    g = np.nanmean(np.array(ms), 0)
                tip = img(g[(g2 >= 0) & (g2 <= 0.3), 2])
                ax.plot(tip[:, 0], tip[:, 1], '-', color='k', lw=0.6, zorder=2)
                draw_skeleton(ax, g, g2)
                pts.append(img(g).reshape(-1, 2))
            ntr = sum(len(a) for e in per.get((c, mod), {}).values() for a in e['skel'])
            ax.set_title(f'{c} · {MOD_LABEL[mod]}', color=k.COHORT_COLOR[c])
            ax.text(0.02, 0.03, f'{len(ms)} mice, {ntr} trials', transform=ax.transAxes, fontsize=5.5)
            ax.set_xlabel('image x (mm; lateral = left)')
            ax.set_ylabel('image y (mm; forward = down)')
    square_limits(axes, pts, pad=0.8)
    for ax in axes:
        ax.invert_yaxis()
    skel_legend(fig)
    fig.suptitle(wrap(f'{k.STAGE_LABEL[stage]}: whisker sweep (top view, image orientation), mean skeleton across mice, '
                 f'each mouse centred on its whisker base at baseline.\nBlack line = mean tip path 0-300 ms, thin = '
                 f'mouse tip paths', fig), fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    save(fig, f'001_{stage}_whisker')


def fig_timecourses(stage):
    sess = k.sessions_and_cohorts()
    acc, extras, _ = k.collect(sess[sess.stage == stage])
    g = extras['grid']
    vars_ = [v for v in k.VARS if v[0] not in ('tongue_ext', 'wh_curv')]
    fig, axs = plt.subplots(len(vars_), 2, figsize=(5.2, 1.05 * len(vars_) + 0.6), sharex=True, squeeze=False)
    for j, mod in enumerate(k.MODS):
        for r, (key, lab, unit, _) in enumerate(vars_):
            ax = axs[r, j]
            for c in k.COHORTS:
                mice, mat = k.mouse_matrix(acc, c, f'tc|{key}|{k.HIT[mod]}')
                if mat is None:
                    continue
                m, se = k.mean_sem(mat)
                ax.fill_between(g, m - se, m + se, color=k.COHORT_COLOR[c], alpha=0.25, lw=0)
                ax.plot(g, m, color=k.COHORT_COLOR[c], lw=1, label=f'{c} ({len(mice)})')
            ax.axvline(0, color='k', lw=0.5, ls=':')
            if j == 0:
                ax.set_ylabel(f'{lab}\n({unit})' if unit else lab, fontsize=6)
            if r == 0:
                ax.set_title(MOD_LABEL[mod])
                ax.legend(frameon=False, loc='upper left')
        axs[-1, j].set_xlabel('time from stimulus (s)')
        axs[-1, j].set_xlim(-0.2, 1.0)
    for r in range(len(vars_)):
        lo = min(a.get_ylim()[0] for a in axs[r])
        hi = max(a.get_ylim()[1] for a in axs[r])
        for a in axs[r]:
            a.set_ylim(lo, hi)
    fig.suptitle(wrap(f'{k.STAGE_LABEL[stage]}: evoked kinematics on hit trials, mean +- SEM across mice (n mice). '
                 f'Whisker angle at 0 ms on whisker trials includes the stimulus', fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f'001_{stage}_timecourses')


def main():
    for stage in k.STAGES:
        per, g2 = load_stage(stage)
        fig_examples(per, g2, stage)
        fig_tongue(per, stage)
        fig_whisker(per, g2, stage)
        fig_timecourses(stage)
        print(stage, 'done', flush=True)


if __name__ == '__main__':
    main()
