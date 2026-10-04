"""012 -- Reaction-time distributions for the overlay figures (user 2026-09-30: "add the distribution of reaction times,
in grey bands, with its own invisible y axis at the bottom of the x axis, like in the time-resolved decoding project").
Run as a script (haas): builds ../artifacts/012_rt_cache.parquet from the learning-day sessions of both cohorts, one
row per trial: session_id, subject_id, cohort, kind, rt_ms, where
  kind "whisker_hit"  = whisker hit trials (prep_hitmiss_trials, licked) -> for hit vs miss (stimulus-aligned);
  kind "licked_stim"  = all licked whisker + auditory trials (prep_lick_aligned_trials) -> for whisker vs auditory
                        (stimulus-aligned: rt_ms; first-lick-aligned: -rt_ms, the stimulus time relative to the lick).
Reaction time = corrected first lick - trial start (add_first_lick_time / prep_lick_aligned_trials first_lick_time,
the corrected first lick of the SSL analyses).
Imported by 008 / 011: rt_values(target, cohort) and add_rt_band(ax, values, view) -- grey histogram (20-ms bins) drawn
on a twin y axis that is invisible, in a strip below the curves (bottom ~18% of the panel); 006 draws one per cohort.
Run (haas, repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/012_rt_band.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
CACHE = HERE.parent / "artifacts" / "012_rt_cache.parquet"
KIND = {"hitmiss": ("whisker_hit", 1), "modality_stim": ("licked_stim", 1), "modality_lick": ("licked_stim", -1)}
BAND_FRAC = 0.18


def build():
    import os
    os.chdir(ROOT_REPO)
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    rows = []
    for r in sess.itertuples():
        hm = T.prep_hitmiss_trials(root, r.session_id, st, tt)
        if hm is not None and len(hm):
            h = T.add_first_lick_time(hm[hm.lick_flag == 1])
            rows += [dict(session_id=r.session_id, subject_id=r.subject_id, cohort=r.reward_group, kind="whisker_hit",
                          rt_ms=float(v) * 1000) for v in h.reaction_time.to_numpy()]
        lk = T.prep_lick_aligned_trials(root, r.session_id, st, tt)
        if lk is not None and len(lk):
            rt = (lk.first_lick_time - lk.start_time).to_numpy() * 1000
            rows += [dict(session_id=r.session_id, subject_id=r.subject_id, cohort=r.reward_group, kind="licked_stim",
                          trial_type=tp, rt_ms=float(v)) for v, tp in zip(rt, lk.trial_type)]
    d = pd.DataFrame(rows)
    d.to_parquet(CACHE, index=False)
    print(d.groupby(["cohort", "kind"]).agg(n_trials=("rt_ms", "size"), n_sessions=("session_id", "nunique"),
                                             median_ms=("rt_ms", "median")))


def rt_values(target, cohort):
    """Reaction-time values (ms) for a target and cohort ("avg"/"pooled" = both cohorts), sign-flipped for lick-aligned."""
    if not CACHE.exists():
        return np.array([])
    d = pd.read_parquet(CACHE)
    kind, sign = KIND[target]
    d = d[d.kind == kind]
    if cohort in ("R+", "R-"):
        d = d[d.cohort == cohort]
    return sign * d.rt_ms.to_numpy()


def add_rt_band(ax, values, view, frac=BAND_FRAC, binw=20, expand=True):
    """RT histogram(s) on an invisible twin y axis, filling the bottom `frac` of the panel, behind the curves.
    values: one array (grey bars) or a list of (array, colour) pairs -- one histogram per cohort (006), drawn as
    translucent filled steps, each normalised to its own trial count (density) so cohorts of different size compare.
    expand=True lowers the data axis' bottom so the curves keep their space above the band; the left spine is bounded
    to the original data range (the band has no visible axis)."""
    sets = [(values, "#b0b0b0")] if not isinstance(values, list) else values
    bins = np.unique(np.r_[np.arange(view[0], view[1], binw), view[1]])          # clamped to the view
    hs = []
    for v, c in sets:
        v = np.asarray(v, float)
        v = v[(v >= view[0]) & (v <= view[1])]
        if len(v):
            hs.append((np.histogram(v, bins=bins, density=len(sets) > 1)[0], c))
    if not hs:
        return None
    if expand:
        y0, y1 = ax.get_ylim()
        lo, hi = ax.spines["left"].get_bounds() if ax.spines["left"].get_bounds() is not None else (y0, y1)
        ax.set_ylim(y0 - frac * (y1 - y0) / (1 - frac), y1)
        ax.spines["left"].set_bounds(max(lo, y0), hi)
        ax.set_yticks([t for t in ax.get_yticks() if y0 - 1e-9 <= t <= y1 + 1e-9])
    axh = ax.twinx()
    if ax.get_box_aspect() is not None:              # square panels: the twin must share the box, else it spans the cell
        axh.set_box_aspect(ax.get_box_aspect())
    for h, c in hs:
        if len(hs) == 1:
            axh.bar(bins[:-1], h, width=np.diff(bins), align="edge", color=c, lw=0, zorder=0)
        else:
            axh.fill_between(bins, np.r_[h, h[-1]], step="post", color=c, alpha=0.25, lw=0)
            axh.step(bins, np.r_[h, h[-1]], where="post", color=c, lw=0.6, alpha=0.8)
    axh.set_ylim(0, max(h.max() for h, _ in hs) / frac)
    axh.set_yticks([])
    for s in axh.spines.values():
        s.set_visible(False)
    # band drawn ABOVE the data axis (its own strip below the curves, so it hides nothing; keeps it visible over
    # background spans such as analysis-window shading); x limits restored to the view
    axh.set_zorder(ax.get_zorder() + 1)
    axh.patch.set_visible(False)
    ax.set_xlim(*view)

    return axh


if __name__ == "__main__":
    build()
