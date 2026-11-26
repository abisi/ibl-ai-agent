"""Supplementary figure: controls for the passive-response change (Figure 4 A-H), rebuilt on the 179 pipeline (user 2026-11-26:
"we do the control plan"; content of the earlier Figma SuppFigure4_passive_controls: baseline rate change across area groups;
effect not explained by the number of whisker stimuli, the baseline rate change, the change in whisker speed trial by trial,
the pre-stimulus whisking trial by trial; robustness to unit quality (good / stable / all units, effect vs CI) and to random
subsets of units per mouse).

Effect = the cohort x stimulus interaction of Figure 4G: per session the whisker minus auditory change (post - pre) of the
population evoked response (5-35 ms minus each trial's own baseline, spikes/s, mean over neurons), then R+ minus R- (mean over
sessions). 95 % CI: bootstrap over sessions within cohort (2000). P: larger of Mann-Whitney and Welch on the session values
(R+ vs R-), uncorrected.
Data: 179 unit table (stable good + MUA, learners, whisker day 0) and 208 (tables/208_passive_controls_{trials,sessions}:
trial-level population responses for three unit sets and per-trial DLC; ephys_utilities trials and units, as 179).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from shared_passive_cd import *  # noqa: F401,F403
from shared_passive_cd import (COH, COLORS, FS, LAB, META, TAB, W_COL, A_COL, _ag_units, _mm_insets, area_palette,  # noqa: E501
                               cohort_ticks, compute_ag_areas, learners, light, mean_sem, p_text, pmax, stars, tests,
                               two_group_tests)

N_BOOT = 2000
N_PERM_CTL = 5000                                                # cohort-label permutations per estimate
SUBSET_K = (10, 25, 50, 100, 200, 500)
N_SUBSET = 200


# ------------------------------------------------------------------ helpers

def _interaction(d, col="WA"):
    """d: one row per session with cohort and col (= dW - dA) -> effect R+ - R-, bootstrap CI, larger P of MW / Welch"""
    a, b = d[d.cohort == "R+"][col].dropna().to_numpy(), d[d.cohort == "R-"][col].dropna().to_numpy()
    rng = np.random.default_rng(0)
    boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    t = two_group_tests(pd.Series(a), pd.Series(b))
    v = np.r_[a, b]
    g = np.r_[["R+"] * len(a), ["R-"] * len(b)]
    _, p_perm, null = perm_interaction(v, g, n=N_PERM_CTL, return_null=True)
    return dict(effect=float(a.mean() - b.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                p=float(pmax(t["mannwhitney_p"], t["welch_p"])), null_lo=float(np.percentile(null, 2.5)),
                null_hi=float(np.percentile(null, 97.5)), p_perm=float(p_perm), n_Rp=len(a), n_Rm=len(b))


def _sessions_179():
    """per session: mean over stable units of the evoked change dW, dA (spikes/s) and of the baseline rate change"""
    u = _ag_units()
    u = u.assign(dW=u.ev_post_W - u.ev_pre_W, dA=u.ev_post_A - u.ev_pre_A,
                 dB=u[["base_post_W", "base_post_A"]].mean(1) - u[["base_pre_W", "base_pre_A"]].mean(1),
                 rW=(u.ev_post_W + u.base_post_W) - (u.ev_pre_W + u.base_pre_W),
                 rA=(u.ev_post_A + u.base_post_A) - (u.ev_pre_A + u.base_pre_A))
    d = u.groupby(["session_id", "mouse_id", "cohort"])[["dW", "dA", "dB", "rW", "rA"]].mean().reset_index()
    d["WA"] = d.dW - d.dA
    return d


def _trials_208():
    """208 trial table restricted to the sessions of the 179 unit table (same sessions as Fig. 4: >= 3 hits and misses)"""
    t = pd.read_parquet(TAB / "208_passive_controls_trials.parquet")
    return t[t.mouse_id.isin(learners()) & t.session_id.isin(_ag_units().session_id.unique())]


def _ci_rows(ax, rows, labels, title, ylabel="Interaction (spks/s)\nR+ − R− of W − A change"):
    """per row: grey bar = 2.5-97.5 % of the cohort-permutation null of the interaction, black dot = observed value,
    permutation P above (user 2026-11-26: "the CIs ... are the cohort permutation null distribution")"""
    for k, r in enumerate(rows):
        ax.plot([k, k], [r["null_lo"], r["null_hi"]], color="0.75", lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot(k, r["effect"], "o", ms=3, color="k", zorder=3)
        top = max(r["effect"], r["null_hi"])
        ax.text(k, top + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]), p_text(r["p_perm"]).replace("p = ", "P = ").replace(
            "p < ", "P < "), ha="center", va="bottom", fontsize=FS - 1.2, rotation=90)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks(range(len(rows)), labels, fontsize=FS - 0.5, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel(ylabel)


def _set_ylim_ci(ax, rows):
    lo = min(min(min(r["null_lo"], r["effect"]) for r in rows), 0)
    hi = max(max(max(r["null_hi"], r["effect"]) for r in rows), 0)
    sp = hi - lo
    ax.set_ylim(lo - 0.1 * sp, hi + 0.55 * sp)


def _scatter_two(ax, d, x, y, xlabel, ylabel):
    """scatter per cohort + per-cohort OLS line (solid if p < 0.05), r and p per cohort"""
    txt = []
    for c in COH:
        g = d[d.cohort == c][[x, y]].dropna()
        ax.scatter(g[x], g[y], s=4, color=COLORS[c], lw=0, zorder=3)
        lr = stats.linregress(g[x], g[y])
        xs = np.linspace(g[x].min(), g[x].max(), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=COLORS[c], lw=0.7, ls="-" if lr.pvalue < 0.05 else (0, (3, 2)))
        txt.append((c, len(g), lr.rvalue, lr.pvalue))
    for k, (c, n, r, p) in enumerate(txt):
        ax.text(0.03, 0.97 - 0.09 * k, f"{LAB[c]}: n = {n}, r = {r:.2f}, {p_text(p)}", transform=ax.transAxes, ha="left",
                va="top", fontsize=FS - 1, color=COLORS[c])
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (1, 1.5)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    v = d[y].dropna()
    lo, hi = float(v.min()), float(np.percentile(v, 98))        # cut only at the top (single outlier sessions)
    sp = hi - lo
    ax.set_ylim(lo - 0.05 * sp, hi + 0.45 * sp)
    if v.max() > hi + 0.45 * sp:
        from shared_passive_cd import _axis_break
        _axis_break(ax, "top")


# ------------------------------------------------------------------ a. baseline rate change per area group

def compute_ctl_baseline_areas():
    """per area group (>= 3 mice per cohort, each >= 10 neurons in the area), per session the mean over the area's neurons of
    the baseline rate change (post - pre; -1 s..-15 ms; mean over whisker and auditory trials; spikes/s)"""
    d = compute_ag_areas()
    META["ctl_baseline_areas"] = META["ag_areas"]
    return d[["session_id", "mouse_id", "cohort", "area_group", "n", "B"]]


def plot_ctl_baseline_areas(ax, df):
    """x = area groups (custom order), R+ / R- mean +- s.e.m. over sessions; filled = R+ vs R- differ (larger P, uncorrected)"""
    col, order, _ = area_palette()
    areas = [a_ for a_ in order if a_ in set(df.area_group)] + sorted(set(df.area_group) - set(order))
    a = ax
    for k, a_ in enumerate(areas):
        g = df[df.area_group == a_]
        t = two_group_tests(g[g.cohort == "R+"].B, g[g.cohort == "R-"].B)
        sig = pmax(t["mannwhitney_p"], t["welch_p"]) < 0.05
        for c, off in (("R+", -0.17), ("R-", 0.17)):
            m, s = mean_sem(g[g.cohort == c].B)
            a.errorbar(k + off, m, s, fmt="o", ms=2.4, color=COLORS[c], mfc=COLORS[c] if sig else "white", mew=0.6, lw=0.6,
                       capsize=0)
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_xticks(range(len(areas)), [AREA_SHORT(x) for x in areas], rotation=60, ha="right", fontsize=FS - 0.5)
    for tl, a_ in zip(a.get_xticklabels(), areas):
        tl.set_color(col.get(a_, "0.2"))
    a.set_xlim(-0.6, len(areas) - 0.4)
    a.set_ylabel("Baseline rate change\n(spks/s, post − pre)")
    a.text(0.99, 0.97, "R+", color=COLORS["R+"], transform=a.transAxes, ha="right", va="top", fontsize=FS)
    a.text(0.99, 0.88, "R−", color=COLORS["R-"], transform=a.transAxes, ha="right", va="top", fontsize=FS)
    a.text(0.01, 0.97, "filled: R+ ≠ R− (P < 0.05)", transform=a.transAxes, ha="left", va="top", fontsize=FS - 1)


def AREA_SHORT(a):
    from shared_passive_cd import area_label
    return area_label(a)


# ------------------------------------------------------------------ b, c. number of whisker stimuli, baseline rate change

def _adjusted(d, cov):
    r = smf.ols(f"WA ~ C(cohort) + {cov}", d.dropna(subset=["WA", cov])).fit()
    k = [x for x in r.params.index if x.startswith("C(cohort)")][0]
    return dict(beta=float(-r.params[k]), p=float(r.pvalues[k]), p_cov=float(r.pvalues[cov]))   # R+ - R-


def compute_ctl_n_stim():
    """per session: number of whisker stimuli (active + passive, trials table) and the whisker response change dW; cohort
    effect on W - A adjusted for the number of stimuli (OLS)"""
    d = _sessions_179().merge(pd.read_parquet(TAB / "208_passive_controls_sessions.parquet")
                              [["session_id", "n_whisker_stim_total", "n_whisker_active"]], on="session_id", how="inner")
    META["ctl_n_stim"] = dict(n_stim=tests(d, "n_whisker_stim_total"), adjusted=_adjusted(d, "n_whisker_stim_total"),
                              unadjusted=_interaction(d))
    return d


def plot_ctl_n_stim(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    _scatter_two(a, df, "n_whisker_stim_total", "dW", "Whisker stimuli in the session", "Whisker response change\n(spks/s)")
    m = META.get("ctl_n_stim", {}).get("adjusted")


def compute_ctl_baseline():
    """per session: baseline rate change and the whisker response change; cohort effect on W - A adjusted for it (OLS)"""
    d = _sessions_179()
    META["ctl_baseline"] = dict(dB=tests(d, "dB"), adjusted=_adjusted(d, "dB"), unadjusted=_interaction(d))
    return d


def plot_ctl_baseline(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    _scatter_two(a, df, "dB", "dW", "Baseline rate change (spks/s)", "Whisker response change\n(spks/s)")


# ------------------------------------------------------------------ d. no baseline correction

def compute_ctl_nobase():
    """raw rate in the 5-35 ms window (evoked + baseline, no baseline subtraction), change post - pre, whisker and auditory"""
    d = _sessions_179()
    d["WA_raw"] = d.rW - d.rA
    obs, p = perm_interaction(d.WA_raw, d.cohort)
    META["ctl_nobase"] = dict(perm=dict(diff=obs, p=p), inter=_interaction(d, "WA_raw"))
    return d.melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=["rW", "rA"], var_name="stim").assign(
        stim=lambda x: x.stim.str[1])


def plot_ctl_nobase(ax, df):
    from shared_passive_cd import plot_ag_interaction
    plot_ag_interaction(ax, df)
    ax.child_axes[0].set_ylabel("Raw rate change, 5–35 ms\n(spks/s, post − pre)") if ax.child_axes else None


# ------------------------------------------------------------------ e. movement, trial by trial

def _session_change(t, ev="ev_stable", covs=None, keep=None):
    """per session x stim: post - pre of the population response; covs -> epoch coefficient of an OLS within the session
    (trials of that stim, both epochs): ev ~ post + covs; keep -> boolean mask of trials used"""
    rows = []
    for (sid, coh), g in t.groupby(["session_id", "cohort"]):
        r = dict(session_id=sid, cohort=coh)
        for s in "WA":
            h = g[g.stim == s]
            if keep is not None:
                h = h[keep.loc[h.index]]
            if covs:
                h = h.dropna(subset=[ev] + covs)
            if (h.epoch == "pre").sum() < 3 or (h.epoch == "post").sum() < 3:
                r[s] = np.nan
                continue
            if covs:
                X = np.column_stack([np.ones(len(h)), (h.epoch == "post").to_numpy(float)] + [h[c].to_numpy() for c in covs])
                beta, *_ = np.linalg.lstsq(X, h[ev].to_numpy(), rcond=None)
                r[s] = float(beta[1])
            else:
                r[s] = float(h[h.epoch == "post"][ev].mean() - h[h.epoch == "pre"][ev].mean())
        rows.append(r)
    d = pd.DataFrame(rows)
    d["WA"] = d.W - d.A
    return d


def _still_mask(t):
    """still trials: within session, a movement score = sum of |pre-stimulus whisker speed|, |evoked whisker speed| and
    |evoked jaw opening|, each divided by its s.d. over the session's passive trials (both epochs and stimuli); trials at or
    below the session's median score (the three-way conjunction kept 19 % of trials, too few per condition)"""
    sc = pd.Series(0.0, index=t.index)
    for c in ("ws_pre", "ws_ev", "jaw_ev"):
        v = t[c].abs()
        sc += v / v.groupby(t.session_id).transform("std")
    return sc <= sc.groupby(t.session_id).transform("median")


def compute_ctl_movement():
    """interaction (R+ - R- of the W - A change) with three estimates of the per-session change: all trials (mean post - mean
    pre); adjusted trial by trial for whisker speed before and after the stimulus (least squares within session and
    stimulus: response ~ 1 + post + ws_pre + ws_ev, epoch coefficient); adjusted also for the evoked jaw opening (+ jaw_ev).
    Same sessions (with DLC) in every estimate. (Still trials and the random-half reference removed, user 2026-11-26.)"""
    t = _trials_208()
    t = t[t.session_id.isin(t.dropna(subset=["ws_pre", "ws_ev", "jaw_ev"]).session_id.unique())]
    est = [("all", dict()), ("whisk", dict(covs=["ws_pre", "ws_ev"])), ("whisk_jaw", dict(covs=["ws_pre", "ws_ev", "jaw_ev"]))]
    rows, meta = [], {}
    for name, kw in est:
        d = _session_change(t, **kw).dropna(subset=["WA"])
        r = _interaction(d)
        meta[name] = r
        rows.append(dict(estimate=name, **r))
    META["ctl_movement"] = meta
    return pd.DataFrame(rows)


def plot_ctl_movement(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    rows = df.to_dict("records")
    _set_ylim_ci(a, rows)
    _ci_rows(a, rows, ["All trials", "Adjusted:\nwhisking", "Adjusted:\nwhisking + jaw"], "Movement, trial by trial")


# ------------------------------------------------------------------ f. unit quality

def compute_ctl_units():
    """interaction for three unit sets (208): good (stable good), stable (stable good + MUA, main), all (no selection)"""
    t = _trials_208()
    rows = []
    for s in ("good", "stable", "all"):
        d = _session_change(t, ev=f"ev_{s}")
        n = int(t.drop_duplicates("session_id")[f"n_{s}"].sum())
        rows.append(dict(unit_set=s, n_units=n, **_interaction(d.dropna(subset=["WA"]))))
    META["ctl_units"] = {r["unit_set"]: r for r in rows}
    return pd.DataFrame(rows)


def plot_ctl_units(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    rows = df.to_dict("records")
    _set_ylim_ci(a, rows)
    lab = {"good": "Good units", "stable": "Stable (main)", "all": "All units"}
    _ci_rows(a, rows, [f"{lab[r['unit_set']]}\n({r['n_units'] / 1000:.0f}k)" for r in rows], "Unit quality")


# ------------------------------------------------------------------ g. random subsets of units per mouse

def compute_ctl_subsets():
    """per session (= mouse), k stable units drawn at random without replacement (sessions with >= k units), N_SUBSET draws:
    interaction per draw; median, 2.5-97.5 % of draws, fraction of draws with P < 0.05, sessions kept"""
    u = _ag_units()
    u = u.assign(dW=u.ev_post_W - u.ev_pre_W, dA=u.ev_post_A - u.ev_pre_A)
    groups = {(s, c): g[["dW", "dA"]].to_numpy() for (s, c), g in u.groupby(["session_id", "cohort"])}
    full = _interaction(_sessions_179())
    rng = np.random.default_rng(0)
    rows = []
    for k in SUBSET_K:
        keep = {key: v for key, v in groups.items() if len(v) >= k}
        effs, sig = [], []
        for _ in range(N_SUBSET):
            recs = []
            for (s, c), v in keep.items():
                m = v[rng.choice(len(v), k, replace=False)].mean(0)
                recs.append(dict(session_id=s, cohort=c, WA=m[0] - m[1]))
            d = pd.DataFrame(recs)
            a, b = d[d.cohort == "R+"].WA, d[d.cohort == "R-"].WA
            if len(a) < 3 or len(b) < 3:
                continue
            effs.append(a.mean() - b.mean())
            t = two_group_tests(a, b)
            sig.append(pmax(t["mannwhitney_p"], t["welch_p"]) < 0.05)
        if effs:
            # all units of the same sessions (sessions with >= k units): the reference for this k
            fs = pd.DataFrame([dict(cohort=c, WA=v[:, 0].mean() - v[:, 1].mean()) for (s, c), v in keep.items()])
            rows.append(dict(k=k, median=float(np.median(effs)), lo=float(np.percentile(effs, 2.5)),
                             hi=float(np.percentile(effs, 97.5)), frac_sig=float(np.mean(sig)), n_sessions=len(keep),
                             frac_sessions=len(keep) / len(groups),
                             full_same=float(fs[fs.cohort == "R+"].WA.mean() - fs[fs.cohort == "R-"].WA.mean())))
    META["ctl_subsets"] = dict(full=full, n_draws=N_SUBSET)
    return pd.DataFrame(rows).assign(full=full["effect"])


def plot_ctl_subsets(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    x = np.arange(len(df))
    a.fill_between(x, df.lo, df.hi, color="0.88", lw=0, edgecolor="none", label="2.5–97.5 % of draws")
    a.plot(x, df["median"], "o-", color="k", ms=2.5, lw=0.8, label="Random subsets, median")
    a.plot(x, df.full_same, "_", color="#B5524A", ms=7, mew=1.0, label="All units, same sessions")
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_xticks(x, [str(k) for k in df.k])
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + 0.25 * (hi - lo))
    for xx, f, fs in zip(x, df.frac_sig, df.frac_sessions):
        a.text(xx, 1.0, f"{f:.0%}", transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 1.2)
    a.text(-0.5, 1.09, "P < 0.05:", transform=a.get_xaxis_transform(), ha="left", va="bottom", fontsize=FS - 1.2)
    a.set_xlabel("Units per mouse")
    a.set_ylabel("Interaction (spks/s)\nR+ − R− of W − A change")
    a.legend(fontsize=FS - 1.2, frameon=False, loc="lower center", bbox_to_anchor=(0.6, 0.0), handlelength=1.0, labelspacing=0.2)


# ------------------------------------------------------------------ h. licked passive trials

def compute_ctl_licked():
    """passive trials removed because of a lick, per session, epoch and stimulus (counts; fraction of the epoch's trials)"""
    s = pd.read_parquet(TAB / "208_passive_controls_sessions.parquet")
    s = s[s.mouse_id.isin(learners())]
    rows = []
    for e in ("pre", "post"):
        for st in "WA":
            rows.append(s[["session_id", "cohort"]].assign(epoch=e, stim=st, n=s[f"n_passive_licked_{e}_{st}"],
                                                            frac=s[f"n_passive_licked_{e}_{st}"] / s[f"n_passive_{e}_{st}"]))
    d = pd.concat(rows, ignore_index=True)
    META["ctl_licked"] = {f"{e}_{st}": tests(d[(d.epoch == e) & (d.stim == st)].rename(columns={"frac": "value"}), "value")
                          for e in ("pre", "post") for st in "WA"}
    return d


def plot_ctl_licked(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    xs, labels = [], []
    for k, (e, st) in enumerate([(e, st) for st in "WA" for e in ("pre", "post")]):
        for q, c in enumerate(COH):
            v = df[(df.epoch == e) & (df.stim == st) & (df.cohort == c)].frac * 100
            m, s = mean_sem(v)
            x = k + (q - 0.5) * 0.36
            a.bar(x, m, 0.32, color=light(COLORS[c], 0.55), edgecolor=COLORS[c], lw=0.5)
            a.errorbar(x, m, s, color="0.25", lw=0.7, capsize=0)
        labels.append(f"{'W' if st == 'W' else 'A'} {e}")
    a.set_xticks(range(4), labels, fontsize=FS - 0.5)
    a.set_ylabel("Passive trials removed\nfor a lick (%)")


# ------------------------------------------------------------------ baseline rate change on Allen coronal sections (user
# 2026-11-26: "A: more condensed, replace by maps (allen sections)")

MAP_RES = 25.0
MAP_N_AP = 4
MAP_MID = 5700.0                                                  # midline (um), 25-um atlas


def _map_parent():
    """Allen structure -> structure drawn on the map: cortical layers merged into their area, barrel columns into SSp-bfd
    (Blue Brain barrel atlas structures.csv, as the Fig. 4 area maps)"""
    from common import axel_bisi_root
    st = pd.read_csv(axel_bisi_root() / "Anatomy" / "allen_mouse_bluebrain_barrels_10um_v1.0" / "structures.csv")
    name, acr = dict(zip(st.id, st.name)), dict(zip(st.id, st.acronym))
    par = dict(zip(st.id, st.parent_structure_id.fillna(-1).astype(int)))
    bfd = int(st.loc[st.acronym == "SSp-bfd", "id"].iloc[0])
    out = {}
    for i in st.id:
        j = i
        while j in name and "layer" in str(name[j]).lower() and par.get(j, -1) in name:
            j = par[j]
        if str(acr.get(j, "")).startswith("SSp-bfd"):
            j = bfd
        out[int(i)] = int(j)
    return out


def compute_ctl_baseline_map():
    """baseline rate change (post - pre; spikes/s) per fine area (area_acronym_custom) and cohort: per session the mean over
    the area's neurons (>= 5), then the mean over sessions; areas with >= 3 mice per cohort. Map: each atlas region (layers
    merged, barrels -> SSp-bfd) takes the fine area holding most of the recorded neurons located in it (ROC table
    ccf_atlas_id); sections at the 20/40/60/80 % quantiles of the neurons' AP position. Rows: kind = area / region / ap."""
    from common import axel_bisi_root
    u = _ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    u["dB"] = u[["base_post_W", "base_post_A"]].mean(1) - u[["base_pre_W", "base_pre_A"]].mean(1)
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "ccf_atlas_id", "ccf_atlas_ap"])
            rows.append(r.drop_duplicates(["session_id", "electrode_group", "cluster_id"]))
    c = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    c["cid"] = c.cid.astype(int)
    u = u.merge(c, on=["session_id", "electrode_group", "cid"], how="left")
    col = "area_acronym_custom"
    d = u.groupby(["session_id", "mouse_id", "cohort", col]).agg(n=("cid", "size"), dB=("dB", "mean")).reset_index()
    d = d[d.n >= 5]
    k = d.groupby([col, "cohort"]).mouse_id.nunique().unstack().fillna(0)
    keep = k[(k.get("R+", 0) >= 3) & (k.get("R-", 0) >= 3)].index
    d = d[d[col].isin(keep)]
    area = d.groupby([col, "cohort"]).dB.mean().reset_index().rename(columns={col: "area", "dB": "value"})
    pm = _map_parent()
    u["rid"] = u.ccf_atlas_id.map(lambda x: pm.get(int(x), int(x)) if np.isfinite(x) else 0)
    maj = u[u[col].isin(keep) & (u.rid != 0)].groupby("rid")[col].agg(lambda x: x.value_counts().index[0])
    reg = pd.DataFrame(dict(rid=maj.index.astype(int), area=maj.values))
    aps = np.nanpercentile(u.ccf_atlas_ap.dropna(), np.linspace(20, 80, MAP_N_AP))
    META["ctl_baseline_map"] = dict(n_areas=len(keep), n_regions=len(reg),
                                    vmax=float(np.nanpercentile(np.abs(area.value), 95)))
    return pd.concat([area.assign(kind="area"), reg.assign(kind="region"),
                      pd.DataFrame(dict(kind="ap", value=aps))], ignore_index=True)


def plot_ctl_baseline_map(ax, df):
    """rows R+ / R-, columns coronal sections (one hemisphere), regions painted with the fine area's baseline change (diverging,
    same scale); light grey, no data; grey lines, region contours"""
    import tifffile
    from pathlib import Path
    from matplotlib import colors as mcolors
    from scipy import ndimage
    import matplotlib.pyplot as plt
    ann = tifffile.imread(Path.home() / ".brainglobe" / "allen_mouse_25um_v1.2" / "annotation.tiff")   # (AP, DV, ML)
    pm = _map_parent()
    area = df[df.kind == "area"]
    reg = df[df.kind == "region"]
    aps = df[df.kind == "ap"].value.to_numpy()
    vmax = float(np.nanpercentile(np.abs(area.value), 95))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(-vmax, vmax)
    midv = int(MAP_MID / MAP_RES)
    ax.axis("off")
    W_mm = ax.get_position().width * ax.figure.get_figwidth() * 25.4
    cb_mm = 6.0
    nc = len(aps)
    cw = (W_mm - cb_mm) / nc
    for i, c in enumerate(COH):
        val = area[area.cohort == c].set_index("area").value
        for j, ap in enumerate(aps):
            sl = ann[int(np.clip(ap / MAP_RES, 0, ann.shape[0] - 1))][:, midv:]
            uq, inv = np.unique(sl, return_inverse=True)
            lab = np.array([pm.get(int(x), int(x)) for x in uq])[inv].reshape(sl.shape)
            rgb = np.ones(lab.shape + (3,))
            inside = lab != 0
            rgb[inside] = 0.93
            for rid, a_ in zip(reg.rid.astype(int), reg.area):
                if a_ in val.index:
                    m = lab == rid
                    if m.any():
                        rgb[m] = cmap(norm(val[a_]))[:3]
            bnd = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
            rgb[bnd & ndimage.binary_dilation(inside)] = 0.6
            a = ax.inset_axes([j * cw / W_mm, 0.5 * (1 - i) + 0.02, cw / W_mm * 0.98, 0.46])
            a.imshow(rgb, interpolation="nearest")
            a.set_axis_off()
            if i == 0:
                a.set_title(f"AP {ap / 1000:.1f} mm", fontsize=FS - 1, pad=1)
            if j == 0:
                a.text(-0.04, 0.5, LAB[c], transform=a.transAxes, ha="right", va="center", color=COLORS[c], fontsize=FS + 0.5)
    cax = ax.inset_axes([1 - (cb_mm - 1.5) / W_mm, 0.15, 1.0 / W_mm, 0.7])
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=cax)
    cb.ax.tick_params(labelsize=FS - 1, length=1, pad=0.5)
    cb.outline.set_linewidth(0.3)
    cb.set_label("Baseline rate change\n(spks/s, post − pre)", fontsize=FS - 0.5, labelpad=1)


# ------------------------------------------------------------------ whisking vs the whisker response change, across sessions
# (user 2026-11-26: "instead [of H] have another correlation figures: whisker response change vs evoked whisker speed change,
# same with baseline whisker speed change")

def _whisking_sessions():
    """per session (whisker trials, 208 per-trial DLC): change post - pre of the evoked whisker speed (5-35 ms minus -0.2..0 s,
    deg/s) and of the pre-stimulus whisker speed (-0.2..0 s, deg/s); with the whisker response change dW (179)"""
    t = _trials_208()
    t = t[t.stim == "W"]
    w = t.groupby(["session_id", "epoch"])[["ws_ev", "ws_pre"]].mean().unstack("epoch")
    d = pd.DataFrame({"d_ws_ev": w[("ws_ev", "post")] - w[("ws_ev", "pre")],
                      "d_ws_pre": w[("ws_pre", "post")] - w[("ws_pre", "pre")]}).reset_index()
    return _sessions_179().merge(d, on="session_id", how="inner").dropna(subset=["d_ws_ev", "d_ws_pre"])


def compute_ctl_whisk_evoked():
    d = _whisking_sessions()
    META["ctl_whisk_evoked"] = dict(d_ws=tests(d, "d_ws_ev"), adjusted=_adjusted(d, "d_ws_ev"))
    return d


def plot_ctl_whisk_evoked(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    _scatter_two(a, df, "d_ws_ev", "dW", "Evoked whisker speed change\n(°/s, post − pre)", "Whisker response change\n(spks/s)")


def compute_ctl_whisk_pre():
    d = _whisking_sessions()
    META["ctl_whisk_pre"] = dict(d_ws=tests(d, "d_ws_pre"), adjusted=_adjusted(d, "d_ws_pre"))
    return d


def plot_ctl_whisk_pre(ax, df):
    a = _mm_insets(ax, [ax.get_position().width * ax.figure.get_figwidth() * 25.4], [])[0]
    _scatter_two(a, df, "d_ws_pre", "dW", "Pre-stimulus whisker speed change\n(°/s, post − pre)",
                 "Whisker response change\n(spks/s)")
