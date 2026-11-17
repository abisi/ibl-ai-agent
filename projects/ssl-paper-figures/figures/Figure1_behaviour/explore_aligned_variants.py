"""Exploration (user 2026-11-16): every variant of Figure 1 panels G-H (licking aligned at the start of whisker learning),
to show to colleagues. Learners, whisker day 0 untrimmed (auditory warm-up kept), early-lick trials dropped; exact
learning curves (learning_curve_exact, sigma 1) per trial type on the same session. R+ / R- mean +- s.e.m. over mice.

One page per alignment x x-axis (2 x 3 = 6 pages) after a cover page:
  alignment  first whisker trial | first whisker hit
  x-axis     trials of the type from the alignment (0 = first trial of the type at or after it)
             | absolute time from the alignment (min) | session time normalised by the whisker period (first to last
               whisker trial), 0 = the alignment
Rows: data, single trials (trial axis only; time axes need bins) | data, binned (trials: -10..-6, -5..-1, 0, 1-5, 6-10, ...;
minutes: 2 min; normalised: 0.05) | learning curves (exact fit of each trial type, evaluated at its trials or interpolated
onto a time grid).
Columns: P(lick) whisker | false alarm | auditory | delta P(lick) = whisker - false alarm | d' = z(whisker) - z(false
alarm). On the trial axis, delta P and d' use the whisker-trial index and the false-alarm trials falling in the same span
of time (data) or the false-alarm curve interpolated at each whisker trial (curves); data d' with the log-linear
correction (hits + 0.5) / (n + 1); curve d' from probabilities clipped to [0.01, 0.99]. No statistics on these pages.
Output: Figures/_explore/aligned_variants_<date>.pdf (+ _table.parquet, per mouse values).
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_aligned_variants.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import compute as fig1  # noqa: E402
import learning_curve_exact as lce  # noqa: E402
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import COLORS, MM, light, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
TT = ["whisker_trial", "no_stim_trial", "auditory_trial"]
MEASURES = [("whisker_trial", "P(lick), whisker"), ("no_stim_trial", "P(lick), false alarm"),
            ("auditory_trial", "P(lick), auditory"), ("dp", "Δ P(lick), W − FA"), ("dprime", "d′, W vs FA")]
ALIGN = {"first_whisker_trial": "1st whisker trial", "first_whisker_hit": "1st whisker hit"}
XAXES = {"trial": ("Trials from {}", (-10, 30)), "minute": ("Time from {} (min)", (-10, 30)),
         "norm": ("Session time from {} (norm.)", (-0.3, 0.85))}
TRIAL_EDGES = np.r_[-10, -5, 0, 1, np.arange(6, 32, 5)]
TIME_BIN = {"minute": 2.0, "norm": 0.05}
CURVE_GRID = {"minute": np.round(np.arange(-10, 30.01, 0.25), 3), "norm": np.round(np.arange(-0.3, 0.851, 0.01), 3)}
ROWS = ["single trials", "binned", "learning curves"]
YLABEL = {"dp": "Δ P(lick)", "dprime": "d′"}  # P(lick) otherwise (user 2026-11-17: y labels)


def color(tt, cohort):
    if tt == "auditory_trial":
        return COLORS["auditory"] if cohort == "R+" else light(COLORS["auditory"], 0.5)
    if tt == "no_stim_trial":
        return COLORS["no_stim"] if cohort == "R+" else "0.6"
    return COLORS[cohort]


def zclip(p):
    return stats.norm.ppf(np.clip(p, 0.01, 0.99))


def session_rows(st):
    """All values of one session: rows (alignment, xaxis, rep, measure, x, value)."""
    st = st.sort_values("start_time")
    out = lce.session_learning_curves_exact(st, sigma=fig1.CURVE_PARAMS["sigma"], conf_int=fig1.CURVE_PARAMS["conf_int"])
    if "whisker_trial" not in out or not (np.asarray(out["whisker_trial"]["outcomes"]) == 1).any():
        return []
    w = out["whisker_trial"]
    tw = np.asarray(w["start_time"], float)
    span = tw[-1] - tw[0]
    ref = {"first_whisker_trial": tw[0], "first_whisker_hit": tw[int(np.argmax(np.asarray(w["outcomes"]) == 1))]}
    typ = {tt: (np.asarray(out[tt]["start_time"], float), np.asarray(out[tt]["outcomes"], float),
                np.asarray(out[tt]["p_mean"], float)) for tt in TT if tt in out}
    rows = []

    def add(align, xaxis, rep, measure, x, v):
        rows.append(pd.DataFrame({"alignment": align, "xaxis": xaxis, "rep": rep, "measure": measure,
                                  "x": np.asarray(x, float), "value": np.asarray(v, float)}))

    for align, t0 in ref.items():
        # ---- trial axis
        for tt, (t, y, pm) in typ.items():
            rel = np.arange(len(y)) - int(np.sum(t < t0))
            k = (rel >= -10) & (rel <= 30)
            add(align, "trial", "single trials", tt, rel[k], y[k])
            b = np.searchsorted(TRIAL_EDGES, rel[k], side="right") - 1
            ok = (b >= 0) & (b < len(TRIAL_EDGES) - 1)
            g = pd.Series(y[k][ok]).groupby(b[ok]).mean()
            add(align, "trial", "binned", tt, (TRIAL_EDGES[g.index] + TRIAL_EDGES[g.index + 1] - 1) / 2, g.values)
            kc = k & ((rel >= 0) if tt == "whisker_trial" and align == "first_whisker_hit" else True)
            add(align, "trial", "learning curves", tt, rel[kc], pm[kc])
        if "no_stim_trial" in typ:
            tf, yf, pf = typ["no_stim_trial"]
            relw = np.arange(len(tw)) - int(np.sum(tw < t0))
            yw, pw = typ["whisker_trial"][1], typ["whisker_trial"][2]
            k = (relw >= -10) & (relw <= 30)
            pf_w = np.interp(tw, tf, pf)  # false-alarm curve at the whisker trials
            add(align, "trial", "learning curves", "dp", relw[k], (pw - pf_w)[k])
            add(align, "trial", "learning curves", "dprime", relw[k], (zclip(pw) - zclip(pf_w))[k])
            b = np.searchsorted(TRIAL_EDGES, relw, side="right") - 1
            for bi in np.unique(b[(b >= 0) & (b < len(TRIAL_EDGES) - 1)]):
                sel = b == bi
                lo, hi = tw[sel].min(), tw[sel].max()
                f = yf[(tf >= lo - 10) & (tf <= hi + 10)]  # FA trials in the bin's span (+- 10 s)
                if len(f) == 0:
                    continue
                xw = (TRIAL_EDGES[bi] + TRIAL_EDGES[bi + 1] - 1) / 2
                add(align, "trial", "binned", "dp", [xw], [yw[sel].mean() - f.mean()])
                hr, fr = (yw[sel].sum() + 0.5) / (sel.sum() + 1), (f.sum() + 0.5) / (len(f) + 1)
                add(align, "trial", "binned", "dprime", [xw], [stats.norm.ppf(hr) - stats.norm.ppf(fr)])
        # ---- time axes
        for xaxis, scale in (("minute", 60.0), ("norm", span)):
            xr = XAXES[xaxis][1]
            edges = np.arange(xr[0], xr[1] + 1e-9, TIME_BIN[xaxis])
            grid = CURVE_GRID[xaxis]
            vals = {}
            for tt, (t, y, pm) in typ.items():
                x = (t - t0) / scale
                b = np.searchsorted(edges, x, side="right") - 1
                ok = (b >= 0) & (b < len(edges) - 1)
                g = pd.DataFrame({"b": b[ok], "y": y[ok]}).groupby("b").y.agg(["mean", "sum", "size"])
                vals[tt] = g
                add(align, xaxis, "binned", tt, edges[g.index] + TIME_BIN[xaxis] / 2, g["mean"].values)
                inside = (grid >= x.min()) & (grid <= x.max())
                if tt == "whisker_trial" and align == "first_whisker_hit":
                    inside &= grid >= 0
                add(align, xaxis, "learning curves", tt, grid[inside], np.interp(grid[inside], x, pm))
            if "no_stim_trial" in vals:
                gw, gf = vals["whisker_trial"], vals["no_stim_trial"]
                both = gw.index.intersection(gf.index)
                add(align, xaxis, "binned", "dp", edges[both] + TIME_BIN[xaxis] / 2,
                    gw.loc[both, "mean"].values - gf.loc[both, "mean"].values)
                hr = (gw.loc[both, "sum"] + 0.5) / (gw.loc[both, "size"] + 1)
                fr = (gf.loc[both, "sum"] + 0.5) / (gf.loc[both, "size"] + 1)
                add(align, xaxis, "binned", "dprime", edges[both] + TIME_BIN[xaxis] / 2,
                    stats.norm.ppf(hr.values) - stats.norm.ppf(fr.values))
                xw, xf = (tw - t0) / scale, (typ["no_stim_trial"][0] - t0) / scale
                inside = (grid >= max(xw.min(), xf.min())) & (grid <= min(xw.max(), xf.max()))
                cw = np.interp(grid[inside], xw, typ["whisker_trial"][2])
                cf = np.interp(grid[inside], xf, typ["no_stim_trial"][2])
                add(align, xaxis, "learning curves", "dp", grid[inside], cw - cf)
                add(align, xaxis, "learning curves", "dprime", grid[inside], zclip(cw) - zclip(cf))
    return rows


def build_table():
    learners = set(fig1.day0_table(shared_behaviour.load(), learners_only=True).mouse_id)
    d0 = shared_behaviour.load_untrimmed_day0()
    d0 = d0[d0.mouse_id.isin(learners) & (d0.early_lick == 0)]
    parts = []
    for sid, st in d0.groupby("session_id"):
        r = session_rows(st)
        if r:
            parts.append(pd.concat(r, ignore_index=True).assign(mouse_id=st.mouse_id.iloc[0], session_id=sid,
                                                                cohort=st.cohort.iloc[0]))
    return pd.concat(parts, ignore_index=True)


def page(pdf, df, align, xaxis):
    fig, axes = plt.subplots(3, 5, figsize=(183 * MM, 120 * MM))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.08, wspace=0.45, hspace=0.55)
    xlabel, xr = XAXES[xaxis]
    for r, rep in enumerate(ROWS):
        for c, (m, mlabel) in enumerate(MEASURES):
            a = axes[r][c]
            if r == 0:
                a.set_title(mlabel, fontsize=5.5)
            if c == 0:
                a.text(-0.55, 0.5, rep.capitalize(), transform=a.transAxes, rotation=90, ha="center", va="center",
                       fontsize=6, fontweight="bold")
            d = df[(df.alignment == align) & (df.xaxis == xaxis) & (df.rep == rep) & (df.measure == m)]
            if not len(d):
                a.text(0.5, 0.5, "n/a\n(needs bins)" if rep == "single trials" else "n/a", ha="center", va="center",
                       fontsize=5, color="0.5", transform=a.transAxes)
                a.set_xticks([]), a.set_yticks([])
                for side in a.spines.values():
                    side.set_color("0.85")
                continue
            for cohort in ("R+", "R-"):
                col = color(m if m in TT else "whisker_trial", cohort)
                g = d[d.cohort == cohort].groupby(["mouse_id", "x"]).value.mean().groupby("x")
                mu, se = g.mean(), g.sem()
                cnt = g.size()
                keep = cnt >= 5  # at least 5 mice per point
                mu, se = mu[keep], se[keep]
                if rep == "learning curves":
                    a.fill_between(mu.index, mu - se, mu + se, color=light(col, 0.75), lw=0)
                    a.plot(mu.index, mu, color=col, lw=0.7)
                else:
                    a.errorbar(mu.index, mu, se, color=col, marker="o", ms=1.2 if rep == "single trials" else 1.8,
                               lw=0.4 if rep == "binned" else 0, elinewidth=0.4, capsize=0)
            a.axvline(0, color="0.6", lw=0.4, ls=":")
            a.axhline(0, color="0.85", lw=0.3)
            a.set_xlim(*xr)
            if m in TT:
                a.set_ylim(-0.03, 1.03)
            a.tick_params(labelsize=4.5)
            a.set_ylabel(YLABEL.get(m, "P(lick)"), fontsize=5, labelpad=1)
            if r == 2:
                a.set_xlabel(xlabel.format(ALIGN[align]), fontsize=4.8)
    n = df[df.alignment == align].groupby("cohort").mouse_id.nunique()
    fig.suptitle(f"Aligned to the {ALIGN[align]}; x = {xaxis_name(xaxis)}.  Learners, whisker day 0, R+ (green) n = "
                 f"{n.get('R+', 0)}, R− (magenta) n = {n.get('R-', 0)}; mean ± s.e.m. over mice (points with ≥ 5 mice)",
                 fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def xaxis_name(x):
    return {"trial": "trials of each type", "minute": "absolute time (min)", "norm": "normalised session time"}[x]


def cover(pdf):
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    fig.text(0.05, 0.95, "Figure 1 G-H variants: licking around the start of whisker learning (user 2026-11-16)",
             fontsize=8, fontweight="bold", va="top")
    fig.text(0.05, 0.88, __doc__.split("Output:")[0], fontsize=5.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    set_nature_rc()
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"aligned_variants_{datetime.date.today()}"
    if "--replot" in sys.argv:  # plot only, from the last saved table
        table = sorted(OUT.glob("aligned_variants_*_table.parquet"))[-1]
        df = pd.read_parquet(table)
        print("replot from", table)
    else:
        df = build_table()
        df.to_parquet(f"{stem}_table.parquet")
    with PdfPages(f"{stem}.pdf") as pdf:
        cover(pdf)
        for align in ALIGN:
            for xaxis in XAXES:
                page(pdf, df, align, xaxis)
    print(stem)


if __name__ == "__main__":
    main()
