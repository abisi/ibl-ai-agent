"""Exploration (user 2026-11-09): the beh_correlates measures (behaviour_analysis/beh_correlates/
beh_correlates_lick_decoding.py) averaged over mice in normalised session time, next to the learning curves.

Per whisker trial of whisker day 0 (shared behaviour table, early licks dropped; video sessions as in shared_movements:
dual-view mice, data_utils.keep_video_data): posture (window mean) and movement (mean |frame-to-frame difference|) of
jaw distance, nose (side + top, dlc_helpers.nose_norm), whisker angle and pupil area, in the beh_correlates windows:
pre [-0.2, 0) s (pupil [-2, 0)), post [0, 1) s (pupil [0, 5)). DLC through ephys_utilities dlc_helpers
(shared_movements.dlc_signals: physical units, reference fix, cleaning, pupil QC), not NWB_reader.get_dlc_data_dict as in
the original script.
Each metric is z-scored within the session, then binned in 20 bins of normalised session time (whisker trial index /
(n - 1)); learning curves (exact whisker curve, chance curve, difference) binned the same way; mean +- s.e.m. over mice,
per cohort. Learners and all mice.
Output: Figures/_explore/movement_session_time_<date>_{learners,all}.{pdf,png} and the CSV.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_movement_session.py
"""

import datetime
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
import shared_movements as sm  # noqa: E402
from common import FIGURES_ROOT, REFERENCE_SHEET  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

N_BINS = 20
OUT = FIGURES_ROOT / "_explore"
BODYPARTS = {"jaw": "jaw_distance", "nose": "nose", "whisker": "whisker_angle", "pupil": "pupil_area"}
WINDOWS = {"pre": {"jaw": (-0.2, 0.0), "nose": (-0.2, 0.0), "whisker": (-0.2, 0.0), "pupil": (-2.0, 0.0)},
           "post": {"jaw": (0.0, 1.0), "nose": (0.0, 1.0), "whisker": (0.0, 1.0), "pupil": (0.0, 5.0)}}


def one_session(args):
    sid, trials = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = sm.dlc_signals(sid)
    except Exception as err:  # noqa: BLE001 - unreadable DLC: reported, skipped
        print("skip", sid, type(err).__name__, err)
        return None
    dh = sm.dlc_helpers()
    out = trials[["mouse_id", "session_id", "cohort", "start_time"]].copy()
    for bp, src in BODYPARTS.items():
        for win, w in WINDOWS.items():
            for kind in ("posture", "movement"):
                col = f"{bp}_{win}_{kind}"
                if src not in sig:
                    out[col] = np.nan
                    continue
                x, ts, _ = sig[src]
                out[col] = dh.window_metric(x, ts, out.start_time.to_numpy(), w[bp], kind=kind)
    return out


def main():
    from compute import session_curves  # exact curves (Figure 1)
    table = shared_behaviour.load()
    ref = pd.read_excel(REFERENCE_SHEET)
    dual = set(ref[ref.with_dual_view == 1].mouse_id)
    d0 = table[(table.behavior == "whisker") & (table.day == 0) & (table.early_lick == 0)
               & table.mouse_id.isin(dual)].sort_values("start_time")
    w = sm.data_utils().keep_video_data(d0[d0.trial_type == "whisker_trial"], views=("side", "top"))
    with Pool(16) as pool:
        parts = [p for p in pool.map(one_session, list(w.groupby("session_id"))) if p is not None]
    mv = pd.concat(parts, ignore_index=True)
    rows, ctx = [], {}
    metrics = [c for c in mv.columns if c.endswith(("posture", "movement"))]
    for sid, m in mv.groupby("session_id"):
        st = d0[d0.session_id == sid]
        curves, _, _ = session_curves(st, ctx)
        n = len(m)
        if len(curves) != n:
            print("length mismatch", sid, len(curves), n)
            continue
        b = np.minimum((np.arange(n) / max(n - 1, 1) * N_BINS).astype(int), N_BINS - 1)
        z = (m[metrics] - m[metrics].mean()) / m[metrics].std()
        vals = {**{c: z[c].to_numpy() for c in metrics}, "p_whisker": curves.p_mean.to_numpy(),
                "p_chance": curves.p_chance.to_numpy(), "p_diff": (curves.p_mean - curves.p_chance).to_numpy()}
        for k, v in vals.items():
            g = pd.Series(v).groupby(b).mean()
            rows.append(pd.DataFrame({"mouse_id": m.mouse_id.iloc[0], "session_id": sid, "cohort": m.cohort.iloc[0],
                                      "learning_category": ref.set_index("mouse_id").learning_category.get(m.mouse_id.iloc[0]),
                                      "measure": k, "bin": g.index, "value": g.to_numpy()}))
    df = pd.concat(rows, ignore_index=True)
    df["time_norm"] = (df["bin"] + 0.5) / N_BINS
    stem = OUT / f"movement_session_time_{datetime.date.today()}"
    df.to_csv(f"{stem}.csv", index=False)
    for name, sel in (("learners", df.learning_category.isin(["good", "moderate"])), ("all", df.index == df.index)):
        plot(df[sel], f"{stem}_{name}", name)
        plot_thirds(df[sel], f"{stem}_{name}_thirds", name)
    print(stem, df.groupby("cohort").mouse_id.nunique().to_dict())


def band(ax, d, color, ls="-"):
    g = d.groupby("time_norm").value
    m, s = g.mean(), g.sem()
    ax.plot(m.index, m, color=color, ls=ls, lw=0.8)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.25, lw=0)


def plot(df, stem, label):
    set_nature_rc()
    cols = [("pre", "posture"), ("pre", "movement"), ("post", "posture"), ("post", "movement")]
    fig, axs = plt.subplots(5, 4, figsize=(183 * MM, 190 * MM), sharex=True)
    for j, (meas, title) in enumerate((("p_whisker", "Whisker curve"), ("p_chance", "Chance (FA) curve"),
                                       ("p_diff", "Whisker − chance"))):
        a = axs[0, j]
        for c in ("R+", "R-"):
            band(a, df[(df.measure == meas) & (df.cohort == c)], COLORS[c])
        a.set_title(title, fontsize=6, pad=2)
        a.set_ylabel("P(lick)" if j == 0 else "", labelpad=1)
    n = df.groupby("cohort").mouse_id.nunique().to_dict()
    axs[0, 3].axis("off")
    axs[0, 3].text(0, 0.6, f"{label}\nR+ n = {n.get('R+', 0)}, R- n = {n.get('R-', 0)} mice\nmetrics z-scored per session"
                   "\n20 bins of normalised session time\nmean ± s.e.m. over mice", fontsize=5.5, va="center")
    for r, bp in enumerate(BODYPARTS, start=1):
        for c, (win, kind) in enumerate(cols):
            a = axs[r, c]
            for co in ("R+", "R-"):
                band(a, df[(df.measure == f"{bp}_{win}_{kind}") & (df.cohort == co)], COLORS[co])
            a.axhline(0, color="0.6", lw=0.3, ls=":")
            if r == 1:
                a.set_title(f"{win}-stimulus {kind}", fontsize=6, pad=2)
            if c == 0:
                a.set_ylabel(f"{bp.capitalize()} (z)", labelpad=1)
    for a in axs[-1]:
        a.set_xlabel("Session time (norm.)", labelpad=1)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


def plot_thirds(df, stem, label):
    """Same measures as plot(), averaged per mouse within early / middle / late thirds of normalised session time."""
    set_nature_rc()
    TH = ["early", "middle", "late"]
    d = df.assign(third=np.array(TH)[np.minimum((df.time_norm * 3).astype(int), 2)])
    m = d.groupby(["mouse_id", "cohort", "measure", "third"]).value.mean().reset_index()
    cols = [("pre", "posture"), ("pre", "movement"), ("post", "posture"), ("post", "movement")]
    fig, axs = plt.subplots(5, 4, figsize=(183 * MM, 170 * MM))
    def dots(a, meas):
        for k, co in enumerate(("R+", "R-")):
            w = m[(m.measure == meas) & (m.cohort == co)].pivot(index="mouse_id", columns="third", values="value")
            w = w.reindex(columns=TH)
            a.errorbar(np.arange(3) + (k - 0.5) * 0.1, w.mean(), w.sem(), color=COLORS[co], marker="o", ms=2, lw=0.8,
                       capsize=0)
        a.set_xticks(range(3), ["Early", "Middle", "Late"])
        a.set_xlim(-0.4, 2.4)
    for j, (meas, title) in enumerate((("p_whisker", "Whisker curve"), ("p_chance", "Chance (FA) curve"),
                                       ("p_diff", "Whisker − chance"))):
        dots(axs[0, j], meas)
        axs[0, j].set_title(title, fontsize=6, pad=2)
    axs[0, 0].set_ylabel("P(lick)", labelpad=1)
    axs[0, 3].axis("off")
    n = df.groupby("cohort").mouse_id.nunique().to_dict()
    axs[0, 3].text(0, 0.5, f"{label}, thirds of normalised session time\nR+ n = {n.get('R+', 0)}, R- n = {n.get('R-', 0)}"
                   " mice\nmetrics z-scored per session\nmean ± s.e.m. over mice", fontsize=5.5, va="center")
    for r, bp in enumerate(BODYPARTS, start=1):
        for c, (win, kind) in enumerate(cols):
            dots(axs[r, c], f"{bp}_{win}_{kind}")
            axs[r, c].axhline(0, color="0.6", lw=0.3, ls=":")
            if r == 1:
                axs[r, c].set_title(f"{win}-stimulus {kind}", fontsize=6, pad=2)
            if c == 0:
                axs[r, c].set_ylabel(f"{bp.capitalize()} (z)", labelpad=1)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
