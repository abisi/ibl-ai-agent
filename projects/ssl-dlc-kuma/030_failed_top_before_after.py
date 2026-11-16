"""The 5 failed top views re-run with the merged model (user 2026-11-10): before (NWB) vs after, and the direction of
protraction in the NWB whisker angle.

Sessions: MH009 0207 / 0208, MH025 0421 / 0422, MH027 0422 (top DLC failed with the old shuffle 5 best-540; setup 2,
camera upside down). After = CSPNeXt-M snapshot-330 (023), rebuilt through the converter exactly as 025 (smoothing,
orientation check, compute_kinematics_alt, real NWB timestamps / conversion) and read with dlc_helpers.
Figure 1 (failed_top_before_after): per session, a frame during whisking with the old (left) and new (right) top
keypoints and vectors; whisker angle aligned to trial start (W hit, A hit, CR; before dashed, after solid, each trial
baseline-subtracted over -1..0 s); % frames whose whisker_base and particle pass the dlc_helpers cutoffs; single
whisker trials before / after (baseline-subtracted).
Figure 2 (whisker_angle_direction): which way protraction moves the stored whisker angle (compute_kinematics_alt). Per
top video of 023 / 030 (new tracks), the angle vs the forward position of the particle relative to whisker_base
(projection on the session's median nose vector, mm): a positive slope means protraction (particle forward) increases
the angle. Frames at the 5th and 95th percentile of the angle in one whisking second, vectors drawn.
Outputs: DLCProjects/merged_training_2026-11-09/pilot_inference/failed_top/
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 030_failed_top_before_after.py
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ba = _load("ba", "025_pilot_before_after.py")
pv, ts, ps, dh = ba.pv, ba.ts_, ba.ps, ba.dh
OUT = pv.OUT / "failed_top"
FAILED = ["MH009_20250207_131953", "MH009_20250208_192403", "MH025_20250421_115741", "MH025_20250422_141134",
          "MH027_20250422_111013"]
BASE = (-1.0, 0.0)


def valid(v):
    return np.all([v.likelihood[l].to_numpy() >= dh.cutoff_for(l) for l in dh.likelihood_series("whisker_angle")], 0)


def baselined(m, t):
    return m - np.nanmean(m[:, (t >= BASE[0]) & (t < BASE[1])], 1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ps.set_nature_rc()
    rows, sess = [], []
    for session in FAILED:
        sdir = pv.PILOT / session
        new_h5 = next(sdir.glob("*top*_filtered.h5"), None)
        if new_h5 is None:
            print(session, "no new top h5 yet", flush=True)
            continue
        old_h5 = pv.old_file(session, "top")
        views = {}
        for name, h5 in (("old", old_h5), ("new", new_h5)):
            refs = ba.build_nwb_like(h5, session, "top", OUT / f"{session}_top_{name}.h5")
            views[name] = ba.load_view(OUT / f"{session}_top_{name}.h5", "top", session)
            print(session, name, refs, flush=True)
        tr = {n: {"angle": ba.series(v, "whisker_angle"), "speed": ba.series(v, "whisker_speed"),
                  "valid": 100 * valid(v).astype(float), "ts": v.timestamps} for n, v in views.items()}
        trials = ba.trials_of(session)
        for n in tr:
            rows.append({"session": session, "model": n, "pct_valid": float(tr[n]["valid"].mean()),
                         "angle_median": float(np.nanmedian(tr[n]["angle"][tr[n]["valid"] > 0]))
                         if (tr[n]["valid"] > 0).any() else np.nan})
        sess.append(dict(session=session, old_h5=old_h5, new_h5=new_h5, tr=tr, trials=trials))
    pd.DataFrame(rows).to_csv(OUT / "failed_top_summary.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string())

    # ---- figure 1
    fig, axs = plt.subplots(len(sess), 6, figsize=(183 * ps.MM, 34 * ps.MM * len(sess)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.2, 1.2, 1, 0.8, 1, 1]))
    for r, s in enumerate(sess):
        h = {"old": pv.wide(s["old_h5"]), "new": pv.wide(s["new_h5"])}
        tn = s["tr"]["new"]
        n = min(len(tn["angle"]), len(s["tr"]["old"]["angle"]))
        cand = np.where(np.isfinite(tn["angle"][:n]) & (tn["speed"][:n] > 500) & (tn["valid"][:n] > 0))[0]
        i = int(rng.choice(cand)) if len(cand) else n // 2
        img = ba.read_frame(ba.video_path(s["session"], s["new_h5"].name.split("DLC_")[0]), i)
        for k, name in enumerate(("old", "new")):
            ax = axs[r, k]
            if img is not None:
                ax.imshow(img)
            ba.draw_keypoints(ax, h[name], i, "top", ba.COLORS[name])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{'Before' if name == 'old' else 'After'}: {s['tr'][name]['valid'].mean():.0f}% frames valid",
                         fontsize=5, pad=1, color=ba.COLORS[name])
        axs[r, 0].set_ylabel(f"{s['session'][:14]}\nframe {i}", fontsize=5.5)
        ax = axs[r, 2]
        for oc, color in ba.OUTCOMES.items():
            ev = s["trials"].loc[s["trials"].outcome == oc, "start_time"].to_numpy()
            for name, ls in (("old", "--"), ("new", "-")):
                x = np.where(s["tr"][name]["valid"] > 0, s["tr"][name]["angle"], np.nan)
                t, m = dh.peth(x, s["tr"][name]["ts"][:len(x)], ev, window=ba.WINDOW)
                mu, se = ba.mean_sem(baselined(m, t))
                ax.plot(t, mu, color=color, ls=ls, lw=0.7 if name == "new" else 0.6)
                if name == "new":
                    ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.4); ax.axhline(0, color="0.7", lw=0.4)
        ax.set_ylabel("Δ whisker angle (deg)", fontsize=5)
        ax = axs[r, 3]
        ev = s["trials"].start_time.to_numpy()
        for name, ls in (("old", "--"), ("new", "-")):
            t, m = dh.peth(s["tr"][name]["valid"], s["tr"][name]["ts"][:len(s["tr"][name]["valid"])], ev,
                           window=ba.WINDOW)
            ax.plot(t, np.nanmean(m, 0), color=ba.COLORS[name], ls=ls, lw=0.7)
        ax.set_ylim(-3, 103); ax.set_ylabel("Frames valid (%)", fontsize=5)
        ev = s["trials"].loc[s["trials"].trial_type == "whisker_trial", "start_time"].to_numpy()
        for k, name in enumerate(("old", "new")):
            ax = axs[r, 4 + k]
            x = np.where(s["tr"][name]["valid"] > 0, s["tr"][name]["angle"], np.nan)
            t, m = dh.peth(x, s["tr"][name]["ts"][:len(x)], ev, window=ba.WINDOW)
            cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.85")
            im = ax.imshow(baselined(m, t), aspect="auto", cmap=cmap, vmin=-25, vmax=25, interpolation="none",
                           extent=(t[0], t[-1], len(ev), 0))
            ax.axvline(0, color="k", lw=0.4)
            ax.set_title(f"{'Before' if name == 'old' else 'After'}: whisker trials", fontsize=5, pad=1,
                         color=ba.COLORS[name])
        if r == 0:
            fig.colorbar(im, ax=axs[r, 5], fraction=0.06, pad=0.02, label="Δ angle (deg)").ax.tick_params(labelsize=4)
    for c in (2, 3, 4, 5):
        axs[-1, c].set_xlabel("Time from trial start (s)")
    handles = [plt.Line2D([], [], color=c, label=o) for o, c in ba.OUTCOMES.items()] + \
              [plt.Line2D([], [], color="k", ls="--", label="before (NWB)"),
               plt.Line2D([], [], color="k", ls="-", label="after (merged model)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"failed_top_before_after.{ext}", dpi=220)
    plt.close(fig)

    # ---- figure 2: direction of protraction in the stored angle
    tops = sorted(pv.PILOT.glob("*/*_top *_filtered.h5"))
    fig, axs = plt.subplots(2, max(len(tops), 2), figsize=(183 * ps.MM, 75 * ps.MM), squeeze=False)
    drows = []
    for c, h5 in enumerate(tops):
        d = pv.wide(h5)
        g = lambda kp, cc: d[(kp, cc)].to_numpy().astype(float)  # noqa: E731
        ok = (g("particle", "likelihood") >= 0.5) & (g("whisker_base", "likelihood") >= 0.5)
        nose = (g("nose_tip", "likelihood") >= 0.5) & (g("nose_base", "likelihood") >= 0.5)
        n = np.array([np.median((g("nose_tip", "x") - g("nose_base", "x"))[nose]),
                      np.median((g("nose_tip", "y") - g("nose_base", "y"))[nose])])
        u = n / np.hypot(*n)
        _, ang = pv.derived(d, "top")
        fwd = ((g("particle", "x") - g("whisker_base", "x")) * u[0] + (g("particle", "y") - g("whisker_base", "y")) * u[1])
        mm = pv.mm_per_px(h5.parent.name.split("_")[0], "top")
        sel = ok & np.isfinite(ang)
        idx = rng.choice(np.where(sel)[0], min(20000, sel.sum()), replace=False)
        slope = np.polyfit(fwd[idx] * mm, ang[idx], 1)[0]
        rr = np.corrcoef(fwd[idx], ang[idx])[0, 1]
        drows.append({"session": h5.parent.name, "slope_deg_per_mm_forward": slope, "r": rr})
        ax = axs[0, c]
        ax.hexbin(fwd[idx] * mm, ang[idx], gridsize=40, bins="log", cmap="Greys", mincnt=1, linewidths=0)
        ax.set_title(f"{h5.parent.name[:14]}\nr = {rr:.2f}", fontsize=5, pad=1)
        ax.set_xlabel("Particle forward of whisker base (mm)", fontsize=5)
        if c == 0:
            ax.set_ylabel("Whisker angle, NWB convention (deg)", fontsize=5)
        # 5th / 95th percentile frames within the most active second
        speed = np.abs(np.diff(ang, prepend=np.nan))
        best = max(range(0, len(ang) - 200, 200), key=lambda s0: np.nansum(speed[s0:s0 + 200]))
        seg = np.arange(best, best + 200)
        seg = seg[np.isfinite(ang[seg])]
        lo, hi = seg[np.argmin(ang[seg])], seg[np.argmax(ang[seg])]
        img = ba.read_frame(ba.video_path(h5.parent.name, h5.name.split("DLC_")[0]), hi)
        ax = axs[1, c]
        if img is not None:
            ax.imshow(img)
        for i, col in ((lo, "#2166ac"), (hi, "#b2182b")):
            x0, y0 = g("whisker_base", "x")[i], g("whisker_base", "y")[i]
            x1, y1 = g("particle", "x")[i], g("particle", "y")[i]
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
            ax.text(x1, y1, f" {ang[i]:.0f}°", color=col, fontsize=5)
        nb = np.array([np.nanmedian(g("nose_base", "x")[nose]), np.nanmedian(g("nose_base", "y")[nose])])
        ax.annotate("", nb + n, nb, arrowprops=dict(arrowstyle="->", color="w", lw=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("blue: lowest, red: highest angle\n(same second; white: nose vector)", fontsize=4.5, pad=1)
    for c in range(len(tops), axs.shape[1]):
        axs[0, c].axis("off"); axs[1, c].axis("off")
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"whisker_angle_direction.{ext}", dpi=220)
    pd.DataFrame(drows).to_csv(OUT / "whisker_angle_direction.csv", index=False)
    print(pd.DataFrame(drows).round(3).to_string())


if __name__ == "__main__":
    main()
